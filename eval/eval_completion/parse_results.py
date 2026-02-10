#!/usr/bin/env python3
"""
Parse evaluation results from lm-evaluation-harness and display as a table.

Usage:
    python parse_results.py [results_dir] [--csv output.csv] [--sort COLUMN]
    
Examples:
    python parse_results.py                                    # Default dir, ASCII table
    python parse_results.py evals_results/my_experiment        # Custom dir
    python parse_results.py --csv results.csv                  # Export to CSV
    python parse_results.py --sort humaneval_pass@1            # Sort by metric

If no results_dir is provided, defaults to 'evals_results/remasking_full_traj'
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def find_result_files(base_dir):
    """Find all results*.json files in the output directory."""
    result_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory '{base_dir}' does not exist.")
        return []
    
    # lm-eval saves results as: results_<timestamp>.json or results.json
    for results_file in base_path.rglob("results*.json"):
        # Skip samples files
        if "samples_" in results_file.name:
            continue
        result_files.append(results_file)
    
    return sorted(result_files)


def extract_remasker_short_name(path):
    """Extract a short identifier from the remasker checkpoint path."""
    import re
    
    if not path or path == "None":
        return "none"
    
    # Extract the checkpoint directory name and step
    # e.g., ".../remasker-training-open-dcoder-0.5B-layers12-lr1e-5-...-confidence_conditioning/step_5000"
    parts = path.rstrip("/").split("/")
    
    # Get step number if present
    step = ""
    if parts and parts[-1].startswith("step_"):
        step = parts[-1]
        parts = parts[:-1]
    
    # Get the checkpoint name (last directory)
    if parts:
        ckpt_name = parts[-1]
        # Extract key identifiers from the name
        identifiers = []
        
        # Conditioning options
        has_time = "time_conditioning" in ckpt_name
        has_conf = "confidence_conditioning" in ckpt_name
        if has_time and has_conf:
            identifiers.append("time+conf")
        elif has_time:
            identifiers.append("time")
        elif has_conf:
            identifiers.append("conf")
        
        # Hidden states
        if "no_hidden_states" in ckpt_name:
            identifiers.append("no_hs")
        
        # Several steps
        match = re.search(r'several_steps(\d+)', ckpt_name)
        if match:
            identifiers.append(f"nsteps{match.group(1)}")
        
        # Denoising params (t ranges)
        match = re.search(r'denoising-t([\d.]+)-t([\d.]+)', ckpt_name)
        if match:
            identifiers.append(f"t{match.group(1)}-{match.group(2)}")
        
        if identifiers:
            short_name = "_".join(identifiers)
        else:
            # Fallback: use last 40 chars of the name
            short_name = ckpt_name[-40:] if len(ckpt_name) > 40 else ckpt_name
        
        if step:
            short_name = f"{short_name}/{step}"
        
        return short_name
    
    return path[-50:] if len(path) > 50 else path


def extract_params_from_config(data):
    """Extract hyperparameters from the config and configs sections."""
    params = {}
    
    # Get config section
    config = data.get("config", {})
    
    # Parse model_args string from config
    model_args = config.get("model_args", "")
    if model_args:
        for arg in model_args.split(","):
            if "=" in arg:
                key, value = arg.split("=", 1)
                params[key.strip()] = value.strip()
    
    # Get seed from config (random_seed, numpy_seed, or torch_seed)
    for seed_key in ["random_seed", "numpy_seed", "torch_seed", "seed"]:
        if seed_key in config:
            params["seed"] = config[seed_key]
            break
    
    # Also try to extract from configs.<task>.metadata (more structured format)
    configs = data.get("configs", {})
    for task_name, task_config in configs.items():
        metadata = task_config.get("metadata", {})
        for key, value in metadata.items():
            if key not in params:  # Don't override existing params
                params[key] = value
    
    # Create a short name for remasker_checkpoint_path
    if "remasker_checkpoint_path" in params:
        params["remasker"] = extract_remasker_short_name(params["remasker_checkpoint_path"])
    
    return params


def get_remasker_display_info(result):
    """Extract remasker display name and optimization steps from result.
    
    - source == "backbone" → name includes "backbone:", opt_steps from path
    - source == "model" → name is the checkpoint directory name, opt_steps from path
    
    Returns:
        (name, opt_steps): e.g., ("eff_bs256-init_from_backbone-denoising-t0.95-t0.05-time_cond", "12k")
    """
    import re
    
    source = result.get("remasking_logits_source", "model")
    ckpt_path = result.get("remasker_checkpoint_path", "") or ""
    
    # Extract optimization steps from checkpoint path (e.g. /step_5000 → "5k")
    step_match = re.search(r'/step_(\d+)', ckpt_path)
    if step_match:
        steps_num = int(step_match.group(1))
        if steps_num >= 1000:
            opt_steps_str = f"{steps_num // 1000}k"
        else:
            opt_steps_str = str(steps_num)
    else:
        opt_steps_str = ""
    
    # Extract the checkpoint directory name (strip common prefix)
    parts = ckpt_path.rstrip("/").split("/")
    # Remove step_* suffix to get directory name
    if parts and parts[-1].startswith("step_"):
        parts = parts[:-1]
    
    if parts:
        ckpt_dir = parts[-1]
        # Strip common prefix "remasker-training-open-dcoder-0.5B-layers12-lr1e-5-"
        # to keep only the distinctive part
        prefix_match = re.match(r'remasker-training-[^-]+-[\d.]+B-layers\d+-lr[\de-]+-(?:bs\d+-ga\d+-rand[\d.]+-rep[\d.]+-ls[\d.]+-)?', ckpt_dir)
        if prefix_match:
            name = ckpt_dir[prefix_match.end():]
        else:
            name = ckpt_dir
    else:
        name = result.get("remasker", "unknown")
    
    # Prefix with source if backbone
    if source == "backbone":
        name = f"backbone: {name}"
    
    return name, opt_steps_str


def _fmt_param(val):
    """Format a numeric parameter, removing unnecessary trailing zeros."""
    if val is None or val == "":
        return ""
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return f"{f:g}"
    except (ValueError, TypeError):
        return str(val)


def parse_result_file(filepath):
    """Parse a single results*.json file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not parse {filepath}: {e}")
        return None
    
    result = {
        "file": str(filepath),
    }
    
    # Extract config/hyperparameters (pass full data for better extraction)
    params = extract_params_from_config(data)
    result.update(params)
    
    # Extract results - handle different lm-eval output formats
    results_data = data.get("results", {})
    for task_name, task_results in results_data.items():
        if isinstance(task_results, dict):
            for metric_name, metric_value in task_results.items():
                # Skip non-metric fields
                if metric_name in ["alias"]:
                    continue
                if isinstance(metric_value, (int, float)):
                    # Clean up metric name (remove ,create_test, ,none suffixes)
                    clean_name = metric_name
                    for suffix in [",create_test", ",none"]:
                        clean_name = clean_name.replace(suffix, "")
                    result[f"{task_name}_{clean_name}"] = metric_value
    
    return result


def get_display_columns(results):
    """Get the columns to display in order."""
    # Default columns to display (most relevant hyperparameters + metrics)
    default_display = [
        "seed",
        "remasking_logits_source",
        "remasker",
        "steps",
        "temperature", 
        "remasking_schedule",
        "remasking_t_on",
        "remasking_t_off",
        "remasking_alpha_on",
        "remasking_temperature",
        "non_remasking_sampling_algorithm",
    ]
    
    # Find all metric columns (anything with task name prefix)
    metric_cols = set()
    for r in results:
        for key in r.keys():
            if "_pass@" in key or "_exact_match" in key or any(m in key for m in ["humaneval", "mbpp"]):
                metric_cols.add(key)
    
    # Build columns list
    columns = [c for c in default_display if any(c in r for r in results)]
    columns.extend(sorted(metric_cols))
    
    return columns


def format_table_pandas(results, display_columns=None):
    """Format results as a nice table using pandas."""
    columns = display_columns or get_display_columns(results)
    
    # Filter results to only include display columns
    filtered_results = []
    for r in results:
        filtered = {k: r.get(k, "") for k in columns}
        filtered_results.append(filtered)
    
    df = pd.DataFrame(filtered_results)
    
    # Format float columns
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    
    print(df.to_string(index=False))


def format_table_ascii(results, display_columns=None):
    """Format results as a nice ASCII table (fallback without pandas)."""
    columns = display_columns or get_display_columns(results)
    
    # Calculate column widths
    col_widths = {}
    for col in columns:
        max_width = len(col)
        for r in results:
            val = r.get(col, "")
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            max_width = max(max_width, len(val_str))
        col_widths[col] = min(max_width, 25)  # Cap at 25 chars
    
    # Print header
    header = " | ".join(col.ljust(col_widths[col])[:col_widths[col]] for col in columns)
    separator = "-+-".join("-" * col_widths[col] for col in columns)
    
    print(separator)
    print(header)
    print(separator)
    
    # Print rows
    for r in results:
        row_values = []
        for col in columns:
            val = r.get(col, "")
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            row_values.append(val_str.ljust(col_widths[col])[:col_widths[col]])
        print(" | ".join(row_values))
    
    print(separator)


def format_table(results, display_columns=None):
    """Format results as a nice table."""
    if not results:
        print("No results found.")
        return
    
    if HAS_PANDAS:
        format_table_pandas(results, display_columns)
    else:
        format_table_ascii(results, display_columns)


def print_summary_stats(results, summary_csv_path=None):
    """Print summary in pivoted format with steps as column groups.
    
    Output format matches the spreadsheet layout:
      remasker | opt_steps | T | t_on | t_off | 128 steps (pass@1,std,n) | 64 steps | ...
    
    Rows are grouped by (remasker, opt_steps). Within each group, rows
    vary by (T, t_on, t_off). The remasker/opt_steps columns are only
    printed on the first row of each group.
    """
    if not results:
        return
    
    # Find pass@1 metric columns (not stderr)
    metric_cols = []
    for r in results:
        for key in r.keys():
            if "_pass@" in key and "stderr" not in key and key not in metric_cols:
                metric_cols.append(key)
    
    if not metric_cols:
        return
    
    # Use first metric for display (usually humaneval_pass@1)
    primary_metric = metric_cols[0]
    
    # Detect step values from data (sorted descending)
    step_set = set()
    for r in results:
        s = r.get("steps")
        if s is not None:
            try:
                step_set.add(int(float(s)))
            except (ValueError, TypeError):
                pass
    step_values = sorted(step_set, reverse=True) if step_set else [128, 64, 32, 16, 8]
    
    # Enrich results with display info
    for r in results:
        name, opt = get_remasker_display_info(r)
        r["_display_name"] = name
        r["_opt_steps"] = opt
    
    # Group results by (name, opt, T, t_on, t_off, step) and compute stats
    groups = defaultdict(list)
    remasker_order = []
    remasker_set = set()
    
    for r in results:
        try:
            step_val = int(float(r.get("steps", 0)))
        except (ValueError, TypeError):
            step_val = 0
        key = (
            r["_display_name"],
            r["_opt_steps"],
            _fmt_param(r.get("temperature", "")),
            _fmt_param(r.get("remasking_t_on", "")),
            _fmt_param(r.get("remasking_t_off", "")),
            step_val,
        )
        groups[key].append(r)
        
        rg = (r["_display_name"], r["_opt_steps"])
        if rg not in remasker_set:
            remasker_order.append(rg)
            remasker_set.add(rg)
    
    # Compute stats: mean, std, n for each group
    stats = {}
    for key, group_results in groups.items():
        metric_stats = {}
        for metric in metric_cols:
            values = [r.get(metric) for r in group_results
                      if metric in r and r.get(metric) is not None]
            if values:
                mean_val = sum(values) / len(values)
                std_val = 0.0
                if len(values) > 1:
                    std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
                metric_stats[metric] = (mean_val, std_val, len(values))
        stats[key] = metric_stats
    
    # Collect unique (T, t_on, t_off) combos per remasker group
    param_combos = defaultdict(set)
    for key in stats:
        name, opt, T, t_on, t_off, step = key
        param_combos[(name, opt)].add((T, t_on, t_off))
    
    # --- Print table ---
    print("\n" + "=" * 70)
    print("SUMMARY (averaged over seeds)")
    print("=" * 70)
    
    # Compute dynamic width for remasker name column
    max_name_len = max((len(name) for name, _ in remasker_order), default=10)
    W_NAME = max(max_name_len + 2, 12)
    W_OPT = 12
    W_T = 5
    W_TON = 5
    W_TOFF = 6
    W_PASS = 8
    W_STD = 7
    W_N = 4
    
    # Header line 1: step groups
    fixed_width = W_NAME + W_OPT + W_T + W_TON + W_TOFF + 8  # 8 = separators
    h1 = " " * fixed_width
    for step in step_values:
        step_label = f"{step} steps"
        group_w = W_PASS + W_STD + W_N + 2
        h1 += f"  {step_label:^{group_w}}"
    
    # Header line 2: sub-column names
    h2 = (f"{'remasker':<{W_NAME}}  {'opt_steps':<{W_OPT}}  "
          f"{'T':>{W_T}}  {'t_on':>{W_TON}}  {'t_off':>{W_TOFF}}")
    for _ in step_values:
        h2 += f"  {'pass@1':>{W_PASS}} {'std':>{W_STD}} {'n':>{W_N}}"
    
    sep = "-" * len(h2)
    
    print()
    print(h1)
    print(h2)
    print(sep)
    
    # Print data rows grouped by remasker
    csv_rows = []
    for rg in remasker_order:
        name, opt = rg
        combos = sorted(param_combos.get(rg, set()),
                        key=lambda x: (float(x[0]) if x[0] else 0,
                                       float(x[1]) if x[1] else 0,
                                       float(x[2]) if x[2] else 0))
        
        first_row = True
        for T, t_on, t_off in combos:
            if first_row:
                row = f"{name:<{W_NAME}}  {opt:<{W_OPT}}"
                first_row = False
            else:
                row = f"{'':<{W_NAME}}  {'':<{W_OPT}}"
            
            row += f"  {T:>{W_T}}  {t_on:>{W_TON}}  {t_off:>{W_TOFF}}"
            
            csv_row = {
                "remasker": name,
                "opt_steps": opt,
                "T": T,
                "t_on": t_on,
                "t_off": t_off,
            }
            
            for step in step_values:
                key = (name, opt, T, t_on, t_off, step)
                if key in stats and primary_metric in stats[key]:
                    mean_val, std_val, n = stats[key][primary_metric]
                    pass_str = f"{mean_val * 100:.2f}%"
                    std_str = f"{std_val * 100:.2f}%"
                    n_str = str(n)
                    row += f"  {pass_str:>{W_PASS}} {std_str:>{W_STD}} {n_str:>{W_N}}"
                    csv_row[f"{step}_pass@1"] = mean_val
                    csv_row[f"{step}_std"] = std_val
                    csv_row[f"{step}_n"] = n
                else:
                    row += f"  {'':>{W_PASS}} {'':>{W_STD}} {'':>{W_N}}"
            
            print(row)
            csv_rows.append(csv_row)
        
        # Separator between remasker groups
        print(sep)
    
    # Export to CSV
    if summary_csv_path:
        csv_cols = ["remasker", "opt_steps", "T", "t_on", "t_off"]
        for step in step_values:
            csv_cols.extend([f"{step}_pass@1", f"{step}_std", f"{step}_n"])
        
        with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"\nSummary exported to: {summary_csv_path}")


def export_csv(results, output_file, columns=None):
    """Export results to a CSV file."""
    if not results:
        print("No results to export.")
        return
    
    # Determine all columns if not specified
    if columns is None:
        columns = set()
        for r in results:
            columns.update(r.keys())
        columns.discard("file")  # Don't include file path
        columns = sorted(columns)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse lm-evaluation-harness results and display as a table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python parse_results.py                                    # Default dir, exports summary.csv
    python parse_results.py evals_results/my_experiment        # Custom dir
    python parse_results.py --csv results.csv                  # Also export detailed results to CSV
    python parse_results.py --summary-csv custom.csv           # Custom summary CSV filename
    python parse_results.py --sort humaneval_pass@1            # Sort by metric (descending)
    python parse_results.py --no-summary                       # Skip summary stats and CSV
        """
    )
    parser.add_argument(
        "results_dir", 
        nargs="?", 
        default="evals_results/remasking_full_traj",
        help="Directory containing evaluation results (default: evals_results/remasking_full_traj)"
    )
    parser.add_argument(
        "--csv", 
        metavar="FILE",
        help="Export detailed results to CSV file"
    )
    parser.add_argument(
        "--summary-csv", 
        metavar="FILE",
        nargs="?",
        const="summary.csv",
        default="summary.csv",
        help="Export summary statistics (averaged over seeds) to CSV file (default: summary.csv)"
    )
    parser.add_argument(
        "--sort", 
        metavar="COLUMN",
        help="Sort results by column (descending for metrics, ascending for params)"
    )
    parser.add_argument(
        "--no-summary", 
        action="store_true",
        help="Skip printing summary statistics"
    )
    
    args = parser.parse_args()
    
    print(f"Searching for results in: {args.results_dir}")
    print()
    
    # Find all result files
    result_files = find_result_files(args.results_dir)
    
    if not result_files:
        print(f"No results.json files found in '{args.results_dir}'")
        print("\nMake sure you have run the evaluation script first:")
        print("  bash run_eval_remasking.sh")
        return 1
    
    print(f"Found {len(result_files)} result file(s)")
    print()
    
    # Parse all results
    results = []
    for rf in result_files:
        parsed = parse_result_file(rf)
        if parsed:
            results.append(parsed)
    
    if not results:
        print("No valid results could be parsed.")
        return 1
    
    # Sort results if requested
    if args.sort:
        try:
            # Determine sort direction: descending for metrics, ascending for params
            is_metric = "pass@" in args.sort or "exact_match" in args.sort
            results.sort(
                key=lambda x: (x.get(args.sort) is None, x.get(args.sort, 0)),
                reverse=is_metric
            )
        except Exception as e:
            print(f"Warning: Could not sort by '{args.sort}': {e}")
    
    # Export to CSV if requested
    if args.csv:
        export_csv(results, args.csv)
    
    # Display as table
    print("=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    format_table(results)
    
    # Print summary statistics (and export CSV unless --no-summary)
    if not args.no_summary:
        print_summary_stats(results, summary_csv_path=args.summary_csv)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
