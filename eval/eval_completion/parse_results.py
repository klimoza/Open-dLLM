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


def print_summary_stats(results):
    """Print summary statistics grouped by key hyperparameters."""
    if not results:
        return
    
    # Find metric columns (only pass@1, not stderr)
    metric_cols = []
    for r in results:
        for key in r.keys():
            if "_pass@" in key and "stderr" not in key and key not in metric_cols:
                metric_cols.append(key)
    
    if not metric_cols:
        return
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS (averaged over seeds)")
    print("=" * 70)
    
    # Parameters to group by (excluding seed) - only include varying ones
    group_params = [
        "remasker", "steps", "temperature", "remasking_schedule", 
        "remasking_t_on", "remasking_t_off", "remasking_alpha_on",
        "remasking_temperature", "non_remasking_sampling_algorithm"
    ]
    
    # Group results
    groups = defaultdict(list)
    for r in results:
        key_parts = []
        for param in group_params:
            if param in r:
                key_parts.append((param, r[param]))
        
        key = tuple(key_parts)
        groups[key].append(r)
    
    # Find which params actually vary
    all_values = {p: set() for p in group_params}
    for r in results:
        for p in group_params:
            if p in r:
                all_values[p].add(str(r[p]))
    
    varying_params = [p for p in group_params if len(all_values[p]) > 1]
    
    # Build summary rows
    summary_rows = []
    for key, group_results in sorted(groups.items()):
        row = dict(key)
        for metric in metric_cols:
            values = [r.get(metric) for r in group_results if metric in r and r.get(metric) is not None]
            if values:
                mean_val = sum(values) / len(values)
                if len(values) > 1:
                    std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
                    row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    row[metric] = f"{mean_val:.4f}"
                row["n"] = len(values)
        summary_rows.append(row)
    
    # Print as table - only show varying params + metrics
    display_cols = varying_params + metric_cols + ["n"]
    
    # Calculate column widths
    col_widths = {}
    for col in display_cols:
        max_width = len(col)
        for row in summary_rows:
            val_str = str(row.get(col, ""))
            max_width = max(max_width, len(val_str))
        col_widths[col] = min(max_width + 1, 45)
    
    # Print header
    header = " | ".join(col.ljust(col_widths[col])[:col_widths[col]] for col in display_cols)
    separator = "-+-".join("-" * col_widths[col] for col in display_cols)
    
    print()
    print(separator)
    print(header)
    print(separator)
    
    # Print rows
    for row in summary_rows:
        row_values = []
        for col in display_cols:
            val = row.get(col, "")
            val_str = str(val)
            row_values.append(val_str.ljust(col_widths[col])[:col_widths[col]])
        print(" | ".join(row_values))
    
    print(separator)


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
    python parse_results.py                                    # Default dir, ASCII table
    python parse_results.py evals_results/my_experiment        # Custom dir
    python parse_results.py --csv results.csv                  # Export to CSV
    python parse_results.py --sort humaneval_pass@1            # Sort by metric (descending)
    python parse_results.py --no-summary                       # Skip summary statistics
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
        help="Export results to CSV file"
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
    
    # Print summary statistics
    if not args.no_summary:
        print_summary_stats(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
