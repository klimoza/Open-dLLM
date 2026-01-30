"""
HumanEval Results Parser
Parse and aggregate results from multiple seeds across different models
"""

import json
import re
from pathlib import Path
import pandas as pd
import numpy as np


def get_short_model_name(full_path):
    """Extract a meaningful short name from the model path.
    
    Handles patterns like:
    - HuggingFace: "fredzzp/open-dcoder-0.5B" -> "open-dcoder-0.5B"
    - Local checkpoint: ".../logs/.../experiment_name/checkpoints/global_step_N/..." 
      -> "experiment_name_stepNk"
    """
    # Check for local checkpoint path with step number
    # Pattern: .../logs/user/experiment_name/checkpoints/global_step_N/...
    checkpoint_match = re.search(r'/logs/[^/]+/([^/]+)/checkpoints/global_step_(\d+)', full_path)
    if checkpoint_match:
        experiment_name = checkpoint_match.group(1)
        step = int(checkpoint_match.group(2))
        # Format step as Nk (e.g., 2000 -> 2k, 10000 -> 10k)
        step_str = f"{step // 1000}k" if step >= 1000 else str(step)
        # Remove common prefixes to shorten the name
        short_exp = experiment_name
        for prefix in ["open_dcoder_0.5B_", "open_dcoder_"]:
            if short_exp.startswith(prefix):
                short_exp = short_exp[len(prefix):]
                break
        return f"{short_exp}_step{step_str}"
    
    # HuggingFace model (e.g., "fredzzp/open-dcoder-0.5B")
    if "/" in full_path and not full_path.startswith("/"):
        return full_path.split("/")[-1]
    
    # Fallback: return last path component
    return Path(full_path).name


def parse_results(results_dir):
    """Parse all results from the given directory."""
    results_dir = Path(results_dir)
    results_data = []

    for model_dir in results_dir.iterdir():
        if model_dir.is_dir():
            for results_file in model_dir.glob("results_*.json"):
                with open(results_file) as f:
                    data = json.load(f)

                full_model_name = data.get("model_name", model_dir.name)
                short_name = get_short_model_name(full_model_name)
                seed = data["config"]["random_seed"]
                pass_at_1 = data["results"]["humaneval"]["pass@1,create_test"]

                results_data.append({
                    "model": short_name,
                    "full_path": full_model_name,
                    "seed": seed,
                    "pass@1": pass_at_1
                })

    return pd.DataFrame(results_data)


def main():
    # Path to results directory
    RESULTS_DIR = Path("evals_results/humaneval-p2-improved")

    # Parse results
    df = parse_results(RESULTS_DIR)
    print(f"Loaded {len(df)} results\n")

    # Pivot table: models as rows, seeds as columns
    pivot_df = df.pivot_table(index="model", columns="seed", values="pass@1", aggfunc="mean")
    print("Pass@1 scores by model and seed:")
    print(pivot_df.round(4).to_string())
    print()

    # Compute mean and std across seeds for each model
    summary_df = df.groupby("model")["pass@1"].agg(["mean", "std", "count"])
    summary_df["mean"] = summary_df["mean"] * 100  # Convert to percentage
    summary_df["std"] = summary_df["std"] * 100
    summary_df = summary_df.round(2)
    summary_df.columns = ["Mean (%)", "Std (%)", "# Seeds"]
    summary_df = summary_df.sort_values("Mean (%)", ascending=False)

    # Print formatted summary
    print("=" * 80)
    print("HumanEval Pass@1 Results Summary")
    print("=" * 80)
    for model in summary_df.index:
        mean = summary_df.loc[model, "Mean (%)"]
        std = summary_df.loc[model, "Std (%)"]
        n = int(summary_df.loc[model, "# Seeds"])
        print(f"{model:40s}: {mean:.2f} ± {std:.2f} (n={n})")
    print("=" * 80)


if __name__ == "__main__":
    main()

