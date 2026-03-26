"""
Standalone evaluation script for trained survival models.

Loads checkpoints from all folds, computes c-index, and generates KM plots.

Usage:
    python tools/eval_survival.py --exp_dir .scratch/experiments/amil_surv_baseline/v1
"""

import argparse
from pathlib import Path

from project.survival.evaluate import summarize_folds


def main():
    parser = argparse.ArgumentParser(description="Evaluate survival model results")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to experiment directory containing fold_* subdirectories",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return

    mean_ci, std_ci, fold_results = summarize_folds(exp_dir)
    if mean_ci is not None:
        print(f"\nFinal: {mean_ci:.4f} +/- {std_ci:.4f}")


if __name__ == "__main__":
    main()
