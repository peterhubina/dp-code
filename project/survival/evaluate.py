"""
Post-training evaluation utilities for survival models.

Provides c-index computation on held-out data, Kaplan-Meier visualization,
and cross-fold result aggregation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored


def compute_cindex(model, dataloader, device):
    """Run inference and compute concordance index.

    Args:
        model: Trained survival model.
        dataloader: DataLoader yielding (features, label, event_time, censorship).
        device: torch.device.

    Returns:
        cindex: Concordance index value.
        results_df: DataFrame with per-patient predictions.
    """
    model.eval()
    all_risk = []
    all_events = []
    all_times = []

    with torch.no_grad():
        for features, label, event_time, censorship in dataloader:
            features = features.to(device)
            hazards, S, _, _ = model(features)
            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk.append(risk)
            all_events.append(censorship.numpy())
            all_times.append(event_time.numpy())

    all_risk = np.concatenate(all_risk)
    all_events = np.concatenate(all_events)
    all_times = np.concatenate(all_times)

    # event indicator: True = event occurred (not censored)
    event_indicator = (1 - all_events).astype(bool)
    cindex = concordance_index_censored(event_indicator, all_times, all_risk)[0]

    results_df = pd.DataFrame({
        "risk_score": all_risk,
        "event_time": all_times,
        "censorship": all_events,
        "event": event_indicator,
    })

    return cindex, results_df


def plot_kaplan_meier(results_df, output_path, title="Kaplan-Meier by Risk Group"):
    """Plot Kaplan-Meier curves stratified by median risk score.

    Args:
        results_df: DataFrame with columns [risk_score, event_time, censorship].
        output_path: Path to save the figure.
        title: Plot title.
    """
    try:
        from lifelines import KaplanMeierFitter
        import matplotlib.pyplot as plt
    except ImportError:
        print("lifelines/matplotlib not available, skipping KM plot.")
        return

    median_risk = results_df["risk_score"].median()
    low_risk = results_df[results_df["risk_score"] <= median_risk]
    high_risk = results_df[results_df["risk_score"] > median_risk]

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    # Low risk group
    kmf.fit(
        low_risk["event_time"],
        event_observed=(1 - low_risk["censorship"]).astype(bool),
        label="Low Risk",
    )
    kmf.plot_survival_function(ax=ax, ci_show=True)

    # High risk group
    kmf.fit(
        high_risk["event_time"],
        event_observed=(1 - high_risk["censorship"]).astype(bool),
        label="High Risk",
    )
    kmf.plot_survival_function(ax=ax, ci_show=True)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    ax.set_title(title)
    ax.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"KM plot saved to {output_path}")


def summarize_folds(results_dir):
    """Aggregate c-index results across folds.

    Args:
        results_dir: Directory containing fold_0/, fold_1/, etc. subdirectories.

    Returns:
        mean_cindex, std_cindex, per_fold_results dict
    """
    results_dir = Path(results_dir)
    fold_dirs = sorted(results_dir.glob("fold_*"))

    fold_results = {}
    for fold_dir in fold_dirs:
        csv_path = fold_dir / "training.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "cindex_val" in df.columns:
                best_cindex = df["cindex_val"].max()
                fold_idx = int(fold_dir.name.split("_")[1])
                fold_results[fold_idx] = best_cindex

    if fold_results:
        values = list(fold_results.values())
        mean_ci = np.mean(values)
        std_ci = np.std(values)
        print(f"Results from {len(fold_results)} folds:")
        for fold, ci in sorted(fold_results.items()):
            print(f"  Fold {fold}: {ci:.4f}")
        print(f"  Mean: {mean_ci:.4f} +/- {std_ci:.4f}")
        return mean_ci, std_ci, fold_results
    else:
        print("No fold results found.")
        return None, None, {}
