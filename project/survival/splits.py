"""
Cross-validation split generation and loading for survival experiments.

Generates patient-level stratified splits ensuring balanced event/censored
ratios and consistent discrete label distribution across folds.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def generate_stratified_splits(labels_df, n_folds=5, seed=42):
    """Generate stratified k-fold splits at patient level.

    Stratifies on disc_label to ensure balanced survival bin distribution per fold.

    Args:
        labels_df: DataFrame with at least [case_id, disc_label] columns.
        n_folds: Number of cross-validation folds.
        seed: Random seed for reproducibility.

    Returns:
        List of (train_case_ids, val_case_ids) tuples.
    """
    patients = labels_df.drop_duplicates("case_id")[["case_id", "disc_label"]].copy()
    patients = patients.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in skf.split(patients["case_id"], patients["disc_label"]):
        train_ids = patients.iloc[train_idx]["case_id"].tolist()
        val_ids = patients.iloc[val_idx]["case_id"].tolist()
        splits.append((train_ids, val_ids))

    return splits


def save_splits(splits, output_dir):
    """Save splits to CSV files.

    Args:
        splits: List of (train_ids, val_ids) tuples.
        output_dir: Directory to save split CSVs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (train_ids, val_ids) in enumerate(splits):
        # Pad shorter list with empty strings
        max_len = max(len(train_ids), len(val_ids))
        train_padded = train_ids + [""] * (max_len - len(train_ids))
        val_padded = val_ids + [""] * (max_len - len(val_ids))

        df = pd.DataFrame({"train": train_padded, "val": val_padded})
        df.to_csv(output_dir / f"splits_{i}.csv", index=False)

    print(f"Saved {len(splits)} splits to {output_dir}")


def load_splits(splits_dir, fold_idx):
    """Load a single split from CSV.

    Args:
        splits_dir: Directory containing split CSVs.
        fold_idx: Index of fold to load.

    Returns:
        (train_case_ids, val_case_ids) tuple.
    """
    df = pd.read_csv(Path(splits_dir) / f"splits_{fold_idx}.csv")
    train_ids = df["train"].dropna().tolist()
    val_ids = df["val"].dropna().tolist()
    # Remove empty strings from padding
    train_ids = [x for x in train_ids if x]
    val_ids = [x for x in val_ids if x]
    return train_ids, val_ids
