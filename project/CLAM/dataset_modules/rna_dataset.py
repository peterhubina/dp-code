from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


RNA_METADATA_COLUMNS = ("case_id", "sample", "label", "sample_type_code")


def read_rna_clam_table(csv_path: str | Path, label_dict: dict[str, int]):
    """Read the prepared TCGA-BRCA RNA table and return metadata, labels and features."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"RNA table not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_cols = [col for col in ("case_id", "sample", "label") if col not in df.columns]
    if missing_cols:
        raise ValueError(f"RNA table is missing required columns: {missing_cols}")

    df = df[df["label"].isin(label_dict)].copy()
    if df.empty:
        raise ValueError("No RNA rows remain after applying the label dictionary.")

    feature_cols = [col for col in df.columns if col not in RNA_METADATA_COLUMNS]
    if not feature_cols:
        raise ValueError("RNA table does not contain expression feature columns.")

    metadata = df[["case_id", "sample", "label"]].copy()
    if "sample_type_code" in df.columns:
        metadata["sample_type_code"] = df["sample_type_code"]

    metadata["label_idx"] = metadata["label"].map(label_dict).astype(np.int64)
    features = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    labels = metadata["label_idx"].to_numpy(dtype=np.int64)

    return metadata.reset_index(drop=True), features, labels, feature_cols


@dataclass
class RNAFeatureTransform:
    selected_idx: np.ndarray
    selected_feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(
        cls,
        x_train: np.ndarray,
        feature_names: list[str],
        top_n_genes: int = 0,
        eps: float = 1e-6,
    ) -> "RNAFeatureTransform":
        if top_n_genes and top_n_genes > 0 and top_n_genes < x_train.shape[1]:
            variances = np.nanvar(x_train, axis=0)
            variances = np.nan_to_num(variances, nan=-np.inf)
            selected_idx = np.argsort(variances)[::-1][:top_n_genes]
            selected_idx = np.sort(selected_idx)
        else:
            selected_idx = np.arange(x_train.shape[1])

        x_selected = x_train[:, selected_idx]
        mean = np.nanmean(x_selected, axis=0)
        std = np.nanstd(x_selected, axis=0)

        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
        std[std < eps] = 1.0

        selected_feature_names = [feature_names[i] for i in selected_idx.tolist()]
        return cls(
            selected_idx=selected_idx.astype(np.int64),
            selected_feature_names=selected_feature_names,
            mean=mean,
            std=std,
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_selected = x[:, self.selected_idx]
        x_scaled = (x_selected - self.mean) / self.std
        return np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class RNATabularDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        features: np.ndarray,
        labels: np.ndarray,
    ):
        if len(metadata) != len(features) or len(metadata) != len(labels):
            raise ValueError("metadata, features and labels must have the same length.")

        self.metadata = metadata.reset_index(drop=True)
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))
        self.sample_ids = self.metadata["sample"].astype(str).tolist()
        self.case_ids = self.metadata["case_id"].astype(str).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.sample_ids[idx], self.case_ids[idx]
