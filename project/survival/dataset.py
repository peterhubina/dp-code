"""
Survival dataset for loading pre-extracted H5 patch embeddings with clinical labels.

Handles patient-level indexing (one patient may have multiple slides),
discrete survival time binning, and H5 file I/O.
"""

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SurvivalDataset(Dataset):
    """Patient-level survival dataset loading UNI2-h embeddings from H5 files.

    Each item returns all patch embeddings for a patient (concatenated across slides)
    along with the discrete survival label, event time, and censorship status.

    Args:
        data_dir: Path to directory containing H5 embedding files.
        labels_df: DataFrame with columns [case_id, slide_id, h5_path, event_time,
                   event_status, disc_label].
        patient_ids: List of case_ids to include in this dataset split.
    """

    def __init__(self, data_dir, labels_df, patient_ids):
        self.data_dir = Path(data_dir)
        self.patient_ids = list(patient_ids)

        # Build patient -> slide mapping
        self.labels_df = labels_df[labels_df["case_id"].isin(self.patient_ids)].copy()
        self.patient_data = {}
        for case_id in self.patient_ids:
            patient_rows = self.labels_df[self.labels_df["case_id"] == case_id]
            # event_status: 1=event occurred, 0=censored
            # MCAT convention: censorship=1 means censored, censorship=0 means event
            event_status = float(patient_rows["event_status"].iloc[0])
            self.patient_data[case_id] = {
                "slide_paths": patient_rows["h5_path"].tolist(),
                "disc_label": int(patient_rows["disc_label"].iloc[0]),
                "event_time": float(patient_rows["event_time"].iloc[0]),
                "censorship": 1.0 - event_status,
            }

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        case_id = self.patient_ids[idx]
        data = self.patient_data[case_id]

        # Load and concatenate embeddings from all slides for this patient
        all_features = []
        for h5_path in data["slide_paths"]:
            with h5py.File(h5_path, "r") as f:
                features = f["features"][:]  # shape: (1, N, D) or (N, D)
                if features.ndim == 3:
                    features = features.squeeze(0)  # (N, D)
                all_features.append(torch.from_numpy(features).float())

        features = torch.cat(all_features, dim=0)  # (total_patches, D)

        return (
            features,
            torch.tensor(data["disc_label"], dtype=torch.long),
            torch.tensor(data["event_time"], dtype=torch.float),
            torch.tensor(data["censorship"], dtype=torch.float),
        )

    @staticmethod
    def prepare_labels(labels_csv, embeddings_dir, n_bins=4, eps=1e-6,
                       time_col="OS_TIME", status_col="OS_STATUS"):
        """Join clinical labels with available embeddings and create discrete survival bins.

        Supports any survival endpoint (OS, DFI, PFI, DSS) by specifying the
        appropriate time and status column names.

        Args:
            labels_csv: Path to CSV with at least [case_id, <time_col>, <status_col>].
            embeddings_dir: Path to directory containing H5 embedding files.
            n_bins: Number of discrete survival time bins.
            eps: Small value for bin edge adjustment.
            time_col: Column name for survival time (e.g. "OS_TIME", "DFI").
            status_col: Column name for event status (e.g. "OS_STATUS", "DFI_STATUS").

        Returns:
            labels_df: DataFrame with columns [case_id, slide_id, h5_path,
                       event_time, event_status, disc_label].
            bin_edges: Array of bin boundaries.
        """
        labels = pd.read_csv(labels_csv)

        # Scan H5 files and extract case_id + slide_id
        embeddings_dir = Path(embeddings_dir)
        h5_files = sorted(embeddings_dir.glob("*.h5"))
        slides = []
        for h5_path in h5_files:
            slide_id = h5_path.stem
            case_id = slide_id[:12]
            slides.append({"case_id": case_id, "slide_id": slide_id, "h5_path": str(h5_path)})
        slides_df = pd.DataFrame(slides)

        # Join: keep only patients with both labels and embeddings
        merged = slides_df.merge(
            labels[["case_id", time_col, status_col]], on="case_id", how="inner"
        )
        merged = merged.dropna(subset=[time_col, status_col])

        # Rename to generic columns
        merged = merged.rename(columns={time_col: "event_time", status_col: "event_status"})
        merged["event_status"] = merged["event_status"].astype(int)

        # Discrete survival binning (MCAT approach):
        # Compute bin edges from uncensored (event=1) patients only
        patients = merged.drop_duplicates("case_id")
        uncensored = patients[patients["event_status"] == 1]
        _, bin_edges = pd.qcut(uncensored["event_time"], q=n_bins, retbins=True, labels=False)
        bin_edges[-1] = patients["event_time"].max() + eps
        bin_edges[0] = patients["event_time"].min() - eps

        # Apply bins to all patients
        disc_labels = pd.cut(
            patients["event_time"], bins=bin_edges, labels=False, right=False, include_lowest=True
        )
        patient_bins = patients[["case_id"]].copy()
        patient_bins["disc_label"] = disc_labels.values.astype(int)

        merged = merged.merge(patient_bins, on="case_id", how="left")

        n_patients = merged["case_id"].nunique()
        n_events = int(patients["event_status"].sum())
        print(f"Prepared dataset: {n_patients} patients, {len(merged)} slides, "
              f"{n_events} events ({n_events/n_patients*100:.1f}%), endpoint={status_col}")
        print(f"Bin edges (days): {bin_edges}")
        print(f"Bin distribution:\n{merged.drop_duplicates('case_id')['disc_label'].value_counts().sort_index()}")

        return merged, bin_edges


def collate_survival(batch):
    """Collate function for SurvivalDataset.

    Since bag sizes vary, we cannot stack features into a single tensor.
    With batch_size=1, this simply unpacks the single item.
    For batch_size>1, returns a list of feature tensors.
    """
    features = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    event_times = torch.stack([item[2] for item in batch])
    censorships = torch.stack([item[3] for item in batch])

    if len(features) == 1:
        features = features[0]

    return features, labels, event_times, censorships
