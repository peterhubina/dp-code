from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset_modules.dataset_generic import Generic_MIL_Dataset, Generic_Split
from dataset_modules.rna_dataset import RNAFeatureTransform


BASE_METADATA_COLUMNS = ("case_id", "sample", "label", "label_idx", "sample_type_code")


def read_tabular_feature_table(
    csv_path: str | Path,
    label_dict: dict[str, int],
    case_id_col: str = "case_id",
    label_col: str = "label",
):
    """Read a per-patient tabular table and return metadata plus raw features."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Tabular feature table not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_cols = [col for col in (case_id_col, label_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Tabular feature table is missing required columns: {missing_cols}")

    df = df.rename(columns={case_id_col: "case_id", label_col: "label"}).copy()
    if "sample" not in df.columns:
        df["sample"] = df["case_id"].astype(str)

    df = df[df["label"].isin(label_dict)].copy()
    if df.empty:
        raise ValueError("No tabular rows remain after applying the label dictionary.")

    metadata_cols = [col for col in BASE_METADATA_COLUMNS if col in df.columns]
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    if not feature_cols:
        raise ValueError("Tabular feature table does not contain feature columns.")

    metadata = df[[col for col in BASE_METADATA_COLUMNS if col in df.columns]].copy()
    metadata["case_id"] = metadata["case_id"].astype(str)
    metadata["sample"] = metadata["sample"].astype(str)
    metadata["label_idx"] = metadata["label"].map(label_dict).astype(np.int64)

    features = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    return metadata.reset_index(drop=True), features, feature_cols


@dataclass
class TabularFeatureStore:
    metadata: pd.DataFrame
    features: np.ndarray
    feature_names: list[str]
    transform: RNAFeatureTransform | None = None

    def __post_init__(self):
        metadata = self.metadata.reset_index(drop=True).copy()
        metadata["case_id"] = metadata["case_id"].astype(str)

        duplicated = metadata["case_id"].duplicated(keep="first")
        if duplicated.any():
            keep = ~duplicated.to_numpy()
            metadata = metadata.loc[keep].reset_index(drop=True)
            self.features = self.features[keep]

        self.metadata = metadata
        self.features = self.features.astype(np.float32, copy=False)
        self.case_ids = self.metadata["case_id"].astype(str).tolist()
        self.case_to_index = {case_id: idx for idx, case_id in enumerate(self.case_ids)}
        self.label_by_case = dict(
            zip(self.metadata["case_id"].astype(str), self.metadata["label_idx"].astype(int))
        )

    @property
    def available_cases(self) -> set[str]:
        return set(self.case_to_index)

    @property
    def input_dim(self) -> int:
        if self.transform is not None:
            return len(self.transform.selected_feature_names)
        return int(self.features.shape[1])

    def fit_transform(self, case_ids: pd.Series | list[str], top_n_features: int = 0) -> RNAFeatureTransform:
        unique_cases = pd.Series(case_ids, dtype=str).drop_duplicates()
        missing = sorted(set(unique_cases) - self.available_cases)
        if missing:
            raise ValueError(f"{len(missing)} training cases are missing tabular features. Examples: {missing[:5]}")

        indices = [self.case_to_index[case_id] for case_id in unique_cases]
        self.transform = RNAFeatureTransform.fit(
            self.features[indices],
            self.feature_names,
            top_n_genes=top_n_features,
        )
        return self.transform

    def set_transform(self, transform: RNAFeatureTransform) -> None:
        self.transform = transform

    def get(self, case_id: str) -> torch.Tensor:
        if self.transform is None:
            raise RuntimeError("Tabular transform has not been fitted for this fold.")

        idx = self.case_to_index[str(case_id)]
        features = self.transform.transform(self.features[idx : idx + 1])[0]
        return torch.from_numpy(features)

    def transform_dict(self) -> dict[str, object] | None:
        if self.transform is None:
            return None

        return {
            "selected_feature_names": self.transform.selected_feature_names,
            "selected_idx": self.transform.selected_idx.tolist(),
            "mean": self.transform.mean.tolist(),
            "std": self.transform.std.tolist(),
        }


class Generic_Multimodal_MIL_Dataset(Generic_MIL_Dataset):
    """MIL dataset that pairs each WSI bag with case-matched RNA/tabular features."""

    def __init__(
        self,
        data_dir,
        tabular_csv: str | Path,
        tabular_case_id_col: str = "case_id",
        tabular_label_col: str = "label",
        **kwargs,
    ):
        print_info = kwargs.get("print_info", True)
        kwargs["print_info"] = False

        label_dict = kwargs.get("label_dict")
        if label_dict is None:
            raise ValueError("label_dict is required for multimodal datasets.")

        metadata, features, feature_names = read_tabular_feature_table(
            tabular_csv,
            label_dict=label_dict,
            case_id_col=tabular_case_id_col,
            label_col=tabular_label_col,
        )
        self.tabular_store = TabularFeatureStore(metadata, features, feature_names)

        super().__init__(data_dir=data_dir, **kwargs)
        self._filter_to_tabular_cases()
        self.patient_data_prep()
        self.cls_ids_prep()

        if print_info:
            self.summarize()
            print(
                "multimodal matched cases: {}, tabular features: {}".format(
                    self.slide_data["case_id"].nunique(),
                    len(self.tabular_store.feature_names),
                )
            )

    def _filter_to_tabular_cases(self):
        slide_cases = self.slide_data["case_id"].astype(str)
        matched_mask = slide_cases.isin(self.tabular_store.available_cases)
        n_missing = int((~matched_mask).sum())
        if n_missing:
            print(f"Excluding {n_missing} WSI slides without matched tabular features.")

        slide_data = self.slide_data.loc[matched_mask].copy()
        if slide_data.empty:
            raise ValueError("No WSI slides have matched tabular features.")

        label_by_case = self.tabular_store.label_by_case
        mismatched = slide_data[
            slide_data.apply(lambda row: int(row["label"]) != label_by_case[str(row["case_id"])], axis=1)
        ]
        if not mismatched.empty:
            examples = mismatched[["case_id", "slide_id", "label"]].head().to_dict("records")
            raise ValueError(f"WSI and tabular labels disagree for matched cases. Examples: {examples}")

        self.slide_data = slide_data.reset_index(drop=True)

    def _make_split(self, slide_data: pd.DataFrame):
        if len(slide_data) == 0:
            return None
        return Generic_Multimodal_Split(
            slide_data.reset_index(drop=True),
            data_dir=self.data_dir,
            num_classes=self.num_classes,
            use_h5=self.use_h5,
            tabular_store=self.tabular_store,
        )

    def get_split_from_df(self, all_splits, split_key="train"):
        split = all_splits[split_key].dropna().reset_index(drop=True)
        if len(split) == 0:
            return None

        mask = self.slide_data["slide_id"].isin(split.tolist())
        return self._make_split(self.slide_data[mask])

    def get_merged_split_from_df(self, all_splits, split_keys=["train"]):
        merged_split = []
        for split_key in split_keys:
            split = all_splits[split_key].dropna().reset_index(drop=True).tolist()
            merged_split.extend(split)

        if len(merged_split) == 0:
            return None

        mask = self.slide_data["slide_id"].isin(merged_split)
        return self._make_split(self.slide_data[mask])

    def return_splits(self, from_id=True, csv_path=None):
        if not from_id:
            return super().return_splits(from_id=from_id, csv_path=csv_path)

        train_split = self._make_split(self.slide_data.loc[self.train_ids]) if len(self.train_ids) > 0 else None
        val_split = self._make_split(self.slide_data.loc[self.val_ids]) if len(self.val_ids) > 0 else None
        test_split = self._make_split(self.slide_data.loc[self.test_ids]) if len(self.test_ids) > 0 else None
        return train_split, val_split, test_split

    def __getitem__(self, idx):
        wsi_features, label = super().__getitem__(idx)
        case_id = self.slide_data["case_id"][idx]
        tabular_features = self.tabular_store.get(case_id)
        return (wsi_features, tabular_features), label


class Generic_Multimodal_Split(Generic_Split):
    def __init__(self, slide_data, data_dir=None, num_classes=2, use_h5=False, tabular_store=None):
        super().__init__(slide_data, data_dir=data_dir, num_classes=num_classes, use_h5=use_h5)
        if tabular_store is None:
            raise ValueError("tabular_store is required for a multimodal split.")
        self.tabular_store = tabular_store

    @property
    def tabular_feature_dim(self) -> int:
        return self.tabular_store.input_dim

    def fit_tabular_transform(self, top_n_features: int = 0) -> RNAFeatureTransform:
        return self.tabular_store.fit_transform(
            self.slide_data["case_id"].astype(str),
            top_n_features=top_n_features,
        )

    def set_tabular_transform(self, transform: RNAFeatureTransform) -> None:
        self.tabular_store.set_transform(transform)

    def tabular_transform_dict(self) -> dict[str, object] | None:
        return self.tabular_store.transform_dict()

    def __getitem__(self, idx):
        wsi_features, label = super().__getitem__(idx)
        case_id = self.slide_data["case_id"][idx]
        tabular_features = self.tabular_store.get(case_id)
        return (wsi_features, tabular_features), label
