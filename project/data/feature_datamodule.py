import torch
import numpy as np
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit


class FeatureDataModule:
    """Loads cached feature embeddings and creates patient-level train/val splits.

    The ``.pt`` file is expected to contain::

        {
            "embeddings": tensor [N, D],
            "labels": tensor [N],
            "slide_ids": list[str] of length N,
            "label_map": dict
        }

    Splitting is done at the **slide level**: all patches from a given slide
    appear in the same fold to prevent spatial-autocorrelation data leakage.
    When multiple slides exist, uses patient-level split. With single slide,
    falls back to stratified split on samples.
    """

    def __init__(self, features_path, batch_size=256, val_split=0.2, num_workers=4, seed=42):
        data = torch.load(features_path, map_location="cpu")
        self.embeddings = data["embeddings"]
        self.labels = data["labels"]
        self.slide_ids = data["slide_ids"]
        self.label_map = data["label_map"]
        self.num_features = self.embeddings.shape[1]

        # Patient-level stratified split
        groups = np.array(self.slide_ids)
        labels_np = self.labels.numpy()

        # Majority label per slide for stratification
        slide_majority = {}
        unique_slides = np.unique(groups)
        
        for sid in unique_slides:
            mask = groups == sid
            counts = Counter(labels_np[mask].tolist())
            slide_majority[sid] = counts.most_common(1)[0][0]
        strat_labels = np.array([slide_majority[sid] for sid in groups])

        
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
        train_idx, val_idx = next(splitter.split(self.embeddings, strat_labels, groups))

        train_idx = torch.tensor(train_idx, dtype=torch.long)
        val_idx = torch.tensor(val_idx, dtype=torch.long)

        # Compute class weights from training set only
        train_labels = self.labels[train_idx]
        class_counts = torch.bincount(train_labels.long())
        n_total = train_labels.shape[0]
        n_classes = len(class_counts)
        self.class_weights = n_total / (n_classes * class_counts.float())

        # Build datasets
        ds_train = TensorDataset(self.embeddings[train_idx], self.labels[train_idx])
        ds_val = TensorDataset(self.embeddings[val_idx], self.labels[val_idx])

        self.dataloader_train = DataLoader(
            ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.dataloader_val = DataLoader(
            ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Store split info for reporting
        train_slides = set(groups[train_idx.numpy()])
        val_slides = set(groups[val_idx.numpy()])
        self.train_slides = train_slides
        self.val_slides = val_slides

        print(f" > Features: {self.embeddings.shape[0]} samples, {self.num_features} dims")
        print(f" > Train: {len(train_idx)} samples ({len(train_slides)} slides)")
        print(f" > Val:   {len(val_idx)} samples ({len(val_slides)} slides)")
        print(f" > Class weights: {self.class_weights.tolist()}")
        overlap = train_slides & val_slides
        if overlap:
            print(f" WARNING: slide overlap in train/val: {overlap}")
        else:
            print(f" > No slide overlap between train and val (OK)")
