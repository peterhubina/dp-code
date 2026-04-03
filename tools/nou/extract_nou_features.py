"""
Phase 3: UNI2-h Feature Extraction for NOU Dataset
===================================================
Extracts UNI2-h (1536-dim) patch-level embeddings for each NOU slide
and saves them in HDF5 format matching the TCGA-BRCA embedding structure.

Output format per slide:
  {slide_id}.h5 with datasets:
    - features: (1, N, 1536) float32
    - coords:   (1, N, 2) int64

This format is required by the CLAM pipeline (dataset_generic.py).
"""

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from project.UNI.uni.get_encoder.get_encoder import get_encoder


class SlidePatchDataset(Dataset):
    """Load all patches for a single slide from its metadata.csv."""

    def __init__(self, slide_dir, transform=None):
        self.slide_dir = slide_dir
        self.transform = transform
        meta_path = os.path.join(slide_dir, "metadata.csv")
        self.meta = pd.read_csv(meta_path)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        # path column contains "slide_id/patch_xN_yM.png"
        patch_name = os.path.basename(row["path"])
        img = Image.open(os.path.join(self.slide_dir, patch_name)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        x, y = int(row["x"]), int(row["y"])
        return img, x, y


def parse_args():
    parser = argparse.ArgumentParser(description="Extract UNI2-h features for NOU slides")
    parser.add_argument("--patch_dir", type=str,
                        default=".scratch/nou_validation/patches_448",
                        help="Root directory containing slide patch subdirs")
    parser.add_argument("--output_dir", type=str,
                        default=".scratch/nou_validation/features_448",
                        help="Output directory (h5_files/ and pt_files/ created inside)")
    parser.add_argument("--assets_dir", type=str, default=".scratch/checkpoints",
                        help="Directory containing UNI2-h model weights")
    parser.add_argument("--encoder", type=str, default="uni2-h",
                        choices=["uni", "uni2-h"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def find_slide_dirs(patch_dir):
    """Find slide directories that contain metadata.csv."""
    dirs = []
    for entry in sorted(os.listdir(patch_dir)):
        candidate = os.path.join(patch_dir, entry)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "metadata.csv")):
            dirs.append(candidate)
    return dirs


def extract_slide_features(model, slide_dir, transform, batch_size, num_workers, device):
    """Extract features for all patches in a slide directory."""
    dataset = SlidePatchDataset(slide_dir, transform=transform)
    if len(dataset) == 0:
        return None, None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_features = []
    all_coords = []

    with torch.inference_mode():
        for imgs, xs, ys in loader:
            imgs = imgs.to(device)
            features = model(imgs)  # [B, 1536]
            all_features.append(features.cpu())
            coords = torch.stack([xs, ys], dim=1)  # [B, 2]
            all_coords.append(coords)

    features = torch.cat(all_features, dim=0)  # [N, 1536]
    coords = torch.cat(all_coords, dim=0)      # [N, 2]
    return features, coords


def main():
    args = parse_args()

    h5_dir = os.path.join(args.output_dir, "h5_files")
    pt_dir = os.path.join(args.output_dir, "pt_files")
    os.makedirs(h5_dir, exist_ok=True)
    os.makedirs(pt_dir, exist_ok=True)

    # Find slides
    slide_dirs = find_slide_dirs(args.patch_dir)
    print(f"Found {len(slide_dirs)} slide directories")

    # Check which slides already have features (resume support)
    existing = {f.replace(".h5", "") for f in os.listdir(h5_dir) if f.endswith(".h5")}
    to_process = [d for d in slide_dirs if os.path.basename(d) not in existing]
    if existing:
        print(f"Skipping {len(existing)} already processed slides")
    print(f"Processing {len(to_process)} slides")

    if not to_process:
        print("All slides already processed.")
        return

    # Load encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading UNI2-h encoder...")
    model, eval_transform = get_encoder(
        enc_name=args.encoder,
        assets_dir=args.assets_dir,
        device=device,
    )

    for i, slide_dir in enumerate(to_process):
        slide_id = os.path.basename(slide_dir)

        features, coords = extract_slide_features(
            model, slide_dir, eval_transform,
            args.batch_size, args.num_workers, device,
        )

        if features is None:
            print(f"  [{i + 1}/{len(to_process)}] {slide_id}: SKIPPED (no patches)")
            continue

        n_patches = features.shape[0]

        # Save h5 matching TCGA format: features (1, N, 1536), coords (1, N, 2)
        h5_path = os.path.join(h5_dir, f"{slide_id}.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("features", data=features.numpy().reshape(1, n_patches, -1),
                             dtype="float32")
            f.create_dataset("coords", data=coords.numpy().reshape(1, n_patches, 2),
                             dtype="int64")

        # Save pt backup
        pt_path = os.path.join(pt_dir, f"{slide_id}.pt")
        torch.save(features, pt_path)

        if (i + 1) % 10 == 0 or (i + 1) == len(to_process):
            print(f"  [{i + 1}/{len(to_process)}] {slide_id}: {n_patches} patches, "
                  f"features {features.shape}")

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
