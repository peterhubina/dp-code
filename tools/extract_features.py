"""Extract UNI2-h features from all patched slides and save to a .pt file."""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from project.data.patch_dataset import PatchDataset, LABEL_MAP
from project.UNI.uni.get_encoder.get_encoder import get_encoder, get_eval_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Extract UNI2-h features from patch images")
    parser.add_argument("--patch_root", type=str, required=True,
                        help="Root directory containing slide subdirs with metadata.csv")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pt file path")
    parser.add_argument("--assets_dir", type=str, default=".scratch/checkpoints",
                        help="Directory containing UNI2-h model weights")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--encoder", type=str, default="uni2-h",
                        choices=["uni", "uni2-h"])
    return parser.parse_args()


def find_patch_dirs(patch_root):
    """Find all subdirectories of patch_root that contain a metadata.csv."""
    dirs = []
    for entry in sorted(os.listdir(patch_root)):
        candidate = os.path.join(patch_root, entry)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "metadata.csv")):
            dirs.append(candidate)
    return dirs


def main():
    args = parse_args()

    # Discover slide directories
    patch_dirs = find_patch_dirs(args.patch_root)
    if not patch_dirs:
        print(f"No patch directories found under {args.patch_root}")
        sys.exit(1)
    print(f"Found {len(patch_dirs)} slide directories: {[os.path.basename(d) for d in patch_dirs]}")

    # Load encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, eval_transform = get_encoder(
        enc_name=args.encoder,
        assets_dir=args.assets_dir,
        device=device,
    )

    # Create dataset with only in_situ and infiltrant classes
    classes = list(LABEL_MAP.keys())
    dataset = PatchDataset(patch_dirs, classes=classes, transform=eval_transform)
    print(f"Dataset: {len(dataset)} samples")

    if len(dataset) == 0:
        print("No samples found matching target classes. Exiting.")
        sys.exit(1)

    # Custom collate to handle slide_id strings
    def collate_fn(batch):
        imgs = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        slide_ids = [b[2] for b in batch]
        return imgs, labels, slide_ids

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Extract features
    all_embeddings = []
    all_labels = []
    all_slide_ids = []

    print("Extracting features...")
    with torch.inference_mode():
        for batch_idx, (imgs, labels, slide_ids) in enumerate(dataloader):
            imgs = imgs.to(device)
            features = model(imgs)  # [B, D]
            all_embeddings.append(features.cpu())
            all_labels.append(labels)
            all_slide_ids.extend(slide_ids)

            if (batch_idx + 1) % 10 == 0:
                n_done = (batch_idx + 1) * args.batch_size
                print(f"  Processed {min(n_done, len(dataset))}/{len(dataset)} samples")

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {torch.bincount(labels).tolist()}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "embeddings": embeddings,
        "labels": labels,
        "slide_ids": all_slide_ids,
        "label_map": LABEL_MAP,
    }
    torch.save(output, args.output)
    print(f"Saved features to {args.output}")


if __name__ == "__main__":
    main()
