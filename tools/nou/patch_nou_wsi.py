"""
Phase 2-WSI: NOU Whole-Slide Image Patching
=============================================
Tiles NOU SVS whole-slide images into non-overlapping 224x224 patches at
20x magnification using OpenSlide, with Otsu-based tissue segmentation.

The NOU SVS files are scanned at 20x (Aperio AT2, MPP ~0.504), matching
the TCGA-BRCA feature extraction protocol exactly.  This eliminates the
magnification mismatch present in the TIFF-crop pipeline.

Output: per-slide directories with patches + metadata CSVs compatible
with extract_nou_features.py.
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

try:
    import openslide
except ImportError:
    raise ImportError("openslide-python is required: pip install openslide-python")


def parse_args():
    parser = argparse.ArgumentParser(description="Patch NOU SVS WSIs into 224x224 tiles at 20x")
    parser.add_argument("--svs_dir", type=str, default=".datasets/nou/data",
                        help="Directory containing NOU SVS files")
    parser.add_argument("--manifest", type=str,
                        default=".scratch/nou_validation/metadata/nou_slide_manifest.csv",
                        help="Slide manifest for label lookup")
    parser.add_argument("--output_dir", type=str,
                        default=".scratch/nou_validation/patches_wsi")
    parser.add_argument("--patch_size", type=int, default=224,
                        help="Patch size in pixels at level 0 (native 20x)")
    parser.add_argument("--bg_threshold", type=int, default=230)
    parser.add_argument("--dark_threshold", type=int, default=10)
    parser.add_argument("--min_tissue", type=float, default=0.5)
    parser.add_argument("--thumb_scale", type=int, default=64,
                        help="Downscale factor for tissue mask thumbnail")
    return parser.parse_args()


def otsu_threshold(gray_img):
    hist, _ = np.histogram(gray_img.ravel(), bins=256, range=(0, 256))
    total = gray_img.size
    sum_total = np.dot(np.arange(256), hist)
    sum_bg = 0.0
    weight_bg = 0
    max_var = 0.0
    threshold = 0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def create_tissue_mask(slide, thumb_scale):
    """Create tissue mask from WSI thumbnail."""
    w, h = slide.dimensions
    tw, th = w // thumb_scale, h // thumb_scale
    if tw == 0 or th == 0:
        return np.ones((h // thumb_scale + 1, w // thumb_scale + 1), dtype=bool), thumb_scale

    thumb = slide.get_thumbnail((tw, th))
    gray = np.array(thumb.convert("L"))
    thresh = otsu_threshold(gray)
    mask = gray < thresh
    return mask, thumb_scale


def patch_wsi(svs_path, slide_id, output_dir, patch_size, bg_threshold,
              dark_threshold, min_tissue, thumb_scale):
    """Tile a single WSI and save valid tissue patches."""
    slide = openslide.OpenSlide(svs_path)
    w, h = slide.dimensions
    mask, scale = create_tissue_mask(slide, thumb_scale)

    slide_dir = os.path.join(output_dir, slide_id)
    os.makedirs(slide_dir, exist_ok=True)

    patches = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            mx = x // scale
            my = y // scale
            mx_end = min(mx + max(1, patch_size // scale), mask.shape[1])
            my_end = min(my + max(1, patch_size // scale), mask.shape[0])
            tissue_frac = mask[my:my_end, mx:mx_end].mean()
            if tissue_frac < min_tissue:
                continue

            region = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            patch_np = np.array(region)
            mean_intensity = patch_np.mean()
            if mean_intensity > bg_threshold or mean_intensity < dark_threshold:
                continue

            patch_name = f"patch_x{x}_y{y}.png"
            region.save(os.path.join(slide_dir, patch_name))

            patches.append({
                "path": os.path.join(slide_id, patch_name),
                "slide_id": slide_id,
                "x": x,
                "y": y,
                "patch_size": patch_size,
            })

    slide.close()

    if patches:
        meta_df = pd.DataFrame(patches)
        meta_df.to_csv(os.path.join(slide_dir, "metadata.csv"), index=False)

    return patches


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    patient_labels = manifest[manifest["pam50_label"].notna()].drop_duplicates("case_id")
    labeled_patients = set(patient_labels["case_id"].astype(str))

    svs_files = sorted(f for f in os.listdir(args.svs_dir) if f.endswith(".svs"))
    print(f"Found {len(svs_files)} SVS files, {len(labeled_patients)} patients with PAM50 labels")

    all_patches = []
    processed = 0
    for i, fname in enumerate(svs_files):
        case_id = fname.replace(".svs", "")
        if case_id not in labeled_patients:
            continue

        svs_path = os.path.join(args.svs_dir, fname)
        slide_id = f"wsi_{case_id}"

        patches = patch_wsi(
            svs_path, slide_id, args.output_dir,
            args.patch_size, args.bg_threshold,
            args.dark_threshold, args.min_tissue, args.thumb_scale,
        )
        all_patches.extend(patches)
        processed += 1

        if processed % 5 == 0 or processed == len(labeled_patients):
            print(f"  [{processed}] {slide_id}: {len(patches)} patches")

    if not all_patches:
        print("No patches extracted.")
        return

    master_df = pd.DataFrame(all_patches)
    master_path = os.path.join(args.output_dir, "master_metadata.csv")
    master_df.to_csv(master_path, index=False)

    patches_per_slide = master_df.groupby("slide_id").size()
    print(f"\n=== WSI Patching Summary ===")
    print(f"Total patches: {len(master_df)}")
    print(f"Slides processed: {patches_per_slide.shape[0]}")
    print(f"Patches per slide: min={patches_per_slide.min()}, "
          f"median={patches_per_slide.median():.0f}, "
          f"max={patches_per_slide.max()}")

    wsi_dataset = []
    for case_id in sorted(labeled_patients):
        sid = f"wsi_{case_id}"
        if sid in set(master_df["slide_id"]):
            row = patient_labels[patient_labels["case_id"].astype(str) == case_id].iloc[0]
            wsi_dataset.append({
                "case_id": case_id,
                "slide_id": sid,
                "label": int(row["pam50_label"]),
            })

    wsi_csv = pd.DataFrame(wsi_dataset)
    wsi_csv_path = os.path.join(os.path.dirname(args.manifest), "nou_pam50_wsi_dataset.csv")
    wsi_csv.to_csv(wsi_csv_path, index=False)
    print(f"\nSaved WSI dataset CSV: {wsi_csv_path} ({len(wsi_csv)} slides)")


if __name__ == "__main__":
    main()
