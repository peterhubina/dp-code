"""
Phase 2: NOU Tissue Patching
=============================
Tiles NOU TIFF tissue crops into non-overlapping patches with tissue
content filtering. Produces per-slide patch directories with metadata
CSVs and a consolidated master metadata file.

NOU images are tissue crops (~6000-9000px) scanned at ~40x native
resolution.  The default patch size of 448px at 40x yields a physical
field of view of ~112 um per patch, matching the TCGA-BRCA pipeline
(224px patches at 20x).  The UNI2-h eval transform then resizes 448px
patches to 224px, producing an effective 20x representation.

NOU images are not full WSIs, so we use simple grid-based tiling with
Otsu thresholding for tissue detection instead of OpenSlide-based
processing.
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

# Large NOU images trigger decompression bomb warning
Image.MAX_IMAGE_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser(description="Patch NOU tissue crops into 224x224 tiles")
    parser.add_argument("--manifest", type=str,
                        default=".scratch/nou_validation/metadata/nou_slide_manifest.csv")
    parser.add_argument("--tiff_dir", type=str, default=".datasets/nou/crops - Copy")
    parser.add_argument("--output_dir", type=str, default=".scratch/nou_validation/patches")
    parser.add_argument("--patch_size", type=int, default=448,
                        help="Patch size in pixels. Default 448 for NOU at ~40x native "
                             "resolution: 448px @ 40x covers ~112um, matching TCGA's "
                             "224px @ 20x (~0.5um/px). UNI2-h eval transform resizes "
                             "to 224, giving an effective 2x downsample (= 20x equiv).")
    parser.add_argument("--bg_threshold", type=int, default=230,
                        help="Mean intensity above this -> background")
    parser.add_argument("--dark_threshold", type=int, default=10,
                        help="Mean intensity below this -> artifact")
    parser.add_argument("--min_tissue", type=float, default=0.5,
                        help="Minimum tissue fraction (Otsu-based)")
    return parser.parse_args()


def otsu_threshold(gray_img):
    """Compute Otsu threshold on a grayscale numpy array."""
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


def create_tissue_mask(image_np, scale_factor=16):
    """Create binary tissue mask using Otsu thresholding on downscaled image."""
    h, w = image_np.shape[:2]
    small_h, small_w = h // scale_factor, w // scale_factor
    if small_h == 0 or small_w == 0:
        return np.ones((h, w), dtype=bool)

    small = Image.fromarray(image_np).resize((small_w, small_h), Image.LANCZOS)
    gray = np.array(small.convert("L"))
    thresh = otsu_threshold(gray)

    mask_small = gray < thresh
    mask_img = Image.fromarray(mask_small.astype(np.uint8) * 255)
    mask_full = np.array(mask_img.resize((w, h), Image.NEAREST)) > 0
    return mask_full


def patch_slide(image_path, slide_id, output_dir, patch_size, bg_threshold,
                dark_threshold, min_tissue):
    """Tile a single TIFF image and save valid patches."""
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    tissue_mask = create_tissue_mask(img_np)

    slide_dir = os.path.join(output_dir, slide_id)
    os.makedirs(slide_dir, exist_ok=True)

    patches = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = img_np[y:y + patch_size, x:x + patch_size]
            mean_intensity = patch.mean()

            if mean_intensity > bg_threshold or mean_intensity < dark_threshold:
                continue

            mask_region = tissue_mask[y:y + patch_size, x:x + patch_size]
            tissue_frac = mask_region.mean()
            if tissue_frac < min_tissue:
                continue

            patch_name = f"patch_x{x}_y{y}.png"
            patch_img = Image.fromarray(patch)
            patch_img.save(os.path.join(slide_dir, patch_name))

            patches.append({
                "path": os.path.join(slide_id, patch_name),
                "slide_id": slide_id,
                "x": x,
                "y": y,
                "patch_size": patch_size,
            })

    if patches:
        meta_df = pd.DataFrame(patches)
        meta_df.to_csv(os.path.join(slide_dir, "metadata.csv"), index=False)

    return patches


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    valid_slides = manifest[manifest["is_valid"]].copy()
    print(f"Processing {len(valid_slides)} valid slides...")

    all_patches = []
    for i, (_, row) in enumerate(valid_slides.iterrows()):
        slide_id = row["slide_id"]
        tiff_path = os.path.join(args.tiff_dir, row["filename"])

        patches = patch_slide(
            tiff_path, slide_id, args.output_dir,
            args.patch_size, args.bg_threshold,
            args.dark_threshold, args.min_tissue,
        )
        all_patches.extend(patches)

        if (i + 1) % 20 == 0 or (i + 1) == len(valid_slides):
            print(f"  [{i + 1}/{len(valid_slides)}] {slide_id}: {len(patches)} patches")

    master_df = pd.DataFrame(all_patches)
    master_path = os.path.join(args.output_dir, "master_metadata.csv")
    master_df.to_csv(master_path, index=False)

    # Summary
    patches_per_slide = master_df.groupby("slide_id").size()
    print(f"\n=== Patching Summary ===")
    print(f"Total patches: {len(master_df)}")
    print(f"Slides processed: {patches_per_slide.shape[0]}")
    print(f"Patches per slide: min={patches_per_slide.min()}, "
          f"median={patches_per_slide.median():.0f}, "
          f"max={patches_per_slide.max()}")


if __name__ == "__main__":
    main()
