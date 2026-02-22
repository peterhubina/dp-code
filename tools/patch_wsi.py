"""
Extract patches from annotated tissue regions in a WSI (.mrxs).

Usage:
    python tools/patch_wsi.py \
        --wsi   .datasets/breast_cancer/25.mrxs \
        --geojson .datasets/breast_cancer/25.geojson \
        --out   .datasets/patches/25
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import openslide
from shapely.geometry import box, shape
from shapely.ops import unary_union
from tqdm import tqdm


def load_annotations(path):
    with open(path) as f:
        data = json.load(f)
    polys = {}
    for feat in data["features"]:
        cls = feat["properties"].get("classification", {}).get("name", "unknown")
        polys.setdefault(cls, []).append(shape(feat["geometry"]))
    return polys


def extract_patches(wsi_path, geojson_path, out_dir, patch_size=256, level=0,
                    overlap=0.0, min_tissue=0.5):
    slide = openslide.OpenSlide(wsi_path)
    ds = slide.level_downsamples[level]
    ps0 = int(patch_size * ds)          # patch size in level-0 coords
    stride = int(ps0 * (1 - overlap))   # stride in level-0 coords

    annotations = load_annotations(geojson_path)
    slide_id = Path(wsi_path).stem
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Level {level}  downsample={ds:.0f}x  patch_l0={ps0}px  stride={stride}px")

    # Metadata CSV alongside the patches
    csv_path = out_root / "metadata.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["path", "label", "slide_id", "x", "y", "level", "patch_size",
                     "tissue_ratio"])

    total = 0
    for cls, polygons in annotations.items():
        union = unary_union([p.buffer(0) for p in polygons])
        cls_dir = out_root / cls.replace(" ", "_")
        cls_dir.mkdir(parents=True, exist_ok=True)

        # Candidate grid from each polygon's bounding box
        coords = set()
        for poly in polygons:
            x0, y0, x1, y1 = (int(v) for v in poly.bounds)
            for y in range(y0, y1, stride):
                for x in range(x0, x1, stride):
                    coords.add((x, y))

        saved = 0
        for x, y in tqdm(coords, desc=cls):
            patch_box = box(x, y, x + ps0, y + ps0)
            tissue_ratio = union.intersection(patch_box).area / patch_box.area
            if tissue_ratio < min_tissue:
                continue

            img = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            if np.asarray(img).mean() > 230:   # skip near-white background
                continue

            fname = f"x{x}_y{y}.png"
            img.save(cls_dir / fname)

            rel_path = str(Path(cls.replace(" ", "_")) / fname)
            writer.writerow([rel_path, cls, slide_id, x, y, level, patch_size,
                             f"{tissue_ratio:.3f}"])
            saved += 1

        print(f"  {cls}: {saved} patches -> {cls_dir}")
        total += saved

    csv_file.close()
    slide.close()
    print(f"\nTotal: {total} patches")
    print(f"Metadata: {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wsi",      required=True)
    p.add_argument("--geojson",  required=True)
    p.add_argument("--out",      required=True)
    p.add_argument("--patch_size", type=int,   default=224)
    p.add_argument("--level",      type=int,   default=1)
    p.add_argument("--overlap",    type=float, default=0.0)
    p.add_argument("--min_tissue", type=float, default=0.5)
    a = p.parse_args()
    extract_patches(a.wsi, a.geojson, a.out, a.patch_size, a.level, a.overlap, a.min_tissue)
