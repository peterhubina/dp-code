"""
Extract patches from annotated tissue regions in WSI files (.mrxs).

Supports both single-WSI and batch processing modes.

Usage:
    # Single WSI mode
    python tools/patch_wsi.py single \
        --wsi .datasets/breast_cancer/25.mrxs \
        --geojson .datasets/breast_cancer/25.geojson \
        --out .datasets/patches/25

    # Batch mode (processes all WSIs in a directory)
    python tools/patch_wsi.py batch \
        --wsi_dir ".datasets/PKG - HistologyHSI-BC-Recurrence/01_01_Histological_Images" \
        --geojson_dir ".datasets/PKG - HistologyHSI-BC-Recurrence/01_02_Tissue_Annotations" \
        --out_dir .datasets/wsi_patches \
        --patch_size 224 --level 0 --overlap 0.0 --min_tissue 0.5

    # Exclude normal tissue from patching
    python tools/patch_wsi.py batch \
        --wsi_dir ./wsi_folder --geojson_dir ./annotations --out_dir ./patches \
        --exclude_classes "Normal"
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from shapely.geometry import box, shape
from shapely.ops import unary_union
from tqdm import tqdm


def load_annotations(path):
    """Load GeoJSON annotations and group polygons by tissue class."""
    with open(path) as f:
        data = json.load(f)
    polys = {}
    for feat in data["features"]:
        cls = feat["properties"].get("classification", {}).get("name", "unknown")
        polys.setdefault(cls, []).append(shape(feat["geometry"]))
    return polys


def extract_patches(wsi_path, geojson_path, out_dir, patch_size=256, level=0,
                    overlap=0.0, min_tissue=0.5, exclude_classes=None):
    """
    Extract patches from a single WSI based on GeoJSON annotations.

    Args:
        wsi_path: Path to WSI file (.mrxs)
        geojson_path: Path to GeoJSON annotation file
        out_dir: Output directory for patches
        patch_size: Size of extracted patches in pixels
        level: Pyramid level to extract from (0 = highest resolution)
        overlap: Overlap ratio between patches (0.0 - 1.0)
        min_tissue: Minimum tissue ratio required to save patch (0.0 - 1.0)
        exclude_classes: List of tissue class names to skip (e.g., ["Normal"])

    Returns:
        int: Total number of patches extracted
    """
    exclude_classes = set(exclude_classes or [])
    slide = openslide.OpenSlide(wsi_path)

    # Get coordinate offset from MRXS metadata (tissue origin within slide canvas)
    # MRXS files store tissue at an offset position; annotations reference the tissue
    # but read_region uses the full canvas coordinates, so we must add the offset
    offset_x = int(slide.properties.get(
        'mirax.NONHIERLAYER_0_LEVEL_0_SECTION.COMPRESSED_STITCHING_ORIG_SLIDE_SCANNED_AREA_IN_PIXELS__LEFT', 0))
    offset_y = int(slide.properties.get(
        'mirax.NONHIERLAYER_0_LEVEL_0_SECTION.COMPRESSED_STITCHING_ORIG_SLIDE_SCANNED_AREA_IN_PIXELS__TOP', 0))

    if offset_x != 0 or offset_y != 0:
        print(f"Applying coordinate offset: ({offset_x}, {offset_y})")

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
    writer.writerow(["path", "label", "slide_id", "x", "y", "level", "patch_size"])

    total = 0
    for cls, polygons in annotations.items():
        if cls in exclude_classes:
            print(f"  Skipping excluded class: {cls}")
            continue
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

            img = slide.read_region((x + offset_x, y + offset_y), level, (patch_size, patch_size)).convert("RGB")
            if np.asarray(img).mean() > 230:   # skip near-white background
                continue

            fname = f"x{x}_y{y}.png"
            img.save(cls_dir / fname)

            rel_path = str(Path(cls.replace(" ", "_")) / fname)
            writer.writerow([rel_path, cls.replace(" ", "_"), slide_id, x, y, level, patch_size])
            saved += 1

        print(f"  {cls}: {saved} patches -> {cls_dir}")
        total += saved

    csv_file.close()
    slide.close()
    print(f"\nTotal: {total} patches")
    print(f"Metadata: {csv_path}")

    return total


def create_master_metadata(patches_dir, output_path=None):
    """
    Consolidate all per-slide metadata.csv files into one master file.

    Args:
        patches_dir: Root directory containing per-slide patch folders
        output_path: Output path for master CSV (default: patches_dir/master_metadata.csv)

    Returns:
        pd.DataFrame: Master metadata dataframe
    """
    patches_dir = Path(patches_dir)
    output_path = output_path or patches_dir / "master_metadata.csv"

    # Find all metadata.csv files
    metadata_files = sorted(patches_dir.glob("*/metadata.csv"))
    print(f"\nConsolidating metadata from {len(metadata_files)} slides...")

    if not metadata_files:
        print("No metadata files found!")
        return None

    # Read and concatenate all metadata
    all_dfs = []
    for meta_file in metadata_files:
        slide_id = meta_file.parent.name
        df = pd.read_csv(meta_file)
        # Add full path relative to patches_dir
        df['full_path'] = df['path'].apply(lambda x: f"{slide_id}/{x}")
        all_dfs.append(df)

    # Combine all dataframes
    master_df = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns
    cols = ['full_path', 'slide_id', 'label', 'x', 'y', 'level', 'patch_size']
    master_df = master_df[cols]

    # Save master metadata
    master_df.to_csv(output_path, index=False)

    # Print summary
    print(f"\n{'='*80}")
    print(f"MASTER METADATA CREATED: {output_path}")
    print(f"{'='*80}")
    print(f"Total patches: {len(master_df)}")
    print(f"Total slides: {master_df['slide_id'].nunique()}")
    print(f"\nPatches per tissue type:")
    for label, count in master_df['label'].value_counts().items():
        print(f"  {label}: {count} ({count/len(master_df)*100:.1f}%)")

    return master_df


def batch_extract_patches(wsi_dir, geojson_dir, out_dir, patch_size=224, level=0,
                          overlap=0.0, min_tissue=0.5, exclude_classes=None):
    """
    Process all WSI files that have corresponding GeoJSON annotations.
    Creates master metadata after processing all slides.

    Args:
        wsi_dir: Directory containing .mrxs files
        geojson_dir: Directory containing .geojson annotation files
        out_dir: Output directory for all patches
        patch_size: Size of extracted patches in pixels
        level: Pyramid level to extract from (0 = highest resolution)
        overlap: Overlap ratio between patches (0.0 - 1.0)
        min_tissue: Minimum tissue ratio required to save patch (0.0 - 1.0)
        exclude_classes: List of tissue class names to skip (e.g., ["Normal"])
    """
    wsi_dir = Path(wsi_dir)
    geojson_dir = Path(geojson_dir)
    out_dir = Path(out_dir)

    # Debug output
    print(f"DEBUG: wsi_dir = {wsi_dir}")
    print(f"DEBUG: wsi_dir.resolve() = {wsi_dir.resolve()}")
    print(f"DEBUG: wsi_dir.exists() = {wsi_dir.exists()}")
    print(f"DEBUG: wsi_dir.is_dir() = {wsi_dir.is_dir()}")

    # Find all WSI files
    wsi_files = sorted(wsi_dir.glob("*.mrxs"))
    print(f"Found {len(wsi_files)} WSI files in {wsi_dir}")

    if not wsi_files:
        print("No WSI files found!")
        return

    # Process each WSI that has a corresponding annotation
    processed = 0
    skipped = 0
    errors = []
    total_patches = 0

    for idx, wsi_path in enumerate(wsi_files):
        slide_id = wsi_path.stem
        geojson_path = geojson_dir / f"{slide_id}.geojson"

        if not geojson_path.exists():
            print(f"\nSkipping {slide_id}: No annotation found at {geojson_path}")
            skipped += 1
            continue

        slide_out_dir = out_dir / slide_id

        # Skip already-patched WSIs (metadata.csv is written only on successful completion)
        if (slide_out_dir / "metadata.csv").exists():
            print(f"\nSkipping {slide_id}: already patched ({slide_out_dir})")
            skipped += 1
            continue

        print(f"\n{'='*80}")
        print(f"Processing {slide_id} ({processed + 1}/{len(wsi_files) - skipped})")
        print(f"  WSI:      {wsi_path}")
        print(f"  GeoJSON:  {geojson_path}")
        print(f"  Output:   {slide_out_dir}")
        print(f"{'='*80}")

        try:
            patches = extract_patches(
                wsi_path=str(wsi_path),
                geojson_path=str(geojson_path),
                out_dir=str(slide_out_dir),
                patch_size=patch_size,
                level=level,
                overlap=overlap,
                min_tissue=min_tissue,
                exclude_classes=exclude_classes
            )
            processed += 1
            total_patches += patches
        except Exception as e:
            error_msg = f"Error processing {slide_id}: {e}"
            print(f"\n[ERROR] {error_msg}")
            errors.append(error_msg)
            continue

    # Create master metadata
    if processed > 0:
        create_master_metadata(out_dir)

    # Final summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Newly processed: {processed}/{len(wsi_files)} slides")
    print(f"Skipped (already patched or no annotation): {skipped}")
    print(f"Errors: {len(errors)}")
    print(f"Total patches extracted: {total_patches}")

    if errors:
        print(f"\nErrors encountered:")
        for err in errors:
            print(f"  - {err}")

    print(f"\nAll patches saved to: {out_dir}")
    print(f"Master metadata: {out_dir / 'master_metadata.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract patches from annotated WSI files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single WSI
  python tools/patch_wsi.py single --wsi slide.mrxs --geojson slide.geojson --out ./patches

  # Batch processing
  python tools/patch_wsi.py batch --wsi_dir ./wsi_folder --geojson_dir ./annotations --out_dir ./patches
        """
    )

    subparsers = parser.add_subparsers(dest="mode", help="Processing mode")

    # Single WSI mode
    single_parser = subparsers.add_parser("single", help="Process a single WSI file")
    single_parser.add_argument("--wsi", required=True, help="Path to WSI file (.mrxs)")
    single_parser.add_argument("--geojson", required=True, help="Path to GeoJSON annotation file")
    single_parser.add_argument("--out", required=True, help="Output directory for patches")
    single_parser.add_argument("--patch_size", type=int, default=224, help="Patch size in pixels (default: 224)")
    single_parser.add_argument("--level", type=int, default=0, help="Pyramid level, 0=highest res (default: 0)")
    single_parser.add_argument("--overlap", type=float, default=0.0, help="Overlap ratio 0.0-1.0 (default: 0.0)")
    single_parser.add_argument("--min_tissue", type=float, default=0.5, help="Min tissue ratio 0.0-1.0 (default: 0.5)")
    single_parser.add_argument("--exclude_classes", type=str, default=None,
                               help="Comma-separated tissue classes to skip (e.g., 'Normal' or 'Normal,Stroma')")

    # Batch mode
    batch_parser = subparsers.add_parser("batch", help="Process all WSIs in a directory")
    batch_parser.add_argument("--wsi_dir", required=True, help="Directory containing .mrxs files")
    batch_parser.add_argument("--geojson_dir", required=True, help="Directory containing .geojson files")
    batch_parser.add_argument("--out_dir", required=True, help="Output directory for all patches")
    batch_parser.add_argument("--patch_size", type=int, default=224, help="Patch size in pixels (default: 224)")
    batch_parser.add_argument("--level", type=int, default=0, help="Pyramid level, 0=highest res (default: 0)")
    batch_parser.add_argument("--overlap", type=float, default=0.0, help="Overlap ratio 0.0-1.0 (default: 0.0)")
    batch_parser.add_argument("--min_tissue", type=float, default=0.5, help="Min tissue ratio 0.0-1.0 (default: 0.5)")
    batch_parser.add_argument("--exclude_classes", type=str, default='normal tissue',
                              help="Comma-separated tissue classes to skip (e.g., 'Normal' or 'Normal,Stroma')")

    args = parser.parse_args()

    # Parse exclude_classes from comma-separated string
    exclude_classes = None
    if hasattr(args, 'exclude_classes') and args.exclude_classes:
        exclude_classes = [c.strip() for c in args.exclude_classes.split(',')]

    if args.mode == "single":
        extract_patches(
            wsi_path=args.wsi,
            geojson_path=args.geojson,
            out_dir=args.out,
            patch_size=args.patch_size,
            level=args.level,
            overlap=args.overlap,
            min_tissue=args.min_tissue,
            exclude_classes=exclude_classes
        )
    elif args.mode == "batch":
        batch_extract_patches(
            wsi_dir=args.wsi_dir,
            geojson_dir=args.geojson_dir,
            out_dir=args.out_dir,
            patch_size=args.patch_size,
            level=args.level,
            overlap=args.overlap,
            min_tissue=args.min_tissue,
            exclude_classes=exclude_classes
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
