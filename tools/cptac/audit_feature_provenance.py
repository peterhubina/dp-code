"""
Phase 1: provenance audit of the CPTAC-BRCA UNI2-h feature store
=================================================================
The TCGA-BRCA training features were produced by Trident at 20x / 256px / 0px
overlap. The published CPTAC features were tiled with CLAM's create_patches_fp.py
instead, so their coords.attrs use a different schema. This script checks that the
*effective* geometry still matches: 256px tiles at 20x with zero overlap, 1536-dim
UNI2-h embeddings, across every slide -- including the 157 slides scanned at 20x
rather than 40x, which must be tiled differently to land on the same field of view.

Writes a per-slide audit table and prints the distribution.
"""

import argparse
import os
from pathlib import Path

import h5py
import pandas as pd

TCGA_REF = {"patch_px": 256, "magnification": 20, "overlap": 0, "feat_dim": 1536}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit CPTAC feature provenance")
    parser.add_argument("--feature_dir", type=str,
                        default=".datasets/cptac-brca/embeddings")
    parser.add_argument("--wsi_manifest", type=str,
                        default=".datasets/cptac-brca/wsi_manifest.csv")
    parser.add_argument("--out_csv", type=str,
                        default=".scratch/cptac_validation/metadata/feature_provenance.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    man = pd.read_csv(args.wsi_manifest)
    man["stem"] = man["filename"].str.replace(r"\.svs$", "", regex=True)
    mpp = dict(zip(man["stem"], man["mpp_x"]))

    rows = []
    for path in sorted(Path(args.feature_dir).glob("*.h5")):
        with h5py.File(path, "r") as f:
            attrs = dict(f["coords"].attrs)
            rows.append({
                "slide_id": path.stem,
                "n_patches": f["features"].shape[1],
                "feat_dim": f["features"].shape[2],
                "patch_size": attrs.get("patch_size"),
                "step_size": attrs.get("step_size"),
                "custom_downsample": attrs.get("custom_downsample"),
                "patch_level": attrs.get("patch_level"),
                "mpp_x": mpp.get(path.stem),
            })

    df = pd.DataFrame(rows)
    print(f"audited {len(df)} feature files in {args.feature_dir}\n")

    df["eff_patch_px"] = df["patch_size"] / df["custom_downsample"]
    df["overlap"] = df["patch_size"] - df["step_size"]
    df["eff_mpp"] = df["mpp_x"] * df["custom_downsample"]
    df["eff_mag"] = (10.0 / df["eff_mpp"]).round(1)

    print("tiling config by native scanner mpp:")
    print(df.groupby(["mpp_x", "patch_size", "custom_downsample"], dropna=False)
            .size().rename("slides").to_string())
    print()
    print(f"effective patch px (target {TCGA_REF['patch_px']}):")
    print(df["eff_patch_px"].value_counts(dropna=False).to_string())
    print(f"\neffective magnification (target {TCGA_REF['magnification']}x):")
    print(df["eff_mag"].value_counts(dropna=False).to_string())
    print(f"\noverlap (target {TCGA_REF['overlap']}):")
    print(df["overlap"].value_counts(dropna=False).to_string())
    print(f"\nfeature dim (target {TCGA_REF['feat_dim']}):")
    print(df["feat_dim"].value_counts().to_string())
    print(f"\npatches per slide: median {df['n_patches'].median():.0f}, "
          f"min {df['n_patches'].min()}, max {df['n_patches'].max()}")

    off = df[(df["eff_patch_px"] != TCGA_REF["patch_px"])
             | (df["overlap"] != TCGA_REF["overlap"])
             | (df["feat_dim"] != TCGA_REF["feat_dim"])
             | (df["eff_mag"].notna() & ((df["eff_mag"] - 20).abs() > 1))]
    print()
    if len(off):
        print(f"*** {len(off)} slides deviate from the TCGA training config ***")
        print(off.to_string(index=False))
    else:
        print("VERDICT: all slides match the TCGA training geometry "
              "(256px @ ~20x, 0 overlap, 1536-dim UNI2-h).")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
