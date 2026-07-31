"""
Phase 2a: CPTAC-BRCA coverage reconciliation + CLAM manifest
=============================================================
Joins the downloaded UNI2-h feature store against wsi_manifest.csv (391 slides
from PathDB) and cohort.csv (145 cases with cBioPortal genomic PAM50 calls),
then writes a CLAM-format dataset CSV restricted to slides that have BOTH
features and a 4-class PAM50 label.

Also emits a per-slide coverage table so every dropped slide has a named reason.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

CLASSES = ["LumA", "LumB", "Basal", "Her2"]
LABEL_IDX = {name: i for i, name in enumerate(CLASSES)}


def parse_args():
    parser = argparse.ArgumentParser(description="Build the CPTAC-BRCA CLAM manifest")
    parser.add_argument("--feature_dir", type=str,
                        default=".datasets/cptac-brca/embeddings",
                        help="Directory holding the per-slide .h5 feature files")
    parser.add_argument("--wsi_manifest", type=str,
                        default=".datasets/cptac-brca/wsi_manifest.csv")
    parser.add_argument("--cohort", type=str,
                        default=".datasets/cptac-brca/cohort.csv")
    parser.add_argument("--dataset_csv", type=str,
                        default=".datasets/cptac-brca/cptac_brca_pam50_dataset.csv")
    parser.add_argument("--coverage_csv", type=str,
                        default=".scratch/cptac_validation/metadata/coverage_report.csv")
    return parser.parse_args()


def find_h5_files(feature_dir):
    """All readable .h5 files under feature_dir, keyed by filename stem."""
    paths = sorted(Path(feature_dir).rglob("*.h5"), key=lambda p: len(p.parts))
    stems = {}
    for path in paths:
        if path.is_file() and path.stem not in stems:
            stems[path.stem] = str(path)
    return stems


def pick_join_key(wsi, stems):
    """Choose the manifest column whose values match the h5 stems."""
    candidates = {
        "filename_stem": wsi["filename"].str.replace(r"\.svs$", "", regex=True),
        "slide_id": wsi["slide_id"].astype(str),
    }
    scored = {k: v.isin(stems).sum() for k, v in candidates.items()}
    best = max(scored, key=scored.get)
    print("join-key match counts against feature stems:")
    for k, n in scored.items():
        print(f"  {k}: {n}/{len(wsi)}")
    if scored[best] == 0:
        raise SystemExit(
            "No manifest column matches the feature filenames. "
            f"Example h5 stems: {sorted(stems)[:5]}"
        )
    print(f"  -> using {best}\n")
    return best, candidates[best]


def main():
    args = parse_args()

    wsi = pd.read_csv(args.wsi_manifest)
    cohort = pd.read_csv(args.cohort)
    stems = find_h5_files(args.feature_dir)
    print(f"feature store: {len(stems)} .h5 files under {args.feature_dir}")
    print(f"wsi_manifest:  {len(wsi)} slides / {wsi['case_id'].nunique()} cases")
    print(f"cohort:        {len(cohort)} cases\n")

    key_name, key_vals = pick_join_key(wsi, stems)
    wsi = wsi.assign(feature_key=key_vals)
    wsi["has_features"] = wsi["feature_key"].isin(stems)
    wsi["feature_path"] = wsi["feature_key"].map(stems)

    labels = cohort.set_index("case_id")["PAM50"]
    wsi["pam50"] = wsi["case_id"].map(labels)
    wsi["has_label"] = wsi["pam50"].isin(CLASSES)

    def reason(row):
        if not row["has_features"]:
            return "no_features_in_tarball"
        if pd.isna(row["pam50"]):
            return "no_pam50_call"
        if row["pam50"] not in CLASSES:
            return f"class_out_of_scope({row['pam50']})"
        return "included"

    wsi["status"] = wsi.apply(reason, axis=1)

    # --- coverage report -------------------------------------------------
    print("=== Slide-level coverage (391 cohort slides) ===")
    print(wsi["status"].value_counts().to_string())
    print()

    per_case = wsi.groupby("case_id").agg(
        n_slides=("slide_id", "count"),
        n_with_features=("has_features", "sum"),
        pam50=("pam50", "first"),
    )
    per_case["coverage"] = pd.cut(
        per_case["n_with_features"] / per_case["n_slides"],
        [-0.01, 0.0, 0.999, 1.0],
        labels=["none", "partial", "full"],
    )
    print("=== Case-level feature coverage (119 cases with slides) ===")
    print(per_case["coverage"].value_counts().to_string())
    print()
    partial = per_case[per_case["coverage"] == "partial"]
    if len(partial):
        print(f"partially covered cases ({len(partial)}):")
        print(partial[["n_slides", "n_with_features", "pam50"]].to_string())
        print()
    missing = per_case[per_case["coverage"] == "none"]
    if len(missing):
        print(f"cases with no features at all ({len(missing)}):")
        print(missing[["n_slides", "pam50"]].to_string())
        print()

    os.makedirs(os.path.dirname(args.coverage_csv), exist_ok=True)
    wsi[["case_id", "slide_id", "filename", "feature_key", "mpp_x",
         "has_features", "pam50", "status"]].to_csv(args.coverage_csv, index=False)
    print(f"wrote coverage report: {args.coverage_csv}")

    # --- CLAM manifest ---------------------------------------------------
    keep = wsi[wsi["status"] == "included"].copy()
    manifest = pd.DataFrame({
        "case_id": keep["case_id"],
        "slide_id": keep["feature_key"],
        "label": keep["pam50"].map(LABEL_IDX),
        "label_name": keep["pam50"],
        # Published UNI2-h features: same encoder and geometry as the TCGA
        # training set (256px @ 20x, 0 overlap), but tiled with CLAM
        # create_patches_fp.py rather than Trident. See audit_feature_provenance.py.
        "provenance": "hf_uni2h_clam_20x_256px",
    }).sort_values(["case_id", "slide_id"]).reset_index(drop=True)

    manifest.to_csv(args.dataset_csv, index=False)
    print(f"wrote manifest: {args.dataset_csv}")
    print(f"  {len(manifest)} slides / {manifest['case_id'].nunique()} cases")
    print()
    print("slides per class:")
    print(manifest["label_name"].value_counts().to_string())
    print("cases per class:")
    print(manifest.drop_duplicates("case_id")["label_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
