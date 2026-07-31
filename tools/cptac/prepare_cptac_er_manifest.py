"""
CPTAC-BRCA ER-status CLAM manifest
==================================
Builds the external-validation manifest for binary ER status, mirroring
prepare_cptac_manifest.py (PAM50) but with two deliberate differences:

  scope     PAM50 used wsi_manifest.csv, which download_cptac.py had already
            filtered to cases carrying RNA + a PAM50 call (391 slides / 119
            cases). ER needs neither, so this script indexes the feature store
            directly (653 h5 files / 198 cases) and keeps every slide whose case
            has an ER call. That buys ~4 extra cases and, more importantly, 3
            extra ER-negatives -- the minority class that bounds the external CI.

  geometry  The feature store mixes two tiling configurations, because CPTAC
            mixes 20x and 40x scans:
                mpp 0.4942 (20x) -> patch_size 256, custom_downsample 1
                mpp 0.2501 (40x) -> patch_size 512, custom_downsample 2
            Both land at 256 effective px / ~127 um FOV, matching the TCGA
            training set. The correspondence is exact on all 391 slides whose
            mpp is known, so it is a deterministic pipeline rule and is asserted
            here for every slide -- a silent magnification mismatch would void
            the external numbers without changing any label.

ER calls come from cBioPortal brca_cptac_2020 (ER_UPDATED_CLINICAL_STATUS); the
CPTAC pan-cancer clinical table carries no receptor status at all. Label
encoding matches project/CLAM/main.py for --task tcga_brca_er exactly:
{'ER-negative': 0, 'ER-positive': 1}.

    python tools/cptac/prepare_cptac_er_manifest.py
"""

import argparse
import os
import re
from pathlib import Path

import h5py
import pandas as pd

CLASSES = ["ER-negative", "ER-positive"]
LABEL_IDX = {name: i for i, name in enumerate(CLASSES)}

# CPTAC case ids look like 01BR001; slide stems are <case>-<uuid-fragment>.
CASE_RE = re.compile(r"^\d{2}BR\d{3}$")

# Effective patch size every slide must reduce to, and the TCGA training FOV.
EFFECTIVE_PX = 256
TCGA_FOV_UM = 128.0


def parse_args():
    parser = argparse.ArgumentParser(description="Build the CPTAC-BRCA ER CLAM manifest")
    parser.add_argument("--feature_dir", type=str,
                        default=".datasets/cptac-brca/embeddings")
    parser.add_argument("--labels_csv", type=str,
                        default=".datasets/cptac-brca/clinical/cbioportal_labels.csv")
    parser.add_argument("--wsi_manifest", type=str,
                        default=".datasets/cptac-brca/wsi_manifest.csv",
                        help="only used to attach mpp for the geometry audit")
    parser.add_argument("--pam50_csv", type=str,
                        default=".datasets/cptac-brca/cptac_brca_pam50_dataset.csv",
                        help="reported as an overlap column so the ER and PAM50 "
                             "external results can be compared on a common subset")
    parser.add_argument("--dataset_csv", type=str,
                        default=".datasets/cptac-brca/cptac_brca_er_dataset.csv")
    parser.add_argument("--coverage_csv", type=str,
                        default=".scratch/cptac_validation/metadata/er_coverage_report.csv")
    return parser.parse_args()


def index_features(feature_dir):
    """Map h5 stem -> path. Shallower paths win; broken symlinks are skipped."""
    index = {}
    for path in sorted(Path(feature_dir).rglob("*.h5"), key=lambda p: len(p.parts)):
        if path.is_file() and path.stem not in index:
            index[path.stem] = str(path)
    return index


def read_geometry(h5_path):
    """Tiling parameters and patch count from a feature file's coords attrs."""
    with h5py.File(h5_path, "r") as handle:
        attrs = handle["coords"].attrs
        return {
            "patch_size": int(attrs["patch_size"]),
            "custom_downsample": float(attrs["custom_downsample"]),
            "patch_level": int(attrs["patch_level"]),
            "n_patches": int(handle["features"].shape[1]),
        }


def normalise_er(value):
    """cBioPortal mixes 'Negative' and 'negative'; anything else is missing."""
    text = str(value).strip().lower()
    if text == "positive":
        return "ER-positive"
    if text == "negative":
        return "ER-negative"
    return None


def main():
    args = parse_args()

    features = index_features(args.feature_dir)
    print(f"feature store: {len(features)} h5 files under {args.feature_dir}")

    slides = pd.DataFrame({"slide_id": sorted(features)})
    slides["feature_path"] = slides["slide_id"].map(features)
    slides["case_id"] = slides["slide_id"].str.split("-").str[0]

    malformed = slides.loc[~slides["case_id"].str.match(CASE_RE), "slide_id"]
    if len(malformed):
        raise SystemExit(
            f"{len(malformed)} slide stems do not yield a CPTAC case id "
            f"(e.g. {malformed.head(3).tolist()}); the naming assumption is wrong."
        )
    print(f"               {slides['case_id'].nunique()} distinct cases\n")

    # --- geometry audit --------------------------------------------------
    geometry = pd.DataFrame([read_geometry(p) for p in slides["feature_path"]])
    slides = pd.concat([slides, geometry], axis=1)
    slides["effective_px"] = slides["patch_size"] / slides["custom_downsample"]

    wrong = slides.loc[slides["effective_px"] != EFFECTIVE_PX]
    if len(wrong):
        raise SystemExit(
            f"{len(wrong)} slides do not reduce to {EFFECTIVE_PX}px "
            f"(e.g. {wrong['slide_id'].head(3).tolist()}); magnification mismatch."
        )

    wsi = pd.read_csv(args.wsi_manifest)
    wsi["slide_id"] = wsi["filename"].str.replace(r"\.svs$", "", regex=True)
    slides = slides.merge(wsi[["slide_id", "mpp_x"]], on="slide_id", how="left")
    slides["fov_um"] = (slides["patch_size"] * slides["mpp_x"]).round(1)

    print(f"=== Geometry audit ({len(slides)} slides) ===")
    print("all slides reduce to 256 effective px: OK")
    combos = slides.groupby(["patch_size", "custom_downsample"]).size()
    print("tiling configurations (patch_size, custom_downsample):")
    print(combos.to_string())
    known = slides[slides["mpp_x"].notna()]
    print(f"\nmpp known for {len(known)}/{len(slides)} slides; FOV vs TCGA "
          f"reference {TCGA_FOV_UM:.0f} um:")
    print(known["fov_um"].value_counts().to_string())
    # The rule is only trustworthy for mpp-less slides if it is exact where mpp
    # IS known, so check that patch_size partitions magnification cleanly.
    leak = known.groupby("patch_size")["mpp_x"].nunique()
    if (leak > 1).any():
        print("\nWARNING: patch_size does not partition mpp cleanly; the tiling "
              "rule cannot be extrapolated to slides with unknown mpp:")
        print(known.groupby(["patch_size", "mpp_x"]).size().to_string())
    else:
        print("\npatch_size partitions mpp exactly -> rule extrapolates to the "
              f"{slides['mpp_x'].isna().sum()} slides with no manifest entry")
    print()

    # --- ER labels -------------------------------------------------------
    labels = pd.read_csv(args.labels_csv)
    if "ER_UPDATED_CLINICAL_STATUS" not in labels.columns:
        raise SystemExit(f"{args.labels_csv} has no ER_UPDATED_CLINICAL_STATUS column")
    er = labels.set_index("case_id")["ER_UPDATED_CLINICAL_STATUS"].map(normalise_er)
    print(f"ER calls: {er.notna().sum()}/{len(er)} cases in {os.path.basename(args.labels_csv)}")

    slides["er"] = slides["case_id"].map(er)
    slides["has_label"] = slides["er"].notna()

    def status(row):
        if row["case_id"] not in er.index:
            return "case_absent_from_cbioportal"
        if not row["has_label"]:
            return "er_call_missing_or_equivocal"
        return "included"

    slides["status"] = slides.apply(status, axis=1)

    print("\n=== Slide-level coverage ===")
    print(slides["status"].value_counts().to_string())

    per_case = slides.groupby("case_id").agg(
        n_slides=("slide_id", "count"),
        er=("er", "first"),
        status=("status", "first"),
    )
    print("\n=== Case-level coverage ===")
    print(per_case["status"].value_counts().to_string())

    os.makedirs(os.path.dirname(args.coverage_csv), exist_ok=True)
    slides[["case_id", "slide_id", "patch_size", "custom_downsample", "mpp_x",
            "fov_um", "n_patches", "er", "status"]].to_csv(args.coverage_csv, index=False)
    print(f"\nwrote coverage report: {args.coverage_csv}")

    # --- manifest --------------------------------------------------------
    keep = slides[slides["status"] == "included"].copy()
    manifest = pd.DataFrame({
        "case_id": keep["case_id"],
        "slide_id": keep["slide_id"],
        "label": keep["er"].map(LABEL_IDX),
        "label_name": keep["er"],
        # Same encoder and effective geometry as the TCGA training set, but tiled
        # with CLAM create_patches_fp.py rather than Trident.
        "provenance": "hf_uni2h_clam_20x_256px",
    }).sort_values(["case_id", "slide_id"]).reset_index(drop=True)

    if os.path.exists(args.pam50_csv):
        pam50_cases = set(pd.read_csv(args.pam50_csv)["case_id"])
        manifest["in_pam50_cohort"] = manifest["case_id"].isin(pam50_cases)
        overlap = manifest.loc[manifest["in_pam50_cohort"], "case_id"].nunique()
        print(f"\noverlap with the PAM50 external cohort: {overlap} cases")

    manifest.to_csv(args.dataset_csv, index=False)
    print(f"wrote manifest: {args.dataset_csv}")
    print(f"  {len(manifest)} slides / {manifest['case_id'].nunique()} cases")
    print("\nslides per class:")
    print(manifest["label_name"].value_counts().to_string())
    print("cases per class:")
    print(manifest.drop_duplicates("case_id")["label_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
