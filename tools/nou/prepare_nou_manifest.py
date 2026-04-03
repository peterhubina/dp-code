"""
Phase 1: NOU Dataset Manifest Preparation
==========================================
Parses NOU TIFF filenames, maps annotations from Excel (Anotacia_CTC_blocky.xls),
and produces structured CSV manifests for downstream processing.

Annotation mapping chain:
  TIFF filename -> poloha number -> Blocky sheet (poloha -> TRU cislo)
  -> databaza sheet (TRU cislo -> molecular subtype + clinical data)

PAM50 labels are assigned at the *patient level* via majority vote across
all tissue blocks for each case_id.  The per-block IHC annotation is retained
in the manifest (``block_pam50_*`` columns) for traceability.

CTC labels are extracted directly from TIFF filenames (EP/EMT/ANY fields).

Outputs:
  - nou_slide_manifest.csv: Full manifest with all annotations
  - nou_pam50_dataset.csv: CLAM-format CSV for PAM50 validation
"""

import argparse
from collections import Counter
import os
import re
import sys

import pandas as pd
from PIL import Image

CORRUPTED_FILES = {
    "1005164-0-poloha-287-1-1-EP-0-EMT-0-ANY-0.tiff",
    "1005192-3-poloha-373-1-1-EP-0-EMT-0-ANY-0.tiff",
    "1005199-1-poloha-467-1-1-EP-0-EMT-0-ANY-0.tiff",
}

NOU_SUBTYPE_TO_PAM50 = {
    1.0: ("LumA", 0),
    2.0: ("LumB", 1),
    3.0: ("Her2", 3),
    4.0: ("Basal", 2),  # TN ≈ Basal-like
}

PAM50_LABEL_TO_NAME = {v[1]: v[0] for v in NOU_SUBTYPE_TO_PAM50.values()}

FILENAME_RE = re.compile(
    r"^(?P<case_id>\d+)-(?P<img_idx>\d+)-poloha-(?P<poloha>\d+)"
    r"-(?P<attr1>\d+)-(?P<attr2>\d+)"
    r"-EP-(?P<EP>\d+)-EMT-(?P<EMT>\d+)-ANY-(?P<ANY>\d+)\.tiff$"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare NOU dataset manifest")
    parser.add_argument(
        "--tiff_dir",
        type=str,
        default=".datasets/nou/crops - Copy",
        help="Directory containing NOU TIFF files",
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        default=".datasets/nou/Anotacia_CTC_blocky.xls",
        help="Path to annotation Excel file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".scratch/nou_validation/metadata",
        help="Output directory for manifest CSVs",
    )
    parser.add_argument(
        "--label_strategy",
        type=str,
        default="patient_majority",
        choices=["patient_majority", "per_block"],
        help="PAM50 label assignment: 'patient_majority' assigns a single "
             "subtype per patient via majority vote across tissue blocks "
             "(default); 'per_block' uses original per-block IHC annotations",
    )
    return parser.parse_args()


def read_blocky_mapping(excel_path):
    """Read poloha -> TRU cislo mapping from 'Blocky' sheet."""
    df = pd.read_excel(excel_path, sheet_name="Blocky", header=None)
    mapping = {}
    for _, row in df.iterrows():
        poloha = row.iloc[0]
        tru = row.iloc[2]
        if pd.notna(poloha) and pd.notna(tru):
            try:
                mapping[int(poloha)] = int(float(tru))
            except (ValueError, TypeError):
                continue
    return mapping


def read_databaza(excel_path):
    """Read clinical data from 'databaza' sheet, indexed by TRU cislo."""
    df = pd.read_excel(excel_path, sheet_name="databaza", header=None)
    records = {}
    for r in range(4, len(df)):
        tru_raw = df.iloc[r, 3]
        if pd.isna(tru_raw) or tru_raw == "":
            continue
        try:
            tru = int(float(tru_raw))
        except (ValueError, TypeError):
            continue

        subtype_raw = df.iloc[r, 52]
        mol_subtype = None
        if pd.notna(subtype_raw) and subtype_raw not in ("", "NA", "X", "bez malignity"):
            try:
                mol_subtype = float(subtype_raw)
            except (ValueError, TypeError):
                pass

        grade_raw = df.iloc[r, 27]
        grade = None
        if pd.notna(grade_raw) and grade_raw != "":
            try:
                grade = int(float(grade_raw))
            except (ValueError, TypeError):
                pass

        hist_type_raw = df.iloc[r, 25]
        hist_type = None
        if pd.notna(hist_type_raw) and hist_type_raw != "":
            try:
                hist_type = int(float(hist_type_raw))
            except (ValueError, TypeError):
                pass

        records[tru] = {
            "mol_subtype_raw": mol_subtype,
            "grade": grade,
            "hist_type": hist_type,
        }
    return records


def validate_image(path):
    """Check that image opens correctly, return (width, height) or None."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def compute_patient_labels(manifest):
    """Compute a single PAM50 label per patient via majority vote.

    For each case_id, collects all valid per-block PAM50 labels and picks
    the most frequent one (ties broken by lower class index = more common
    subtype in the general population: LumA=0 > LumB=1 > Basal=2 > Her2=3).

    Returns a dict {case_id: (pam50_name, pam50_label)} for patients that
    have at least one block with a valid subtype annotation.
    """
    valid = manifest[manifest["is_valid"] & manifest["block_pam50_label"].notna()]
    patient_labels = {}
    for case_id, group in valid.groupby("case_id"):
        counts = Counter(int(v) for v in group["block_pam50_label"])
        if not counts:
            continue
        winner = min(counts.keys(), key=lambda k: (-counts[k], k))
        patient_labels[case_id] = (PAM50_LABEL_TO_NAME[winner], winner)
    return patient_labels


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load annotation mappings ---
    print("Reading annotations from Excel...")
    poloha_to_tru = read_blocky_mapping(args.excel_path)
    print(f"  Blocky: {len(poloha_to_tru)} poloha -> TRU mappings")

    databaza = read_databaza(args.excel_path)
    print(f"  databaza: {len(databaza)} TRU -> clinical records")

    # --- Parse TIFF filenames ---
    print(f"\nScanning TIFFs in {args.tiff_dir}...")
    tiff_files = sorted(f for f in os.listdir(args.tiff_dir) if f.endswith(".tiff"))
    print(f"  Found {len(tiff_files)} TIFF files")

    rows = []
    for fname in tiff_files:
        m = FILENAME_RE.match(fname)
        if not m:
            print(f"  WARNING: Could not parse filename: {fname}")
            continue

        case_id = m.group("case_id")
        poloha = int(m.group("poloha"))
        ep = int(m.group("EP"))
        emt = int(m.group("EMT"))
        any_ctc = int(m.group("ANY"))
        slide_id = fname.replace(".tiff", "")

        is_corrupted = fname in CORRUPTED_FILES

        tru_cislo = poloha_to_tru.get(poloha)
        clinical = databaza.get(tru_cislo, {}) if tru_cislo else {}

        mol_subtype_raw = clinical.get("mol_subtype_raw")
        block_pam50_label = None
        block_pam50_name = None
        if mol_subtype_raw in NOU_SUBTYPE_TO_PAM50:
            block_pam50_name, block_pam50_label = NOU_SUBTYPE_TO_PAM50[mol_subtype_raw]

        width, height = None, None
        is_valid = not is_corrupted
        if is_valid:
            size = validate_image(os.path.join(args.tiff_dir, fname))
            if size is None:
                print(f"  WARNING: Image validation failed: {fname}")
                is_valid = False
            else:
                width, height = size

        rows.append({
            "slide_id": slide_id,
            "case_id": case_id,
            "filename": fname,
            "poloha": poloha,
            "tru_cislo": tru_cislo,
            "EP": ep,
            "EMT": emt,
            "ANY": any_ctc,
            "mol_subtype_raw": mol_subtype_raw,
            "block_pam50_name": block_pam50_name,
            "block_pam50_label": block_pam50_label,
            "grade": clinical.get("grade"),
            "hist_type": clinical.get("hist_type"),
            "is_valid": is_valid,
            "width": width,
            "height": height,
        })

    manifest = pd.DataFrame(rows)

    # --- Assign patient-level PAM50 labels ---
    use_patient = args.label_strategy == "patient_majority"
    if use_patient:
        patient_labels = compute_patient_labels(manifest)
        manifest["pam50_name"] = manifest["case_id"].map(
            lambda c: patient_labels[c][0] if c in patient_labels else None
        )
        manifest["pam50_label"] = manifest["case_id"].map(
            lambda c: patient_labels[c][1] if c in patient_labels else None
        )
        print(f"\nLabel strategy: patient_majority ({len(patient_labels)} patients with labels)")
    else:
        manifest["pam50_name"] = manifest["block_pam50_name"]
        manifest["pam50_label"] = manifest["block_pam50_label"]
        print("\nLabel strategy: per_block (original per-tissue-block annotations)")

    # --- Summary statistics ---
    n_valid = manifest["is_valid"].sum()
    n_corrupted = (~manifest["is_valid"]).sum()
    n_with_pam50 = manifest.loc[manifest["is_valid"], "pam50_label"].notna().sum()
    n_patients = manifest.loc[manifest["is_valid"], "case_id"].nunique()

    print(f"\n=== Manifest Summary ===")
    print(f"Total images: {len(manifest)}")
    print(f"Valid images: {n_valid}")
    print(f"Corrupted/invalid: {n_corrupted}")
    print(f"With PAM50 label: {n_with_pam50}")
    print(f"Unique patients (valid): {n_patients}")

    pam50_valid = manifest.loc[manifest["is_valid"] & manifest["pam50_label"].notna()]

    print(f"\nPAM50 distribution (valid images):")
    print(pam50_valid["pam50_name"].value_counts().to_string(header=False))

    print(f"\nPAM50 distribution (unique patients):")
    patient_pam50 = pam50_valid.drop_duplicates("case_id")["pam50_name"].value_counts()
    print(patient_pam50.to_string(header=False))

    if use_patient:
        block_with = manifest.loc[
            manifest["is_valid"] & manifest["block_pam50_label"].notna()
        ]
        n_block_labels = block_with["block_pam50_label"].nunique()
        mixed = block_with.groupby("case_id")["block_pam50_label"].nunique()
        n_mixed = (mixed > 1).sum()
        print(f"\n  (Block-level annotation had {n_block_labels} distinct subtypes, "
              f"{n_mixed}/{len(mixed)} patients with mixed per-block labels)")

    print(f"\nCTC ANY distribution (valid images):")
    ctc_valid = manifest.loc[manifest["is_valid"]]
    print(ctc_valid["ANY"].value_counts().sort_index().to_string(header=False))

    # --- Save manifest ---
    manifest_path = os.path.join(args.output_dir, "nou_slide_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"\nSaved manifest: {manifest_path}")

    # --- PAM50 dataset CSV (CLAM format) ---
    pam50_df = pam50_valid[["case_id", "slide_id", "pam50_label"]].copy()
    pam50_df = pam50_df.rename(columns={"pam50_label": "label"})
    pam50_df["label"] = pam50_df["label"].astype(int)
    pam50_path = os.path.join(args.output_dir, "nou_pam50_dataset.csv")
    pam50_df.to_csv(pam50_path, index=False)
    print(f"Saved PAM50 dataset: {pam50_path} ({len(pam50_df)} slides)")

    # --- CTC dataset CSV (CLAM format) ---
    ctc_df = ctc_valid[["case_id", "slide_id", "ANY"]].copy()
    ctc_df = ctc_df.rename(columns={"ANY": "label"})
    ctc_path = os.path.join(args.output_dir, "nou_ctc_dataset.csv")
    ctc_df.to_csv(ctc_path, index=False)
    print(f"Saved CTC dataset: {ctc_path} ({len(ctc_df)} slides)")


if __name__ == "__main__":
    main()
