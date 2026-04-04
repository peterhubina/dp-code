"""
Prepare HSI-BC-Recurrence manifest for CLAM PAM50 pipeline.

Reads the clinical Excel file, maps IHC-surrogate molecular subtypes to
PAM50-equivalent labels, and outputs:
  1. CLAM dataset CSV (case_id, slide_id, label) for inference
  2. Full clinical metadata CSV for multimodal experiments
"""

import argparse
import os

import pandas as pd

IHC_TO_PAM50 = {
    0: "LumA",   # Luminal A: ER+/PR+/HER2-/KI67 low
    1: "LumB",   # Luminal B HER2-: ER+/PR±/HER2-/KI67 high
    2: "LumB",   # Luminal B HER2+: ER+/PR+/HER2+/KI67 high → merged into LumB
    3: "Her2",   # HER2-enriched: ER-/PR-/HER2+
    4: "Basal",  # Triple-negative: ER-/PR-/HER2-
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare HSI-BC manifest for CLAM")
    parser.add_argument(
        "--excel_path", type=str,
        default=".datasets/HistologyHSI-BC-Recurrence/Histology_HSI_BRCA_Recurrence.xlsx",
    )
    parser.add_argument(
        "--wsi_dir", type=str,
        default=".datasets/HistologyHSI-BC-Recurrence/01_01_Histological_Images",
    )
    parser.add_argument(
        "--clam_csv_path", type=str,
        default="project/CLAM/dataset_csv/hsi_bc_pam50.csv",
    )
    parser.add_argument(
        "--clinical_csv_path", type=str,
        default=".scratch/hsi_bc_recurrence/metadata/hsi_bc_clinical.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    clinical = pd.read_excel(args.excel_path, sheet_name="Clinical Data")
    print(f"Loaded clinical data: {len(clinical)} patients")

    # Verify all WSI files exist on disk
    missing = []
    for case_id in clinical["Case ID"]:
        mrxs = os.path.join(args.wsi_dir, f"{case_id}.mrxs")
        if not os.path.isfile(mrxs):
            missing.append(case_id)
    if missing:
        print(f"WARNING: {len(missing)} slides not found on disk: {missing}")

    # Map IHC subtypes to PAM50 labels
    clinical["pam50_label"] = clinical["Molecular_subtype"].map(IHC_TO_PAM50)
    unmapped = clinical["pam50_label"].isna().sum()
    if unmapped:
        print(f"WARNING: {unmapped} patients have unmapped molecular subtypes")

    # Build CLAM dataset CSV (matches tcga_brca_subtyping.csv format)
    clam_rows = []
    for _, row in clinical.iterrows():
        case_id = str(row["Case ID"])
        clam_rows.append({
            "case_id": case_id,
            "slide_id": case_id,
            "label": row["pam50_label"],
        })

    clam_df = pd.DataFrame(clam_rows)
    os.makedirs(os.path.dirname(args.clam_csv_path), exist_ok=True)
    clam_df.to_csv(args.clam_csv_path, index=False)
    print(f"\nCLAM dataset CSV saved: {args.clam_csv_path}")
    print(f"  {len(clam_df)} slides")
    print(f"  Label distribution:\n{clam_df['label'].value_counts().to_string()}")

    # Save full clinical metadata
    os.makedirs(os.path.dirname(args.clinical_csv_path), exist_ok=True)
    clinical.to_csv(args.clinical_csv_path, index=False)
    print(f"\nClinical metadata saved: {args.clinical_csv_path}")


if __name__ == "__main__":
    main()
