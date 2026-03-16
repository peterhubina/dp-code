"""
Map UNI2-h .h5 embeddings to TCGA-BRCA recurrence labels.

Steps:
1. List all .h5 files in .datasets/embeddings/
2. Extract patient ID (first 12 chars of filename)
3. Join with tools/data/tcga_brca_labels.csv on patient ID
4. Save project/CLAM/dataset_csv/tcga_brca_recurrence.csv

Output CSV columns: case_id, slide_id, label
"""

import os
import glob
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_DIR = os.path.join(REPO_ROOT, ".datasets", "embeddings")
LABELS_CSV = os.path.join(REPO_ROOT, "tools", "data", "tcga_brca_labels.csv")
OUTPUT_CSV = os.path.join(REPO_ROOT, "project", "CLAM", "dataset_csv", "tcga_brca_recurrence.csv")


def main():
    # 1. Find all .h5 files (exclude TCGA subdirectory with archives)
    h5_files = glob.glob(os.path.join(EMBEDDINGS_DIR, "*.h5"))
    print(f"Found {len(h5_files)} .h5 files in {EMBEDDINGS_DIR}")

    if len(h5_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in {EMBEDDINGS_DIR}")

    # 2. Build slide DataFrame
    records = []
    for h5_path in sorted(h5_files):
        filename = os.path.basename(h5_path)
        slide_id = filename.replace(".h5", "")
        # Patient ID = first 12 characters (TCGA-XX-XXXX)
        case_id = slide_id[:12]
        records.append({"case_id": case_id, "slide_id": slide_id})

    slides_df = pd.DataFrame(records)
    print(f"Unique patients in embeddings: {slides_df['case_id'].nunique()}")

    # 3. Load labels
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(
            f"Labels file not found: {LABELS_CSV}\n"
            "Run tools/fetch_tcga_labels.py first."
        )
    labels_df = pd.read_csv(LABELS_CSV)[["case_id", "DFI", "DFI_STATUS", "label"]]
    print(f"Loaded {len(labels_df)} labelled patients from {LABELS_CSV}")

    # 4. Join
    merged = slides_df.merge(labels_df, on="case_id", how="left")

    # Report unmatched slides
    unmatched = merged[merged["label"].isna()]
    matched = merged[merged["label"].notna()]
    print(f"\nMatched: {len(matched)} slides ({matched['case_id'].nunique()} patients)")
    print(f"Unmatched (no clinical label): {len(unmatched)} slides")
    if len(unmatched) > 0:
        print("Unmatched case IDs (first 10):")
        print(unmatched["case_id"].unique()[:10].tolist())

    # 5. Save only matched slides
    output_df = matched[["case_id", "slide_id", "label"]].reset_index(drop=True)

    print(f"\nLabel distribution:")
    print(output_df["label"].value_counts())
    recurrence_rate = (output_df["label"] == "recurrence").mean() * 100
    print(f"Recurrence rate: {recurrence_rate:.1f}%")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV} ({len(output_df)} rows)")


if __name__ == "__main__":
    main()
