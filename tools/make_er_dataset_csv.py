#!/usr/bin/env python
"""Build the CLAM ``tcga_brca_er.csv`` manifest for binary ER-status subtyping.

Joins the ER label table (case_id, label) against the WSI embedding index,
keeping only primary-tumour slides, and emits one row per slide with columns
``case_id,slide_id,label`` -- the same shape as ``tcga_brca_subtyping.csv``.

Run from the repo root::

    python tools/make_er_dataset_csv.py
"""

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_LABELS = os.path.join(REPO_ROOT, "tools", "data", "tcga_brca_er_labels.csv")
DEFAULT_INDEX = os.path.join(
    REPO_ROOT, ".scratch", "TCGA-BRCA-rna", "TCGA_BRCA_WSI_embedding_index.csv"
)
DEFAULT_OUTPUT = os.path.join(
    REPO_ROOT, "project", "CLAM", "dataset_csv", "tcga_brca_er.csv"
)

PRIMARY_TUMOR_CODE = "01"
ER_LABELS = ("ER-negative", "ER-positive")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=DEFAULT_LABELS,
                        help="ER label CSV with columns case_id,label")
    parser.add_argument("--embedding_index", default=DEFAULT_INDEX,
                        help="WSI embedding index CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="destination dataset_csv path")
    return parser.parse_args()


def build_dataset_csv(labels_path, index_path, output_path):
    labels = pd.read_csv(labels_path, dtype=str)
    labels["case_id"] = labels["case_id"].str.strip()
    labels["label"] = labels["label"].str.strip()

    # Reading sample_type_code as string protects against pandas inferring an
    # int and dropping the leading zero (e.g. "01" -> 1); zfill re-pads either.
    index = pd.read_csv(index_path, dtype=str)
    index["sample_type_code"] = index["sample_type_code"].str.strip().str.zfill(2)
    primary = index[index["sample_type_code"] == PRIMARY_TUMOR_CODE].copy()

    primary["slide_id"] = primary["embedding_file"].str.replace(
        r"\.h5$", "", regex=True
    )
    primary["case_id"] = primary["patient"].str.strip()
    primary = primary[["case_id", "slide_id"]]

    merged = primary.merge(labels[["case_id", "label"]], on="case_id", how="inner")
    merged = merged[["case_id", "slide_id", "label"]].sort_values(
        ["case_id", "slide_id"]
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)

    labelled_cases = set(labels["case_id"])
    primary_cases = set(primary["case_id"])
    matched_cases = set(merged["case_id"])

    print(f"Wrote {output_path}")
    print(f"  slides written : {len(merged)}")
    print(f"  unique cases   : {merged['case_id'].nunique()}")
    print("  class balance  :")
    counts = merged["label"].value_counts()
    for name in ER_LABELS:
        n = int(counts.get(name, 0))
        pct = (100.0 * n / len(merged)) if len(merged) else 0.0
        print(f"    {name:12s}: {n:5d} ({pct:5.1f}%)")
    other = counts.drop(labels=[c for c in ER_LABELS if c in counts.index],
                        errors="ignore")
    for name, n in other.items():
        pct = (100.0 * int(n) / len(merged)) if len(merged) else 0.0
        print(f"    {name:12s}: {int(n):5d} ({pct:5.1f}%)  [unexpected label]")
    print(f"  label-cases with no matching embedding : "
          f"{len(labelled_cases - matched_cases)}")
    print(f"  primary embeddings with no ER label    : "
          f"{len(primary_cases - labelled_cases)}")

    return merged


def main():
    args = parse_args()

    if not os.path.exists(args.labels):
        print(f"ER label file not found: {args.labels}")
        print("It is produced concurrently; this script does NOT run until it exists.")
        print("Once the labels are in place, build the manifest with:")
        print("    python tools/make_er_dataset_csv.py")
        return 1

    if not os.path.exists(args.embedding_index):
        print(f"Embedding index not found: {args.embedding_index}")
        return 1

    build_dataset_csv(args.labels, args.embedding_index, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
