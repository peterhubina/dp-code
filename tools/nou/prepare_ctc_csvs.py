#!/usr/bin/env python3
"""Generate CLAM-format dataset CSVs for NOU CTC binary classification tasks.

Reads the NOU slide manifest and produces three CSVs:
  - nou_ctc_ep.csv   : EP=0 ('no_ep') vs EP=1 ('ep')
  - nou_ctc_emt.csv  : EMT=0 ('no_emt') vs EMT=1 ('emt')
  - nou_ctc_any.csv  : ANY=0 ('no_any') vs ANY=1 ('any')

Only slides with is_valid=True are included.

Output location: project/CLAM/dataset_csv/
"""

import argparse
import os

import pandas as pd


def build_dataset_csv(df, source_col, label_map, output_path, case_id_col="case_id"):
    dataset_df = df[["slide_id", source_col]].copy()
    dataset_df["case_id"] = df[case_id_col].values
    dataset_df["label"] = dataset_df[source_col].map(label_map)
    dataset_df = dataset_df[["case_id", "slide_id", "label"]]
    dataset_df = dataset_df.sort_values(["case_id", "slide_id"])
    dataset_df.to_csv(output_path, index=False)
    print(f"\nWrote {output_path}")
    print(f"  Label distribution: {dataset_df['label'].value_counts().to_dict()}")
    print(f"  Unique case_id values: {dataset_df['case_id'].nunique()}")
    return dataset_df


def main():
    parser = argparse.ArgumentParser(description="Generate CTC dataset CSVs for CLAM")
    parser.add_argument(
        "--manifest",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            ".scratch",
            "nou_validation",
            "metadata",
            "nou_slide_manifest.csv",
        ),
        help="Path to the NOU slide manifest CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "project",
            "CLAM",
            "dataset_csv",
        ),
        help="Output directory for CLAM dataset CSVs",
    )
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    output_dir = os.path.abspath(args.output_dir)

    print(f"Reading manifest from: {manifest_path}")
    df = pd.read_csv(manifest_path)
    print(f"  Total slides in manifest: {len(df)}")

    # Filter to valid slides only
    df = df[df["is_valid"] == True].copy()
    print(f"  Valid slides: {len(df)}")
    print(f"  Unique patients: {df['case_id'].nunique()}")

    os.makedirs(output_dir, exist_ok=True)

    # --- EP task ---
    build_dataset_csv(
        df=df,
        source_col="EP",
        label_map={0: "no_ep", 1: "ep"},
        output_path=os.path.join(output_dir, "nou_ctc_ep.csv"),
    )

    # --- EMT task ---
    build_dataset_csv(
        df=df,
        source_col="EMT",
        label_map={0: "no_emt", 1: "emt"},
        output_path=os.path.join(output_dir, "nou_ctc_emt.csv"),
    )

    # --- ANY task ---
    # Treat every TIFF crop as its own grouping unit during split generation.
    build_dataset_csv(
        df=df,
        source_col="ANY",
        label_map={0: "no_any", 1: "any"},
        output_path=os.path.join(output_dir, "nou_ctc_any.csv"),
        case_id_col="slide_id",
    )


if __name__ == "__main__":
    main()
