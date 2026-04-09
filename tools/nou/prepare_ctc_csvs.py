#!/usr/bin/env python3
"""Generate CLAM-format dataset CSVs for NOU CTC binary classification tasks.

Reads the NOU slide manifest and produces two CSVs:
  - nou_ctc_ep.csv   : EP=0 ('no_ep') vs EP=1 ('ep')
  - nou_ctc_emt.csv  : EMT=0 ('no_emt') vs EMT=1 ('emt')

Only slides with is_valid=True are included.

Output location: project/CLAM/dataset_csv/
"""

import argparse
import os
import sys

import pandas as pd


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
    ep_df = df[["case_id", "slide_id", "EP"]].copy()
    ep_df["label"] = ep_df["EP"].map({0: "no_ep", 1: "ep"})
    ep_df = ep_df[["case_id", "slide_id", "label"]].sort_values(
        ["case_id", "slide_id"]
    )

    ep_path = os.path.join(output_dir, "nou_ctc_ep.csv")
    ep_df.to_csv(ep_path, index=False)
    print(f"\nWrote {ep_path}")
    print(f"  EP distribution: {ep_df['label'].value_counts().to_dict()}")

    # --- EMT task ---
    emt_df = df[["case_id", "slide_id", "EMT"]].copy()
    emt_df["label"] = emt_df["EMT"].map({0: "no_emt", 1: "emt"})
    emt_df = emt_df[["case_id", "slide_id", "label"]].sort_values(
        ["case_id", "slide_id"]
    )

    emt_path = os.path.join(output_dir, "nou_ctc_emt.csv")
    emt_df.to_csv(emt_path, index=False)
    print(f"\nWrote {emt_path}")
    print(f"  EMT distribution: {emt_df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
