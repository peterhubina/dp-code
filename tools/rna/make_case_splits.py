#!/usr/bin/env python3
"""Convert CLAM's slide-level split files into case-level ones for train_rna.py.

CLAM's splits_<fold>.csv lists slide ids; the RNA table is one row per case, and
train_rna.py matches split entries against its `sample` column. Rewriting the
splits in case-id space lets the RNA-only model train on exactly the same patient
partition as the fusion model, so their cross-validation numbers are comparable
and no fusion-train patient turns up in an RNA-only test fold.

Cases that appear in the CLAM splits but carry no RNA row are dropped, and the
count is reported per fold.

    python tools/rna/make_case_splits.py
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_dir", default=str(ROOT / "project/CLAM/splits/tcga_brca_subtyping_100"))
    parser.add_argument("--dataset_csv", default=str(ROOT / "project/CLAM/dataset_csv/tcga_brca_subtyping.csv"))
    parser.add_argument("--rna_csv", default=str(ROOT / ".scratch/rna-gdc/TCGA_BRCA_RNA_gdc_4class_clam.csv.gz"))
    parser.add_argument("--out_dir", default=str(ROOT / "project/CLAM/splits/tcga_brca_subtyping_100_case"))
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slide_to_case = pd.read_csv(args.dataset_csv).set_index("slide_id")["case_id"].to_dict()
    rna_cases = set(pd.read_csv(args.rna_csv, usecols=["case_id"])["case_id"])
    print(f"RNA table: {len(rna_cases)} cases")

    for fold in range(args.k):
        source = Path(args.split_dir) / f"splits_{fold}.csv"
        split_df = pd.read_csv(source, dtype=str)

        columns, dropped = {}, {}
        for split in ("train", "val", "test"):
            slides = split_df[split].dropna().astype(str)
            cases = pd.unique(pd.Series([slide_to_case.get(s) for s in slides]).dropna())
            kept = [c for c in cases if c in rna_cases]
            dropped[split] = len(cases) - len(kept)
            columns[split] = pd.Series(kept)

        overlap = set(columns["train"]) & (set(columns["val"]) | set(columns["test"]))
        if overlap:
            raise SystemExit(f"fold {fold}: {len(overlap)} cases appear in both train and val/test")

        pd.DataFrame(columns).to_csv(out_dir / f"splits_{fold}.csv", index=True)
        print(f"fold {fold}: train {len(columns['train'])}, val {len(columns['val'])}, "
              f"test {len(columns['test'])} cases "
              f"(dropped for missing RNA: {dropped})")

    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
