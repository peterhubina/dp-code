#!/usr/bin/env python3
"""Build TCGA-BRCA and CPTAC-BRCA expression tables from one GDC pipeline.

Both cohorts are read from GDC STAR-Counts files (hg38, GENCODE v36), collapsed
to log2(TPM + 1) over a *shared* gene axis, and written in the CLAM tabular
schema (case_id, sample, label, then one column per gene) that
dataset_modules/multimodal_dataset.py consumes.

The point is that the two tables come out of the same quantification pipeline in
the same identifier space, so a fusion head trained on the TCGA table can be
applied to the CPTAC table with no cross-platform normalisation at all. That
removes the confound that would otherwise sit between "the model does not
transfer" and "I mapped the expression scales badly".

    python tools/rna/build_gdc_expression.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["LumA", "LumB", "Basal", "Her2"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcga_rna_dir", default=str(ROOT / ".datasets/tcga-brca/rna"))
    parser.add_argument("--cptac_rna_dir", default=str(ROOT / ".datasets/cptac-brca/rna"))
    parser.add_argument("--tcga_labels", default=str(ROOT / "tools/data/tcga_brca_pam50_labels.csv"))
    parser.add_argument("--cptac_cohort", default=str(ROOT / ".datasets/cptac-brca/cohort.csv"))
    parser.add_argument("--out_dir", default=str(ROOT / ".scratch/rna-gdc"))
    parser.add_argument("--gene_type", default="protein_coding",
                        help="GENCODE biotype to keep; 'all' keeps everything")
    return parser.parse_args()


def read_star_counts(path):
    """One GDC STAR-Counts file -> Series of TPM indexed by unversioned Ensembl id."""
    df = pd.read_csv(path, sep="\t", skiprows=1,
                     usecols=["gene_id", "gene_name", "gene_type", "tpm_unstranded"])
    df = df[df["gene_id"].astype(str).str.startswith("ENSG")]
    df["gene_id"] = df["gene_id"].str.split(".").str[0]
    return df


def load_cohort(rna_dir, tag, gene_type):
    """Read every STAR-Counts file under rna_dir into a cases x genes log2(TPM+1) frame."""
    paths = sorted(Path(rna_dir).glob("*.star_counts.tsv"))
    if not paths:
        raise SystemExit(f"no STAR-Counts files under {rna_dir}")

    columns, gene_index, gene_meta = {}, None, None
    for path in tqdm(paths, desc=f"[{tag}] reading", unit="file"):
        df = read_star_counts(path)
        if gene_type != "all":
            df = df[df["gene_type"] == gene_type]
        df = df.drop_duplicates("gene_id").set_index("gene_id")
        if gene_index is None:
            gene_index = df.index
            gene_meta = df["gene_name"]
        elif not df.index.equals(gene_index):
            df = df.reindex(gene_index)
        case_id = path.name.split(".")[0]
        columns[case_id] = df["tpm_unstranded"].to_numpy(dtype=np.float32)

    matrix = pd.DataFrame(columns, index=gene_index).T          # cases x genes
    matrix = np.log2(matrix + 1.0)
    print(f"[{tag}] {matrix.shape[0]} cases x {matrix.shape[1]} genes "
          f"({gene_type}), log2(TPM+1) range {matrix.to_numpy().min():.2f}"
          f"..{matrix.to_numpy().max():.2f}")
    return matrix, gene_meta


def write_clam_table(matrix, labels, out_path, tag):
    """CLAM tabular schema: case_id, sample, label, then one column per gene."""
    keep = labels[labels["label"].isin(CLASSES)]
    rows = matrix.index.intersection(keep["case_id"])
    dropped_no_label = len(matrix.index) - len(rows)

    table = matrix.loc[rows].round(4).reset_index(names="case_id")
    table.insert(1, "sample", table["case_id"])
    table.insert(2, "label", table["case_id"].map(dict(zip(keep["case_id"], keep["label"]))))
    table.to_csv(out_path, index=False, compression="gzip")

    print(f"[{tag}] wrote {out_path}")
    print(f"[{tag}]   {len(table)} cases kept, {dropped_no_label} dropped "
          f"(no 4-class PAM50 label)")
    print(f"[{tag}]   class counts: {table['label'].value_counts().to_dict()}")
    return table


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tcga, tcga_genes = load_cohort(args.tcga_rna_dir, "tcga", args.gene_type)
    cptac, _ = load_cohort(args.cptac_rna_dir, "cptac", args.gene_type)

    shared = tcga.columns.intersection(cptac.columns)
    print(f"\nshared gene axis: {len(shared)} genes "
          f"(tcga {tcga.shape[1]}, cptac {cptac.shape[1]})")
    tcga, cptac = tcga[shared], cptac[shared]

    tcga_labels = pd.read_csv(args.tcga_labels)[["case_id", "label"]]
    cohort = pd.read_csv(args.cptac_cohort)
    cptac_labels = cohort.rename(columns={"PAM50": "label"})[["case_id", "label"]]

    print()
    write_clam_table(tcga, tcga_labels, out_dir / "TCGA_BRCA_RNA_gdc_4class_clam.csv.gz", "tcga")
    print()
    write_clam_table(cptac, cptac_labels, out_dir / "CPTAC_BRCA_RNA_gdc_4class_clam.csv.gz", "cptac")

    # Cross-cohort distribution check: with one pipeline these should sit on top
    # of each other. A visible offset here means the harmonisation claim is wrong.
    print("\nper-gene mean log2(TPM+1), TCGA vs CPTAC:")
    tm, cm = tcga.mean(axis=0), cptac.mean(axis=0)
    print(f"  TCGA  mean {tm.mean():.3f}  sd {tm.std():.3f}")
    print(f"  CPTAC mean {cm.mean():.3f}  sd {cm.std():.3f}")
    print(f"  correlation of per-gene means: {np.corrcoef(tm, cm)[0, 1]:.4f}")
    print(f"  median per-gene difference (CPTAC - TCGA): {(cm - tm).median():+.4f}")

    pd.DataFrame({"gene_id": shared, "gene_name": tcga_genes.reindex(shared).to_numpy(),
                  "tcga_mean": tm.to_numpy(), "cptac_mean": cm.to_numpy()}
                 ).to_csv(out_dir / "gene_axis.csv", index=False)
    print(f"\nwrote {out_dir / 'gene_axis.csv'}")


if __name__ == "__main__":
    main()
