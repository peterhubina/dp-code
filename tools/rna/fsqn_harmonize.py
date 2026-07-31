#!/usr/bin/env python3
"""Tier-2 sensitivity: map CPTAC RNA onto the UCSC Xena scale with FSQN.

The gated-fusion head published in the thesis was trained on Xena HiSeqV2
(log2 RSEM normalized_count, HUGO symbols of ~2013 vintage). CPTAC RNA is GDC
STAR TPM in current symbol space. Two things therefore have to be fixed before
that model can see CPTAC at all:

  symbols   ~15% of the genes the model selected are deprecated HUGO symbols
            (NOV, MLLT4, FAM134B, ...). They are resolved through the HGNC
            previous-symbol table; whatever still fails to match is left for the
            inference script to impute at the training mean.

  scale     Feature Specific Quantile Normalization (Franks, Cai & Whitfield,
            Bioinformatics 2018, doi:10.1093/bioinformatics/bty026): for each
            gene independently, the target cohort's distribution of that gene is
            mapped onto the training cohort's distribution of the same gene.
            FSQN was developed for exactly this -- applying a classifier trained
            on one expression platform to data from another -- and was
            re-validated for breast PAM50 by Skubleny et al. 2024
            (doi:10.1186/s12859-024-05759-w). It needs >=25 target samples;
            CPTAC has 114.

Unlike the Tier-1 route (tools/rna/build_gdc_expression.py), FSQN uses the target
cohort's own per-gene distribution, so this is unsupervised domain adaptation and
must be reported as such -- it is a weaker claim than the WSI-only result.

    python tools/rna/fsqn_harmonize.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["LumA", "LumB", "Basal", "Her2"]
METADATA = ("case_id", "sample", "label", "sample_type_code")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xena_csv",
                        default=str(ROOT / ".scratch/TCGA-BRCA-rna/TCGA_BRCA_RNA_primary_tumor_4class_clam.csv.gz"),
                        help="Reference: the Xena table the fusion head was trained on")
    parser.add_argument("--cptac_csv",
                        default=str(ROOT / ".scratch/rna-gdc/CPTAC_BRCA_RNA_gdc_4class_clam.csv.gz"),
                        help="CPTAC log2(TPM+1), Ensembl-id columns")
    parser.add_argument("--gene_axis", default=str(ROOT / ".scratch/rna-gdc/gene_axis.csv"))
    parser.add_argument("--hgnc", default=str(ROOT / ".scratch/rna-gdc/hgnc_complete_set.txt"))
    parser.add_argument("--out_csv",
                        default=str(ROOT / ".scratch/rna-gdc/CPTAC_BRCA_RNA_fsqn_xena_clam.csv.gz"))
    return parser.parse_args()


HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"


def load_symbol_aliases(hgnc_path):
    """prev/alias symbol -> current approved symbol."""
    hgnc_path = Path(hgnc_path)
    if not hgnc_path.is_file():
        import urllib.request
        hgnc_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching HGNC complete set -> {hgnc_path}")
        urllib.request.urlretrieve(HGNC_URL, hgnc_path)

    hgnc = pd.read_csv(hgnc_path, sep="\t", low_memory=False,
                       usecols=["symbol", "prev_symbol", "alias_symbol"])
    mapping = {}
    for _, row in hgnc.iterrows():
        current = row["symbol"]
        for field in ("prev_symbol", "alias_symbol"):
            value = row[field]
            if isinstance(value, str):
                for old in value.split("|"):
                    old = old.strip()
                    # never let an alias override a real approved symbol
                    if old and old not in mapping:
                        mapping[old] = current
    return mapping


def fsqn(target, reference):
    """Map each column of `target` onto the same column's distribution in `reference`.

    Rank the target values, convert ranks to plotting positions, and read off the
    corresponding quantiles of the reference distribution.
    """
    out = np.empty_like(target, dtype=np.float32)
    n = target.shape[0]
    positions = (np.arange(1, n + 1) - 0.5) / n
    for j in range(target.shape[1]):
        order = np.argsort(target[:, j], kind="mergesort")
        out[order, j] = np.quantile(reference[:, j], positions).astype(np.float32)
    return out


def main():
    args = parse_args()

    xena = pd.read_csv(args.xena_csv)
    xena_genes = [c for c in xena.columns if c not in METADATA]
    reference = xena[xena_genes].to_numpy(dtype=np.float32)
    print(f"Xena reference: {reference.shape[0]} samples x {len(xena_genes)} genes")

    cptac = pd.read_csv(args.cptac_csv)
    axis = pd.read_csv(args.gene_axis).set_index("gene_id")
    ensembl_to_symbol = axis["gene_name"].to_dict()

    meta = cptac[[c for c in METADATA if c in cptac.columns]].copy()
    expr = cptac.drop(columns=meta.columns)
    expr.columns = [ensembl_to_symbol.get(c, c) for c in expr.columns]
    # a handful of symbols map from several Ensembl ids; keep the highest-expressed
    expr = expr.T.groupby(level=0).max().T
    print(f"CPTAC: {expr.shape[0]} cases x {expr.shape[1]} unique symbols")

    aliases = load_symbol_aliases(args.hgnc)
    print(f"HGNC: {len(aliases)} previous/alias symbols")

    # For each Xena gene find the CPTAC column that carries it.
    available = set(expr.columns)
    resolved, direct, via_alias, unmatched = {}, 0, 0, []
    for gene in xena_genes:
        if gene in available:
            resolved[gene] = gene
            direct += 1
            continue
        current = aliases.get(gene)
        if current and current in available:
            resolved[gene] = current
            via_alias += 1
            continue
        unmatched.append(gene)

    print(f"\nXena genes matched to CPTAC: {len(resolved)}/{len(xena_genes)} "
          f"({100 * len(resolved) / len(xena_genes):.1f}%)")
    print(f"  direct symbol match: {direct}")
    print(f"  via HGNC previous/alias symbol: {via_alias}")
    print(f"  unmatched (left for mean-imputation at inference): {len(unmatched)}")
    print(f"  unmatched examples: {unmatched[:10]}")

    matched_genes = [g for g in xena_genes if g in resolved]
    target = expr[[resolved[g] for g in matched_genes]].to_numpy(dtype=np.float32)
    ref_cols = [xena_genes.index(g) for g in matched_genes]
    normalised = fsqn(target, reference[:, ref_cols])

    print(f"\nbefore FSQN: CPTAC mean {target.mean():.3f}, Xena mean {reference[:, ref_cols].mean():.3f}")
    print(f"after  FSQN: CPTAC mean {normalised.mean():.3f}")

    out = pd.DataFrame(normalised, columns=matched_genes).round(4)
    for col in reversed([c for c in METADATA if c in meta.columns]):
        out.insert(0, col, meta[col].to_numpy())
    out.to_csv(args.out_csv, index=False, compression="gzip")
    print(f"\nwrote {args.out_csv}")
    print(f"  {len(out)} cases x {len(matched_genes)} genes in Xena symbol space")


if __name__ == "__main__":
    main()
