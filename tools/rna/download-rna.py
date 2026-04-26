from pathlib import Path

import pandas as pd

script_dir = Path(__file__).resolve().parent
workspace_root = script_dir.parents[1]
output_dir = workspace_root / ".scratch" / "TCGA-BRCA-rna"
output_dir.mkdir(parents=True, exist_ok=True)

# Processed TCGA-BRCA RNA-seq expression from UCSC Xena
rna_url = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap/HiSeqV2.gz"

# TCGA-BRCA phenotype / clinical matrix
pheno_url = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix"

rna = pd.read_csv(rna_url, sep="\t", index_col=0)
pheno = pd.read_csv(pheno_url, sep="\t", index_col=0)

rna.to_csv(output_dir / "TCGA_BRCA_HiSeqV2.tsv.gz", sep="\t", compression="gzip")
pheno.to_csv(output_dir / "TCGA_BRCA_clinicalMatrix.tsv", sep="\t")

labels_path = (
    workspace_root
    / ".scratch"
    / "TCGA-BRCA-additional"
    / "TCGA_BRCA_PAM50_labels.csv"
)
if labels_path.exists():
    labels = pd.read_csv(labels_path)
    sample_labels = pd.DataFrame({"sample": rna.columns})
    sample_labels["patient"] = sample_labels["sample"].str.slice(0, 12)
    sample_labels = sample_labels.merge(labels, on="patient", how="left")
    sample_labels.to_csv(output_dir / "TCGA_BRCA_sample_PAM50_labels.csv", index=False)

print(rna.shape)
print(pheno.shape)
print(pheno.columns[pheno.columns.str.contains("PAM50|subtype|Subtype", case=False, na=False)])
print(f"Saved outputs to: {output_dir}")