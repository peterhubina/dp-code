from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


CLASS_4 = ["LumA", "LumB", "Basal", "Her2"]
CLASS_5 = [*CLASS_4, "Normal"]
TCGA_BARCODE_RE = re.compile(
    r"^(?P<slide>(?P<patient>TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(?P<sample_type>\d{2})[A-Z]-\d{2}-DX\d)"
)


def sample_type_code(sample: str) -> str:
    parts = sample.split("-")
    return parts[3][:2] if len(parts) > 3 else ""


def build_wsi_index(embeddings_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(embeddings_dir.glob("*.h5")):
        match = TCGA_BARCODE_RE.match(path.name)
        rows.append(
            {
                "embedding_file": path.name,
                "embedding_path": str(path),
                "patient": match.group("patient") if match else path.name[:12],
                "slide": match.group("slide") if match else path.stem.split(".")[0],
                "sample_type_code": match.group("sample_type") if match else "",
                "file_size_bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def save_training_split(
    *,
    x_all: pd.DataFrame,
    labels: pd.DataFrame,
    classes: list[str],
    name: str,
    output_dir: Path,
) -> pd.DataFrame:
    y = labels[
        (labels["sample_type_code"] == "01")
        & labels["BRCA_Subtype_PAM50"].isin(classes)
    ].copy()
    y = y.drop_duplicates(subset=["sample"]).sort_values("sample")
    x = x_all.loc[y["sample"]]

    x.to_csv(output_dir / f"TCGA_BRCA_RNA_primary_tumor_{name}_X.tsv.gz", sep="\t", compression="gzip")
    y.to_csv(output_dir / f"TCGA_BRCA_RNA_primary_tumor_{name}_y.csv", index=False)

    clam = x.reset_index().merge(y, on="sample", how="inner")
    clam = clam.rename(columns={"patient": "case_id", "BRCA_Subtype_PAM50": "label"})
    metadata_cols = ["case_id", "sample", "label", "sample_type_code"]
    feature_cols = [col for col in clam.columns if col not in metadata_cols]
    clam = clam[metadata_cols + feature_cols]
    clam.to_csv(
        output_dir / f"TCGA_BRCA_RNA_primary_tumor_{name}_clam.csv.gz",
        index=False,
        compression="gzip",
    )
    return y


def save_wsi_matches(
    *,
    y: pd.DataFrame,
    wsi_index: pd.DataFrame,
    name: str,
    output_dir: Path,
) -> pd.DataFrame:
    matched_slides = wsi_index.merge(
        y[["sample", "patient", "BRCA_Subtype_PAM50"]],
        on="patient",
        how="inner",
    ).sort_values(["patient", "embedding_file"])

    patient_summary = (
        matched_slides.groupby(["patient", "sample", "BRCA_Subtype_PAM50"], as_index=False)
        .agg(
            n_wsi_embeddings=("embedding_file", "size"),
            embedding_files=("embedding_file", lambda values: ";".join(values)),
        )
        .sort_values("patient")
    )

    matched_slides.to_csv(output_dir / f"TCGA_BRCA_WSI_RNA_matched_slides_{name}.csv", index=False)
    patient_summary.to_csv(output_dir / f"TCGA_BRCA_WSI_RNA_matched_patients_{name}.csv", index=False)

    unmatched_rna = y[~y["patient"].isin(wsi_index["patient"])].sort_values("patient")
    unmatched_wsi = wsi_index[~wsi_index["patient"].isin(y["patient"])].sort_values("patient")
    unmatched_rna.to_csv(output_dir / f"TCGA_BRCA_RNA_unmatched_to_WSI_{name}.csv", index=False)
    unmatched_wsi.to_csv(output_dir / f"TCGA_BRCA_WSI_unmatched_to_RNA_{name}.csv", index=False)

    return matched_slides


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parents[1]
    output_dir = workspace_root / ".scratch" / "TCGA-BRCA-rna"
    embeddings_dir = workspace_root / ".datasets" / "tcga-brca" / "embeddings"

    rna_path = output_dir / "TCGA_BRCA_HiSeqV2.tsv.gz"
    labels_path = output_dir / "TCGA_BRCA_sample_PAM50_labels.csv"

    rna = pd.read_csv(rna_path, sep="\t", index_col=0)
    labels = pd.read_csv(labels_path)
    labels["sample_type_code"] = labels["sample"].map(sample_type_code)

    x_all = rna.T
    x_all.index.name = "sample"

    wsi_index = build_wsi_index(embeddings_dir)
    wsi_index.to_csv(output_dir / "TCGA_BRCA_WSI_embedding_index.csv", index=False)

    y_5class = save_training_split(
        x_all=x_all,
        labels=labels,
        classes=CLASS_5,
        name="5class",
        output_dir=output_dir,
    )
    y_4class = save_training_split(
        x_all=x_all,
        labels=labels,
        classes=CLASS_4,
        name="4class",
        output_dir=output_dir,
    )

    matched_5class = save_wsi_matches(
        y=y_5class,
        wsi_index=wsi_index,
        name="5class",
        output_dir=output_dir,
    )
    matched_4class = save_wsi_matches(
        y=y_4class,
        wsi_index=wsi_index,
        name="4class",
        output_dir=output_dir,
    )

    summary = {
        "rna_matrix_genes_x_samples_shape": list(rna.shape),
        "all_rna_samples": int(x_all.shape[0]),
        "n_genes": int(x_all.shape[1]),
        "primary_tumor_5class_samples": int(len(y_5class)),
        "primary_tumor_5class_label_counts": y_5class["BRCA_Subtype_PAM50"].value_counts().to_dict(),
        "primary_tumor_4class_samples": int(len(y_4class)),
        "primary_tumor_4class_label_counts": y_4class["BRCA_Subtype_PAM50"].value_counts().to_dict(),
        "wsi_embedding_files": int(len(wsi_index)),
        "wsi_embedding_patients": int(wsi_index["patient"].nunique()),
        "matched_5class_slides": int(len(matched_5class)),
        "matched_5class_patients": int(matched_5class["patient"].nunique()),
        "matched_4class_slides": int(len(matched_4class)),
        "matched_4class_patients": int(matched_4class["patient"].nunique()),
    }
    (output_dir / "TCGA_BRCA_RNA_WSI_pipeline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved prepared files to: {output_dir}")


if __name__ == "__main__":
    main()
