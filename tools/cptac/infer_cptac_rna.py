"""
PAM50 RNA-only inference on CPTAC-BRCA
=======================================
Runs the 10-fold RNA MLP trained on TCGA-BRCA over the CPTAC expression table,
frozen, no fine-tuning. This is the attribution control for the fusion result:
it says how much of the multimodal external number the RNA branch carries on its
own, and -- because PAM50 is defined from expression -- it doubles as the check
that the GDC-derived tables really are on a common scale. A collapse here means
the harmonisation failed, not that the model failed.

Predictions are per case (RNA is patient-level). The output schema matches the
WSI/fusion prediction CSVs so the report notebook reads it unchanged; slide_id
is set to the case id.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "CLAM"))

from models.model_rna import RNA_MLP

LABEL_MAP = {0: "LumA", 1: "LumB", 2: "Basal", 3: "Her2"}
LABEL_IDX = {name: i for i, name in LABEL_MAP.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="RNA-only PAM50 inference on CPTAC-BRCA")
    parser.add_argument("--tabular_csv", default=".scratch/rna-gdc/CPTAC_BRCA_RNA_gdc_4class_clam.csv.gz")
    parser.add_argument("--ckpt_dir", default=".scratch/results/pam50_rna_only_gdc_4class_s1")
    parser.add_argument("--output_dir", default=".scratch/cptac_validation/results/predictions_rna")
    parser.add_argument("--n_folds", type=int, default=10)
    return parser.parse_args()


def build_matrix(rna_table, ckpt):
    """Select the fold's genes by name, z-score them, impute absent genes at z=0."""
    names = list(ckpt["selected_feature_names"])
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)

    present = rna_table.columns.intersection(names)
    matrix = np.tile(mean, (len(rna_table), 1))
    if len(present):
        pos = {name: i for i, name in enumerate(names)}
        matrix[:, [pos[n] for n in present]] = rna_table[present].to_numpy(dtype=np.float32)

    return ((matrix - mean) / std).astype(np.float32), len(names) - len(present)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rna = pd.read_csv(args.tabular_csv)
    meta = rna[["case_id", "label"]].copy()
    meta["true_label"] = meta["label"].map(LABEL_IDX)
    features = rna.set_index("case_id").drop(
        columns=[c for c in ("sample", "label", "sample_type_code") if c in rna.columns])
    print(f"CPTAC RNA: {features.shape[0]} cases x {features.shape[1]} genes")
    print(f"class counts: {meta['label'].value_counts().to_dict()}")

    all_fold_probs = []
    for fold in range(args.n_folds):
        ckpt_path = os.path.join(args.ckpt_dir, f"s_{fold}_checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"Fold {fold}: checkpoint not found at {ckpt_path}, skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        matrix, n_missing = build_matrix(features, ckpt)

        model = RNA_MLP(
            input_dim=ckpt["input_dim"],
            hidden_dims=tuple(ckpt["hidden_dims"]),
            dropout=ckpt["dropout"],
            n_classes=len(ckpt["class_names"]),
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device).eval()

        with torch.inference_mode():
            logits = model(torch.from_numpy(matrix).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        fold_df = pd.DataFrame({
            "slide_id": features.index,
            "case_id": features.index,
            "true_label": meta["true_label"].to_numpy(),
            "true_name": meta["label"].to_numpy(),
            "pred_label": probs.argmax(axis=1),
            "pred_name": [LABEL_MAP[i] for i in probs.argmax(axis=1)],
            "p_LumA": probs[:, 0], "p_LumB": probs[:, 1],
            "p_Basal": probs[:, 2], "p_Her2": probs[:, 3],
        })
        fold_df.to_csv(os.path.join(args.output_dir, f"fold_{fold}_predictions.csv"), index=False)
        print(f"Fold {fold}: {n_missing} genes absent from CPTAC, "
              f"accuracy {(fold_df['true_label'] == fold_df['pred_label']).mean():.4f}")
        all_fold_probs.append(probs)

    if all_fold_probs:
        mean_probs = np.mean(all_fold_probs, axis=0)
        ensemble = pd.DataFrame({
            "slide_id": features.index,
            "case_id": features.index,
            "true_label": meta["true_label"].to_numpy(),
            "true_name": meta["label"].to_numpy(),
            "pred_label": mean_probs.argmax(axis=1),
            "pred_name": [LABEL_MAP[i] for i in mean_probs.argmax(axis=1)],
            "p_LumA": mean_probs[:, 0], "p_LumB": mean_probs[:, 1],
            "p_Basal": mean_probs[:, 2], "p_Her2": mean_probs[:, 3],
            "max_prob": mean_probs.max(axis=1),
        })
        path = os.path.join(args.output_dir, "ensemble_predictions.csv")
        ensemble.to_csv(path, index=False)

        print(f"\n=== Ensemble Results (case-level, n = {len(ensemble)}) ===")
        print(f"Accuracy: {(ensemble['true_label'] == ensemble['pred_label']).mean():.4f}")
        print(f"\nPredicted distribution:")
        print(ensemble["pred_name"].value_counts().to_string())
        print(f"\nSaved ensemble predictions: {path}")


if __name__ == "__main__":
    main()
