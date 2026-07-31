"""
PAM50 WSI + RNA fusion inference on CPTAC-BRCA
===============================================
Runs the 10-fold gated-fusion ensemble trained on TCGA-BRCA over CPTAC slides,
with frozen weights and no fine-tuning.

Each fold ships its own RNA transform (s_<fold>_tabular_transform.json): the
10,000 genes it selected by training-fold variance, plus that fold's per-gene
mean and std. Those are applied here by *gene name*, not by the stored column
index, because the CPTAC table has a different column order. Genes the fold
selected that CPTAC does not carry are imputed at the training mean, i.e. z=0,
and the count is reported per fold -- under the GDC-derived tables that count
should be zero, and a non-zero value means the two tables are not on a common
gene axis.

--rna_ablate replaces every RNA input with the training mean (z=0), which turns
the fusion model into its WSI branch plus a constant. Comparing a normal run to
an ablated one shows how much the RNA branch actually contributes on CPTAC.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "CLAM"))

from models.model_multimodal import CLAMRNAFusion

LABEL_MAP = {0: "LumA", 1: "LumB", 2: "Basal", 3: "Her2"}


def parse_args():
    parser = argparse.ArgumentParser(description="PAM50 fusion inference on CPTAC-BRCA")
    parser.add_argument("--feature_dir", default=".datasets/cptac-brca/embeddings")
    parser.add_argument("--dataset_csv", default=".datasets/cptac-brca/cptac_brca_pam50_dataset.csv")
    parser.add_argument("--tabular_csv", default=".scratch/rna-gdc/CPTAC_BRCA_RNA_gdc_4class_clam.csv.gz")
    parser.add_argument("--ckpt_dir", default=".scratch/results/pam50_wsi_rna_gatedfusion_gdc_s1")
    parser.add_argument("--output_dir", default=".scratch/cptac_validation/results/predictions_fusion")
    parser.add_argument("--rna_ablate", action="store_true",
                        help="Replace RNA with the training mean (z=0) to isolate the WSI branch")
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=1536)
    parser.add_argument("--model_size", default="big")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fusion_mode", default="gated")
    parser.add_argument("--fusion_hidden_dim", type=int, default=32)
    parser.add_argument("--tabular_hidden_dim", type=int, default=256)
    parser.add_argument("--tabular_num_layers", type=int, default=2)
    return parser.parse_args()


def index_features(feature_dir):
    paths = sorted(Path(feature_dir).rglob("*.h5"), key=lambda p: len(p.parts))
    index = {}
    for path in paths:
        if path.is_file() and path.stem not in index:
            index[path.stem] = str(path)
    return index


def load_model(ckpt_path, tabular_input_dim, args, device):
    model = CLAMRNAFusion(
        wsi_model_type="clam_mb",
        gate=True,
        size_arg=args.model_size,
        dropout=args.dropout,
        k_sample=4,
        n_classes=args.n_classes,
        subtyping=True,
        embed_dim=args.embed_dim,
        tabular_input_dim=tabular_input_dim,
        tabular_hidden_dim=args.tabular_hidden_dim,
        tabular_num_layers=args.tabular_num_layers,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_mode=args.fusion_mode,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    clean = {k.replace(".module", ""): v for k, v in ckpt.items() if "instance_loss_fn" not in k}
    model.load_state_dict(clean, strict=True)
    model.to(device)
    model.eval()
    return model


def apply_transform(rna_table, transform, ablate=False):
    """Select the fold's genes by name, z-score them, impute absent genes at z=0."""
    names = list(transform["selected_feature_names"])
    mean = np.asarray(transform["mean"], dtype=np.float32)
    std = np.asarray(transform["std"], dtype=np.float32)

    present = rna_table.columns.intersection(names)
    missing = [n for n in names if n not in present]

    # start every gene at its training mean, then overwrite the ones CPTAC carries
    matrix = np.tile(mean, (len(rna_table), 1))
    if not ablate and len(present):
        pos = {name: i for i, name in enumerate(names)}
        cols = [pos[n] for n in present]
        matrix[:, cols] = rna_table[present].to_numpy(dtype=np.float32)

    return ((matrix - mean) / std).astype(np.float32), len(missing)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.rna_ablate:
        print("RNA ABLATION: all RNA inputs held at the training mean (z=0)")

    dataset = pd.read_csv(args.dataset_csv)
    feature_paths = index_features(args.feature_dir)
    missing_wsi = [s for s in dataset["slide_id"] if s not in feature_paths]
    if missing_wsi:
        print(f"WARNING: {len(missing_wsi)} slides missing WSI features, skipping them")
        dataset = dataset[~dataset["slide_id"].isin(missing_wsi)].reset_index(drop=True)

    rna = pd.read_csv(args.tabular_csv)
    rna_by_case = rna.set_index("case_id").drop(
        columns=[c for c in ("sample", "label", "sample_type_code") if c in rna.columns])
    missing_rna = sorted(set(dataset["case_id"]) - set(rna_by_case.index))
    if missing_rna:
        print(f"WARNING: {len(missing_rna)} cases missing RNA, dropping their slides: {missing_rna}")
        dataset = dataset[~dataset["case_id"].isin(missing_rna)].reset_index(drop=True)

    print(f"Dataset: {len(dataset)} slides / {dataset['case_id'].nunique()} cases")
    print(f"RNA table: {rna_by_case.shape[0]} cases x {rna_by_case.shape[1]} genes")

    all_fold_probs = []
    imputed_counts = []

    for fold in range(args.n_folds):
        ckpt_path = os.path.join(args.ckpt_dir, f"s_{fold}_checkpoint.pt")
        transform_path = os.path.join(args.ckpt_dir, f"s_{fold}_tabular_transform.json")
        if not os.path.exists(ckpt_path):
            print(f"Fold {fold}: checkpoint not found at {ckpt_path}, skipping")
            continue

        transform = json.loads(Path(transform_path).read_text())
        rna_matrix, n_missing = apply_transform(rna_by_case, transform, ablate=args.rna_ablate)
        rna_lookup = {case: rna_matrix[i] for i, case in enumerate(rna_by_case.index)}
        imputed_counts.append(n_missing)

        print(f"\nFold {fold}: {len(transform['selected_feature_names'])} genes, "
              f"{n_missing} absent from CPTAC and imputed at training mean")
        model = load_model(ckpt_path, len(transform["selected_feature_names"]), args, device)

        fold_results = []
        for _, row in dataset.iterrows():
            with h5py.File(feature_paths[row["slide_id"]], "r") as f:
                wsi = torch.from_numpy(f["features"][:]).float().to(device)
            tab = torch.from_numpy(rna_lookup[row["case_id"]]).float().unsqueeze(0).to(device)

            with torch.inference_mode():
                _, Y_prob, Y_hat, _, _ = model((wsi, tab))

            probs = Y_prob.cpu().numpy().squeeze()
            fold_results.append({
                "slide_id": row["slide_id"],
                "case_id": row["case_id"],
                "true_label": int(row["label"]),
                "true_name": LABEL_MAP[int(row["label"])],
                "pred_label": Y_hat.item(),
                "pred_name": LABEL_MAP[Y_hat.item()],
                "p_LumA": probs[0], "p_LumB": probs[1],
                "p_Basal": probs[2], "p_Her2": probs[3],
            })

        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(os.path.join(args.output_dir, f"fold_{fold}_predictions.csv"), index=False)
        print(f"  Fold {fold} accuracy: {(fold_df['true_label'] == fold_df['pred_label']).mean():.4f}")
        all_fold_probs.append(fold_df[["p_LumA", "p_LumB", "p_Basal", "p_Her2"]].values)

        del model
        torch.cuda.empty_cache()

    if all_fold_probs:
        mean_probs = np.mean(all_fold_probs, axis=0)
        ensemble = dataset[["slide_id", "case_id", "label"]].rename(columns={"label": "true_label"}).copy()
        ensemble["true_name"] = ensemble["true_label"].map(LABEL_MAP)
        ensemble["pred_label"] = mean_probs.argmax(axis=1)
        ensemble["pred_name"] = ensemble["pred_label"].map(LABEL_MAP)
        ensemble["p_LumA"] = mean_probs[:, 0]
        ensemble["p_LumB"] = mean_probs[:, 1]
        ensemble["p_Basal"] = mean_probs[:, 2]
        ensemble["p_Her2"] = mean_probs[:, 3]
        ensemble["max_prob"] = mean_probs.max(axis=1)

        path = os.path.join(args.output_dir, "ensemble_predictions.csv")
        ensemble.to_csv(path, index=False)

        print(f"\n=== Ensemble Results (slide-level) ===")
        print(f"Slides: {len(ensemble)}  Cases: {ensemble['case_id'].nunique()}")
        print(f"Accuracy: {(ensemble['true_label'] == ensemble['pred_label']).mean():.4f}")
        print(f"Genes imputed per fold: min {min(imputed_counts)}, max {max(imputed_counts)}")
        print(f"\nPredicted distribution:")
        print(ensemble["pred_name"].value_counts().to_string())
        print(f"\nTrue distribution:")
        print(ensemble["true_name"].value_counts().to_string())
        print(f"\nSaved ensemble predictions: {path}")


if __name__ == "__main__":
    main()
