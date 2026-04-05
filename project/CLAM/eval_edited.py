"""
CLAM External Dataset Evaluation
=================================
Runs k-fold CLAM-MB inference on pre-extracted slide features using
trained checkpoints. Produces per-fold and ensemble prediction CSVs
with per-class probabilities.

Expected dataset CSV columns: slide_id, case_id, label
Labels may be integers (0..n_classes-1) or strings matching LABEL_MAP values.

Usage:
    python eval.py \
        --feature_dir /path/to/h5_files \
        --dataset_csv /path/to/dataset.csv \
        --ckpt_dir /path/to/checkpoints \
        --output_dir /path/to/output
"""

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Allow running from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from models.model_clam import CLAM_MB

LABEL_MAP = {0: "LumA", 1: "LumB", 2: "Basal", 3: "Her2"}
LABEL_TO_INT = {v: k for k, v in LABEL_MAP.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLAM-MB k-fold inference on external dataset features"
    )
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory with per-slide .h5 feature files")
    parser.add_argument("--dataset_csv", type=str, required=True,
                        help="CSV with slide_id, case_id, label columns")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Directory containing s_<fold>_checkpoint.pt files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write prediction CSVs")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="Number of cross-validation folds (default: 10)")
    parser.add_argument("--n_classes", type=int, default=4,
                        help="Number of output classes (default: 4)")
    parser.add_argument("--embed_dim", type=int, default=1536,
                        help="Feature embedding dimension (default: 1536)")
    parser.add_argument("--model_size", type=str, default="big",
                        help="CLAM model size: small or big (default: big)")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate (default: 0.5)")
    return parser.parse_args()


def load_model(ckpt_path, n_classes, embed_dim, model_size, dropout, device):
    """Load a trained CLAM_MB checkpoint."""
    model = CLAM_MB(
        gate=True,
        size_arg=model_size,
        dropout=dropout,
        k_sample=4,
        n_classes=n_classes,
        subtyping=True,
        embed_dim=embed_dim,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_clean = {}
    for key in ckpt.keys():
        if "instance_loss_fn" in key:
            continue
        ckpt_clean[key.replace(".module", "")] = ckpt[key]
    model.load_state_dict(ckpt_clean, strict=True)
    model.to(device)
    model.eval()
    return model


def infer_slide(model, h5_path, device):
    """Run inference on a single slide, return probabilities and attention."""
    with h5py.File(h5_path, "r") as f:
        features = f["features"][:]

    features = torch.from_numpy(features).float().to(device)
    # CLAM-MB expects (1, N, D); handle both (N, D) and (1, N, D) inputs
    if features.ndim == 2:
        features = features.unsqueeze(0)

    with torch.inference_mode():
        logits, Y_prob, Y_hat, A_raw, _ = model(features)

    return {
        "probs": Y_prob.cpu().numpy().squeeze(),
        "pred": Y_hat.item(),
        "attention": A_raw.cpu().numpy(),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build probability column names from LABEL_MAP
    prob_columns = [f"p_{LABEL_MAP[i]}" for i in range(args.n_classes)]

    # Load dataset CSV
    dataset = pd.read_csv(args.dataset_csv)
    print(f"Dataset: {len(dataset)} slides")

    # Map string labels to int if needed; keep int labels as-is
    if dataset["label"].dtype == object:
        dataset["label_int"] = dataset["label"].map(LABEL_TO_INT)
    else:
        dataset["label_int"] = dataset["label"].astype(int)

    print(f"Label distribution:\n{dataset['label'].value_counts().to_string()}")

    # Check for missing feature files
    missing = []
    for slide_id in dataset["slide_id"]:
        h5_path = os.path.join(args.feature_dir, f"{slide_id}.h5")
        if not os.path.exists(h5_path):
            missing.append(slide_id)
    if missing:
        print(f"WARNING: {len(missing)} slides missing features, will skip: {missing}")
        dataset = dataset[~dataset["slide_id"].isin(missing)].reset_index(drop=True)

    # Run inference for each fold
    all_fold_probs = []

    for fold in range(args.n_folds):
        ckpt_path = os.path.join(args.ckpt_dir, f"s_{fold}_checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"Fold {fold}: checkpoint not found at {ckpt_path}, skipping")
            continue

        print(f"\nFold {fold}: loading {ckpt_path}")
        model = load_model(ckpt_path, args.n_classes, args.embed_dim,
                           args.model_size, args.dropout, device)

        fold_results = []
        for _, row in dataset.iterrows():
            slide_id = str(row["slide_id"])
            h5_path = os.path.join(args.feature_dir, f"{slide_id}.h5")
            result = infer_slide(model, h5_path, device)

            entry = {
                "slide_id": slide_id,
                "case_id": str(row["case_id"]),
                "true_label": int(row["label_int"]),
                "true_name": LABEL_MAP[int(row["label_int"])],
                "pred_label": result["pred"],
                "pred_name": LABEL_MAP[result["pred"]],
            }
            for i, col in enumerate(prob_columns):
                entry[col] = result["probs"][i]
            fold_results.append(entry)

        fold_df = pd.DataFrame(fold_results)
        fold_path = os.path.join(args.output_dir, f"fold_{fold}_predictions.csv")
        fold_df.to_csv(fold_path, index=False)

        acc = (fold_df["true_label"] == fold_df["pred_label"]).mean()
        print(f"  Fold {fold} accuracy: {acc:.4f}")

        all_fold_probs.append(fold_df[prob_columns].values)

        del model
        torch.cuda.empty_cache()

    # Ensemble predictions (average softmax across folds)
    if all_fold_probs:
        mean_probs = np.mean(all_fold_probs, axis=0)  # (N, n_classes)
        ensemble_preds = mean_probs.argmax(axis=1)

        ensemble_df = dataset[["slide_id", "case_id", "label_int"]].copy()
        ensemble_df = ensemble_df.rename(columns={"label_int": "true_label"})
        ensemble_df["true_name"] = ensemble_df["true_label"].map(LABEL_MAP)
        ensemble_df["pred_label"] = ensemble_preds
        ensemble_df["pred_name"] = ensemble_df["pred_label"].map(LABEL_MAP)
        for i, col in enumerate(prob_columns):
            ensemble_df[col] = mean_probs[:, i]
        ensemble_df["max_prob"] = mean_probs.max(axis=1)

        ensemble_path = os.path.join(args.output_dir, "ensemble_predictions.csv")
        ensemble_df.to_csv(ensemble_path, index=False)

        acc = (ensemble_df["true_label"] == ensemble_df["pred_label"]).mean()
        print(f"\n=== Ensemble Results ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"\nPredicted distribution:")
        print(ensemble_df["pred_name"].value_counts().to_string())
        print(f"\nTrue distribution:")
        print(ensemble_df["true_name"].value_counts().to_string())
        print(f"\nSaved ensemble predictions: {ensemble_path}")
    else:
        print("\nNo fold checkpoints found. Skipping ensemble.")


if __name__ == "__main__":
    main()
