"""
Phase 4A: PAM50 Inference on NOU Dataset
=========================================
Runs 10-fold CLAM-MB inference on NOU slide features using checkpoints
trained on TCGA-BRCA for PAM50 molecular subtyping.

Produces per-fold and ensemble prediction CSVs with class probabilities.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Add CLAM to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "CLAM"))

from models.model_clam import CLAM_MB

LABEL_MAP = {0: "LumA", 1: "LumB", 2: "Basal", 3: "Her2"}


def parse_args():
    parser = argparse.ArgumentParser(description="PAM50 inference on NOU features")
    parser.add_argument("--feature_dir", type=str,
                        default=".scratch/nou_validation/features/h5_files",
                        help="Directory with per-slide .h5 feature files")
    parser.add_argument("--dataset_csv", type=str,
                        default=".scratch/nou_validation/metadata/nou_pam50_dataset.csv",
                        help="CLAM-format CSV with slide_id, case_id, label")
    parser.add_argument("--ckpt_dir", type=str,
                        default=".scratch/results/pam50_final_s1",
                        help="Directory with trained CLAM checkpoints")
    parser.add_argument("--output_dir", type=str,
                        default=".scratch/nou_validation/results/predictions")
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=1536)
    parser.add_argument("--model_size", type=str, default="big")
    parser.add_argument("--dropout", type=float, default=0.5)
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
        features = f["features"][:]  # (1, N, 1536)
    features = torch.from_numpy(features).float().to(device)

    with torch.inference_mode():
        logits, Y_prob, Y_hat, A_raw, _ = model(features)

    return {
        "probs": Y_prob.cpu().numpy().squeeze(),  # (n_classes,)
        "pred": Y_hat.item(),
        "attention": A_raw.cpu().numpy(),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset CSV
    dataset = pd.read_csv(args.dataset_csv)
    print(f"Dataset: {len(dataset)} slides")

    # Verify all h5 files exist
    missing = []
    for slide_id in dataset["slide_id"]:
        h5_path = os.path.join(args.feature_dir, f"{slide_id}.h5")
        if not os.path.exists(h5_path):
            missing.append(slide_id)
    if missing:
        print(f"WARNING: {len(missing)} slides missing features, will skip them")
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
            slide_id = row["slide_id"]
            h5_path = os.path.join(args.feature_dir, f"{slide_id}.h5")
            result = infer_slide(model, h5_path, device)

            fold_results.append({
                "slide_id": slide_id,
                "case_id": row["case_id"],
                "true_label": int(row["label"]),
                "true_name": LABEL_MAP[int(row["label"])],
                "pred_label": result["pred"],
                "pred_name": LABEL_MAP[result["pred"]],
                "p_LumA": result["probs"][0],
                "p_LumB": result["probs"][1],
                "p_Basal": result["probs"][2],
                "p_Her2": result["probs"][3],
            })

        fold_df = pd.DataFrame(fold_results)
        fold_path = os.path.join(args.output_dir, f"fold_{fold}_predictions.csv")
        fold_df.to_csv(fold_path, index=False)

        # Fold accuracy
        acc = (fold_df["true_label"] == fold_df["pred_label"]).mean()
        print(f"  Fold {fold} accuracy: {acc:.4f}")

        all_fold_probs.append(
            fold_df[["p_LumA", "p_LumB", "p_Basal", "p_Her2"]].values
        )

        del model
        torch.cuda.empty_cache()

    # Ensemble predictions (average softmax across folds)
    if all_fold_probs:
        mean_probs = np.mean(all_fold_probs, axis=0)  # (N, 4)
        ensemble_preds = mean_probs.argmax(axis=1)

        ensemble_df = dataset[["slide_id", "case_id", "label"]].copy()
        ensemble_df = ensemble_df.rename(columns={"label": "true_label"})
        ensemble_df["true_name"] = ensemble_df["true_label"].map(LABEL_MAP)
        ensemble_df["pred_label"] = ensemble_preds
        ensemble_df["pred_name"] = ensemble_df["pred_label"].map(LABEL_MAP)
        ensemble_df["p_LumA"] = mean_probs[:, 0]
        ensemble_df["p_LumB"] = mean_probs[:, 1]
        ensemble_df["p_Basal"] = mean_probs[:, 2]
        ensemble_df["p_Her2"] = mean_probs[:, 3]
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


if __name__ == "__main__":
    main()
