"""
CPTAC-BRCA external validation: binary ER status
================================================
Runs the 10 TCGA-trained CLAM-MB folds of the `er_wsi_alone` arm over the CPTAC
feature store. Frozen weights, no fine-tuning, no domain adaptation.

The model is reconstructed to match training exactly, as recorded in
.scratch/results/er/er_wsi_alone_s1/experiment_er_wsi_alone.txt:

    clam_mb, size_arg 'big', dropout 0.5, embed_dim 1536, B (k_sample) 4,
    subtyping False (never passed on the command line), inst_loss svm

`subtyping` and `k_sample` only feed the instance-loss branch, so they cannot
change a forward pass -- they are set for fidelity, not effect. The SVM instance
loss keys are stripped from the state dict before loading, as they are buffers of
the loss object rather than model parameters.

Label encoding follows project/CLAM/main.py --task tcga_brca_er:
{'ER-negative': 0, 'ER-positive': 1}. Index 1 is the positive class throughout;
getting this backwards would silently invert every AUROC.

Two aggregation levels are reported:
  slide  one row per slide, the unit the model actually predicts on
  case   per-case MEAN over that case's slide probabilities -- the convention in
         tools/analyze_er_ablation.py, so the external number is comparable to
         the internal case-level 0.8957

Per-fold metrics are reported alongside the ensemble. Internally the 10 folds
partition the cohort, so each case had exactly one out-of-fold prediction; here
every fold sees every slide, so the folds are an ensemble and their spread is a
measure of checkpoint agreement rather than a cross-validation estimate.

    python tools/cptac/infer_cptac_er.py
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
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "CLAM"))

from models.model_clam import CLAM_MB  # noqa: E402

LABEL_MAP = {0: "ER-negative", 1: "ER-positive"}
POS_INDEX = 1

# Internal pooled case-level AUROC of this same arm, for context in the printout
# (.scratch/results/er/analysis/metrics.json).
INTERNAL_CASE_AUROC = 0.8957


def parse_args():
    parser = argparse.ArgumentParser(description="ER external validation on CPTAC-BRCA")
    parser.add_argument("--feature_dir", type=str,
                        default=".datasets/cptac-brca/embeddings")
    parser.add_argument("--dataset_csv", type=str,
                        default=".datasets/cptac-brca/cptac_brca_er_dataset.csv")
    parser.add_argument("--ckpt_dir", type=str,
                        default=".scratch/results/er/er_wsi_alone_s1")
    parser.add_argument("--output_dir", type=str,
                        default=".scratch/cptac_validation/results/er")
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=1536)
    parser.add_argument("--model_size", type=str, default="big")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--k_sample", type=int, default=4)
    parser.add_argument("--pam50_subset_only", action="store_true",
                        help="restrict to cases shared with the PAM50 external cohort")
    return parser.parse_args()


def load_model(ckpt_path, args, device):
    model = CLAM_MB(
        gate=True,
        size_arg=args.model_size,
        dropout=args.dropout,
        k_sample=args.k_sample,
        n_classes=args.n_classes,
        subtyping=False,
        embed_dim=args.embed_dim,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    cleaned = {k.replace(".module", ""): v for k, v in ckpt.items()
               if "instance_loss_fn" not in k}
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model


def index_features(feature_dir):
    index = {}
    for path in sorted(Path(feature_dir).rglob("*.h5"), key=lambda p: len(p.parts)):
        if path.is_file() and path.stem not in index:
            index[path.stem] = str(path)
    return index


def infer_slide(model, h5_path, device):
    with h5py.File(h5_path, "r") as handle:
        features = handle["features"][:]
    n_patches = features.shape[1]
    features = torch.from_numpy(features).float().to(device)
    with torch.inference_mode():
        _, Y_prob, _, _, _ = model(features)
    return Y_prob.cpu().numpy().reshape(-1), n_patches


def binary_metrics(y_true, p_pos):
    """Threshold-free metrics plus the untuned 0.5 operating point."""
    y_true = np.asarray(y_true)
    p_pos = np.asarray(p_pos)
    pred = (p_pos >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "auroc": float(roc_auc_score(y_true, p_pos)),
        "auprc_pos": float(average_precision_score(y_true, p_pos)),
        "f1_pos": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity_ERpos": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity_ERneg": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
    }


def to_case_level(frame, prob_col="p_ER_positive"):
    """Per-case mean over slide probabilities (analyze_er_ablation.py convention)."""
    return frame.groupby("case_id").agg(
        true_label=("true_label", "first"),
        p_ER_positive=(prob_col, "mean"),
        n_slides=("slide_id", "count"),
    ).reset_index()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = pd.read_csv(args.dataset_csv)
    if args.pam50_subset_only:
        if "in_pam50_cohort" not in dataset.columns:
            raise SystemExit("--pam50_subset_only needs an in_pam50_cohort column")
        dataset = dataset[dataset["in_pam50_cohort"]].reset_index(drop=True)
        print("restricted to the PAM50-shared subset")
    print(f"dataset: {len(dataset)} slides / {dataset['case_id'].nunique()} cases")
    print(dataset["label_name"].value_counts().to_string())

    features = index_features(args.feature_dir)
    missing = sorted(set(dataset["slide_id"]) - set(features))
    if missing:
        print(f"WARNING: {len(missing)} slides have no feature file, dropping them")
        dataset = dataset[~dataset["slide_id"].isin(missing)].reset_index(drop=True)

    fold_prob_matrix = []
    per_fold_metrics = []
    n_patches_by_slide = {}

    for fold in range(args.n_folds):
        ckpt_path = os.path.join(args.ckpt_dir, f"s_{fold}_checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"fold {fold}: no checkpoint at {ckpt_path}, skipping")
            continue

        model = load_model(ckpt_path, args, device)
        probs = np.zeros(len(dataset), dtype=np.float64)
        for i, row in enumerate(dataset.itertuples(index=False)):
            slide_probs, n_patches = infer_slide(model, features[row.slide_id], device)
            probs[i] = slide_probs[POS_INDEX]
            n_patches_by_slide[row.slide_id] = n_patches

        fold_frame = dataset[["case_id", "slide_id", "label"]].rename(
            columns={"label": "true_label"}).copy()
        fold_frame["p_ER_positive"] = probs
        fold_frame.to_csv(os.path.join(args.output_dir, f"fold_{fold}_predictions.csv"),
                          index=False)

        slide_m = binary_metrics(fold_frame["true_label"], fold_frame["p_ER_positive"])
        case_frame = to_case_level(fold_frame)
        case_m = binary_metrics(case_frame["true_label"], case_frame["p_ER_positive"])
        per_fold_metrics.append({"fold": fold, "slide": slide_m, "case": case_m})
        print(f"fold {fold}: slide AUROC {slide_m['auroc']:.4f}  "
              f"case AUROC {case_m['auroc']:.4f}")

        fold_prob_matrix.append(probs)
        del model
        torch.cuda.empty_cache()

    if not fold_prob_matrix:
        raise SystemExit(f"no checkpoints found under {args.ckpt_dir}")

    mean_probs = np.mean(fold_prob_matrix, axis=0)

    slide_df = dataset[["case_id", "slide_id", "label", "label_name"]].rename(
        columns={"label": "true_label", "label_name": "true_name"}).copy()
    slide_df["p_ER_positive"] = mean_probs
    slide_df["pred_label"] = (mean_probs >= 0.5).astype(int)
    slide_df["pred_name"] = slide_df["pred_label"].map(LABEL_MAP)
    slide_df["n_patches"] = slide_df["slide_id"].map(n_patches_by_slide)
    slide_df["prob_sd_across_folds"] = np.std(fold_prob_matrix, axis=0)
    slide_df.to_csv(os.path.join(args.output_dir, "ensemble_slide_predictions.csv"),
                    index=False)

    case_df = to_case_level(slide_df)
    case_df["true_name"] = case_df["true_label"].map(LABEL_MAP)
    case_df["pred_label"] = (case_df["p_ER_positive"] >= 0.5).astype(int)
    case_df["pred_name"] = case_df["pred_label"].map(LABEL_MAP)
    case_df.to_csv(os.path.join(args.output_dir, "ensemble_case_predictions.csv"),
                   index=False)

    slide_m = binary_metrics(slide_df["true_label"], slide_df["p_ER_positive"])
    case_m = binary_metrics(case_df["true_label"], case_df["p_ER_positive"])

    fold_case_aurocs = [m["case"]["auroc"] for m in per_fold_metrics]
    fold_slide_aurocs = [m["slide"]["auroc"] for m in per_fold_metrics]
    summary = {
        "arm": "er_wsi_alone (TCGA-trained, frozen)",
        "cohort": "CPTAC-BRCA",
        "n_folds_used": len(fold_prob_matrix),
        "ensemble": {"slide_level": slide_m, "case_level": case_m},
        "per_fold_auroc": {
            "slide": {"mean": float(np.mean(fold_slide_aurocs)),
                      "std": float(np.std(fold_slide_aurocs)),
                      "values": fold_slide_aurocs},
            "case": {"mean": float(np.mean(fold_case_aurocs)),
                     "std": float(np.std(fold_case_aurocs)),
                     "values": fold_case_aurocs},
        },
        "internal_reference_case_auroc": INTERNAL_CASE_AUROC,
        "external_minus_internal_case_auroc": case_m["auroc"] - INTERNAL_CASE_AUROC,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== Ensemble, slide level ===")
    for key in ("n", "n_pos", "auroc", "auprc_pos", "f1_pos", "balanced_acc",
                "sensitivity_ERpos", "specificity_ERneg"):
        print(f"  {key:20s} {slide_m[key]}")
    print("\n=== Ensemble, case level (comparable to internal) ===")
    for key in ("n", "n_pos", "auroc", "auprc_pos", "f1_pos", "balanced_acc",
                "sensitivity_ERpos", "specificity_ERneg"):
        print(f"  {key:20s} {case_m[key]}")
    print(f"\nper-fold case AUROC: {np.mean(fold_case_aurocs):.4f} "
          f"+/- {np.std(fold_case_aurocs):.4f}")
    print(f"internal case AUROC: {INTERNAL_CASE_AUROC:.4f}  ->  "
          f"external {case_m['auroc']:.4f}  "
          f"(delta {case_m['auroc'] - INTERNAL_CASE_AUROC:+.4f})")
    print("\nNOTE: the 0.5 operating point is untuned; AUROC and AUPRC are the "
          "threshold-free primary metrics.")
    print(f"wrote {args.output_dir}/{{metrics.json, ensemble_*_predictions.csv, "
          f"fold_*_predictions.csv}}")


if __name__ == "__main__":
    main()
