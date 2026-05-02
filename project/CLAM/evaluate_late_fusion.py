from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

from dataset_modules.dataset_generic import Generic_MIL_Dataset
from models.model_clam import CLAM_MB, CLAM_SB
from utils.utils import get_simple_loader


CLASS_NAMES = ["LumA", "LumB", "Basal", "Her2"]
LABEL_DICT = {name: idx for idx, name in enumerate(CLASS_NAMES)}
PROB_COLUMNS = [f"prob_{name}" for name in CLASS_NAMES]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune validation-weighted late probability fusion for matched RNA and WSI branches."
    )
    parser.add_argument("--rna_results_dir", type=str, required=True)
    parser.add_argument("--wsi_ckpt_dir", type=str, required=True)
    parser.add_argument("--data_root_dir", type=str, required=True)
    parser.add_argument("--dataset_csv", type=str, default="dataset_csv/tcga_brca_subtyping.csv")
    parser.add_argument("--split_dir", type=str, default="splits/tcga_brca_subtyping_100")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k_start", type=int, default=0)
    parser.add_argument("--k_end", type=int, default=-1)
    parser.add_argument("--model_type", type=str, choices=["clam_sb", "clam_mb"], default="clam_mb")
    parser.add_argument("--model_size", type=str, choices=["small", "big"], default="big")
    parser.add_argument("--drop_out", type=float, default=0.5)
    parser.add_argument("--embed_dim", type=int, default=1536)
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--objective", type=str, choices=["macro_auc", "macro_f1", "balanced_accuracy", "accuracy"], default="macro_f1")
    parser.add_argument("--alpha_min", type=float, default=0.0)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--alpha_step", type=float, default=0.01)
    return parser.parse_args()


def fold_range(args) -> list[int]:
    end = args.k if args.k_end == -1 else args.k_end
    return list(range(args.k_start, end))


def normalize_checkpoint(checkpoint: dict) -> dict:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    return {
        key.replace(".module", ""): value
        for key, value in checkpoint.items()
        if "instance_loss_fn" not in key
    }


def load_wsi_model(args, ckpt_path: Path, device: torch.device):
    model_kwargs = {
        "size_arg": args.model_size,
        "dropout": args.drop_out,
        "k_sample": args.B,
        "n_classes": len(CLASS_NAMES),
        "instance_loss_fn": nn.CrossEntropyLoss(),
        "subtyping": True,
        "embed_dim": args.embed_dim,
    }
    if args.model_type == "clam_sb":
        model = CLAM_SB(**model_kwargs)
    else:
        model = CLAM_MB(**model_kwargs)

    checkpoint = normalize_checkpoint(torch.load(ckpt_path, map_location=device))
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    return model


def build_wsi_dataset(args):
    dataset = Generic_MIL_Dataset(
        csv_path=args.dataset_csv,
        data_dir=args.data_root_dir,
        shuffle=False,
        seed=1,
        print_info=True,
        label_dict=LABEL_DICT,
        patient_strat=True,
        ignore=["Normal"],
    )
    dataset.load_from_h5(True)
    return dataset


def predict_wsi_split(model, split, device: torch.device) -> pd.DataFrame:
    loader = get_simple_loader(split)
    rows = []
    slide_ids = split.slide_data["slide_id"].astype(str).reset_index(drop=True)
    case_ids = split.slide_data["case_id"].astype(str).reset_index(drop=True)

    with torch.inference_mode():
        for batch_idx, (features, label) in enumerate(loader):
            features = features.to(device)
            logits, y_prob, y_hat, _, _ = model(features)
            probs = y_prob.detach().cpu().numpy().reshape(-1)
            label_idx = int(label.item())
            pred_idx = int(y_hat.item())
            row = {
                "slide_id": slide_ids.iloc[batch_idx],
                "case_id": case_ids.iloc[batch_idx],
                "label_idx": label_idx,
                "label": CLASS_NAMES[label_idx],
                "pred_idx": pred_idx,
                "pred": CLASS_NAMES[pred_idx],
            }
            for class_name, prob in zip(CLASS_NAMES, probs):
                row[f"prob_{class_name}"] = float(prob)
            rows.append(row)
    return pd.DataFrame(rows)


def read_rna_predictions(rna_results_dir: Path, fold: int, split: str) -> pd.DataFrame:
    path = rna_results_dir / f"fold_{fold}_{split}_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing RNA predictions: {path}")
    df = pd.read_csv(path)
    if "slide_id" not in df.columns:
        df["slide_id"] = df["sample"].astype(str)
    return df


def align_predictions(rna_df: pd.DataFrame, wsi_df: pd.DataFrame) -> pd.DataFrame:
    rna_cols = ["slide_id", "case_id", "label_idx", "label", *PROB_COLUMNS]
    wsi_cols = ["slide_id", *PROB_COLUMNS]
    merged = rna_df[rna_cols].merge(
        wsi_df[wsi_cols],
        on="slide_id",
        how="inner",
        suffixes=("_rna", "_wsi"),
    )
    if len(merged) != len(rna_df) or len(merged) != len(wsi_df):
        raise ValueError(
            "RNA/WSI prediction alignment dropped rows: "
            f"rna={len(rna_df)}, wsi={len(wsi_df)}, merged={len(merged)}"
        )
    return merged


def fused_probabilities(aligned: pd.DataFrame, alpha: float) -> np.ndarray:
    rna_probs = aligned[[f"{col}_rna" for col in PROB_COLUMNS]].to_numpy(dtype=np.float64)
    wsi_probs = aligned[[f"{col}_wsi" for col in PROB_COLUMNS]].to_numpy(dtype=np.float64)
    probs = alpha * rna_probs + (1.0 - alpha) * wsi_probs
    return probs / probs.sum(axis=1, keepdims=True)


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    class_auc = {}
    aucs = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        binary_labels = (labels == class_idx).astype(np.int64)
        if binary_labels.min() == binary_labels.max():
            class_auc[class_name] = float("nan")
            continue
        class_auc[class_name] = float(roc_auc_score(binary_labels, probs[:, class_idx]))
        aucs.append(class_auc[class_name])

    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "macro_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "class_auc": class_auc,
        "classification_report": classification_report(
            labels,
            preds,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES)))).tolist(),
    }


def objective_value(metrics: dict, objective: str) -> float:
    return float(metrics[objective])


def tune_alpha(aligned_val: pd.DataFrame, args) -> tuple[float, pd.DataFrame]:
    labels = aligned_val["label_idx"].to_numpy(dtype=np.int64)
    alpha_values = np.arange(args.alpha_min, args.alpha_max + (args.alpha_step / 2.0), args.alpha_step)
    rows = []
    best_alpha = None
    best_score = -np.inf
    for alpha in alpha_values:
        metrics = compute_metrics(labels, fused_probabilities(aligned_val, float(alpha)))
        score = objective_value(metrics, args.objective)
        rows.append(
            {
                "alpha_rna": float(alpha),
                "alpha_wsi": float(1.0 - alpha),
                "objective": args.objective,
                "objective_value": score,
                "macro_auc": metrics["macro_auc"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return float(best_alpha), pd.DataFrame(rows)


def save_predictions(aligned: pd.DataFrame, probs: np.ndarray, output_path: Path, fold: int, split: str, alpha: float) -> pd.DataFrame:
    preds = probs.argmax(axis=1)
    predictions = aligned[["slide_id", "case_id", "label_idx", "label"]].copy()
    predictions.insert(0, "fold", fold)
    predictions.insert(1, "split", split)
    predictions["alpha_rna"] = alpha
    predictions["alpha_wsi"] = 1.0 - alpha
    predictions["pred_idx"] = preds
    predictions["pred"] = [CLASS_NAMES[pred] for pred in preds]
    predictions["correct"] = predictions["pred_idx"] == predictions["label_idx"]
    for class_idx, class_name in enumerate(CLASS_NAMES):
        predictions[f"prob_{class_name}"] = probs[:, class_idx]
    predictions.to_csv(output_path, index=False)
    return predictions


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "late_fusion_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_wsi_dataset(args)
    rna_results_dir = Path(args.rna_results_dir)
    wsi_ckpt_dir = Path(args.wsi_ckpt_dir)
    split_dir = Path(args.split_dir)

    fold_metric_rows = []
    all_test_predictions = []
    for fold in fold_range(args):
        print(f"\nEvaluating late fusion fold {fold}")
        ckpt_path = wsi_ckpt_dir / f"s_{fold}_checkpoint.pt"
        model = load_wsi_model(args, ckpt_path, device)
        train_split, val_split, test_split = dataset.return_splits(from_id=False, csv_path=str(split_dir / f"splits_{fold}.csv"))

        wsi_val = predict_wsi_split(model, val_split, device)
        wsi_test = predict_wsi_split(model, test_split, device)
        wsi_val.to_csv(output_dir / f"fold_{fold}_wsi_val_predictions.csv", index=False)
        wsi_test.to_csv(output_dir / f"fold_{fold}_wsi_test_predictions.csv", index=False)

        rna_val = read_rna_predictions(rna_results_dir, fold, "val")
        rna_test = read_rna_predictions(rna_results_dir, fold, "test")
        aligned_val = align_predictions(rna_val, wsi_val)
        aligned_test = align_predictions(rna_test, wsi_test)

        alpha, alpha_metrics = tune_alpha(aligned_val, args)
        alpha_metrics.to_csv(output_dir / f"fold_{fold}_alpha_sweep.csv", index=False)

        val_probs = fused_probabilities(aligned_val, alpha)
        test_probs = fused_probabilities(aligned_test, alpha)
        val_metrics = compute_metrics(aligned_val["label_idx"].to_numpy(dtype=np.int64), val_probs)
        test_metrics = compute_metrics(aligned_test["label_idx"].to_numpy(dtype=np.int64), test_probs)

        save_predictions(aligned_val, val_probs, output_dir / f"fold_{fold}_val_predictions.csv", fold, "val", alpha)
        test_predictions = save_predictions(aligned_test, test_probs, output_dir / f"fold_{fold}_test_predictions.csv", fold, "test", alpha)
        all_test_predictions.append(test_predictions)

        fold_metric_rows.append(
            {
                "fold": fold,
                "alpha_rna": alpha,
                "alpha_wsi": 1.0 - alpha,
                "val_macro_auc": val_metrics["macro_auc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "test_macro_auc": test_metrics["macro_auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
            }
        )
        print(
            "fold {} alpha_rna={:.2f}: val_{}={:.4f}, test_auc={:.4f}, test_acc={:.4f}, test_macro_f1={:.4f}".format(
                fold,
                alpha,
                args.objective,
                objective_value(val_metrics, args.objective),
                test_metrics["macro_auc"],
                test_metrics["accuracy"],
                test_metrics["macro_f1"],
            )
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    all_predictions = pd.concat(all_test_predictions, ignore_index=True)
    all_predictions.to_csv(output_dir / "all_test_predictions.csv", index=False)

    aggregate_probs = all_predictions[PROB_COLUMNS].to_numpy(dtype=np.float64)
    aggregate_labels = all_predictions["label_idx"].to_numpy(dtype=np.int64)
    aggregate_metrics = compute_metrics(aggregate_labels, aggregate_probs)
    (output_dir / "test_metrics.json").write_text(json.dumps(aggregate_metrics, indent=2) + "\n")

    print("\nAggregate test metrics")
    print(json.dumps({key: aggregate_metrics[key] for key in ("n", "macro_auc", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}, indent=2))


if __name__ == "__main__":
    main()
