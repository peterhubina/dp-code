from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from dataset_modules.dataset_generic import Generic_MIL_Dataset
from models.model_clam import CLAM_MB, CLAM_SB
from utils.utils import get_simple_loader


CLASS_NAMES = ["LumA", "LumB", "Basal", "Her2"]
LABEL_DICT = {name: idx for idx, name in enumerate(CLASS_NAMES)}
PROB_COLUMNS = [f"prob_{name}" for name in CLASS_NAMES]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validation-tuned confidence routing: use RNA softmax when confident, "
            "otherwise use a secondary modality (WSI-from-checkpoint or precomputed multimodal probs)."
        )
    )
    parser.add_argument("--rna_results_dir", type=str, required=True, help="Dir with fold_{k}_val/test_predictions.csv from matched RNA.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--objective", type=str, choices=["macro_auc", "macro_f1", "balanced_accuracy", "accuracy"], default="macro_f1")
    parser.add_argument("--confidence", type=str, choices=["max_prob", "margin", "neg_entropy"], default="max_prob")
    parser.add_argument("--tau_min", type=float, default=0.40)
    parser.add_argument("--tau_max", type=float, default=0.995)
    parser.add_argument("--tau_step", type=float, default=0.005)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k_start", type=int, default=0)
    parser.add_argument("--k_end", type=int, default=-1)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--secondary_pred_dir",
        type=str,
        default=None,
        help=(
            "Directory with fold_{f}_val_predictions.csv and fold_{f}_test_predictions.csv "
            "(e.g. from evaluate_multimodal.py per split)."
        ),
    )
    grp.add_argument("--run_wsi_secondary", action="store_true", help="Use frozen WSI checkpoints as secondary (same as late-fusion setup).")
    parser.add_argument("--data_root_dir", type=str, default=None)
    parser.add_argument("--dataset_csv", type=str, default="dataset_csv/tcga_brca_subtyping.csv")
    parser.add_argument("--split_dir", type=str, default="splits/tcga_brca_subtyping_100")
    parser.add_argument("--wsi_ckpt_dir", type=str, default=None)
    parser.add_argument("--model_type", type=str, choices=["clam_sb", "clam_mb"], default="clam_mb")
    parser.add_argument("--model_size", type=str, choices=["small", "big"], default="big")
    parser.add_argument("--drop_out", type=float, default=0.5)
    parser.add_argument("--embed_dim", type=int, default=1536)
    parser.add_argument("--B", type=int, default=4)
    return parser.parse_args()


def fold_range(args) -> list[int]:
    end = args.k if args.k_end == -1 else args.k_end
    return list(range(args.k_start, end))


def read_rna_predictions(rna_results_dir: Path, fold: int, split: str) -> pd.DataFrame:
    path = rna_results_dir / f"fold_{fold}_{split}_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing RNA predictions: {path}")
    df = pd.read_csv(path)
    if "slide_id" not in df.columns:
        df["slide_id"] = df["sample"].astype(str)
    else:
        df["slide_id"] = df["slide_id"].astype(str)
    return df


def read_secondary_predictions(pred_dir: Path, fold: int, split: str) -> pd.DataFrame:
    path = pred_dir / f"fold_{fold}_{split}_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing secondary predictions: {path}")
    df = pd.read_csv(path)
    df["slide_id"] = df["slide_id"].astype(str)
    return df


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
        print_info=False,
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


def align_rna_secondary(rna_df: pd.DataFrame, sec_df: pd.DataFrame) -> pd.DataFrame:
    rna_cols = ["slide_id", "case_id", "label_idx", "label", *PROB_COLUMNS]
    sec_cols = ["slide_id", *PROB_COLUMNS]
    merged = rna_df[rna_cols].merge(sec_df[sec_cols], on="slide_id", how="inner", suffixes=("_rna", "_sec"))
    if len(merged) != len(rna_df) or len(merged) != len(sec_df):
        raise ValueError(
            "RNA/secondary alignment dropped rows: "
            f"rna={len(rna_df)}, sec={len(sec_df)}, merged={len(merged)}"
        )
    return merged


def rna_confidence(rna_probs: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max_prob":
        return rna_probs.max(axis=1)
    if mode == "margin":
        sorted_probs = np.sort(rna_probs, axis=1)
        return sorted_probs[:, -1] - sorted_probs[:, -2]
    if mode == "neg_entropy":
        p = np.clip(rna_probs, 1e-12, 1.0)
        ent = -np.sum(p * np.log(p), axis=1)
        return -ent
    raise ValueError(mode)


def secondary_prob_matrix(aligned: pd.DataFrame) -> np.ndarray:
    return aligned[[f"{col}_sec" for col in PROB_COLUMNS]].to_numpy(dtype=np.float64)


def rna_prob_matrix(aligned: pd.DataFrame) -> np.ndarray:
    return aligned[[f"{col}_rna" for col in PROB_COLUMNS]].to_numpy(dtype=np.float64)


def route_probs(rna_probs: np.ndarray, sec_probs: np.ndarray, tau: float, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    use_rna = conf >= tau
    routed = np.where(use_rna[:, None], rna_probs, sec_probs)
    return routed, use_rna


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


def tune_tau(aligned_val: pd.DataFrame, args) -> tuple[float, pd.DataFrame]:
    labels = aligned_val["label_idx"].to_numpy(dtype=np.int64)
    rna_p = rna_prob_matrix(aligned_val)
    sec_p = secondary_prob_matrix(aligned_val)
    conf = rna_confidence(rna_p, args.confidence)
    taus = np.arange(args.tau_min, args.tau_max + args.tau_step / 2.0, args.tau_step)
    rows = []
    best_tau = float(args.tau_min)
    best_score = -np.inf
    for tau in taus:
        routed, _ = route_probs(rna_p, sec_p, float(tau), conf)
        metrics = compute_metrics(labels, routed)
        score = objective_value(metrics, args.objective)
        rows.append(
            {
                "tau": float(tau),
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
            best_tau = float(tau)
    return best_tau, pd.DataFrame(rows)


def save_routed_predictions(
    aligned: pd.DataFrame,
    probs: np.ndarray,
    use_rna: np.ndarray,
    conf: np.ndarray,
    fold: int,
    split: str,
    tau: float,
    path: Path,
) -> pd.DataFrame:
    preds = probs.argmax(axis=1)
    out = aligned[["slide_id", "case_id", "label_idx", "label"]].copy()
    out.insert(0, "fold", fold)
    out.insert(1, "split", split)
    out["tau"] = tau
    out["confidence_score"] = conf
    out["used_branch"] = np.where(use_rna, "rna", "secondary")
    out["pred_idx"] = preds
    out["pred"] = [CLASS_NAMES[p] for p in preds]
    out["correct"] = out["pred_idx"] == out["label_idx"]
    out["confidence_routed"] = np.where(use_rna, conf, np.nan)
    for j, class_name in enumerate(CLASS_NAMES):
        out[f"prob_{class_name}"] = probs[:, j]
    out.to_csv(path, index=False)
    return out


def baseline_metrics(labels: np.ndarray, rna_probs: np.ndarray, sec_probs: np.ndarray) -> dict:
    return {
        "rna_only": compute_metrics(labels, rna_probs),
        "secondary_only": compute_metrics(labels, sec_probs),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "routing_config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rna_dir = Path(args.rna_results_dir)

    if args.run_wsi_secondary:
        if not args.data_root_dir or not args.wsi_ckpt_dir:
            raise ValueError("--run_wsi_secondary requires --data_root_dir and --wsi_ckpt_dir")
        dataset = build_wsi_dataset(args)
        split_dir = Path(args.split_dir)
        wsi_ckpt_dir = Path(args.wsi_ckpt_dir)
    else:
        dataset = None
        split_dir = None
        wsi_ckpt_dir = None

    sec_pred_dir = Path(args.secondary_pred_dir) if args.secondary_pred_dir else None

    fold_rows = []
    all_test = []
    summary_rows = []

    for fold in fold_range(args):
        rna_val = read_rna_predictions(rna_dir, fold, "val")
        rna_test = read_rna_predictions(rna_dir, fold, "test")

        if args.run_wsi_secondary:
            ckpt_path = wsi_ckpt_dir / f"s_{fold}_checkpoint.pt"
            model = load_wsi_model(args, ckpt_path, device)
            train_split, val_split, test_split = dataset.return_splits(
                from_id=False, csv_path=str(split_dir / f"splits_{fold}.csv")
            )
            sec_val = predict_wsi_split(model, val_split, device)
            sec_test = predict_wsi_split(model, test_split, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            sec_val = read_secondary_predictions(sec_pred_dir, fold, "val")
            sec_test = read_secondary_predictions(sec_pred_dir, fold, "test")

        aligned_val = align_rna_secondary(rna_val, sec_val)
        aligned_test = align_rna_secondary(rna_test, sec_test)

        tau, tau_sweep = tune_tau(aligned_val, args)
        tau_sweep.to_csv(output_dir / f"fold_{fold}_tau_sweep.csv", index=False)

        labels_val = aligned_val["label_idx"].to_numpy(dtype=np.int64)
        labels_test = aligned_test["label_idx"].to_numpy(dtype=np.int64)
        rna_val_p = rna_prob_matrix(aligned_val)
        sec_val_p = secondary_prob_matrix(aligned_val)
        rna_test_p = rna_prob_matrix(aligned_test)
        sec_test_p = secondary_prob_matrix(aligned_test)

        conf_val = rna_confidence(rna_val_p, args.confidence)
        conf_test = rna_confidence(rna_test_p, args.confidence)

        val_routed, _ = route_probs(rna_val_p, sec_val_p, tau, conf_val)
        test_routed, use_rna_test = route_probs(rna_test_p, sec_test_p, tau, conf_test)

        val_metrics = compute_metrics(labels_val, val_routed)
        test_metrics = compute_metrics(labels_test, test_routed)
        bases = baseline_metrics(labels_test, rna_test_p, sec_test_p)

        frac_secondary = float(1.0 - use_rna_test.mean())

        test_df = save_routed_predictions(
            aligned_test,
            test_routed,
            use_rna_test,
            conf_test,
            fold,
            "test",
            tau,
            output_dir / f"fold_{fold}_test_predictions.csv",
        )
        all_test.append(test_df)

        rna_pred_test = rna_test_p.argmax(axis=1)
        sec_pred_test = sec_test_p.argmax(axis=1)
        switched = ~use_rna_test
        rna_wrong = rna_pred_test != labels_test
        helped = switched & (test_routed.argmax(axis=1) == labels_test) & rna_wrong
        hurt = switched & (test_routed.argmax(axis=1) != labels_test) & (rna_pred_test == labels_test)

        summary_rows.append(
            {
                "fold": fold,
                "tau": tau,
                "frac_secondary_test": frac_secondary,
                "n_helped_when_switched": int(helped.sum()),
                "n_hurt_when_switched": int(hurt.sum()),
                "val_objective": objective_value(val_metrics, args.objective),
                "val_macro_auc": val_metrics["macro_auc"],
                "test_macro_auc": test_metrics["macro_auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "test_rna_only_macro_auc": bases["rna_only"]["macro_auc"],
                "test_rna_only_accuracy": bases["rna_only"]["accuracy"],
                "test_secondary_only_macro_auc": bases["secondary_only"]["macro_auc"],
                "test_secondary_only_accuracy": bases["secondary_only"]["accuracy"],
            }
        )

        fold_rows.append({"fold": fold, "split": "test", **test_metrics})
        print(
            "fold {} tau={:.3f} frac_sec={:.3f} test_auc={:.4f} test_acc={:.4f} "
            "rna_auc={:.4f} sec_auc={:.4f} helped={} hurt={}".format(
                fold,
                tau,
                frac_secondary,
                test_metrics["macro_auc"],
                test_metrics["accuracy"],
                bases["rna_only"]["macro_auc"],
                bases["secondary_only"]["macro_auc"],
                int(helped.sum()),
                int(hurt.sum()),
            )
        )

    fold_metrics = pd.DataFrame(summary_rows)
    fold_metrics.to_csv(output_dir / "fold_routing_summary.csv", index=False)

    all_predictions = pd.concat(all_test, ignore_index=True)
    all_predictions.to_csv(output_dir / "all_test_predictions.csv", index=False)

    aggregate_labels = all_predictions["label_idx"].to_numpy(dtype=np.int64)
    aggregate_probs = all_predictions[PROB_COLUMNS].to_numpy(dtype=np.float64)
    aggregate_metrics = compute_metrics(aggregate_labels, aggregate_probs)
    (output_dir / "test_metrics.json").write_text(json.dumps(aggregate_metrics, indent=2) + "\n")

    print("\nAggregate routed test metrics")
    print(json.dumps({k: aggregate_metrics[k] for k in ("n", "macro_auc", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}, indent=2))
    print("Mean fraction routed to secondary: {:.4f}".format(float(fold_metrics["frac_secondary_test"].mean())))
    print(f"Saved routing evaluation to {output_dir}")


if __name__ == "__main__":
    main()
