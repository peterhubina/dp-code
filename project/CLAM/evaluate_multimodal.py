from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc as calc_auc,
)

from dataset_modules.multimodal_dataset import Generic_Multimodal_MIL_Dataset
from dataset_modules.rna_dataset import RNAFeatureTransform
from models.model_multimodal import CLAMRNAFusion
from utils.utils import get_split_loader

try:
    import wandb
except ImportError:
    wandb = None


CLASS_NAMES = ["LumA", "LumB", "Basal", "Her2"]
LABEL_DICT = {name: idx for idx, name in enumerate(CLASS_NAMES)}
PROB_COLUMNS = [f"prob_{name}" for name in CLASS_NAMES]
FUSION_RESULT_KEYS = (
    "fusion_wsi_gate_mean",
    "fusion_rna_gate_mean",
    "fusion_gate_std",
    "fusion_wsi_to_rna_attention",
    "fusion_rna_to_wsi_attention",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained CLAM + RNA multimodal-fusion checkpoints on held-out splits."
    )
    parser.add_argument("--data_root_dir", type=str, required=True)
    parser.add_argument("--tabular_csv", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_dir", type=str, default="splits/tcga_brca_subtyping_100")
    parser.add_argument("--dataset_csv", type=str, default="dataset_csv/tcga_brca_subtyping.csv")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--fold", type=int, default=-1, help="evaluate a single fold; -1 evaluates k_start:k_end")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k_start", type=int, default=0)
    parser.add_argument("--k_end", type=int, default=-1)
    parser.add_argument("--embed_dim", type=int, default=1536)
    parser.add_argument("--model_type", type=str, choices=["clam_sb", "clam_mb"], default="clam_mb")
    parser.add_argument("--model_size", type=str, choices=["small", "big"], default="big")
    parser.add_argument("--fusion_mode", type=str, choices=["auto", "concat", "gated", "residual", "cross_attention"], default="auto")
    parser.add_argument("--drop_out", type=float, default=0.5)
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--tabular_case_id_col", type=str, default="case_id")
    parser.add_argument("--tabular_label_col", type=str, default="label")
    parser.add_argument("--tabular_hidden_dim", type=int, default=256)
    parser.add_argument("--tabular_num_layers", type=int, default=2)
    parser.add_argument("--fusion_hidden_dim", type=int, default=32)
    parser.add_argument("--rna_hidden_dims", type=str, default="1024,512")
    parser.add_argument("--rna_dropout", type=float, default=0.4)
    parser.add_argument("--residual_scale", type=float, default=0.2)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="clam-brca-subtyping-cv")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default=None)
    return parser.parse_args()


def fold_range(args):
    if args.fold >= 0:
        return [args.fold]
    end = args.k if args.k_end == -1 else args.k_end
    return list(range(args.k_start, end))


def load_transform(path: Path) -> RNAFeatureTransform:
    if not path.is_file():
        raise FileNotFoundError(f"Missing tabular transform: {path}")

    payload = json.loads(path.read_text())
    return RNAFeatureTransform(
        selected_idx=np.asarray(payload["selected_idx"], dtype=np.int64),
        selected_feature_names=list(payload["selected_feature_names"]),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
    )


def normalize_checkpoint(checkpoint: dict) -> dict:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    return {key.replace(".module", ""): value for key, value in checkpoint.items()}


def infer_fusion_mode(checkpoint: dict, ckpt_path: Path) -> str:
    has_concat_head = any(key.startswith("fusion_head.") for key in checkpoint)
    has_cross_attention_head = any(
        key.startswith(("cross_attention.", "cross_attention_norm."))
        for key in checkpoint
    )
    has_gated_head = any(
        key.startswith(("tabular_projection.", "fusion_gate.", "fusion_classifier."))
        for key in checkpoint
    )
    has_residual_head = any(
        key.startswith(("rna_model.", "rna_projection.", "residual_head."))
        for key in checkpoint
    )

    if has_cross_attention_head:
        return "cross_attention"
    if has_residual_head and not has_concat_head and not has_gated_head:
        return "residual"
    if has_concat_head and not has_gated_head:
        return "concat"
    if has_gated_head and not has_concat_head:
        return "gated"
    raise ValueError(
        f"Could not infer fusion mode from checkpoint keys in {ckpt_path}. "
        "Pass --fusion_mode concat, gated, residual or cross_attention explicitly."
    )


def load_model(args, tabular_input_dim: int, ckpt_path: Path, device: torch.device):
    checkpoint = normalize_checkpoint(torch.load(ckpt_path, map_location=device))
    fusion_mode = args.fusion_mode
    inferred_fusion_mode = infer_fusion_mode(checkpoint, ckpt_path)
    if fusion_mode == "auto":
        fusion_mode = inferred_fusion_mode
        print(f"Inferred fusion_mode={fusion_mode} from {ckpt_path.name}")
    elif fusion_mode != inferred_fusion_mode:
        raise ValueError(
            f"Requested fusion_mode={fusion_mode}, but {ckpt_path} contains "
            f"a {inferred_fusion_mode} fusion checkpoint."
        )

    model = CLAMRNAFusion(
        wsi_model_type=args.model_type,
        size_arg=args.model_size,
        dropout=args.drop_out,
        k_sample=args.B,
        n_classes=len(CLASS_NAMES),
        subtyping=True,
        embed_dim=args.embed_dim,
        tabular_input_dim=tabular_input_dim,
        tabular_hidden_dim=args.tabular_hidden_dim,
        tabular_num_layers=args.tabular_num_layers,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_mode=fusion_mode,
        rna_hidden_dims=args.rna_hidden_dims,
        rna_dropout=args.rna_dropout,
        residual_scale=args.residual_scale,
    )

    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    return model


def build_dataset(args):
    dataset = Generic_Multimodal_MIL_Dataset(
        csv_path=args.dataset_csv,
        data_dir=args.data_root_dir,
        tabular_csv=args.tabular_csv,
        tabular_case_id_col=args.tabular_case_id_col,
        tabular_label_col=args.tabular_label_col,
        shuffle=False,
        seed=1,
        print_info=True,
        label_dict=LABEL_DICT,
        patient_strat=True,
        ignore=["Normal"],
    )
    dataset.load_from_h5(True)
    return dataset


def select_split(dataset, args, fold: int):
    split_path = Path(args.split_dir) / f"splits_{fold}.csv"
    train_split, val_split, test_split = dataset.return_splits(from_id=False, csv_path=str(split_path))
    splits = {"train": train_split, "val": val_split, "test": test_split}
    selected = splits[args.split]
    if selected is None:
        raise ValueError(f"Fold {fold} has no {args.split} split.")
    return selected


def compute_auc(labels: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float]]:
    per_class = {}
    if len(np.unique(labels)) < 2:
        return float("nan"), {name: float("nan") for name in CLASS_NAMES}

    for idx, name in enumerate(CLASS_NAMES):
        binary = (labels == idx).astype(np.int64)
        if binary.min() == binary.max():
            per_class[name] = float("nan")
            continue
        per_class[name] = float(roc_auc_score(binary, probs[:, idx]))

    valid = [value for value in per_class.values() if np.isfinite(value)]
    return float(np.mean(valid)) if valid else float("nan"), per_class


def compute_metrics(predictions: pd.DataFrame) -> dict:
    labels = predictions["label_idx"].to_numpy(dtype=np.int64)
    preds = predictions["pred_idx"].to_numpy(dtype=np.int64)
    probs = predictions[PROB_COLUMNS].to_numpy(dtype=np.float32)
    auc, class_auc = compute_auc(labels, probs)

    metrics = {
        "n": int(len(predictions)),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "macro_auc": auc,
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
    for key in FUSION_RESULT_KEYS:
        if key in predictions.columns:
            values = predictions[key].dropna()
            if not values.empty:
                metrics[key] = float(values.mean())
                metrics[f"{key}_slide_std"] = float(values.std(ddof=0))
    return metrics


def evaluate_fold(model, split_dataset, fold: int, split_name: str, device: torch.device) -> pd.DataFrame:
    loader = get_split_loader(split_dataset, training=False, testing=False)
    rows = []

    for batch_idx, (data, label) in enumerate(loader):
        wsi_features, tabular_features = data
        data = (wsi_features.to(device), tabular_features.to(device))
        label = label.to(device)

        with torch.inference_mode():
            _, probs, pred, _, results = model(data)

        slide_row = split_dataset.slide_data.iloc[batch_idx]
        prob_values = probs.cpu().numpy().reshape(-1)
        pred_idx = int(pred.item())
        label_idx = int(label.item())
        row = {
            "fold": fold,
            "split": split_name,
            "slide_id": str(slide_row["slide_id"]),
            "case_id": str(slide_row["case_id"]),
            "label_idx": label_idx,
            "label": CLASS_NAMES[label_idx],
            "pred_idx": pred_idx,
            "pred": CLASS_NAMES[pred_idx],
            "correct": bool(pred_idx == label_idx),
            "confidence": float(prob_values.max()),
        }
        for class_idx, class_name in enumerate(CLASS_NAMES):
            row[f"prob_{class_name}"] = float(prob_values[class_idx])
        for key in FUSION_RESULT_KEYS:
            if key in results:
                value = results[key]
                if torch.is_tensor(value):
                    value = value.detach().cpu().item()
                row[key] = float(value)
        rows.append(row)

    return pd.DataFrame(rows)


def save_metrics(metrics: dict, output_dir: Path, prefix: str) -> None:
    (output_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    report = pd.DataFrame(metrics["classification_report"]).transpose()
    report.to_csv(output_dir / f"{prefix}_classification_report.csv")

    cm = pd.DataFrame(metrics["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES)
    cm.to_csv(output_dir / f"{prefix}_confusion_matrix.csv")


def plot_confusion_matrix(metrics: dict, output_path: Path) -> None:
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.float32)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, matrix, title, fmt in (
        (axes[0], cm, "Counts", ".0f"),
        (axes[1], normalized, "Row-normalized", ".2f"),
    ):
        image = ax.imshow(matrix, cmap="Blues", vmin=0)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc(predictions: pd.DataFrame, output_path: Path) -> None:
    labels = predictions["label_idx"].to_numpy(dtype=np.int64)
    probs = predictions[PROB_COLUMNS].to_numpy(dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6, 5))
    for idx, class_name in enumerate(CLASS_NAMES):
        binary = (labels == idx).astype(np.int64)
        if binary.min() == binary.max():
            continue
        fpr, tpr, _ = roc_curve(binary, probs[:, idx])
        ax.plot(fpr, tpr, label=f"{class_name} AUC={calc_auc(fpr, tpr):.3f}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_fold_metrics(fold_metrics: pd.DataFrame, output_path: Path) -> None:
    metrics = ["accuracy", "balanced_accuracy", "macro_auc", "macro_f1"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(fold_metrics))
    width = 0.18
    for offset, metric in enumerate(metrics):
        ax.bar(x + (offset - 1.5) * width, fold_metrics[metric], width=width, label=metric)
    ax.set_xticks(x, [str(fold) for fold in fold_metrics["fold"]])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("Held-out test metrics by fold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confidence(predictions: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    correct = predictions[predictions["correct"]]["confidence"]
    wrong = predictions[~predictions["correct"]]["confidence"]
    ax.hist(correct, bins=20, alpha=0.7, label="Correct", color="#2ca02c")
    ax.hist(wrong, bins=20, alpha=0.7, label="Incorrect", color="#d62728")
    ax.set_xlabel("Max predicted probability")
    ax.set_ylabel("Slides")
    ax.set_title("Prediction confidence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_gate_weights(predictions: pd.DataFrame, output_path: Path) -> bool:
    if "fusion_wsi_gate_mean" not in predictions.columns:
        return False

    gate_values = predictions["fusion_wsi_gate_mean"].dropna()
    if gate_values.empty:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    correct = predictions[predictions["correct"]]["fusion_wsi_gate_mean"].dropna()
    wrong = predictions[~predictions["correct"]]["fusion_wsi_gate_mean"].dropna()
    axes[0].hist(correct, bins=20, alpha=0.7, label="Correct", color="#2ca02c")
    axes[0].hist(wrong, bins=20, alpha=0.7, label="Incorrect", color="#d62728")
    axes[0].axvline(0.5, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Mean WSI gate")
    axes[0].set_ylabel("Slides")
    axes[0].set_title("Gated fusion weight distribution")
    axes[0].legend()

    box_data = []
    box_labels = []
    for class_name in CLASS_NAMES:
        class_values = predictions.loc[
            predictions["label"] == class_name,
            "fusion_wsi_gate_mean",
        ].dropna()
        if not class_values.empty:
            box_data.append(class_values.to_numpy())
            box_labels.append(class_name)

    if box_data:
        axes[1].boxplot(box_data, tick_labels=box_labels, showfliers=False)
        axes[1].axhline(0.5, color="0.4", linestyle="--", linewidth=1)
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("True subtype")
        axes[1].set_ylabel("Mean WSI gate")
        axes[1].set_title("Gate by subtype")
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def save_gate_summary(predictions: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    if "fusion_wsi_gate_mean" not in predictions.columns:
        return

    gate_predictions = predictions.dropna(subset=["fusion_wsi_gate_mean"])
    if gate_predictions.empty:
        return

    by_label = (
        gate_predictions.groupby("label")["fusion_wsi_gate_mean"]
        .agg(["count", "mean", "std", "min", "max"])
        .reindex(CLASS_NAMES)
    )
    by_label["fusion_rna_gate_mean"] = 1.0 - by_label["mean"]
    by_label.to_csv(output_dir / f"{prefix}_gate_summary_by_label.csv")

    by_correct = gate_predictions.groupby("correct")["fusion_wsi_gate_mean"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    by_correct["fusion_rna_gate_mean"] = 1.0 - by_correct["mean"]
    by_correct.to_csv(output_dir / f"{prefix}_gate_summary_by_correctness.csv")


def load_training_history(ckpt_dir: Path, folds: list[int], output_dir: Path) -> pd.DataFrame:
    histories = []
    for fold in folds:
        history_path = ckpt_dir / f"fold_{fold}_history.csv"
        if not history_path.is_file():
            continue
        history = pd.read_csv(history_path)
        if "fold" not in history.columns:
            history["fold"] = fold
        histories.append(history)

    if not histories:
        return pd.DataFrame()

    history = pd.concat(histories, ignore_index=True)
    history.to_csv(output_dir / "training_history.csv", index=False)
    return history


def plot_training_history(history: pd.DataFrame, output_path: Path) -> None:
    if history.empty:
        return

    has_gate = any(
        column in history.columns
        for column in ("train/fusion_wsi_gate_mean", "val/fusion_wsi_gate_mean")
    )
    ncols = 3 if has_gate else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    for fold, fold_history in history.groupby("fold"):
        label = f"fold {fold}"
        if "train/loss" in fold_history:
            axes[0].plot(fold_history["epoch"], fold_history["train/loss"], label=f"{label} train")
        if "val/loss" in fold_history:
            axes[0].plot(fold_history["epoch"], fold_history["val/loss"], linestyle="--", label=f"{label} val")
        if "val/auc" in fold_history:
            axes[1].plot(fold_history["epoch"], fold_history["val/auc"], label=f"{label} val AUC")
        if "val/accuracy" in fold_history:
            axes[1].plot(
                fold_history["epoch"],
                fold_history["val/accuracy"],
                linestyle="--",
                label=f"{label} val acc",
            )
        if has_gate and "train/fusion_wsi_gate_mean" in fold_history:
            axes[2].plot(
                fold_history["epoch"],
                fold_history["train/fusion_wsi_gate_mean"],
                label=f"{label} train",
            )
        if has_gate and "val/fusion_wsi_gate_mean" in fold_history:
            axes[2].plot(
                fold_history["epoch"],
                fold_history["val/fusion_wsi_gate_mean"],
                linestyle="--",
                label=f"{label} val",
            )

    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation performance")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    if has_gate:
        axes[2].axhline(0.5, color="0.4", linestyle="--", linewidth=1)
        axes[2].set_title("Mean WSI gate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Gate value")
        axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def maybe_init_wandb(args, output_dir: Path):
    if not args.wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is required when --wandb is set.")

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name or f"multimodal_eval_{args.split}",
        "tags": ["multimodal", "evaluation", args.split, args.fusion_mode],
        "dir": str(output_dir),
        "config": vars(args),
    }
    if args.wandb_mode is not None:
        init_kwargs["mode"] = args.wandb_mode
    return wandb.init(**init_kwargs)


def log_to_wandb(
    run,
    metrics: dict,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    training_history: pd.DataFrame,
    output_dir: Path,
    split_name: str,
):
    if run is None:
        return

    flat_metrics = {
        f"{split_name}/n": metrics["n"],
        f"{split_name}/accuracy": metrics["accuracy"],
        f"{split_name}/balanced_accuracy": metrics["balanced_accuracy"],
        f"{split_name}/macro_auc": metrics["macro_auc"],
        f"{split_name}/macro_f1": metrics["macro_f1"],
        f"{split_name}/weighted_f1": metrics["weighted_f1"],
    }
    for class_name, value in metrics["class_auc"].items():
        flat_metrics[f"{split_name}_auc/{class_name}"] = value
    for key in FUSION_RESULT_KEYS:
        if key in metrics:
            flat_metrics[f"{split_name}/{key}"] = metrics[key]

    payload = {
        **flat_metrics,
        "plots/confusion_matrix": wandb.Image(str(output_dir / f"{split_name}_confusion_matrix.png")),
        "plots/roc": wandb.Image(str(output_dir / f"{split_name}_roc.png")),
        "plots/fold_metrics": wandb.Image(str(output_dir / "fold_metrics.png")),
        "plots/confidence": wandb.Image(str(output_dir / f"{split_name}_confidence.png")),
        "tables/fold_metrics": wandb.Table(dataframe=fold_metrics),
        f"tables/{split_name}_predictions": wandb.Table(dataframe=predictions),
    }
    gate_plot = output_dir / f"{split_name}_gate_weights.png"
    if gate_plot.is_file():
        payload["plots/gate_weights"] = wandb.Image(str(gate_plot))
    gate_by_label = output_dir / f"{split_name}_gate_summary_by_label.csv"
    gate_by_correctness = output_dir / f"{split_name}_gate_summary_by_correctness.csv"
    if gate_by_label.is_file():
        payload["tables/gate_summary_by_label"] = wandb.Table(dataframe=pd.read_csv(gate_by_label))
    if gate_by_correctness.is_file():
        payload["tables/gate_summary_by_correctness"] = wandb.Table(dataframe=pd.read_csv(gate_by_correctness))
    if not training_history.empty:
        payload["plots/training_history"] = wandb.Image(str(output_dir / "training_history.png"))
        payload["tables/training_history"] = wandb.Table(dataframe=training_history)

    run.log(payload)

    artifact = wandb.Artifact(f"{run.name}_outputs", type="multimodal-evaluation")
    for path in output_dir.glob("*"):
        if path.is_file():
            artifact.add_file(str(path))
    run.log_artifact(artifact)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args)

    fold_rows = []
    fold_metric_rows = []
    for fold in fold_range(args):
        split_dataset = select_split(dataset, args, fold)
        transform = load_transform(Path(args.ckpt_dir) / f"s_{fold}_tabular_transform.json")
        split_dataset.set_tabular_transform(transform)

        ckpt_path = Path(args.ckpt_dir) / f"s_{fold}_checkpoint.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        model = load_model(args, split_dataset.tabular_feature_dim, ckpt_path, device)
        predictions = evaluate_fold(model, split_dataset, fold, args.split, device)
        predictions.to_csv(output_dir / f"fold_{fold}_{args.split}_predictions.csv", index=False)

        fold_metrics = compute_metrics(predictions)
        save_metrics(fold_metrics, output_dir, f"fold_{fold}_{args.split}")
        fold_metric_row = {
            "fold": fold,
            "n": fold_metrics["n"],
            "accuracy": fold_metrics["accuracy"],
            "balanced_accuracy": fold_metrics["balanced_accuracy"],
            "macro_auc": fold_metrics["macro_auc"],
            "macro_f1": fold_metrics["macro_f1"],
            "weighted_f1": fold_metrics["weighted_f1"],
        }
        for key in FUSION_RESULT_KEYS:
            if key in fold_metrics:
                fold_metric_row[key] = fold_metrics[key]
        fold_metric_rows.append(fold_metric_row)
        fold_rows.append(predictions)
        print(
            "Fold {} {}: n={}, acc={:.4f}, bal_acc={:.4f}, macro_auc={:.4f}, macro_f1={:.4f}".format(
                fold,
                args.split,
                fold_metrics["n"],
                fold_metrics["accuracy"],
                fold_metrics["balanced_accuracy"],
                fold_metrics["macro_auc"],
                fold_metrics["macro_f1"],
            )
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_predictions = pd.concat(fold_rows, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    aggregate_metrics = compute_metrics(all_predictions)

    all_predictions.to_csv(output_dir / f"all_{args.split}_predictions.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    save_metrics(aggregate_metrics, output_dir, args.split)

    plot_confusion_matrix(aggregate_metrics, output_dir / f"{args.split}_confusion_matrix.png")
    plot_roc(all_predictions, output_dir / f"{args.split}_roc.png")
    plot_fold_metrics(fold_metrics_df, output_dir / "fold_metrics.png")
    plot_confidence(all_predictions, output_dir / f"{args.split}_confidence.png")
    plot_gate_weights(all_predictions, output_dir / f"{args.split}_gate_weights.png")
    save_gate_summary(all_predictions, output_dir, args.split)
    training_history = load_training_history(Path(args.ckpt_dir), fold_range(args), output_dir)
    if not training_history.empty:
        plot_training_history(training_history, output_dir / "training_history.png")

    print("\nAggregate {} metrics:".format(args.split))
    print(
        "n={n}, acc={accuracy:.4f}, bal_acc={balanced_accuracy:.4f}, "
        "macro_auc={macro_auc:.4f}, macro_f1={macro_f1:.4f}, weighted_f1={weighted_f1:.4f}".format(
            **aggregate_metrics
        )
    )
    if "fusion_wsi_gate_mean" in aggregate_metrics:
        print(
            "fusion_wsi_gate_mean={fusion_wsi_gate_mean:.4f}, "
            "fusion_rna_gate_mean={fusion_rna_gate_mean:.4f}, "
            "fusion_gate_std={fusion_gate_std:.4f}".format(**aggregate_metrics)
        )
    if "fusion_wsi_to_rna_attention" in aggregate_metrics:
        print(
            "fusion_wsi_to_rna_attention={fusion_wsi_to_rna_attention:.4f}, "
            "fusion_rna_to_wsi_attention={fusion_rna_to_wsi_attention:.4f}".format(**aggregate_metrics)
        )
    print(f"Saved evaluation outputs to: {output_dir}")

    run = maybe_init_wandb(args, output_dir)
    try:
        log_to_wandb(run, aggregate_metrics, fold_metrics_df, all_predictions, training_history, output_dir, args.split)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
