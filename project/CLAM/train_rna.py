from __future__ import annotations

import argparse
import json
import math
import os
import random
from copy import deepcopy
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
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset_modules.rna_dataset import RNATabularDataset, RNAFeatureTransform, read_rna_clam_table
from models.model_rna import RNA_MLP

try:
    import wandb
except ImportError:
    wandb = None


CLASS_NAMES = {
    "4class": ["LumA", "LumB", "Basal", "Her2"],
    "5class": ["LumA", "LumB", "Basal", "Her2", "Normal"],
}


def parse_hidden_dims(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    if value is None or str(value).strip() == "":
        return ()
    return tuple(int(dim.strip()) for dim in str(value).split(",") if dim.strip())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def require_wandb(args) -> None:
    if args.wandb and wandb is None:
        raise ImportError("wandb is not installed. Install it or rerun without --wandb.")


def init_wandb_run(args, settings: dict, fold: int | None, results_dir: Path) -> bool:
    if not args.wandb:
        return False

    require_wandb(args)
    wandb_root = results_dir / "wandb"
    wandb_root.mkdir(parents=True, exist_ok=True)
    for env_name, env_path in {
        "WANDB_DIR": wandb_root,
        "WANDB_CONFIG_DIR": wandb_root / "config",
        "WANDB_CACHE_DIR": wandb_root / "cache",
        "WANDB_DATA_DIR": wandb_root / "data",
    }.items():
        env_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(env_path))
    os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")

    run_name = f"{args.exp_code}_{args.class_set}_summary" if fold is None else f"{args.exp_code}_{args.class_set}_fold{fold}"
    tags = list(args.wandb_tags or [])
    if fold is None:
        tags.append("summary")

    if wandb.run is not None:
        wandb.config.update(settings if fold is None else {**settings, "fold": int(fold)}, allow_val_change=True)
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        return False

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": run_name,
        "tags": tags,
        "group": f"{args.exp_code}_{args.class_set}_s{args.seed}",
        "config": settings if fold is None else {**settings, "fold": int(fold)},
        "dir": str(results_dir),
        "reinit": True,
        "settings": wandb.Settings(start_method="thread", _disable_service=True),
    }
    if args.wandb_mode is not None:
        init_kwargs["mode"] = args.wandb_mode

    wandb.init(**init_kwargs)
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    return True


def finish_wandb_run(owns_run: bool) -> None:
    if owns_run and wandb is not None and wandb.run is not None:
        wandb.finish()


def wandb_log(metrics: dict) -> None:
    if wandb is not None and wandb.run is not None:
        wandb.log(metrics)


def flatten_final_metrics(prefix: str, metrics: dict, class_names: list[str]) -> dict:
    return {
        f"{prefix}/loss": metrics["loss"],
        f"{prefix}/error": metrics["error"],
        f"{prefix}/acc": metrics["acc"],
        f"{prefix}/balanced_acc": metrics["balanced_acc"],
        f"{prefix}/auc": metrics["auc"],
        f"{prefix}/macro_f1": metrics["macro_f1"],
        f"{prefix}/weighted_f1": metrics["weighted_f1"],
        f"{prefix}/n": metrics["n"],
    }


def resolve_data_path(args) -> Path:
    if args.data_path:
        return Path(args.data_path)
    return Path(args.rna_dir) / f"TCGA_BRCA_RNA_primary_tumor_{args.class_set}_clam.csv.gz"


def resolve_split_dir(args, clam_dir: Path) -> Path:
    default_name = f"tcga_brca_rna_{args.class_set}_{int(args.label_frac * 100)}"
    if args.split_dir is None:
        return clam_dir / "splits" / default_name

    split_dir = Path(args.split_dir)
    if split_dir.is_absolute():
        return split_dir
    if split_dir.parts and split_dir.parts[0] == "splits":
        return clam_dir / split_dir
    return clam_dir / "splits" / split_dir


def split_counts(n_items: int, val_frac: float, test_frac: float) -> tuple[int, int]:
    n_val = max(1, int(round(n_items * val_frac))) if val_frac > 0 else 0
    n_test = max(1, int(round(n_items * test_frac))) if test_frac > 0 else 0

    while n_val + n_test >= n_items and n_test > 0:
        n_test -= 1
    while n_val + n_test >= n_items and n_val > 0:
        n_val -= 1
    return n_val, n_test


def create_rna_splits(
    metadata: pd.DataFrame,
    split_dir: Path,
    class_names: list[str],
    k: int,
    seed: int,
    val_frac: float,
    test_frac: float,
    label_frac: float,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)

    label_per_case = metadata.groupby("case_id")["label"].nunique()
    ambiguous_cases = label_per_case[label_per_case > 1]
    if not ambiguous_cases.empty:
        raise ValueError(
            "Cannot create patient-stratified splits because some case_id values have multiple labels: "
            f"{ambiguous_cases.index[:5].tolist()}"
        )

    case_labels = metadata[["case_id", "label"]].drop_duplicates().reset_index(drop=True)
    sample_lookup = metadata.groupby("case_id")["sample"].apply(list).to_dict()

    for fold in range(k):
        rng = np.random.default_rng(seed + fold)
        split_cases = {"train": [], "val": [], "test": []}

        for class_name in class_names:
            class_cases = case_labels.loc[case_labels["label"] == class_name, "case_id"].to_numpy()
            class_cases = rng.permutation(class_cases)
            n_val, n_test = split_counts(len(class_cases), val_frac, test_frac)

            val_cases = class_cases[:n_val]
            test_cases = class_cases[n_val : n_val + n_test]
            train_cases = class_cases[n_val + n_test :]

            if label_frac < 1.0:
                n_train = max(1, int(math.ceil(len(train_cases) * label_frac)))
                train_cases = train_cases[:n_train]

            split_cases["train"].extend(train_cases.tolist())
            split_cases["val"].extend(val_cases.tolist())
            split_cases["test"].extend(test_cases.tolist())

        split_samples = {
            split: [sample for case_id in case_ids for sample in sample_lookup[case_id]]
            for split, case_ids in split_cases.items()
        }
        for split in split_samples:
            split_samples[split] = sorted(split_samples[split])

        split_df = pd.DataFrame({key: pd.Series(values) for key, values in split_samples.items()})
        split_df.to_csv(split_dir / f"splits_{fold}.csv", index=False)

        bool_df = pd.DataFrame(index=metadata["sample"].astype(str), columns=["train", "val", "test"])
        bool_df.loc[:, :] = False
        for split, samples in split_samples.items():
            bool_df.loc[samples, split] = True
        bool_df.to_csv(split_dir / f"splits_{fold}_bool.csv")

        descriptor = pd.DataFrame(index=class_names, columns=["train", "val", "test"], data=0)
        for split, samples in split_samples.items():
            labels = metadata.loc[metadata["sample"].isin(samples), "label"]
            counts = labels.value_counts()
            for class_name, count in counts.items():
                descriptor.loc[class_name, split] = int(count)
        descriptor.to_csv(split_dir / f"splits_{fold}_descriptor.csv")


def read_split(split_path: Path, metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    split_df = pd.read_csv(split_path, dtype=str)
    sample_to_idx = pd.Series(metadata.index, index=metadata["sample"].astype(str))
    split_indices = {}

    for split in ("train", "val", "test"):
        if split not in split_df.columns:
            raise ValueError(f"{split_path} is missing the '{split}' column.")

        sample_ids = split_df[split].dropna().astype(str)
        missing = sorted(set(sample_ids) - set(sample_to_idx.index))
        if missing:
            raise ValueError(
                f"{split_path} contains {len(missing)} {split} samples not found in the RNA table. "
                f"Examples: {missing[:5]}"
            )
        split_indices[split] = sample_to_idx.loc[sample_ids].to_numpy(dtype=np.int64)

    return split_indices


def make_loader(
    dataset: RNATabularDataset,
    batch_size: int,
    training: bool,
    weighted_sample: bool,
    num_workers: int,
) -> DataLoader:
    if training and weighted_sample:
        labels = dataset.labels.numpy()
        class_counts = np.bincount(labels)
        sample_weights = np.array([1.0 / class_counts[label] for label in labels], dtype=np.float64)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=num_workers)


def class_weight_tensor(labels: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_auc(labels: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")

    try:
        if n_classes == 2:
            return float(roc_auc_score(labels, probs[:, 1]))

        aucs = []
        for class_idx in range(n_classes):
            binary_labels = (labels == class_idx).astype(np.int64)
            if binary_labels.min() == binary_labels.max():
                continue
            aucs.append(roc_auc_score(binary_labels, probs[:, class_idx]))
        return float(np.mean(aucs)) if aucs else float("nan")
    except ValueError:
        return float("nan")


def evaluate(model, loader, loss_fn, n_classes: int, device: torch.device, class_names: list[str]):
    model.eval()
    losses = []
    all_probs = []
    all_labels = []
    all_preds = []
    all_samples = []
    all_cases = []

    with torch.inference_mode():
        for features, labels, samples, cases in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = loss_fn(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            losses.append(loss.item() * labels.numel())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_samples.extend(samples)
            all_cases.extend(cases)

    labels_np = np.concatenate(all_labels)
    preds_np = np.concatenate(all_preds)
    probs_np = np.concatenate(all_probs)
    loss = float(np.sum(losses) / len(labels_np))

    metrics = {
        "loss": loss,
        "error": float(1.0 - accuracy_score(labels_np, preds_np)),
        "acc": float(accuracy_score(labels_np, preds_np)),
        "balanced_acc": float(balanced_accuracy_score(labels_np, preds_np)),
        "macro_f1": float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels_np, preds_np, average="weighted", zero_division=0)),
        "auc": compute_auc(labels_np, probs_np, n_classes),
        "n": int(len(labels_np)),
        "classification_report": classification_report(
            labels_np,
            preds_np,
            labels=list(range(n_classes)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels_np, preds_np, labels=list(range(n_classes))).tolist(),
    }

    predictions = pd.DataFrame(
        {
            "sample": all_samples,
            "case_id": all_cases,
            "label": [class_names[label] for label in labels_np],
            "label_idx": labels_np,
            "pred": [class_names[pred] for pred in preds_np],
            "pred_idx": preds_np,
        }
    )
    for class_idx, class_name in enumerate(class_names):
        predictions[f"prob_{class_name}"] = probs_np[:, class_idx]

    return metrics, predictions


def train_one_fold(
    fold: int,
    args,
    metadata: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    split_dir: Path,
    results_dir: Path,
    device: torch.device,
):
    split_indices = read_split(split_dir / f"splits_{fold}.csv", metadata)
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    transform = RNAFeatureTransform.fit(
        features[train_idx],
        feature_names,
        top_n_genes=args.top_n_genes,
    )
    x_train = transform.transform(features[train_idx])
    x_val = transform.transform(features[val_idx])
    x_test = transform.transform(features[test_idx])

    train_dataset = RNATabularDataset(metadata.iloc[train_idx], x_train, labels[train_idx])
    val_dataset = RNATabularDataset(metadata.iloc[val_idx], x_val, labels[val_idx])
    test_dataset = RNATabularDataset(metadata.iloc[test_idx], x_test, labels[test_idx])

    train_loader = make_loader(train_dataset, args.batch_size, True, args.weighted_sample, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, False, args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, False, False, args.num_workers)

    model = RNA_MLP(
        input_dim=x_train.shape[1],
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.drop_out,
        n_classes=len(class_names),
    ).to(device)

    loss_weight = class_weight_tensor(labels[train_idx], len(class_names), device) if args.weighted_loss else None
    loss_fn = nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=args.label_smoothing)

    if args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError(args.opt)

    best_score = -float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.max_epochs):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for batch_features, batch_labels, _, _ in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_features)
            loss = loss_fn(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item() * batch_labels.numel()
            n_seen += batch_labels.numel()

        train_loss = running_loss / max(n_seen, 1)
        val_metrics, _ = evaluate(model, val_loader, loss_fn, len(class_names), device, class_names)
        score = val_metrics["auc"]
        if not np.isfinite(score):
            score = -val_metrics["loss"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_balanced_acc": val_metrics["balanced_acc"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_auc": val_metrics["auc"],
            }
        )
        wandb_log(
            {
                "epoch": epoch,
                "train/loss": float(train_loss),
                "val/loss": val_metrics["loss"],
                "val/acc": val_metrics["acc"],
                "val/balanced_acc": val_metrics["balanced_acc"],
                "val/macro_f1": val_metrics["macro_f1"],
                "val/weighted_f1": val_metrics["weighted_f1"],
                "val/auc": val_metrics["auc"],
            }
        )

        print(
            "Fold {}, epoch {}: train_loss={:.4f}, val_loss={:.4f}, val_acc={:.4f}, "
            "val_bal_acc={:.4f}, val_auc={:.4f}".format(
                fold,
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["acc"],
                val_metrics["balanced_acc"],
                val_metrics["auc"],
            )
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.early_stopping and epochs_without_improvement >= args.patience:
            print(f"Fold {fold}: early stopping at epoch {epoch}")
            break

    if best_state is None:
        best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    val_metrics, val_predictions = evaluate(model, val_loader, loss_fn, len(class_names), device, class_names)
    test_metrics, test_predictions = evaluate(model, test_loader, loss_fn, len(class_names), device, class_names)

    checkpoint = {
        "model_state_dict": best_state,
        "input_dim": int(x_train.shape[1]),
        "hidden_dims": parse_hidden_dims(args.hidden_dims),
        "dropout": args.drop_out,
        "class_names": class_names,
        "label_dict": {class_name: i for i, class_name in enumerate(class_names)},
        "selected_feature_names": transform.selected_feature_names,
        "selected_idx": transform.selected_idx.tolist(),
        "mean": transform.mean.tolist(),
        "std": transform.std.tolist(),
        "args": vars(args),
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, results_dir / f"s_{fold}_checkpoint.pt")

    pd.DataFrame(history).to_csv(results_dir / f"fold_{fold}_history.csv", index=False)
    val_predictions_path = results_dir / f"fold_{fold}_val_predictions.csv"
    test_predictions_path = results_dir / f"fold_{fold}_test_predictions.csv"
    val_predictions.to_csv(val_predictions_path, index=False)
    test_predictions.to_csv(test_predictions_path, index=False)

    fold_results = {
        "fold": fold,
        "best_epoch": best_epoch,
        "n_train": int(len(train_dataset)),
        "n_val": int(len(val_dataset)),
        "n_test": int(len(test_dataset)),
        "n_features": int(x_train.shape[1]),
        "val": val_metrics,
        "test": test_metrics,
    }
    (results_dir / f"fold_{fold}_results.json").write_text(json.dumps(fold_results, indent=2) + "\n")

    final_log = {
        "best_epoch": best_epoch,
        "n_train": int(len(train_dataset)),
        "n_val": int(len(val_dataset)),
        "n_test": int(len(test_dataset)),
        "n_features": int(x_train.shape[1]),
    }
    final_log.update(flatten_final_metrics("final/val", val_metrics, class_names))
    final_log.update(flatten_final_metrics("final/test", test_metrics, class_names))
    wandb_log(final_log)

    if getattr(args, "wandb_log_artifacts", False) and wandb is not None and wandb.run is not None:
        artifact = wandb.Artifact(
            f"{args.exp_code}_{args.class_set}_fold{fold}_rna_outputs",
            type="rna-fold-results",
        )
        artifact.add_file(str(results_dir / f"fold_{fold}_history.csv"))
        artifact.add_file(str(results_dir / f"fold_{fold}_results.json"))
        artifact.add_file(str(val_predictions_path))
        artifact.add_file(str(test_predictions_path))
        wandb.log_artifact(artifact)

    print(
        "Fold {} best_epoch={}: val_auc={:.4f}, val_acc={:.4f}, test_auc={:.4f}, test_acc={:.4f}".format(
            fold,
            best_epoch,
            val_metrics["auc"],
            val_metrics["acc"],
            test_metrics["auc"],
            test_metrics["acc"],
        )
    )

    return fold_results


def build_arg_parser(default_rna_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RNA-only TCGA-BRCA molecular subtype training")
    parser.add_argument("--rna_dir", type=str, default=str(default_rna_dir))
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--class_set", type=str, choices=["4class", "5class"], default="4class")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k_start", type=int, default=-1)
    parser.add_argument("--k_end", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label_frac", type=float, default=1.0)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--split_dir", type=str, default=None)
    parser.add_argument("--force_splits", action="store_true", default=False)
    parser.add_argument("--no_auto_splits", action="store_true", default=False)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--exp_code", type=str, default="tcga_brca_rna")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--opt", type=str, choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--hidden_dims", type=str, default="512,256")
    parser.add_argument("--drop_out", type=float, default=0.25)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--top_n_genes", type=int, default=0)
    parser.add_argument("--weighted_loss", action="store_true", default=False)
    parser.add_argument("--weighted_sample", action="store_true", default=False)
    parser.add_argument("--early_stopping", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="rna-subtyping")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None)
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--wandb_log_artifacts", action="store_true", default=False)
    return parser


def run_experiment(args, clam_dir: Path | None = None) -> pd.DataFrame:
    if clam_dir is None:
        clam_dir = Path(__file__).resolve().parent

    seed_everything(args.seed)
    require_wandb(args)

    class_names = CLASS_NAMES[args.class_set]
    label_dict = {class_name: i for i, class_name in enumerate(class_names)}
    data_path = resolve_data_path(args)
    split_dir = resolve_split_dir(args, clam_dir)

    metadata, features, labels, feature_names = read_rna_clam_table(data_path, label_dict)

    expected_split_files = [split_dir / f"splits_{fold}.csv" for fold in range(args.k)]
    missing_split_files = [path for path in expected_split_files if not path.is_file()]
    need_splits = args.force_splits or not split_dir.is_dir() or bool(missing_split_files)

    if need_splits:
        if args.no_auto_splits:
            missing = ", ".join(str(path) for path in missing_split_files[:5])
            raise FileNotFoundError(f"RNA split files are missing under {split_dir}: {missing}")
        print(f"Creating RNA splits in {split_dir}")
        create_rna_splits(
            metadata=metadata,
            split_dir=split_dir,
            class_names=class_names,
            k=args.k,
            seed=args.seed,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            label_frac=args.label_frac,
        )

    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end
    folds = np.arange(start, end)

    results_root = Path(args.results_dir)
    if not results_root.is_absolute():
        results_root = clam_dir / results_root
    results_dir = results_root / f"{args.exp_code}_{args.class_set}_s{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    settings = vars(args).copy()
    settings.update(
        {
            "data_path": str(data_path),
            "split_dir": str(split_dir),
            "results_dir": str(results_dir),
            "class_names": class_names,
            "n_samples": int(len(metadata)),
            "n_genes": int(features.shape[1]),
        }
    )
    (results_dir / f"experiment_{args.exp_code}.json").write_text(json.dumps(settings, indent=2) + "\n")

    print("################# RNA Settings ###################")
    for key, value in settings.items():
        print(f"{key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    fold_results = []
    for fold in folds:
        seed_everything(args.seed + int(fold))
        owns_wandb_run = init_wandb_run(args, settings, int(fold), results_dir)
        try:
            fold_results.append(
                train_one_fold(
                    fold=int(fold),
                    args=args,
                    metadata=metadata,
                    features=features,
                    labels=labels,
                    feature_names=feature_names,
                    class_names=class_names,
                    split_dir=split_dir,
                    results_dir=results_dir,
                    device=device,
                )
            )
        finally:
            finish_wandb_run(owns_wandb_run)

    summary_rows = []
    for result in fold_results:
        summary_rows.append(
            {
                "folds": result["fold"],
                "best_epoch": result["best_epoch"],
                "n_train": result["n_train"],
                "n_val": result["n_val"],
                "n_test": result["n_test"],
                "n_features": result["n_features"],
                "val_auc": result["val"]["auc"],
                "val_acc": result["val"]["acc"],
                "val_balanced_acc": result["val"]["balanced_acc"],
                "val_macro_f1": result["val"]["macro_f1"],
                "val_weighted_f1": result["val"]["weighted_f1"],
                "val_loss": result["val"]["loss"],
                "test_auc": result["test"]["auc"],
                "test_acc": result["test"]["acc"],
                "test_balanced_acc": result["test"]["balanced_acc"],
                "test_macro_f1": result["test"]["macro_f1"],
                "test_weighted_f1": result["test"]["weighted_f1"],
                "test_loss": result["test"]["loss"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    if len(folds) != args.k:
        summary_name = f"summary_partial_{start}_{end}.csv"
    else:
        summary_name = "summary.csv"
    summary.to_csv(results_dir / summary_name, index=False)

    if args.wandb and len(fold_results) > 1:
        owns_wandb_run = init_wandb_run(args, settings, None, results_dir)
        try:
            summary_metrics = {
                "mean_val_auc": float(summary["val_auc"].mean()),
                "std_val_auc": float(summary["val_auc"].std(ddof=0)),
                "mean_val_acc": float(summary["val_acc"].mean()),
                "std_val_acc": float(summary["val_acc"].std(ddof=0)),
                "mean_val_balanced_acc": float(summary["val_balanced_acc"].mean()),
                "std_val_balanced_acc": float(summary["val_balanced_acc"].std(ddof=0)),
                "mean_val_macro_f1": float(summary["val_macro_f1"].mean()),
                "std_val_macro_f1": float(summary["val_macro_f1"].std(ddof=0)),
                "mean_val_weighted_f1": float(summary["val_weighted_f1"].mean()),
                "std_val_weighted_f1": float(summary["val_weighted_f1"].std(ddof=0)),
                "mean_test_auc": float(summary["test_auc"].mean()),
                "std_test_auc": float(summary["test_auc"].std(ddof=0)),
                "mean_test_acc": float(summary["test_acc"].mean()),
                "std_test_acc": float(summary["test_acc"].std(ddof=0)),
                "mean_test_balanced_acc": float(summary["test_balanced_acc"].mean()),
                "std_test_balanced_acc": float(summary["test_balanced_acc"].std(ddof=0)),
                "mean_test_macro_f1": float(summary["test_macro_f1"].mean()),
                "std_test_macro_f1": float(summary["test_macro_f1"].std(ddof=0)),
                "mean_test_weighted_f1": float(summary["test_weighted_f1"].mean()),
                "std_test_weighted_f1": float(summary["test_weighted_f1"].std(ddof=0)),
            }
            wandb_log(summary_metrics)
            if getattr(args, "wandb_log_artifacts", False):
                artifact = wandb.Artifact(
                    f"{args.exp_code}_{args.class_set}_rna_summary",
                    type="rna-summary",
                )
                artifact.add_file(str(results_dir / summary_name))
                artifact.add_file(str(results_dir / f"experiment_{args.exp_code}.json"))
                wandb.log_artifact(artifact)
        finally:
            finish_wandb_run(owns_wandb_run)

    print(summary.to_string(index=False))
    print(f"Saved RNA results to: {results_dir}")
    return summary


def main():
    clam_dir = Path(__file__).resolve().parent
    workspace_root = clam_dir.parents[1]
    default_rna_dir = workspace_root / ".scratch" / "TCGA-BRCA-rna"
    parser = build_arg_parser(default_rna_dir)
    args = parser.parse_args()
    run_experiment(args, clam_dir=clam_dir)


if __name__ == "__main__":
    main()
