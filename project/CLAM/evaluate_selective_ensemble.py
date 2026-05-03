from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


CLASS_NAMES = ["LumA", "LumB", "Basal", "Her2"]
PROB_COLUMNS = [f"prob_{name}" for name in CLASS_NAMES]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validation-tuned probability ensemble for RNA + WSI multimodal runs. "
            "Each fold chooses convex branch weights on the validation split, then "
            "applies those weights to that fold's held-out test predictions."
        )
    )
    parser.add_argument("--rna_results_dir", type=str, required=True)
    parser.add_argument("--wsi_pred_dir", type=str, required=True)
    parser.add_argument("--concat_val_dir", type=str, required=True)
    parser.add_argument("--concat_test_dir", type=str, required=True)
    parser.add_argument("--gated_val_dir", type=str, required=True)
    parser.add_argument("--gated_test_dir", type=str, required=True)
    parser.add_argument("--residual_val_dir", type=str, required=True)
    parser.add_argument("--residual_test_dir", type=str, required=True)
    parser.add_argument("--cross_attention_val_dir", type=str, default=None)
    parser.add_argument("--cross_attention_test_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["weights", "stacker"],
        default="weights",
        help=(
            "weights tunes a convex branch average on validation. "
            "stacker trains a regularized logistic meta-classifier on validation branch probabilities."
        ),
    )
    parser.add_argument("--objective", type=str, choices=["macro_f1", "balanced_accuracy", "accuracy", "macro_auc"], default="balanced_accuracy")
    parser.add_argument("--weight_step", type=float, default=0.1, help="Coarse simplex grid step over all branches.")
    parser.add_argument("--rna_heavy_step", type=float, default=0.05, help="Extra finer grid step for RNA-heavy mixtures.")
    parser.add_argument("--max_aux_weight", type=float, default=0.5, help="Maximum non-RNA weight in the finer RNA-heavy grid.")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Temperature-calibrate each branch on the validation fold before selection.",
    )
    parser.add_argument("--temperature_min", type=float, default=0.5)
    parser.add_argument("--temperature_max", type=float, default=3.0)
    parser.add_argument("--temperature_step", type=float, default=0.05)
    parser.add_argument(
        "--stacker_c_values",
        type=str,
        default="0.01,0.03,0.1,0.3,1.0",
        help="Comma-separated inverse regularization strengths for stacker CV.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k_start", type=int, default=0)
    parser.add_argument("--k_end", type=int, default=-1)
    return parser.parse_args()


def fold_range(args) -> list[int]:
    end = args.k if args.k_end == -1 else args.k_end
    return list(range(args.k_start, end))


def simplex_grid(n_items: int, step: float):
    units = int(round(1.0 / step))
    for parts in itertools.product(range(units + 1), repeat=n_items - 1):
        total = sum(parts)
        if total <= units:
            yield np.asarray([*parts, units - total], dtype=np.float64) / units


def build_weight_grid(args, branch_names: list[str]) -> np.ndarray:
    weights = list(simplex_grid(len(branch_names), args.weight_step))

    if args.rna_heavy_step > 0 and args.max_aux_weight > 0:
        aux_names = branch_names[1:]
        values = np.arange(0.0, args.max_aux_weight + args.rna_heavy_step / 2.0, args.rna_heavy_step)
        for aux_weights in itertools.product(values, repeat=len(aux_names)):
            aux_total = float(sum(aux_weights))
            if aux_total <= args.max_aux_weight + 1e-12:
                weights.append(np.asarray([1.0 - aux_total, *aux_weights], dtype=np.float64))

    return np.unique(np.round(np.vstack(weights), 6), axis=0)


def read_prediction(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path)
    if "slide_id" not in df.columns:
        if "sample" not in df.columns:
            raise ValueError(f"{path} has neither slide_id nor sample column.")
        df["slide_id"] = df["sample"].astype(str)
    df["slide_id"] = df["slide_id"].astype(str)
    missing = [col for col in PROB_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing probability columns: {missing}")
    return df


def prediction_path(args, branch_name: str, fold: int, split: str) -> Path:
    if branch_name == "rna":
        return Path(args.rna_results_dir) / f"fold_{fold}_{split}_predictions.csv"
    if branch_name == "wsi":
        return Path(args.wsi_pred_dir) / f"fold_{fold}_wsi_{split}_predictions.csv"
    if branch_name == "concat":
        root = Path(args.concat_val_dir if split == "val" else args.concat_test_dir)
        return root / f"fold_{fold}_{split}_predictions.csv"
    if branch_name == "gated":
        root = Path(args.gated_val_dir if split == "val" else args.gated_test_dir)
        return root / f"fold_{fold}_{split}_predictions.csv"
    if branch_name == "residual":
        root = Path(args.residual_val_dir if split == "val" else args.residual_test_dir)
        return root / f"fold_{fold}_{split}_predictions.csv"
    if branch_name == "cross_attention":
        root = Path(args.cross_attention_val_dir if split == "val" else args.cross_attention_test_dir)
        return root / f"fold_{fold}_{split}_predictions.csv"
    raise ValueError(branch_name)


def align_predictions(args, branch_names: list[str], fold: int, split: str) -> tuple[pd.DataFrame, list[np.ndarray]]:
    base = read_prediction(prediction_path(args, branch_names[0], fold, split))
    meta_cols = ["slide_id", "case_id", "label_idx", "label"]
    missing_meta = [col for col in meta_cols if col not in base.columns]
    if missing_meta:
        raise ValueError(f"Base prediction file for fold {fold} {split} is missing {missing_meta}")

    metadata = base[meta_cols].copy()
    probs = []
    for branch_name in branch_names:
        df = read_prediction(prediction_path(args, branch_name, fold, split))
        merged = metadata[["slide_id"]].merge(df[["slide_id", *PROB_COLUMNS]], on="slide_id", how="inner")
        if len(merged) != len(metadata):
            raise ValueError(
                f"Alignment dropped rows for branch={branch_name}, fold={fold}, split={split}: "
                f"base={len(metadata)}, merged={len(merged)}"
            )
        probs.append(merged[PROB_COLUMNS].to_numpy(dtype=np.float64))
    return metadata, probs


def weighted_probs(branch_probs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    probs = np.zeros_like(branch_probs[0], dtype=np.float64)
    for weight, branch_prob in zip(weights, branch_probs):
        probs += float(weight) * branch_prob
    return probs / probs.sum(axis=1, keepdims=True)


def clip_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-12, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.power(clip_probs(probs), 1.0 / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def temperature_grid(args) -> np.ndarray:
    return np.arange(
        args.temperature_min,
        args.temperature_max + args.temperature_step / 2.0,
        args.temperature_step,
        dtype=np.float64,
    )


def tune_temperature(labels: np.ndarray, probs: np.ndarray, temps: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    best_temp = 1.0
    best_nll = float("inf")
    for temp in temps:
        calibrated = temperature_scale_probs(probs, float(temp))
        value = float(log_loss(labels, calibrated, labels=list(range(len(CLASS_NAMES)))))
        metrics = compute_metrics(labels, calibrated, include_report=False)
        rows.append(
            {
                "temperature": float(temp),
                "nll": value,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_auc": metrics["macro_auc"],
                "ece": metrics["ece"],
                "brier": metrics["brier"],
            }
        )
        if value < best_nll:
            best_nll = value
            best_temp = float(temp)
    return best_temp, pd.DataFrame(rows)


def calibrate_branch_probs(
    labels_val: np.ndarray,
    val_probs: list[np.ndarray],
    test_probs: list[np.ndarray],
    branch_names: list[str],
    args,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float], pd.DataFrame]:
    if not args.calibrate:
        return val_probs, test_probs, {name: 1.0 for name in branch_names}, pd.DataFrame()

    temps = temperature_grid(args)
    calibrated_val = []
    calibrated_test = []
    best_temps = {}
    rows = []
    for name, probs_val, probs_test in zip(branch_names, val_probs, test_probs):
        best_temp, sweep = tune_temperature(labels_val, probs_val, temps)
        sweep.insert(0, "branch", name)
        rows.append(sweep)
        best_temps[name] = best_temp
        calibrated_val.append(temperature_scale_probs(probs_val, best_temp))
        calibrated_test.append(temperature_scale_probs(probs_test, best_temp))

    return calibrated_val, calibrated_test, best_temps, pd.concat(rows, ignore_index=True)


def multiclass_brier(labels: np.ndarray, probs: np.ndarray) -> float:
    target = np.eye(len(CLASS_NAMES), dtype=np.float64)[labels]
    return float(np.mean(np.sum((target - probs) ** 2, axis=1)))


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = preds == labels
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (conf >= lower) & (conf <= upper)
        else:
            mask = (conf >= lower) & (conf < upper)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def compute_auc(labels: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float]]:
    class_auc = {}
    aucs = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        binary = (labels == class_idx).astype(np.int64)
        if binary.min() == binary.max():
            class_auc[class_name] = float("nan")
            continue
        value = float(roc_auc_score(binary, probs[:, class_idx]))
        class_auc[class_name] = value
        aucs.append(value)
    return float(np.mean(aucs)) if aucs else float("nan"), class_auc


def compute_metrics(labels: np.ndarray, probs: np.ndarray, include_report: bool = True) -> dict:
    probs = clip_probs(probs)
    preds = probs.argmax(axis=1)
    macro_auc, class_auc = compute_auc(labels, probs)
    metrics = {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "macro_auc": macro_auc,
        "nll": float(log_loss(labels, probs, labels=list(range(len(CLASS_NAMES))))),
        "brier": multiclass_brier(labels, probs),
        "ece": expected_calibration_error(labels, probs),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "class_auc": class_auc,
    }
    if include_report:
        metrics["classification_report"] = classification_report(
            labels,
            preds,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        metrics["confusion_matrix"] = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES)))).tolist()
    return metrics


def compute_fast_objectives(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def objective_value(metrics: dict, objective: str) -> float:
    return float(metrics[objective])


def tune_weights(labels: np.ndarray, branch_probs: list[np.ndarray], weight_grid: np.ndarray, objective: str):
    best = None
    for weights in weight_grid:
        probs = weighted_probs(branch_probs, weights)
        metrics = (
            compute_metrics(labels, probs, include_report=False)
            if objective == "macro_auc"
            else compute_fast_objectives(labels, probs)
        )
        score = objective_value(metrics, objective)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "weights": weights.copy(),
                "metrics": compute_metrics(labels, probs, include_report=False),
            }
    return best


def branch_entropy(probs: np.ndarray) -> np.ndarray:
    clipped = clip_probs(probs)
    return -np.sum(clipped * np.log(clipped), axis=1)


def branch_margin(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def stacker_features(branch_probs: list[np.ndarray]) -> np.ndarray:
    parts = []
    for probs in branch_probs:
        probs = clip_probs(probs)
        parts.append(probs)
        parts.append(np.log(probs))
        parts.append(probs.max(axis=1, keepdims=True))
        parts.append(branch_margin(probs).reshape(-1, 1))
        parts.append(branch_entropy(probs).reshape(-1, 1))
    return np.hstack(parts)


def parse_c_values(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("--stacker_c_values must contain at least one float")
    return values


def make_stacker(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=5000,
        solver="lbfgs",
    )


def stacker_cv_score(labels: np.ndarray, features: np.ndarray, c_value: float, objective: str) -> float:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    min_count = int(counts[counts > 0].min()) if np.any(counts > 0) else 0
    if min_count < 2 or len(labels) < 20:
        model = make_stacker(c_value)
        model.fit(features, labels)
        probs = model.predict_proba(features)
        return objective_value(compute_metrics(labels, probs, include_report=False), objective)

    n_splits = min(5, min_count)
    scores = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=17)
    for train_idx, heldout_idx in cv.split(features, labels):
        model = make_stacker(c_value)
        model.fit(features[train_idx], labels[train_idx])
        probs = model.predict_proba(features[heldout_idx])
        scores.append(objective_value(compute_metrics(labels[heldout_idx], probs, include_report=False), objective))
    return float(np.mean(scores))


def fit_stacker(labels: np.ndarray, val_probs: list[np.ndarray], args):
    features = stacker_features(val_probs)
    rows = []
    best_c = None
    best_score = -np.inf
    for c_value in parse_c_values(args.stacker_c_values):
        score = stacker_cv_score(labels, features, c_value, args.objective)
        rows.append({"C": c_value, "objective": args.objective, "cv_objective": score})
        if score > best_score:
            best_score = score
            best_c = c_value

    model = make_stacker(float(best_c))
    model.fit(features, labels)
    train_probs = model.predict_proba(features)
    summary = {
        "score": best_score,
        "C": float(best_c),
        "metrics": compute_metrics(labels, train_probs, include_report=False),
        "cv_sweep": pd.DataFrame(rows),
    }
    return model, summary


def stacker_predict(model: LogisticRegression, branch_probs: list[np.ndarray]) -> np.ndarray:
    probs = model.predict_proba(stacker_features(branch_probs))
    if probs.shape[1] == len(CLASS_NAMES):
        return probs

    full = np.zeros((probs.shape[0], len(CLASS_NAMES)), dtype=np.float64)
    for out_idx, class_idx in enumerate(model.classes_):
        full[:, int(class_idx)] = probs[:, out_idx]
    return clip_probs(full)


def save_predictions(metadata: pd.DataFrame, probs: np.ndarray, fold: int, output_path: Path) -> pd.DataFrame:
    preds = probs.argmax(axis=1)
    out = metadata.copy()
    out.insert(0, "fold", fold)
    out["pred_idx"] = preds
    out["pred"] = [CLASS_NAMES[pred] for pred in preds]
    out["correct"] = out["pred_idx"] == out["label_idx"]
    for class_idx, class_name in enumerate(CLASS_NAMES):
        out[f"prob_{class_name}"] = probs[:, class_idx]
    out.to_csv(output_path, index=False)
    return out


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selective_ensemble_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    if bool(args.cross_attention_val_dir) != bool(args.cross_attention_test_dir):
        raise ValueError("--cross_attention_val_dir and --cross_attention_test_dir must be provided together.")

    branch_names = ["rna", "wsi", "concat", "gated", "residual"]
    if args.cross_attention_val_dir:
        branch_names.append("cross_attention")
    weight_grid = build_weight_grid(args, branch_names)

    fold_rows = []
    all_test_predictions = []
    for fold in fold_range(args):
        val_metadata, val_probs = align_predictions(args, branch_names, fold, "val")
        labels_val = val_metadata["label_idx"].to_numpy(dtype=np.int64)
        test_metadata, test_probs = align_predictions(args, branch_names, fold, "test")
        labels_test = test_metadata["label_idx"].to_numpy(dtype=np.int64)

        val_probs, test_probs, temperatures, temp_sweep = calibrate_branch_probs(
            labels_val, val_probs, test_probs, branch_names, args
        )
        if not temp_sweep.empty:
            temp_sweep.to_csv(output_dir / f"fold_{fold}_temperature_sweep.csv", index=False)

        if args.mode == "weights":
            best = tune_weights(labels_val, val_probs, weight_grid, args.objective)
            probs_test = weighted_probs(test_probs, best["weights"])
            val_selector_metrics = best["metrics"]
            val_selector_score = best["score"]
            selector_fields = {
                f"weight_{name}": float(weight)
                for name, weight in zip(branch_names, best["weights"])
            }
            printed_selector = {
                name: round(float(weight), 3)
                for name, weight in zip(branch_names, best["weights"])
                if weight > 0
            }
        else:
            stacker, best = fit_stacker(labels_val, val_probs, args)
            best["cv_sweep"].to_csv(output_dir / f"fold_{fold}_stacker_cv_sweep.csv", index=False)
            probs_test = stacker_predict(stacker, test_probs)
            val_selector_metrics = best["metrics"]
            val_selector_score = best["score"]
            selector_fields = {"stacker_C": best["C"]}
            printed_selector = {"C": round(float(best["C"]), 4)}

        test_metrics = compute_metrics(labels_test, probs_test, include_report=False)

        test_df = save_predictions(test_metadata, probs_test, fold, output_dir / f"fold_{fold}_test_predictions.csv")
        all_test_predictions.append(test_df)

        row = {
            "fold": fold,
            "mode": args.mode,
            "calibrated": bool(args.calibrate),
            "objective": args.objective,
            "val_objective": val_selector_score,
        }
        row.update(selector_fields)
        row.update({f"temperature_{name}": float(temperatures[name]) for name in branch_names})
        row.update({f"val_{key}": value for key, value in val_selector_metrics.items() if key != "class_auc"})
        row.update({f"test_{key}": value for key, value in test_metrics.items() if key != "class_auc"})
        fold_rows.append(row)

        print(
            "fold {} selector={} val_{}={:.4f} test_auc={:.4f} test_acc={:.4f} test_bal_acc={:.4f} test_macro_f1={:.4f} test_ece={:.4f}".format(
                fold,
                printed_selector,
                args.objective,
                val_selector_score,
                test_metrics["macro_auc"],
                test_metrics["accuracy"],
                test_metrics["balanced_accuracy"],
                test_metrics["macro_f1"],
                test_metrics["ece"],
            )
        )

    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(output_dir / "fold_selector_summary.csv", index=False)
    if args.mode == "weights":
        fold_summary.to_csv(output_dir / "fold_weight_summary.csv", index=False)

    all_predictions = pd.concat(all_test_predictions, ignore_index=True)
    all_predictions.to_csv(output_dir / "all_test_predictions.csv", index=False)

    aggregate_labels = all_predictions["label_idx"].to_numpy(dtype=np.int64)
    aggregate_probs = all_predictions[PROB_COLUMNS].to_numpy(dtype=np.float64)
    aggregate_metrics = compute_metrics(aggregate_labels, aggregate_probs)
    (output_dir / "test_metrics.json").write_text(json.dumps(aggregate_metrics, indent=2) + "\n")

    print("\nAggregate selective ensemble test metrics")
    print(json.dumps({key: aggregate_metrics[key] for key in ("n", "macro_auc", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "nll", "brier", "ece")}, indent=2))
    print(f"Saved selective ensemble outputs to {output_dir}")


if __name__ == "__main__":
    main()
