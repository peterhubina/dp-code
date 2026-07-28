#!/usr/bin/env python
"""Tabular-only (no-WSI) baselines for the TCGA-BRCA ER-status ablation.

The three-arm ablation (WSI-alone, WSI+RNA gated, WSI+clinicopath gated) shows
fusion beats the image. It cannot show whether fusion beats the *table alone*,
because no tabular-only arm was ever run. This script fills that hole using the
same 10 tissue-site-holdout folds, the same case-level aggregation convention,
and the same fold-local feature transform (variance top-N selection +
standardisation fitted on the training fold only) that the fusion arms used.

Nothing is trained on the GPU and nothing in the repo is modified: this reads
the split CSVs, the dataset manifest, the two tabular tables, and the saved
fusion predictions, then fits sklearn probes on CPU.

Design decisions (stated so they can be reproduced):
  * Validation cases are LEFT OUT of probe training. The fusion arms fit their
    tabular transform on the train split only and used val purely for early
    stopping, so train-only keeps the fitting set identical.
  * Feature transform mirrors project/CLAM/dataset_modules/rna_dataset.py
    RNAFeatureTransform exactly (np.nanvar ranking, top-N, nan-safe mean/std,
    std floor 1e-6 -> 1.0, nan_to_num on the output).
  * top_n = 10000 for RNA and 0 (keep all 24) for clinicopath, matching the
    deployed fusion settings recorded in the experiment_*.txt files.

Run from the repo root:  python tools/diagnostics/tabular_only_probes.py
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier

import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_DIR = os.path.join(REPO_ROOT, "project", "CLAM", "splits", "tcga_brca_er_100")
DATASET_CSV = os.path.join(REPO_ROOT, "project", "CLAM", "dataset_csv", "tcga_brca_er.csv")
RESULTS_ROOT = os.path.join(REPO_ROOT, ".scratch", "results", "er")
OUT_DIR = os.path.join(RESULTS_ROOT, "diagnostics")

LABEL_DICT = {"ER-negative": 0, "ER-positive": 1}
N_FOLDS = 10
SEED = 1
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

MODALITIES = {
    "rna": {
        "table": os.path.join(REPO_ROOT, ".scratch", "TCGA-BRCA-rna", "tcga_brca_er_rna_clam.csv.gz"),
        "top_n": 10000,
        "fusion_arm": "er_wsi_rna_gated_s1",
        "fusion_label": "WSI+RNA gated",
    },
    "clinicopath": {
        "table": os.path.join(REPO_ROOT, "tools", "data", "tcga_brca_clinicopath_clam.csv"),
        "top_n": 0,
        "fusion_arm": "er_wsi_clinpath_gated_s1",
        "fusion_label": "WSI+clinicopath gated",
    },
}

# Metadata columns of the CLAM tabular tables (see rna_dataset.RNA_METADATA_COLUMNS
# and multimodal_dataset.BASE_METADATA_COLUMNS).
METADATA_COLUMNS = ("case_id", "sample", "label", "label_idx", "sample_type_code")


# --------------------------------------------------------------------------- #
# Feature transform: a faithful re-implementation of RNAFeatureTransform
# --------------------------------------------------------------------------- #
class FoldTransform:
    def __init__(self, selected_idx, mean, std):
        self.selected_idx = selected_idx
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, x_train: np.ndarray, top_n: int = 0, eps: float = 1e-6) -> "FoldTransform":
        if top_n and 0 < top_n < x_train.shape[1]:
            variances = np.nan_to_num(np.nanvar(x_train, axis=0), nan=-np.inf)
            selected_idx = np.sort(np.argsort(variances)[::-1][:top_n])
        else:
            selected_idx = np.arange(x_train.shape[1])

        x_sel = x_train[:, selected_idx]
        mean = np.nan_to_num(np.nanmean(x_sel, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        std = np.nan_to_num(np.nanstd(x_sel, axis=0), nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
        std[std < eps] = 1.0
        return cls(selected_idx.astype(np.int64), mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        scaled = (x[:, self.selected_idx] - self.mean) / self.std
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# DeLong paired AUROC test (Sun & Xu 2014 fast algorithm)
# --------------------------------------------------------------------------- #
def _midrank(x):
    order = np.argsort(x)
    ranked = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks + 1
    return out


def _fast_delong(preds_sorted, n_pos):
    n_neg = preds_sorted.shape[1] - n_pos
    pos = preds_sorted[:, :n_pos]
    neg = preds_sorted[:, n_pos:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1.0) / 2.0 / n_neg
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    cov = np.cov(v01) / n_pos + np.cov(v10) / n_neg
    return aucs, np.atleast_2d(cov)


def delong_test(y_true, p1, p2):
    """Two-sided DeLong p-value for AUROC(p1) - AUROC(p2) on paired data."""
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)
    n_pos = int(y_true.sum())
    preds = np.vstack((np.asarray(p1), np.asarray(p2)))[:, order]
    aucs, cov = _fast_delong(preds, n_pos)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = 0.0 if var <= 0 else (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * stats.norm.sf(abs(z)))


# --------------------------------------------------------------------------- #
# Metrics (same definitions as tools/analyze_er_ablation.py)
# --------------------------------------------------------------------------- #
def point_metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    yhat = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc_pos": float(average_precision_score(y, p)),
        "f1_pos": float(f1_score(y, yhat, pos_label=1)),
        "balanced_acc": float((sens + spec) / 2),
        "sensitivity_ERpos": float(sens),
        "specificity_ERneg": float(spec),
        "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
    }


def per_fold_auroc(df):
    aurocs = {}
    for fold, g in df.groupby("fold"):
        aurocs[int(fold)] = float(roc_auc_score(g.y, g.p)) if g.y.nunique() > 1 else float("nan")
    vals = [v for v in aurocs.values() if not np.isnan(v)]
    return {
        "per_fold": aurocs,
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "n_folds_scored": len(vals),
    }


def block(df):
    return {
        "pooled_case_level": point_metrics(df.y.values, df.p.values),
        "per_fold_auroc": per_fold_auroc(df),
    }


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_manifest():
    ds = pd.read_csv(DATASET_CSV, dtype=str)
    ds["y"] = ds["label"].map(LABEL_DICT)
    if ds["y"].isna().any():
        raise ValueError("Unmapped labels in the dataset manifest.")
    slide_to_case = dict(zip(ds.slide_id, ds.case_id))
    case_to_y = dict(zip(ds.case_id, ds.y.astype(int)))
    return ds, slide_to_case, case_to_y


def load_table(path):
    df = pd.read_csv(path)
    df = df[df["label"].isin(LABEL_DICT)].copy()
    df["case_id"] = df["case_id"].astype(str)
    df = df[~df["case_id"].duplicated(keep="first")].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    features = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = df["label"].map(LABEL_DICT).to_numpy(dtype=np.int64)
    case_index = {c: i for i, c in enumerate(df["case_id"])}
    return case_index, features, y, feature_cols


def fold_cases(fold, slide_to_case):
    path = os.path.join(SPLIT_DIR, f"splits_{fold}.csv")
    sp = pd.read_csv(path)
    out = {}
    for key in ("train", "val", "test"):
        slides = sp[key].dropna().astype(str).tolist()
        missing = [s for s in slides if s not in slide_to_case]
        if missing:
            raise ValueError(f"fold {fold} {key}: {len(missing)} slide_ids absent from the manifest")
        cases = list(dict.fromkeys(slide_to_case[s] for s in slides))
        out[key] = cases
    return out


def load_fusion_cases(arm_dir):
    """Case-level out-of-fold predictions for a fusion arm (mean over slides)."""
    rows = []
    paths = sorted(glob.glob(os.path.join(RESULTS_ROOT, arm_dir, "split_*_results.pkl")))
    if len(paths) != N_FOLDS:
        raise ValueError(f"{arm_dir}: expected {N_FOLDS} result pickles, found {len(paths)}")
    for path in paths:
        fold = int(os.path.basename(path).split("split_")[1].split("_results")[0])
        with open(path, "rb") as handle:
            fold_results = pickle.load(handle)
        for record in fold_results.values():
            slide_id = str(np.asarray(record["slide_id"]))
            prob = np.asarray(record["prob"]).ravel()
            rows.append((slide_id, fold, int(record["label"]), float(prob[1])))
    slides = pd.DataFrame(rows, columns=["slide_id", "fold", "y", "p"])
    ds = pd.read_csv(DATASET_CSV, dtype=str)
    slides["case_id"] = slides.slide_id.map(dict(zip(ds.slide_id, ds.case_id)))
    cases = (
        slides.groupby("case_id")
        .agg(fold=("fold", "first"), y=("y", "first"), p=("p", "mean"))
        .reset_index()
    )
    return cases


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def fit_logreg(x_tr, y_tr):
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        LogisticRegression(
            penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=3000, random_state=SEED
        ),
        {"C": C_GRID},
        scoring="roc_auc",
        cv=inner,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(x_tr, y_tr)
    return grid.best_estimator_, {"best_C": float(grid.best_params_["C"]), "inner_cv_auroc": float(grid.best_score_)}


def fit_mlp(x_tr, y_tr):
    clf = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        alpha=1e-3,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.15,
        random_state=SEED,
    )
    clf.fit(x_tr, y_tr)
    return clf, {"n_iter": int(clf.n_iter_), "best_validation_score": float(clf.best_validation_score_)}


PROBES = {"logreg_l2": fit_logreg, "mlp_128": fit_mlp}


# --------------------------------------------------------------------------- #
def run_modality(name, cfg, slide_to_case, case_to_y, wsi_cases):
    table_path = cfg["table"]
    if not os.path.isfile(table_path):
        raise FileNotFoundError(table_path)

    print(f"\n=== {name} ===\nloading {table_path}", flush=True)
    case_index, features, table_y, feature_cols = load_table(table_path)
    print(f"table: {len(case_index)} cases, {len(feature_cols)} feature columns", flush=True)

    fusion_cases = load_fusion_cases(cfg["fusion_arm"])
    fusion_case_set = set(fusion_cases.case_id)
    print(f"fusion arm {cfg['fusion_arm']}: {len(fusion_case_set)} case-level predictions", flush=True)

    preds = {k: [] for k in PROBES}
    fold_meta = {k: {} for k in PROBES}

    for fold in range(N_FOLDS):
        t0 = time.time()
        splits = fold_cases(fold, slide_to_case)
        train_cases = [c for c in splits["train"] if c in case_index]
        test_cases = [c for c in splits["test"] if c in case_index]
        n_dropped_tr = len(splits["train"]) - len(train_cases)
        n_dropped_te = len(splits["test"]) - len(test_cases)

        tr_idx = [case_index[c] for c in train_cases]
        te_idx = [case_index[c] for c in test_cases]
        y_tr = table_y[tr_idx]
        y_te = np.array([case_to_y[c] for c in test_cases])
        if not np.array_equal(y_te, table_y[te_idx]):
            raise ValueError(f"fold {fold}: manifest and table labels disagree on test cases")

        transform = FoldTransform.fit(features[tr_idx], top_n=cfg["top_n"])
        x_tr = transform.transform(features[tr_idx])
        x_te = transform.transform(features[te_idx])

        for probe_name, fitter in PROBES.items():
            clf, meta = fitter(x_tr, y_tr)
            p = clf.predict_proba(x_te)[:, 1]
            preds[probe_name].append(
                pd.DataFrame({"case_id": test_cases, "fold": fold, "y": y_te, "p": p})
            )
            meta.update(
                n_train=len(train_cases),
                n_test=len(test_cases),
                n_train_dropped_no_table=n_dropped_tr,
                n_test_dropped_no_table=n_dropped_te,
                n_selected_features=int(len(transform.selected_idx)),
            )
            fold_meta[probe_name][fold] = meta

        print(
            f"fold {fold}: train={len(train_cases)} (dropped {n_dropped_tr}) "
            f"test={len(test_cases)} (dropped {n_dropped_te}) "
            f"feat={len(transform.selected_idx)}  [{time.time() - t0:.1f}s]",
            flush=True,
        )

    out = {
        "table_path": os.path.relpath(table_path, REPO_ROOT),
        "top_n_features": cfg["top_n"],
        "n_table_cases": len(case_index),
        "n_raw_feature_columns": len(feature_cols),
        "fusion_arm": cfg["fusion_arm"],
        "fusion_label": cfg["fusion_label"],
        "n_fusion_cases": len(fusion_case_set),
        "probes": {},
    }

    for probe_name, frames in preds.items():
        df = pd.concat(frames, ignore_index=True)
        if df.case_id.duplicated().any():
            raise ValueError(f"{name}/{probe_name}: test folds are not disjoint")
        df.to_csv(os.path.join(OUT_DIR, f"case_predictions_{name}_{probe_name}.csv"), index=False)

        matched = df[df.case_id.isin(fusion_case_set)].reset_index(drop=True)
        entry = {
            "all_table_cases": block(df),
            "matched_to_fusion_cases": block(matched),
            "per_fold_details": fold_meta[probe_name],
        }

        merged = matched.merge(
            fusion_cases[["case_id", "y", "p"]].rename(columns={"p": "p_fusion"}),
            on=["case_id", "y"],
            how="inner",
        )
        auc_tab, auc_fus, z_tab_minus_fus, pval = delong_test(
            merged.y.values, merged.p.values, merged.p_fusion.values
        )
        entry["paired_delong_vs_fusion"] = {
            "n_matched_cases": int(len(merged)),
            "auroc_tabular_only": auc_tab,
            "auroc_fusion": auc_fus,
            "delta_auroc_fusion_minus_tabular": auc_fus - auc_tab,
            "z_fusion_minus_tabular": -z_tab_minus_fus,
            "p_value": pval,
        }

        merged_wsi = matched.merge(
            wsi_cases[["case_id", "y", "p"]].rename(columns={"p": "p_wsi"}),
            on=["case_id", "y"],
            how="inner",
        )
        auc_tab_w, auc_wsi, z_tab_minus_wsi, pval_w = delong_test(
            merged_wsi.y.values, merged_wsi.p.values, merged_wsi.p_wsi.values
        )
        entry["paired_delong_vs_wsi_alone"] = {
            "n_matched_cases": int(len(merged_wsi)),
            "auroc_tabular_only": auc_tab_w,
            "auroc_wsi_alone": auc_wsi,
            "delta_auroc_tabular_minus_wsi": auc_tab_w - auc_wsi,
            "z_tabular_minus_wsi": z_tab_minus_wsi,
            "p_value": pval_w,
        }
        out["probes"][probe_name] = entry

    out["fusion_arm_recomputed"] = block(fusion_cases)
    fusion_cases.to_csv(os.path.join(OUT_DIR, f"case_predictions_fusion_{name}.csv"), index=False)
    return out


def render_markdown(report):
    L = []
    L.append("# Tabular-only baselines for the TCGA-BRCA ER ablation\n")
    L.append(f"Generated: {report['generated']}\n")
    L.append(
        "Purpose: the published three-arm ablation (WSI-alone, WSI+RNA, WSI+clinicopath) "
        "has no tabular-only arm, so it cannot say whether fusion beats simply using the "
        "table. These probes close that gap on the identical 10 tissue-site-holdout folds "
        "(`project/CLAM/splits/tcga_brca_er_100`), the identical case-level aggregation "
        "convention, and the identical fold-local feature transform (variance top-N "
        "selection + standardisation fitted on the training fold only).\n"
    )
    L.append("## Protocol\n")
    L.append(
        "- Fold train/test CASE sets are the split's slide_ids mapped through "
        "`project/CLAM/dataset_csv/tcga_brca_er.csv` and deduplicated.\n"
        "- **Validation cases were left out of probe training** (not folded into train), "
        "mirroring the fusion arms, which fit their tabular transform on the train split "
        "alone and used val only for early stopping.\n"
        "- The transform is a line-for-line re-implementation of `RNAFeatureTransform` "
        "(`project/CLAM/dataset_modules/rna_dataset.py`): `np.nanvar` ranking, top-N, "
        "nan-safe mean/std, std floor. Fitted on train cases only, applied to test.\n"
        f"- Probe A: L2 logistic regression, `class_weight='balanced'`, C chosen from "
        f"{C_GRID} by 5-fold stratified inner CV on the training cases only (scoring AUROC).\n"
        "- Probe B: `MLPClassifier`, one hidden layer of 128 units, alpha 1e-3, "
        "early stopping on an internal 15% slice of the training cases.\n"
        "- The 10 test folds are a disjoint cover, so pooling gives one out-of-fold "
        "prediction per case (asserted in code).\n"
        "- Fusion-arm case probabilities are the per-case MEAN over that case's slide "
        "probabilities, the convention used by `tools/analyze_er_ablation.py`.\n"
    )

    L.append("## Headline (matched-to-fusion case sets)\n")
    L.append("| Modality | Probe | N | AUROC | AUPRC+ | F1+ | BalAcc | Per-fold AUROC |")
    L.append("|---|---|---|---|---|---|---|---|")
    for mod, m in report["modalities"].items():
        for probe, e in m["probes"].items():
            b = e["matched_to_fusion_cases"]
            pm = b["pooled_case_level"]
            pf = b["per_fold_auroc"]
            L.append(
                f"| {mod} | {probe} | {pm['n']} | {pm['auroc']:.4f} | {pm['auprc_pos']:.4f} | "
                f"{pm['f1_pos']:.4f} | {pm['balanced_acc']:.4f} | "
                f"{pf['mean']:.4f} ± {pf['std']:.4f} |"
            )
        fb = m["fusion_arm_recomputed"]
        L.append(
            f"| {mod} | *{m['fusion_label']} (recomputed)* | {fb['pooled_case_level']['n']} | "
            f"{fb['pooled_case_level']['auroc']:.4f} | {fb['pooled_case_level']['auprc_pos']:.4f} | "
            f"{fb['pooled_case_level']['f1_pos']:.4f} | {fb['pooled_case_level']['balanced_acc']:.4f} | "
            f"{fb['per_fold_auroc']['mean']:.4f} ± {fb['per_fold_auroc']['std']:.4f} |"
        )
    L.append("")

    L.append("## Full-table case sets (probes evaluated on every case the table covers)\n")
    L.append("| Modality | Probe | N | AUROC | AUPRC+ | F1+ | BalAcc | Per-fold AUROC |")
    L.append("|---|---|---|---|---|---|---|---|")
    for mod, m in report["modalities"].items():
        for probe, e in m["probes"].items():
            b = e["all_table_cases"]
            pm = b["pooled_case_level"]
            pf = b["per_fold_auroc"]
            L.append(
                f"| {mod} | {probe} | {pm['n']} | {pm['auroc']:.4f} | {pm['auprc_pos']:.4f} | "
                f"{pm['f1_pos']:.4f} | {pm['balanced_acc']:.4f} | "
                f"{pf['mean']:.4f} ± {pf['std']:.4f} |"
            )
    L.append("")

    L.append("## Paired DeLong: fusion vs tabular-only (same cases, same folds)\n")
    L.append("| Modality | Probe | N | Tabular-only AUROC | Fusion AUROC | Δ (fusion − tabular) | z | p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for mod, m in report["modalities"].items():
        for probe, e in m["probes"].items():
            d = e["paired_delong_vs_fusion"]
            L.append(
                f"| {mod} | {probe} | {d['n_matched_cases']} | {d['auroc_tabular_only']:.4f} | "
                f"{d['auroc_fusion']:.4f} | {d['delta_auroc_fusion_minus_tabular']:+.4f} | "
                f"{d['z_fusion_minus_tabular']:.3f} | {d['p_value']:.3g} |"
            )
    L.append("")

    L.append("## Paired DeLong: tabular-only vs WSI-alone (same cases, same folds)\n")
    L.append("| Modality | Probe | N | Tabular-only AUROC | WSI-alone AUROC | Δ (tabular − WSI) | z | p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for mod, m in report["modalities"].items():
        for probe, e in m["probes"].items():
            d = e["paired_delong_vs_wsi_alone"]
            L.append(
                f"| {mod} | {probe} | {d['n_matched_cases']} | {d['auroc_tabular_only']:.4f} | "
                f"{d['auroc_wsi_alone']:.4f} | {d['delta_auroc_tabular_minus_wsi']:+.4f} | "
                f"{d['z_tabular_minus_wsi']:.3f} | {d['p_value']:.3g} |"
            )
    L.append("")

    L.append("## Reference numbers from the existing ablation\n")
    L.append(
        "WSI-alone 0.8957, WSI+RNA gated 0.9412 (956 cases), WSI+clinicopath gated 0.8937 "
        "(1003 cases). The fusion AUROCs recomputed here from the saved pickles are the "
        "'recomputed' rows above and should reproduce those values.\n"
    )
    L.append("## Artefacts\n")
    L.append(
        "- `tabular_only_probes.json` — all numbers, machine-readable.\n"
        "- `case_predictions_<modality>_<probe>.csv` — pooled out-of-fold case predictions.\n"
        "- `case_predictions_fusion_<modality>.csv` — the fusion arm aggregated to case level.\n"
        "- Script: `tools/diagnostics/tabular_only_probes.py`.\n"
    )
    return "\n".join(L)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _, slide_to_case, case_to_y = load_manifest()

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "split_dir": os.path.relpath(SPLIT_DIR, REPO_ROOT),
            "dataset_csv": os.path.relpath(DATASET_CSV, REPO_ROOT),
            "n_folds": N_FOLDS,
            "label_dict": LABEL_DICT,
            "validation_cases": "excluded from probe training",
            "logreg_C_grid": C_GRID,
            "inner_cv": "StratifiedKFold(5, shuffle=True, random_state=1) on training cases only",
            "mlp": "MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, early_stopping=True)",
            "seed": SEED,
        },
        "modalities": {},
    }

    wsi_cases = load_fusion_cases("er_wsi_alone_s1")
    report["wsi_alone_recomputed"] = block(wsi_cases)
    print(
        "WSI-alone arm: n={} pooled AUROC={:.4f}".format(
            report["wsi_alone_recomputed"]["pooled_case_level"]["n"],
            report["wsi_alone_recomputed"]["pooled_case_level"]["auroc"],
        )
    )

    for name, cfg in MODALITIES.items():
        report["modalities"][name] = run_modality(name, cfg, slide_to_case, case_to_y, wsi_cases)

    with open(os.path.join(OUT_DIR, "tabular_only_probes.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    with open(os.path.join(OUT_DIR, "tabular_only_probes.md"), "w") as handle:
        handle.write(render_markdown(report))

    print("\n=== summary (matched-to-fusion cases) ===")
    for mod, m in report["modalities"].items():
        for probe, e in m["probes"].items():
            pm = e["matched_to_fusion_cases"]["pooled_case_level"]
            d = e["paired_delong_vs_fusion"]
            print(
                f"{mod:12s} {probe:10s} n={pm['n']:4d} AUROC={pm['auroc']:.4f}  "
                f"fusion={d['auroc_fusion']:.4f}  Δ={d['delta_auroc_fusion_minus_tabular']:+.4f}  "
                f"p={d['p_value']:.3g}"
            )
    print(f"\nwrote {OUT_DIR}/tabular_only_probes.json and .md")


if __name__ == "__main__":
    main()
