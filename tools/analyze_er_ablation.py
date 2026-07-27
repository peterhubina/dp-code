#!/usr/bin/env python
"""Part B analysis for the TCGA-BRCA ER-status three-way ablation.

Reads the per-fold CLAM test predictions (split_{i}_results.pkl) for the three
arms, aggregates to case level, and computes the ablation metrics, the DeLong
paired tests of each fusion arm vs WSI-alone (on the matched case set), the
per-tissue-submitting-site generalisation table (Howard 2021), and calibration.

Nothing here re-trains: it consumes only the saved results, the dataset_csv, and
the split files. Run from the repo root:

    python tools/analyze_er_ablation.py
"""

import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO_ROOT, ".scratch", "results", "er")
DATASET_CSV = os.path.join(REPO_ROOT, "project", "CLAM", "dataset_csv", "tcga_brca_er.csv")
OUT_DIR = os.path.join(RESULTS_ROOT, "analysis")

ARMS = {
    "WSI-alone": "er_wsi_alone_s1",
    "WSI+RNA": "er_wsi_rna_gated_s1",
    "WSI+clinicopath": "er_wsi_clinpath_gated_s1",
}


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
    # preds_sorted: (k_predictors, n_examples), positives (label==1) in the first n_pos columns.
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
    order = np.argsort(-y_true)  # label==1 first
    n_pos = int(y_true.sum())
    preds = np.vstack((np.asarray(p1), np.asarray(p2)))[:, order]
    aucs, cov = _fast_delong(preds, n_pos)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


# --------------------------------------------------------------------------- #
# Loading + aggregation
# --------------------------------------------------------------------------- #
def load_predictions(slide_to_case):
    """Return a case-level frame per arm: case_id, site, fold, y, p (mean over slides)."""
    per_arm = {}
    for name, arm_dir in ARMS.items():
        rows = []
        for path in sorted(glob.glob(os.path.join(RESULTS_ROOT, arm_dir, "split_*_results.pkl"))):
            fold = int(os.path.basename(path).split("split_")[1].split("_results")[0])
            with open(path, "rb") as handle:
                fold_results = pickle.load(handle)
            for record in fold_results.values():
                slide_id = str(np.asarray(record["slide_id"]))
                prob = np.asarray(record["prob"]).ravel()
                rows.append((slide_id, slide_to_case[slide_id], fold, int(record["label"]), float(prob[1])))
        slides = pd.DataFrame(rows, columns=["slide_id", "case_id", "fold", "y", "p"])
        cases = slides.groupby("case_id").agg(
            site=("case_id", lambda s: s.iloc[0].split("-")[1]),
            fold=("fold", "first"),
            y=("y", "first"),
            p=("p", "mean"),
        ).reset_index()
        per_arm[name] = {"slides": slides, "cases": cases}
    return per_arm


def point_metrics(y, p):
    yhat = (np.asarray(p) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")   # ER-positive recall
    spec = tn / (tn + fp) if (tn + fp) else float("nan")   # ER-negative recall
    return {
        "n": int(len(y)),
        "n_pos": int(np.sum(y)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc_pos": float(average_precision_score(y, p)),
        "f1_pos": float(f1_score(y, yhat, pos_label=1)),
        "balanced_acc": float((sens + spec) / 2),
        "sensitivity_ERpos": float(sens),
        "specificity_ERneg": float(spec),
        "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
    }


def per_fold_summary(cases):
    aurocs, auprcs, f1s = [], [], []
    for _, g in cases.groupby("fold"):
        aurocs.append(roc_auc_score(g.y, g.p))
        auprcs.append(average_precision_score(g.y, g.p))
        f1s.append(f1_score(g.y, (g.p >= 0.5).astype(int), pos_label=1))
    mean_std = lambda v: {"mean": float(np.mean(v)), "std": float(np.std(v))}
    return {"auroc": mean_std(aurocs), "auprc_pos": mean_std(auprcs), "f1_pos": mean_std(f1s)}


def expected_calibration_error(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def per_site_table(per_arm, min_n=10):
    sites = sorted(per_arm["WSI-alone"]["cases"]["site"].unique())
    rows = []
    for site in sites:
        base = per_arm["WSI-alone"]["cases"]
        sub = base[base.site == site]
        row = {"site": site, "n_cases": int(len(sub)),
               "n_ERneg": int((sub.y == 0).sum()), "n_ERpos": int((sub.y == 1).sum())}
        both = row["n_ERneg"] > 0 and row["n_ERpos"] > 0 and row["n_cases"] >= min_n
        for name in ARMS:
            c = per_arm[name]["cases"]
            g = c[c.site == site]
            row[f"auroc_{name}"] = float(roc_auc_score(g.y, g.p)) if both and len(g) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = pd.read_csv(DATASET_CSV, dtype=str)
    slide_to_case = dict(zip(ds.slide_id, ds.case_id))

    per_arm = load_predictions(slide_to_case)

    report = {"arms": {}, "delong": {}, "calibration": {}}
    for name in ARMS:
        cases = per_arm[name]["cases"]
        report["arms"][name] = {
            "pooled_case_level": point_metrics(cases.y.values, cases.p.values),
            "per_fold_mean_std": per_fold_summary(cases),
        }
        report["calibration"][name] = {
            "ece_10bin": expected_calibration_error(cases.y.values, cases.p.values)
        }
        cases.to_csv(os.path.join(OUT_DIR, f"case_predictions_{ARMS[name]}.csv"), index=False)

    # DeLong: each fusion arm vs WSI-alone on the matched case set.
    wsi = per_arm["WSI-alone"]["cases"][["case_id", "y", "p"]].rename(columns={"p": "p_wsi"})
    for name in ["WSI+RNA", "WSI+clinicopath"]:
        fus = per_arm[name]["cases"][["case_id", "y", "p"]].rename(columns={"p": "p_fus"})
        merged = wsi.merge(fus, on=["case_id", "y"], how="inner")
        auc_wsi, auc_fus, z_wsi_minus_fus, p = delong_test(
            merged.y.values, merged.p_wsi.values, merged.p_fus.values
        )
        report["delong"][f"{name}_vs_WSI-alone"] = {
            "n_matched_cases": int(len(merged)),
            "auroc_WSI_alone_matched": auc_wsi,
            "auroc_fusion_matched": auc_fus,
            "delta_auroc": auc_fus - auc_wsi,   # positive = fusion better
            "z_fusion_minus_wsi": -z_wsi_minus_fus,
            "p_value": p,
        }

    site_df = per_site_table(per_arm)
    site_df.to_csv(os.path.join(OUT_DIR, "per_site_auroc.csv"), index=False)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    # --- console summary ---
    print("=== Pooled case-level metrics (out-of-fold) ===")
    for name in ARMS:
        m = report["arms"][name]["pooled_case_level"]
        fold = report["arms"][name]["per_fold_mean_std"]
        print(f"{name:16s} n={m['n']:4d}  AUROC={m['auroc']:.4f}  AUPRC={m['auprc_pos']:.4f}  "
              f"F1={m['f1_pos']:.4f}  balAcc={m['balanced_acc']:.4f}  "
              f"| per-fold AUROC {fold['auroc']['mean']:.4f}±{fold['auroc']['std']:.4f}  "
              f"ECE={report['calibration'][name]['ece_10bin']:.4f}")
    print("\n=== DeLong paired test (fusion vs WSI-alone, matched cases) ===")
    for key, d in report["delong"].items():
        print(f"{key:28s} n={d['n_matched_cases']:4d}  WSI={d['auroc_WSI_alone_matched']:.4f}  "
              f"fusion={d['auroc_fusion_matched']:.4f}  Δ={d['delta_auroc']:+.4f}  "
              f"z={d['z_fusion_minus_wsi']:.3f}  p={d['p_value']:.4g}")
    print(f"\n=== Per-site AUROC (sites with both classes, n>=10) -> {os.path.join(OUT_DIR,'per_site_auroc.csv')} ===")
    scored = site_df.dropna(subset=["auroc_WSI-alone"]).sort_values("n_cases", ascending=False)
    print(f"{len(scored)} sites scored; {len(site_df)-len(scored)} too small/single-class.")
    for _, r in scored.iterrows():
        print(f"  {r.site:3s} n={int(r.n_cases):3d} (-{int(r.n_ERneg):2d}/+{int(r.n_ERpos):3d})  "
              f"WSI={r['auroc_WSI-alone']:.3f}  RNA={r['auroc_WSI+RNA']:.3f}  CP={r['auroc_WSI+clinicopath']:.3f}")


if __name__ == "__main__":
    main()
