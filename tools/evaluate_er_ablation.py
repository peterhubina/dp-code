#!/usr/bin/env python
"""Offline ablation report for the TCGA-BRCA ER-status three-arm experiment.

Reads the per-fold CLAM test predictions written by ``tools/train_er_ablation.sh``
(``.scratch/results/er/<exp_code>_s1/split_{i}_results.pkl``) and, with no
retraining, produces the full "Part B" ablation report:

  1. Headline metrics per arm  -- AUROC / AUPRC / F1(ER+) / balanced-acc,
     both per-fold mean+/-std and pooled out-of-fold (single value + bootstrap 95% CI).
  2. Matched paired comparison -- each fusion arm vs the WSI-alone baseline on the
     *same* slides, tested with DeLong's paired-AUROC test (+ paired bootstrap delta).
  3. Per-site generalisation   -- pooled-OOF AUROC per held-out tissue-submitting site.
  4. Calibration               -- reliability curve, ECE, Brier per arm.

Outputs (CSV + markdown + figures) land under ``.scratch/results/er/report/``.

The prediction unit is the *slide* (CLAM predicts per slide); pooled OOF is a
disjoint cover of all test slides across the 10 site-holdout folds, so each slide
contributes exactly one prediction. ER-positive is the positive class (label 1).

Usage (from repo root, venv active):
    python tools/evaluate_er_ablation.py
    python tools/evaluate_er_ablation.py --results_dir .scratch/results/er --k 10
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---- arms ------------------------------------------------------------------ #
ARMS = {  # values are base exp codes; "_s<seed>" is appended at load time
    "wsi_alone": "er_wsi_alone",
    # chapter 1 -- the published gated arms
    "wsi_rna": "er_wsi_rna_gated",
    "wsi_clinpath": "er_wsi_clinpath_gated",
    # chapter 2 -- the novel mechanism, its ablation, and its controls
    "rna_film": "er_wsi_rna_film",
    "clinpath_film": "er_wsi_clinpath_film",
    "rna_delta": "er_wsi_rna_delta",
    "clinpath_delta": "er_wsi_clinpath_delta",
    "rna_gatedcap": "er_wsi_rna_gatedcap",
    "clinpath_gatedcap": "er_wsi_clinpath_gatedcap",
    "rna_coattn": "er_wsi_rna_coattn",
    "clinpath_coattn": "er_wsi_clinpath_coattn",
}
ARM_LABELS = {
    "wsi_alone": "WSI-alone",
    "wsi_rna": "WSI + RNA (gated, dim 32)",
    "wsi_clinpath": "WSI + clinicopath (gated, dim 32)",
    "rna_film": "WSI + RNA (FiLM-attention)",
    "clinpath_film": "WSI + clinicopath (FiLM-attention)",
    "rna_delta": "WSI + RNA (additive-logit, film_rank 0)",
    "clinpath_delta": "WSI + clinicopath (additive-logit, film_rank 0)",
    "rna_gatedcap": "WSI + RNA (gated, dim 96 capacity control)",
    "clinpath_gatedcap": "WSI + clinicopath (gated, dim 96 capacity control)",
    "rna_coattn": "WSI + RNA (adapted co-attention)",
    "clinpath_coattn": "WSI + clinicopath (adapted co-attention)",
}
FUSION_ARMS = [a for a in ARMS if a != "wsi_alone"]
BASELINE = "wsi_alone"
POS_LABEL = 1  # ER-positive

# Tabular-only probes (case level only; produced by tools/diagnostics/tabular_only_probes.py).
# These are the controls the original ablation lacked: they answer "does fusion beat simply
# using the table?", which for RNA is the decisive question.
PROBES = {
    "rna_probe": ("case_predictions_rna_logreg_l2.csv", "RNA table only (logistic regression)"),
    "clinpath_probe": ("case_predictions_clinicopath_logreg_l2.csv",
                       "Clinicopath table only (logistic regression)"),
}

# ---- pre-registered comparisons -------------------------------------------- #
# Fixed in docs/implementation-research/novel-fusion-design.md 6.4 BEFORE any test number
# was read. PRIMARY carries a Holm correction across its two members; everything else is
# secondary and reported without alpha adjustment, labelled as such.
PRIMARY_COMPARISONS = [
    ("rna_film", "wsi_rna"),
    ("clinpath_film", "wsi_clinpath"),
]
SECONDARY_COMPARISONS = [
    ("rna_film", "wsi_alone"),
    ("rna_film", "rna_gatedcap"),
    ("rna_film", "rna_delta"),
    ("rna_film", "rna_coattn"),
    ("rna_film", "rna_probe"),
    ("rna_delta", "wsi_rna"),
    ("rna_coattn", "wsi_rna"),
    ("clinpath_film", "wsi_alone"),
    ("clinpath_film", "clinpath_gatedcap"),
    ("clinpath_film", "clinpath_delta"),
    ("clinpath_film", "clinpath_coattn"),
    ("clinpath_film", "clinpath_probe"),
    ("clinpath_delta", "wsi_clinpath"),
    ("clinpath_coattn", "wsi_clinpath"),
]

N_BOOT = 2000
BOOT_SEED = 1
_DO_BOOTSTRAP = True  # disabled for non-primary seeds; see main()


# ---- loading --------------------------------------------------------------- #
def site_of(case_id: str) -> str:
    """Tissue-submitting site = TCGA barcode field 2 (e.g. TCGA-BH-A0AU -> BH)."""
    return case_id.split("-")[1]


def load_arm(results_dir: Path, exp_dir: str, k: int, slide2case: dict) -> pd.DataFrame:
    """Pool the k per-fold test prediction pkls into one slide-level frame."""
    rows = []
    arm_path = results_dir / exp_dir
    for fold in range(k):
        pkl = arm_path / f"split_{fold}_results.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"missing predictions: {pkl}")
        with open(pkl, "rb") as fh:
            preds = pickle.load(fh)
        for slide_id, rec in preds.items():
            prob = np.asarray(rec["prob"]).ravel()  # [P(neg), P(pos)]
            case_id = slide2case.get(slide_id)
            rows.append(
                {
                    "slide_id": slide_id,
                    "case_id": case_id,
                    "site": site_of(case_id) if case_id else "UNK",
                    "fold": fold,
                    "label": int(rec["label"]),
                    "p_pos": float(prob[POS_LABEL]),
                }
            )
    df = pd.DataFrame(rows)
    dup = df["slide_id"].duplicated().sum()
    if dup:
        raise ValueError(f"{exp_dir}: {dup} slide(s) appear in >1 test fold -- OOF not disjoint")
    return df


# ---- metrics --------------------------------------------------------------- #
def point_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    yhat = (p >= 0.5).astype(int)
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "auroc": np.nan,
        "auprc": np.nan,
        "f1_pos": np.nan,
        "bal_acc": np.nan,
    }
    if len(np.unique(y)) == 2:  # AUROC/AUPRC need both classes present
        out["auroc"] = roc_auc_score(y, p)
        out["auprc"] = average_precision_score(y, p)
        out["bal_acc"] = balanced_accuracy_score(y, yhat)
    out["f1_pos"] = f1_score(y, yhat, pos_label=POS_LABEL, zero_division=0)
    return out


def bootstrap_ci(y: np.ndarray, p: np.ndarray, fn, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Stratified (by label) bootstrap 95% CI for a metric fn(y, p)."""
    if not _DO_BOOTSTRAP:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        bi = np.concatenate(
            [rng.choice(pos_idx, len(pos_idx), replace=True),
             rng.choice(neg_idx, len(neg_idx), replace=True)]
        )
        yb, pb = y[bi], p[bi]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(fn(yb, pb))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ---- DeLong paired AUROC test --------------------------------------------- #
# Fast DeLong (Sun & Xu, 2014); midrank implementation after Xu Sun's reference.
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray):
    """Two-sided DeLong test for AUC(a) - AUC(b) on paired predictions.

    Returns (auc_a, auc_b, z, p_value). Order is by ascending label, so we sort.
    """
    order = np.argsort(-y_true)  # positives (label 1) first
    label_1_count = int((y_true == 1).sum())
    preds = np.vstack((p_a[order], p_b[order]))
    aucs, cov = _fast_delong(preds, label_1_count)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0 if aucs[0] == aucs[1] else np.inf * np.sign(aucs[0] - aucs[1])
        pval = 1.0 if aucs[0] == aucs[1] else 0.0
        return float(aucs[0]), float(aucs[1]), float(z), float(pval)
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    pval = 2.0 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(pval)


def paired_bootstrap_delta(y, p_fusion, p_base, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap 95% CI for AUROC(fusion) - AUROC(base) on paired slides."""
    if not _DO_BOOTSTRAP:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    deltas = []
    for _ in range(n_boot):
        bi = np.concatenate(
            [rng.choice(pos_idx, len(pos_idx), replace=True),
             rng.choice(neg_idx, len(neg_idx), replace=True)]
        )
        yb = y[bi]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(roc_auc_score(yb, p_fusion[bi]) - roc_auc_score(yb, p_base[bi]))
    if not deltas:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(deltas)),
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)))


# ---- calibration ----------------------------------------------------------- #
def calibration(y, p, n_bins=10):
    """Return (bin_centers, empirical_freq, mean_pred, counts, ECE, Brier)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    emp, mean_pred, centers, counts = [], [], [], []
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        centers.append((bins[b] + bins[b + 1]) / 2)
        counts.append(c)
        if c:
            e = float(y[mask].mean())
            mp = float(p[mask].mean())
            emp.append(e)
            mean_pred.append(mp)
            ece += (c / len(y)) * abs(e - mp)
        else:
            emp.append(np.nan)
            mean_pred.append(np.nan)
    return (np.array(centers), np.array(emp), np.array(mean_pred),
            np.array(counts), float(ece), float(brier_score_loss(y, p)))


# ---- figures --------------------------------------------------------------- #
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

_BASE_COLORS = {"wsi_alone": "#4C6EF5", "wsi_rna": "#E8590C", "wsi_clinpath": "#2F9E44"}


class _ArmColors(dict):
    """Stable colours for the three original arms; distinct tab20 colours for the rest."""

    _EXTRA = plt.get_cmap("tab20").colors

    def __missing__(self, arm):
        order = [a for a in ARMS if a not in _BASE_COLORS] + sorted(PROBES)
        idx = order.index(arm) if arm in order else len(self)
        colour = self._EXTRA[idx % len(self._EXTRA)]
        self[arm] = colour
        return colour


COLORS = _ArmColors(_BASE_COLORS)


def fig_roc(data, out, unit="slide"):
    plt.figure(figsize=(5.2, 5))
    for arm, df in data.items():
        fpr, tpr, _ = roc_curve(df["label"], df["p_pos"])
        auc = roc_auc_score(df["label"], df["p_pos"])
        plt.plot(fpr, tpr, color=COLORS[arm], lw=1.8,
                 label=f"{ARM_LABELS[arm]} (AUROC {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"Pooled out-of-fold ROC -- ER status ({unit}-level)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_pr(data, out, unit="slide"):
    plt.figure(figsize=(5.2, 5))
    for arm, df in data.items():
        prec, rec, _ = precision_recall_curve(df["label"], df["p_pos"])
        ap = average_precision_score(df["label"], df["p_pos"])
        plt.plot(rec, prec, color=COLORS[arm], lw=1.8,
                 label=f"{ARM_LABELS[arm]} (AUPRC {ap:.3f})")
    base = data["wsi_alone"]["label"].mean()
    plt.axhline(base, ls="--", color="k", lw=0.8, alpha=0.5, label=f"prevalence {base:.2f}")
    plt.xlabel("Recall (ER+)"); plt.ylabel("Precision (ER+)")
    plt.title(f"Pooled out-of-fold PR -- ER status ({unit}-level)")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_calibration(data, out, unit="slide"):
    plt.figure(figsize=(5.2, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="perfect")
    for arm, df in data.items():
        _, emp, mp, _, ece, _ = calibration(df["label"].values, df["p_pos"].values)
        ok = ~np.isnan(emp)
        plt.plot(mp[ok], emp[ok], "o-", color=COLORS[arm], lw=1.5, ms=4,
                 label=f"{ARM_LABELS[arm]} (ECE {ece:.3f})")
    plt.xlabel("Mean predicted P(ER+)"); plt.ylabel("Observed frequency")
    plt.title(f"Reliability curve -- ER status ({unit}-level)")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_per_fold(per_fold, out, unit="slide"):
    plt.figure(figsize=(6, 4.2))
    arms = list(ARMS.keys())
    for i, arm in enumerate(arms):
        vals = per_fold[per_fold["arm"] == arm]["auroc"].values
        x = np.full(len(vals), i) + np.random.default_rng(0).normal(0, 0.04, len(vals))
        plt.scatter(x, vals, color=COLORS[arm], alpha=0.7, s=28)
        plt.hlines(np.mean(vals), i - 0.2, i + 0.2, color=COLORS[arm], lw=2)
    plt.xticks(range(len(arms)), [ARM_LABELS[a] for a in arms], rotation=12, fontsize=8)
    plt.ylabel("Per-fold test AUROC")
    plt.title(f"Per-fold AUROC across 10 site-holdout folds ({unit}-level)")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_per_site(site_df, out, unit="slide"):
    d = site_df.sort_values("n", ascending=False).head(15)
    x = np.arange(len(d)); w = 0.26
    plt.figure(figsize=(9, 4.4))
    for j, arm in enumerate(ARMS.keys()):
        plt.bar(x + (j - 1) * w, d[f"auroc_{arm}"], w, color=COLORS[arm], label=ARM_LABELS[arm])
    plt.xticks(x, [f"{s}\n(n={n})" for s, n in zip(d["site"], d["n"])], fontsize=7)
    plt.axhline(0.5, ls=":", color="k", lw=0.7)
    plt.ylabel("Pooled-OOF AUROC"); plt.ylim(0.3, 1.02)
    plt.title(f"Per-site AUROC (top 15 sites by {unit} count)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


# ---- unit aggregation ------------------------------------------------------ #
def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def brier_decomposition(y, p, n_bins=10):
    """Murphy decomposition: Brier = reliability - resolution + uncertainty.

    reliability (lower better) is the calibration term; resolution (higher better) is the
    discrimination term. Reporting both separates "this arm ranks cases better" from
    "this arm merely reports better-calibrated numbers", which a single ECE cannot.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    base = y.mean()
    reliability = resolution = 0.0
    for b in range(n_bins):
        mask = idx == b
        n_b = int(mask.sum())
        if not n_b:
            continue
        w = n_b / len(y)
        reliability += w * (p[mask].mean() - y[mask].mean()) ** 2
        resolution += w * (y[mask].mean() - base) ** 2
    uncertainty = base * (1.0 - base)
    return reliability, resolution, uncertainty


def load_probe(diag_dir: Path, filename: str) -> pd.DataFrame:
    """Load a case-level tabular-only probe into the same frame shape as an arm."""
    path = diag_dir / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return pd.DataFrame({
        "key": df["case_id"].astype(str),
        "case_id": df["case_id"].astype(str),
        "slide_id": df["case_id"].astype(str),
        "site": [c.split("-")[1] for c in df["case_id"].astype(str)],
        "fold": df["fold"].astype(int),
        "label": df["y"].astype(int),
        "p_pos": df["p"].astype(float),
    })


def compare_pair(data_u: dict, arm_a: str, arm_b: str) -> dict:
    """Paired DeLong + bootstrap delta of arm_a against arm_b on their shared keys."""
    fa = data_u[arm_a].set_index("key")
    fb = data_u[arm_b].set_index("key")
    common = fa.index.intersection(fb.index)
    y = fa.loc[common, "label"].values
    pa = fa.loc[common, "p_pos"].values
    pb = fb.loc[common, "p_pos"].values
    if not np.array_equal(y, fb.loc[common, "label"].values):
        raise ValueError(f"label mismatch between {arm_a} and {arm_b} on shared keys")

    auc_a, auc_b, z, pval = delong_roc_test(y, pa, pb)
    # Guard against the argument-order bug that bit this analysis once before: DeLong's
    # own AUROCs must agree with independently computed ones, or the labels are swapped.
    for name, got, want in ((arm_a, auc_a, roc_auc_score(y, pa)),
                            (arm_b, auc_b, roc_auc_score(y, pb))):
        if abs(got - want) > 1e-6:
            raise AssertionError(f"DeLong AUROC for {name} ({got:.6f}) != roc_auc_score ({want:.6f})")

    dmean, dlo, dhi = paired_bootstrap_delta(y, pa, pb)
    return {
        "arm": arm_a, "reference": arm_b,
        "label": ARM_LABELS.get(arm_a, arm_a), "reference_label": ARM_LABELS.get(arm_b, arm_b),
        "n_matched": int(len(common)),
        "auroc_arm": auc_a, "auroc_reference": auc_b, "delta_auroc": auc_a - auc_b,
        "delong_z": z, "delong_p": pval,
        "boot_delta_mean": dmean, "boot_delta_ci_lo": dlo, "boot_delta_ci_hi": dhi,
    }


def run_comparisons(data_u: dict) -> pd.DataFrame:
    """The pre-registered comparison table, with Holm applied to the primary endpoint."""
    rows = []
    for a, b in PRIMARY_COMPARISONS:
        if a in data_u and b in data_u:
            rows.append({**compare_pair(data_u, a, b), "tier": "primary"})
    primary_idx = list(range(len(rows)))
    for a, b in SECONDARY_COMPARISONS:
        if a in data_u and b in data_u:
            rows.append({**compare_pair(data_u, a, b), "tier": "secondary"})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["delong_p_holm"] = np.nan
    if primary_idx:
        df.loc[primary_idx, "delong_p_holm"] = holm(df.loc[primary_idx, "delong_p"].values)
    return df


def aggregate_unit(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Return a prediction frame at the requested unit.

    slide: unchanged (one row per slide, key = slide_id).
    case:  mean-pool p_pos across a case's slides (one row per case, key = case_id).
           Label/site/fold are constant within a case (site-holdout keeps a case in
           one fold); a case with an inconsistent ER label would be a data error.
    """
    if unit == "slide":
        out = df.copy()
        out["key"] = out["slide_id"]
        return out
    if unit != "case":
        raise ValueError(f"unknown unit: {unit}")
    g = df.groupby("case_id")
    if (g["label"].nunique() > 1).any():
        raise ValueError("case with inconsistent ER label across slides")
    out = g.agg(
        p_pos=("p_pos", "mean"),
        label=("label", "first"),
        site=("site", "first"),
        fold=("fold", "first"),
        n_slides=("slide_id", "size"),
    ).reset_index()
    out["key"] = out["case_id"]
    return out


# ---- per-unit analysis ----------------------------------------------------- #
def analyze(data_u: dict, do_bootstrap: bool = True) -> dict:
    """Full metric battery for one unit's per-arm prediction frames."""
    global _DO_BOOTSTRAP
    _DO_BOOTSTRAP = do_bootstrap
    per_fold_rows, pooled_rows = [], []
    for arm, df in data_u.items():
        for fold, g in df.groupby("fold"):
            m = point_metrics(g["label"].values, g["p_pos"].values)
            per_fold_rows.append({"arm": arm, "fold": int(fold), **m})
        y, p = df["label"].values, df["p_pos"].values
        m = point_metrics(y, p)
        auroc_ci = bootstrap_ci(y, p, roc_auc_score)
        auprc_ci = bootstrap_ci(y, p, average_precision_score)
        pf = pd.DataFrame([r for r in per_fold_rows if r["arm"] == arm])
        pooled_rows.append({
            "arm": arm, "label": ARM_LABELS[arm], **m,
            "auroc_ci_lo": auroc_ci[0], "auroc_ci_hi": auroc_ci[1],
            "auprc_ci_lo": auprc_ci[0], "auprc_ci_hi": auprc_ci[1],
            "auroc_foldmean": pf["auroc"].mean(), "auroc_foldstd": pf["auroc"].std(),
            "auprc_foldmean": pf["auprc"].mean(), "auprc_foldstd": pf["auprc"].std(),
            "f1_foldmean": pf["f1_pos"].mean(), "f1_foldstd": pf["f1_pos"].std(),
        })
    per_fold = pd.DataFrame(per_fold_rows)
    pooled = pd.DataFrame(pooled_rows)

    # matched paired comparison (DeLong + bootstrap delta), paired on `key`
    base = data_u[BASELINE].set_index("key")
    paired_rows = []
    for arm in FUSION_ARMS:
        fus = data_u[arm].set_index("key")
        common = fus.index.intersection(base.index)
        y = fus.loc[common, "label"].values
        pf_ = fus.loc[common, "p_pos"].values
        pb = base.loc[common, "p_pos"].values
        auc_f, auc_b, z, pval = delong_roc_test(y, pf_, pb)
        dmean, dlo, dhi = paired_bootstrap_delta(y, pf_, pb)
        paired_rows.append({
            "fusion_arm": arm, "label": ARM_LABELS[arm],
            "n_matched": int(len(common)),
            "auroc_fusion": auc_f, "auroc_wsi_alone_matched": auc_b,
            "delta_auroc": auc_f - auc_b,
            "delong_z": z, "delong_p": pval,
            "boot_delta_mean": dmean, "boot_delta_ci_lo": dlo, "boot_delta_ci_hi": dhi,
        })
    paired = pd.DataFrame(paired_rows)

    # per-site AUROC (pooled OOF)
    site_rows = {}
    for arm, df in data_u.items():
        for site, g in df.groupby("site"):
            r = site_rows.setdefault(site, {"site": site, "n": len(g), "n_pos": int(g["label"].sum())})
            r["n"] = max(r["n"], len(g))
            auc = roc_auc_score(g["label"], g["p_pos"]) if g["label"].nunique() == 2 else np.nan
            r[f"auroc_{arm}"] = auc
            r[f"n_{arm}"] = len(g)
    site_df = pd.DataFrame(list(site_rows.values())).sort_values("n", ascending=False)

    # calibration
    cal_rows = []
    for arm, df in data_u.items():
        y, p = df["label"].values, df["p_pos"].values
        _, _, _, _, ece, brier = calibration(y, p)
        rel, res_, unc = brier_decomposition(y, p)
        cal_rows.append({
            "arm": arm, "label": ARM_LABELS.get(arm, arm), "ece": ece, "brier": brier,
            "reliability": rel, "resolution": res_, "uncertainty": unc,
        })
    cal = pd.DataFrame(cal_rows)

    comparisons = run_comparisons(data_u)

    return {"per_fold": per_fold, "pooled": pooled, "paired": paired, "site": site_df,
            "cal": cal, "comparisons": comparisons}


# ---- main ------------------------------------------------------------------ #
# Old single-unit output filenames (pre case-level); removed so the report only
# references the current unit-suffixed artifacts.
_STALE = [
    "metrics_per_fold.csv", "metrics_pooled.csv", "paired_tests.csv",
    "per_site_auroc.csv", "calibration.csv", "fig_roc.png", "fig_pr.png",
    "fig_calibration.png", "fig_per_fold_auroc.png", "fig_per_site_auroc.png",
]
UNIT_NOUN = {"slide": "slides", "case": "cases"}


def _record_seed_rows(seed, unit, res, arm_rows, cmp_rows):
    """Collect one seed's point estimates for the across-seed summary."""
    for _, r in res["pooled"].iterrows():
        arm_rows.append({"seed": seed, "unit": unit, "arm": r["arm"], "label": r["label"],
                         "n": r["n"], "auroc": r["auroc"], "auprc": r["auprc"],
                         "f1_pos": r["f1_pos"]})
    if not res["comparisons"].empty:
        for _, r in res["comparisons"].iterrows():
            cmp_rows.append({"seed": seed, "unit": unit, "tier": r["tier"],
                             "arm": r["arm"], "reference": r["reference"],
                             "label": r["label"], "reference_label": r["reference_label"],
                             "n_matched": r["n_matched"], "auroc_arm": r["auroc_arm"],
                             "auroc_reference": r["auroc_reference"],
                             "delta_auroc": r["delta_auroc"], "delong_p": r["delong_p"]})


def _multiseed_summary(arm_rows, cmp_rows, out_dir, seeds):
    """Across-seed summary. The SEED is the unit of replication: deltas are summarised
    across seeds, never by pooling or averaging predictions (which would make an ensemble)."""
    arms = pd.DataFrame(arm_rows)
    cmps = pd.DataFrame(cmp_rows)
    arms.to_csv(out_dir / "multiseed_arms_per_seed.csv", index=False)
    if not cmps.empty:
        cmps.to_csv(out_dir / "multiseed_comparisons_per_seed.csv", index=False)

    arm_sum = (arms.groupby(["unit", "arm", "label"])["auroc"]
               .agg(n_seeds="count", auroc_mean="mean", auroc_sd="std",
                    auroc_min="min", auroc_max="max").reset_index())
    arm_sum.to_csv(out_dir / "multiseed_arms_summary.csv", index=False)

    cmp_sum = pd.DataFrame()
    if not cmps.empty:
        def _agg(g):
            d = g["delta_auroc"].values
            return pd.Series({
                "n_seeds": len(d),
                "delta_mean": d.mean(),
                "delta_sd": d.std(ddof=1) if len(d) > 1 else np.nan,
                "delta_min": d.min(), "delta_max": d.max(),
                # How often the effect points the same way, and how often it is nominally
                # significant. A conclusion that flips sign across seeds is not a conclusion.
                "seeds_positive": int((d > 0).sum()),
                "seeds_p_lt_05": int((g["delong_p"].values < 0.05).sum()),
                "sign_consistent": bool(np.all(d > 0) or np.all(d < 0)),
            })
        cmp_sum = (cmps.groupby(["unit", "tier", "arm", "reference", "label", "reference_label"])
                   .apply(_agg).reset_index())
        cmp_sum.to_csv(out_dir / "multiseed_comparisons_summary.csv", index=False)

    print(f"\n{'=' * 70}\nACROSS-SEED SUMMARY (seeds: {', '.join(map(str, seeds))})\n{'=' * 70}")
    for unit in sorted(arms["unit"].unique()):
        print(f"\n--- {unit} level: arm AUROC across seeds ---")
        sub = arm_sum[arm_sum["unit"] == unit][["label", "n_seeds", "auroc_mean", "auroc_sd",
                                                "auroc_min", "auroc_max"]]
        print(sub.round(4).to_string(index=False))
        if not cmp_sum.empty:
            for tier in ("primary", "secondary"):
                t = cmp_sum[(cmp_sum["unit"] == unit) & (cmp_sum["tier"] == tier)]
                if t.empty:
                    continue
                print(f"\n--- {unit} level: {tier.upper()} comparisons across seeds ---")
                print(t[["label", "reference_label", "n_seeds", "delta_mean", "delta_sd",
                         "seeds_positive", "seeds_p_lt_05", "sign_consistent"]]
                      .round(4).to_string(index=False))
    return arm_sum, cmp_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=".scratch/results/er")
    ap.add_argument("--dataset_csv", default="project/CLAM/dataset_csv/tcga_brca_er.csv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--unit", choices=["slide", "case", "both"], default="both",
                    help="prediction unit for metrics (default: both)")
    ap.add_argument("--out_dir", default=None, help="default: <results_dir>/report")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1],
                    help="seeds to analyse. The FIRST seed keeps the canonical output paths; "
                         "with more than one, per-seed summaries are also written. The seed is "
                         "the unit of replication -- predictions are never averaged across "
                         "seeds, which would silently turn the arms into an ensemble.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in _STALE:
        (out_dir / stale).unlink(missing_ok=True)

    ds = pd.read_csv(args.dataset_csv)
    slide2case = dict(zip(ds["slide_id"], ds["case_id"]))

    units = ["slide", "case"] if args.unit == "both" else [args.unit]
    bundles = {}
    diag_dir = results_dir / "diagnostics"
    seed_arm_rows, seed_cmp_rows = [], []

    for seed_i, seed in enumerate(args.seeds):
      primary_seed = seed_i == 0
      data_slide = {}
      absent = []
      for arm, exp in ARMS.items():
          run_dir = results_dir / f"{exp}_s{seed}"
          if run_dir.is_dir():
              data_slide[arm] = load_arm(results_dir, f"{exp}_s{seed}", args.k, slide2case)
          else:
              absent.append(arm)
      if absent:
          print(f"[seed {seed}] no run directory for: {', '.join(absent)} -- skipped")
      if not data_slide:
          print(f"[seed {seed}] no arms found; skipping this seed")
          continue

      for unit in units:
        data_u = {arm: aggregate_unit(df, unit) for arm, df in data_slide.items()}
        if unit == "case":
            # Tabular-only probes exist at case level only; they are comparators, not arms,
            # so they join data_u after the per-arm metric tables are built from ARMS.
            for key, (fname, label) in PROBES.items():
                probe = load_probe(diag_dir, fname)
                if probe is not None:
                    data_u[key] = probe
                    ARM_LABELS.setdefault(key, label)
        res = analyze(data_u, do_bootstrap=primary_seed)
        if not primary_seed:
            _record_seed_rows(seed, unit, res, seed_arm_rows, seed_cmp_rows)
            continue
        _record_seed_rows(seed, unit, res, seed_arm_rows, seed_cmp_rows)
        res["per_fold"].to_csv(out_dir / f"metrics_per_fold_{unit}.csv", index=False)
        res["pooled"].to_csv(out_dir / f"metrics_pooled_{unit}.csv", index=False)
        res["paired"].to_csv(out_dir / f"paired_tests_{unit}.csv", index=False)
        res["site"].to_csv(out_dir / f"per_site_auroc_{unit}.csv", index=False)
        res["cal"].to_csv(out_dir / f"calibration_{unit}.csv", index=False)
        if not res["comparisons"].empty:
            res["comparisons"].to_csv(out_dir / f"comparisons_{unit}.csv", index=False)
        fig_roc(data_u, out_dir / f"fig_roc_{unit}.png", unit)
        fig_pr(data_u, out_dir / f"fig_pr_{unit}.png", unit)
        fig_calibration(data_u, out_dir / f"fig_calibration_{unit}.png", unit)
        fig_per_fold(res["per_fold"], out_dir / f"fig_per_fold_auroc_{unit}.png", unit)
        fig_per_site(res["site"], out_dir / f"fig_per_site_auroc_{unit}.png", unit)
        bundles[unit] = res

        print(f"\n=== {unit}-level ===")
        print(res["pooled"][["label", "n", "auroc", "auroc_ci_lo", "auroc_ci_hi",
                             "auprc", "f1_pos", "auroc_foldmean", "auroc_foldstd"]].to_string(index=False))
        print(f"Paired vs WSI-alone (matched {UNIT_NOUN[unit]}):")
        print(res["paired"][["label", "n_matched", "delta_auroc", "delong_p",
                             "boot_delta_ci_lo", "boot_delta_ci_hi"]].to_string(index=False))
        if not res["comparisons"].empty:
            cmp_df = res["comparisons"]
            for tier in ("primary", "secondary"):
                sub = cmp_df[cmp_df["tier"] == tier]
                if sub.empty:
                    continue
                cols = ["label", "reference_label", "n_matched", "auroc_arm", "auroc_reference",
                        "delta_auroc", "delong_p"]
                if tier == "primary":
                    cols.append("delong_p_holm")
                print(f"\nPre-registered {tier.upper()} comparisons ({unit} level):")
                print(sub[cols].to_string(index=False))

    if len(args.seeds) > 1 and seed_arm_rows:
        _multiseed_summary(seed_arm_rows, seed_cmp_rows, out_dir, args.seeds)

    write_report(out_dir, bundles, units)
    print(f"\nReport written to {out_dir}/report.md")


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _headline_table(L, pooled, unit):
    noun = UNIT_NOUN[unit]
    L.append(f"| Arm | N {noun} | AUROC (pooled) | AUROC 95% CI | AUPRC | F1(ER+) | Bal.acc | AUROC (fold mean+/-std) |")
    L.append("|---|--:|--:|:--:|--:|--:|--:|:--:|")
    for _, r in pooled.iterrows():
        L.append(f"| {r['label']} | {int(r['n'])} | **{_fmt(r['auroc'])}** | "
                 f"{_fmt(r['auroc_ci_lo'])}–{_fmt(r['auroc_ci_hi'])} | {_fmt(r['auprc'])} | "
                 f"{_fmt(r['f1_pos'])} | {_fmt(r['bal_acc'])} | "
                 f"{_fmt(r['auroc_foldmean'])} +/- {_fmt(r['auroc_foldstd'])} |")
    L.append("")


def _paired_table(L, paired, unit):
    noun = UNIT_NOUN[unit]
    L.append(f"| Fusion arm | N matched | AUROC fusion | AUROC WSI-alone | delta AUROC | DeLong p | Bootstrap delta (95% CI) |")
    L.append("|---|--:|--:|--:|--:|--:|:--:|")
    for _, r in paired.iterrows():
        sig = " *" if r["delong_p"] < 0.05 else ""
        L.append(f"| {r['label']} | {int(r['n_matched'])} | {_fmt(r['auroc_fusion'])} | "
                 f"{_fmt(r['auroc_wsi_alone_matched'])} | {_fmt(r['delta_auroc'], 3)}{sig} | "
                 f"{_fmt(r['delong_p'], 4)} | {_fmt(r['boot_delta_mean'])} "
                 f"[{_fmt(r['boot_delta_ci_lo'])}, {_fmt(r['boot_delta_ci_hi'])}] |")
    L.append(f"\n(* DeLong p < 0.05; N matched in {noun}.)\n")


def write_report(out_dir, bundles, units):
    L = []
    L.append("# ER-status ablation report (TCGA-BRCA)\n")
    L.append("Three-arm ablation over 10 tissue-submitting-site holdout folds (seed 1). "
             "Pooled out-of-fold (OOF) = each test unit scored once by the fold that held out "
             "its site. Positive class = ER-positive. Metrics are reported at two prediction "
             "units: **slide-level** (CLAM's native unit) and **case-level** "
             "(per-patient, mean-pooling a case's slide probabilities) as a sensitivity check "
             "that the finding is invariant to the unit of analysis.\n")

    L.append("## 1. Headline metrics\n")
    L.append("Pooled-OOF point estimates (bootstrap 95% CI) and per-fold mean +/- std across the 10 folds.\n")
    for unit in units:
        L.append(f"### 1.{units.index(unit) + 1} {unit.capitalize()}-level\n")
        _headline_table(L, bundles[unit]["pooled"], unit)

    L.append("## 2. Matched paired comparison vs WSI-alone\n")
    L.append("Each fusion arm is compared to WSI-alone on **the same units** "
             "(WSI-alone restricted to the fusion arm's matched cohort). "
             "DeLong = paired two-sided AUROC test; bootstrap delta = paired AUROC gain with 95% CI.\n")
    for unit in units:
        L.append(f"### 2.{units.index(unit) + 1} {unit.capitalize()}-level\n")
        _paired_table(L, bundles[unit]["paired"], unit)

    L.append("## 3. Calibration\n")
    L.append("| Unit | Arm | ECE (10-bin) | Brier |")
    L.append("|---|---|--:|--:|")
    for unit in units:
        for _, r in bundles[unit]["cal"].iterrows():
            L.append(f"| {unit} | {r['label']} | {_fmt(r['ece'])} | {_fmt(r['brier'])} |")
    L.append("")

    # Per-site: slide-level is the primary breakdown (case-level CSV is also written).
    site_unit = "slide" if "slide" in units else units[0]
    site_df = bundles[site_unit]["site"]
    L.append(f"## 4. Per-site generalisation ({site_unit}-level, top 15 sites)\n")
    L.append("Pooled-OOF AUROC within each held-out tissue-submitting site "
             "(sites with only one ER class present show n/a). "
             f"Case-level per-site values are in `per_site_auroc_case.csv`.\n")
    L.append(f"| Site | N {UNIT_NOUN[site_unit]} | N pos | WSI-alone | WSI+RNA | WSI+clinpath |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for _, r in site_df.head(15).iterrows():
        L.append(f"| {r['site']} | {int(r['n'])} | {int(r['n_pos'])} | "
                 f"{_fmt(r.get('auroc_wsi_alone'))} | {_fmt(r.get('auroc_wsi_rna'))} | "
                 f"{_fmt(r.get('auroc_wsi_clinpath'))} |")
    L.append("")

    L.append("## Figures\n")
    for unit in units:
        for name, cap in [
            (f"fig_roc_{unit}.png", "Pooled-OOF ROC curves"),
            (f"fig_pr_{unit}.png", "Pooled-OOF precision-recall curves"),
            (f"fig_calibration_{unit}.png", "Reliability curves"),
            (f"fig_per_fold_auroc_{unit}.png", "Per-fold AUROC (10 folds)"),
            (f"fig_per_site_auroc_{unit}.png", "Per-site AUROC (top 15 sites)"),
        ]:
            L.append(f"- `{name}` -- {cap} ({unit}-level)")
    L.append("")
    (out_dir / "report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
