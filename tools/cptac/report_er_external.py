"""
CPTAC-BRCA ER external validation: report tables and figures
============================================================
Completes the external leg with the artefacts that already exist for the internal
run under .scratch/results/er/report/.

Two robustness analyses beyond plain plotting:

  per-site        CPTAC case ids carry a two-digit site prefix (01BR, 05BR, ...).
                  Per-site AUROC is reported where a site has both classes, but the
                  cohort is lopsided -- site 11 alone is 39% of it -- so most sites
                  are too small to read anything into.

  leave-one-site  The honest version of the same question: drop each site in turn
                  and recompute on the remainder. If the headline AUROC survives
                  removing the largest site, it is not a single-site artefact.
                  This is more informative than per-site AUROC at these n.

Figures deliberately include the internal curves on the same axes -- the finding is
a comparison between cohorts, not an absolute number.

    python tools/cptac/report_er_external.py
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (average_precision_score, precision_recall_curve,  # noqa: E402
                             roc_auc_score, roc_curve)

ARMS = {
    "WSI-alone": ".scratch/cptac_validation/results/er/ensemble_case_predictions.csv",
    "WSI+clinicopath": ".scratch/cptac_validation/results/er_clinpath/ensemble_case_predictions.csv",
    "WSI+clinicopath (table absent)":
        ".scratch/cptac_validation/results/er_clinpath_table_absent/ensemble_case_predictions.csv",
}
MIN_SITE_N = 5


def parse_args():
    parser = argparse.ArgumentParser(description="ER external report")
    parser.add_argument("--internal_csv", type=str,
                        default=".scratch/results/er/analysis/case_predictions_er_wsi_alone_s1.csv")
    parser.add_argument("--output_dir", type=str,
                        default=".scratch/cptac_validation/results/er/report")
    parser.add_argument("--n_boot", type=int, default=10000)
    return parser.parse_args()


def load_arms():
    frames = {}
    for name, path in ARMS.items():
        if not os.path.exists(path):
            print(f"skipping {name}: {path} not found")
            continue
        frame = pd.read_csv(path)
        frame["site"] = frame["case_id"].str.slice(0, 2)
        frames[name] = frame
    return frames


def boot_ci(y, p, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    draws = []
    while len(draws) < n_boot:
        sample = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[sample])) < 2:
            continue
        draws.append(roc_auc_score(y[sample], p[sample]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def per_site(frame):
    rows = []
    for site, group in frame.groupby("site"):
        y, p = group.true_label.values, group.p_ER_positive.values
        rows.append({
            "site": site, "n": len(y), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 and len(y) >= MIN_SITE_N
                     else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def leave_one_site_out(frame):
    rows = []
    overall = roc_auc_score(frame.true_label.values, frame.p_ER_positive.values)
    for site in sorted(frame.site.unique()):
        rest = frame[frame.site != site]
        y, p = rest.true_label.values, rest.p_ER_positive.values
        rows.append({
            "dropped_site": site, "n_dropped": int((frame.site == site).sum()),
            "n_remaining": len(y),
            "auroc_remaining": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        })
    out = pd.DataFrame(rows)
    out["delta_vs_full"] = out["auroc_remaining"] - overall
    return out.sort_values("n_dropped", ascending=False).reset_index(drop=True)


def fig_roc(internal, arms, path):
    plt.figure(figsize=(5.2, 5))
    yi, pi = internal
    fpr, tpr, _ = roc_curve(yi, pi)
    plt.plot(fpr, tpr, "k--", lw=1.4, label=f"internal TCGA ({roc_auc_score(yi, pi):.3f})")
    for name, frame in arms.items():
        y, p = frame.true_label.values, frame.p_ER_positive.values
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, lw=1.6, label=f"{name} ({roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], color="0.8", lw=0.8)
    plt.xlabel("1 - specificity"); plt.ylabel("sensitivity")
    plt.title("ER, case level: CPTAC external vs TCGA internal")
    plt.legend(loc="lower right", fontsize=7.5)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def fig_pr(internal, arms, path):
    plt.figure(figsize=(5.2, 5))
    yi, pi = internal
    prec, rec, _ = precision_recall_curve(yi, pi)
    plt.plot(rec, prec, "k--", lw=1.4,
             label=f"internal TCGA ({average_precision_score(yi, pi):.3f})")
    for name, frame in arms.items():
        y, p = frame.true_label.values, frame.p_ER_positive.values
        prec, rec, _ = precision_recall_curve(y, p)
        plt.plot(rec, prec, lw=1.6, label=f"{name} ({average_precision_score(y, p):.3f})")
    plt.xlabel("recall (ER-positive)"); plt.ylabel("precision")
    plt.title("Precision-recall, case level")
    plt.legend(loc="lower left", fontsize=7.5)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def fig_calibration(internal, external, path, n_bins=8):
    plt.figure(figsize=(5.2, 5))
    plt.plot([0, 1], [0, 1], color="0.8", lw=0.9, label="perfect")
    for (y, p), name, style in ((internal, "internal TCGA", "k--o"),
                                (external, "external CPTAC", "-o")):
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        centres, observed = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (p >= lo) & (p <= hi if hi == edges[-1] else p < hi)
            if mask.sum() < 3:
                continue
            centres.append(p[mask].mean()); observed.append(y[mask].mean())
        plt.plot(centres, observed, style, lw=1.5, ms=4, label=name)
    plt.axvline(0.5, color="0.6", lw=0.8, ls=":")
    plt.xlabel("mean predicted P(ER+)"); plt.ylabel("observed fraction ER+")
    plt.title("Reliability, case level (equal-count bins)")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def fig_scores(internal, external, path):
    # Density, not counts: TCGA has 1003 cases against CPTAC's 118, and the point is the
    # shape of each distribution, not how many cases sit under it.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 26)
    for ax, (y, p), title in zip(axes, (internal, external),
                                 ("internal TCGA", "external CPTAC")):
        ax.hist(p[y == 0], bins=bins, density=True, alpha=0.6,
                label=f"ER-negative (n={(y == 0).sum()})", color="#c0392b")
        ax.hist(p[y == 1], bins=bins, density=True, alpha=0.6,
                label=f"ER-positive (n={(y == 1).sum()})", color="#2471a3")
        ax.axvline(0.5, color="k", lw=1.0, ls="--")
        ax.set_title(f"{title}  (range {p.min():.2f}-{p.max():.2f}, "
                     f"mean p|ER- = {p[y == 0].mean():.2f})", fontsize=9.5)
        ax.set_xlabel("predicted P(ER+)")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("density")
    fig.suptitle("Score compression is why the 0.50 threshold stops working", fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def fig_leave_one_site(table, overall, path):
    plt.figure(figsize=(6.4, 4.0))
    keep = table.dropna(subset=["auroc_remaining"])
    plt.bar(keep.dropped_site, keep.auroc_remaining, color="#2471a3", alpha=0.85)
    plt.axhline(overall, color="k", ls="--", lw=1.2, label=f"full cohort ({overall:.3f})")
    for x, (site, value, n) in enumerate(zip(keep.dropped_site, keep.auroc_remaining,
                                             keep.n_dropped)):
        plt.text(x, value + 0.004, f"-{n}", ha="center", fontsize=7)
    plt.ylim(0.80, 1.0)
    plt.ylabel("AUROC on remaining cases"); plt.xlabel("site dropped")
    plt.title("Leave-one-site-out, WSI-alone, case level")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    arms = load_arms()
    if "WSI-alone" not in arms:
        raise SystemExit("WSI-alone predictions are required")

    internal_df = pd.read_csv(args.internal_csv)
    internal = (internal_df.y.values, internal_df.p.values)
    primary = arms["WSI-alone"]
    y, p = primary.true_label.values, primary.p_ER_positive.values
    overall = roc_auc_score(y, p)
    lo, hi = boot_ci(y, p, args.n_boot)

    summary = {"arm": "WSI-alone", "n": int(len(y)), "n_pos": int(y.sum()),
               "auroc": float(overall), "auroc_ci95": [lo, hi],
               "auprc_pos": float(average_precision_score(y, p))}

    site_table = per_site(primary)
    site_table.to_csv(os.path.join(args.output_dir, "per_site_auroc_case.csv"), index=False)
    loso = leave_one_site_out(primary)
    loso.to_csv(os.path.join(args.output_dir, "leave_one_site_out_case.csv"), index=False)

    print(f"=== WSI-alone, case level: AUROC {overall:.4f} [95% CI {lo:.4f}, {hi:.4f}] ===")
    print(f"\n=== Per-site (AUROC only where both classes and n >= {MIN_SITE_N}) ===")
    print(site_table.to_string(index=False))
    print("\n=== Leave-one-site-out ===")
    print(loso.to_string(index=False))
    worst = loso.dropna(subset=["delta_vs_full"]).sort_values("delta_vs_full").iloc[0]
    print(f"\nlargest swing: dropping site {worst.dropped_site} "
          f"({int(worst.n_dropped)} cases) moves AUROC by {worst.delta_vs_full:+.4f} "
          f"to {worst.auroc_remaining:.4f}")
    summary["leave_one_site_out_range"] = [float(loso.auroc_remaining.min()),
                                           float(loso.auroc_remaining.max())]

    fig_roc(internal, arms, os.path.join(args.output_dir, "fig_roc_case.png"))
    fig_pr(internal, arms, os.path.join(args.output_dir, "fig_pr_case.png"))
    fig_calibration(internal, (y, p), os.path.join(args.output_dir, "fig_calibration_case.png"))
    fig_scores(internal, (y, p), os.path.join(args.output_dir, "fig_score_distributions.png"))
    fig_leave_one_site(loso, overall,
                       os.path.join(args.output_dir, "fig_leave_one_site_out.png"))

    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nwrote tables and 5 figures to {args.output_dir}")


if __name__ == "__main__":
    main()
