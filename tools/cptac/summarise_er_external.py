"""
CPTAC-BRCA ER external validation: discrimination vs calibration
===============================================================
Compares the TCGA out-of-fold predictions of the `er_wsi_alone` arm against the
frozen-weight CPTAC predictions, and separates the two failure modes that a bare
AUROC hides:

  discrimination  does the ranking survive the cohort change (AUROC, AUPRC)
  calibration     does the shipped 0.50 threshold still mean anything

They come apart sharply here, which is the whole point of the comparison: a model
can rank almost perfectly on a new cohort and still be unusable at the operating
point it was shipped with.

Three thresholds are reported on CPTAC:
  0.50                   what a naive deployment would use
  internal Youden        chosen on TCGA only -- honest, no external labels needed
  prevalence-matched     the (1 - prevalence) quantile of the external scores;
                         needs only a local ER+ prevalence estimate, not labels,
                         so it is deployable where the Youden transfer is not
An external-Youden row is printed for reference and marked ORACLE: it is chosen
on the test labels and is an upper bound, not an achievable result.

    python tools/cptac/summarise_er_external.py
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score, roc_curve)


def parse_args():
    parser = argparse.ArgumentParser(description="ER external calibration report")
    parser.add_argument("--internal_csv", type=str,
                        default=".scratch/results/er/analysis/case_predictions_er_wsi_alone_s1.csv",
                        help="TCGA out-of-fold case predictions (columns: y, p)")
    parser.add_argument("--external_csv", type=str,
                        default=".scratch/cptac_validation/results/er/ensemble_case_predictions.csv",
                        help="CPTAC ensemble case predictions")
    parser.add_argument("--output_dir", type=str,
                        default=".scratch/cptac_validation/results/er")
    return parser.parse_args()


def operating_point(y, p, threshold):
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "sensitivity_ERpos": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity_ERneg": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
    }


def score_distribution(y, p):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    return {
        "n": int(len(y)),
        "prevalence_ERpos": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc_pos": float(average_precision_score(y, p)),
        "mean_p": float(p.mean()),
        "calibration_bias": float(p.mean() - y.mean()),
        "p_min": float(p.min()),
        "p_max": float(p.max()),
        "p_spread": float(p.max() - p.min()),
        "mean_p_in_ERpos": float(p[y == 1].mean()),
        "mean_p_in_ERneg": float(p[y == 0].mean()),
        "frac_scored_positive_at_0.5": float((p >= 0.5).mean()),
    }


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    internal = pd.read_csv(args.internal_csv)
    yi, pi = internal["y"].values, internal["p"].values
    external = pd.read_csv(args.external_csv)
    ye, pe = external["true_label"].values, external["p_ER_positive"].values

    int_dist, ext_dist = score_distribution(yi, pi), score_distribution(ye, pe)
    thr_internal_youden = youden_threshold(yi, pi)
    thr_prevalence = float(np.quantile(pe, 1.0 - ye.mean()))
    thr_external_youden = youden_threshold(ye, pe)

    report = {
        "internal_tcga": {**int_dist,
                          "at_0.5": operating_point(yi, pi, 0.5),
                          "at_internal_youden": operating_point(yi, pi, thr_internal_youden)},
        "external_cptac": {**ext_dist,
                           "at_0.5": operating_point(ye, pe, 0.5),
                           "at_internal_youden": operating_point(ye, pe, thr_internal_youden),
                           "at_prevalence_matched": operating_point(ye, pe, thr_prevalence),
                           "at_external_youden_ORACLE": operating_point(ye, pe, thr_external_youden)},
        "deltas": {
            "auroc": ext_dist["auroc"] - int_dist["auroc"],
            "auprc_pos": ext_dist["auprc_pos"] - int_dist["auprc_pos"],
            "p_spread": ext_dist["p_spread"] - int_dist["p_spread"],
            "mean_p_in_ERneg": ext_dist["mean_p_in_ERneg"] - int_dist["mean_p_in_ERneg"],
        },
    }
    path = os.path.join(args.output_dir, "external_calibration.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    def row(tag, dist):
        return (f"{tag:24s} n={dist['n']:5d}  prev={dist['prevalence_ERpos']:.3f}  "
                f"AUROC={dist['auroc']:.4f}  AUPRC={dist['auprc_pos']:.4f}")

    print("=== Discrimination ===")
    print(row("internal (TCGA, OOF)", int_dist))
    print(row("external (CPTAC)", ext_dist))
    print(f"{'delta':24s} AUROC {report['deltas']['auroc']:+.4f}   "
          f"AUPRC {report['deltas']['auprc_pos']:+.4f}")

    print("\n=== Score distribution (why the threshold breaks) ===")
    for tag, dist in (("internal", int_dist), ("external", ext_dist)):
        print(f"{tag:9s} range [{dist['p_min']:.3f}, {dist['p_max']:.3f}] "
              f"spread {dist['p_spread']:.3f}  mean_p={dist['mean_p']:.3f} "
              f"(bias {dist['calibration_bias']:+.3f})  "
              f"mean_p|ER-={dist['mean_p_in_ERneg']:.3f}  "
              f"frac>=0.5={dist['frac_scored_positive_at_0.5']:.3f}")

    print("\n=== Operating points ===")
    def show(tag, op, note=""):
        print(f"{tag:38s} thr={op['threshold']:.4f}  sens={op['sensitivity_ERpos']:.3f}  "
              f"spec={op['specificity_ERneg']:.3f}  balacc={op['balanced_acc']:.3f}{note}")
    show("internal @0.5", report["internal_tcga"]["at_0.5"])
    show("internal @internal-Youden", report["internal_tcga"]["at_internal_youden"])
    show("external @0.5", report["external_cptac"]["at_0.5"], "   <-- collapses")
    show("external @internal-Youden", report["external_cptac"]["at_internal_youden"])
    show("external @prevalence-matched", report["external_cptac"]["at_prevalence_matched"])
    show("external @external-Youden", report["external_cptac"]["at_external_youden_ORACLE"],
         "   <-- ORACLE, upper bound only")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
