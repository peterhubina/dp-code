"""
Slide- and case-level metrics for any CPTAC prediction directory
=================================================================
Reads <dir>/ensemble_predictions.csv in the shared schema and reports balanced
accuracy, macro AUROC, per-class one-vs-rest AUROC and the confusion matrix, at
both slide level and case level. Case level averages the ensemble softmax within
a case, which is the number to quote: CPTAC ships up to 10 slides per case, so
slide-level metrics are not patient-weighted.

    python tools/cptac/summarise_predictions.py .scratch/cptac_validation/results/predictions
"""

import sys

import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.preprocessing import label_binarize

CLASSES = ["LumA", "LumB", "Basal", "Her2"]
PROB_COLS = ["p_LumA", "p_LumB", "p_Basal", "p_Her2"]


def metrics(df, tag):
    y = df["true_label"].to_numpy(dtype=int)
    yp = df["pred_label"].to_numpy(dtype=int)
    prob = df[PROB_COLS].to_numpy(dtype=float)
    yb = label_binarize(y, classes=[0, 1, 2, 3])

    print(f"--- {tag} (n = {len(df)}) ---")
    print(f"  macro AUROC        {roc_auc_score(yb, prob, multi_class='ovr', average='macro'):.4f}")
    print(f"  balanced accuracy  {balanced_accuracy_score(y, yp):.4f}")
    print(f"  accuracy           {(y == yp).mean():.4f}")
    print("  per-class one-vs-rest AUROC:")
    for i, name in enumerate(CLASSES):
        if 0 < yb[:, i].sum() < len(y):
            print(f"    {name:<6s} {roc_auc_score(yb[:, i], prob[:, i]):.4f}")
    print("  confusion (rows true, cols predicted, LumA/LumB/Basal/Her2):")
    for row in confusion_matrix(y, yp, labels=[0, 1, 2, 3]):
        print(f"    {row}")
    print(classification_report(y, yp, target_names=CLASSES, zero_division=0))


def main():
    results_dir = sys.argv[1]
    df = pd.read_csv(f"{results_dir}/ensemble_predictions.csv")
    print(f"=== {results_dir} ===\n")
    metrics(df, "slide level")

    if df["case_id"].nunique() < len(df):
        agg = (df.groupby("case_id")
                 .agg(true_label=("true_label", "first"),
                      **{c: (c, "mean") for c in PROB_COLS})
                 .reset_index())
        agg["pred_label"] = agg[PROB_COLS].to_numpy().argmax(axis=1)
        metrics(agg, "case level (mean softmax within case)")
    else:
        print("(one row per case already -- no case-level aggregation needed)")


if __name__ == "__main__":
    main()
