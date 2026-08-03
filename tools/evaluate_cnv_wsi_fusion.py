#!/usr/bin/env python3
"""External validation of WSI + arm-level-CNV fusion for 4-class PAM50, TCGA -> CPTAC.

Reproduces every number in ``docs/cnv-wsi-fusion-external-validation.md``.

    python tools/evaluate_cnv_wsi_fusion.py
    python tools/evaluate_cnv_wsi_fusion.py --internal      # TCGA-only head-to-head as well

Two arms, both trained on TCGA only and applied to CPTAC without refitting:

  WSI  TCGA-trained CLAM-MB + UNI2-h, already inferred over CPTAC by the external-validation
       pipeline. Read from disk; this script never touches a slide.
  CNV  39 chromosome arms (median gene-level log2 per arm) from
       ``tools/download_cnv_mutations.py --representation arm``, into a logistic regression.

The fusion rule is an equal-weight mean of the two probability vectors, fixed before the external
set was scored. Nothing here is tuned on CPTAC.

**Prior balancing.** The CNV arm is fit with ``class_weight='balanced'`` while the WSI arm was
trained under TCGA's natural class frequencies (Her2 = 8.3%). Comparing their argmax decisions
directly is therefore unfair to the WSI arm, so ``WSI (prior-balanced)`` divides its probabilities
by the TCGA training prior and renormalises — matching the decision rule, not fitting anything.
``WSI (SLD-EM)`` is the fully unsupervised alternative (Saerens-Latinne-Decaestecker prior
estimation from the unlabelled CPTAC probabilities). Both are reported because the balanced variant
was run *after* seeing the raw fusion underperform, and should be read as a post-hoc control.
"""

import argparse
import glob
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
CLASSES = np.array(["Basal", "Her2", "LumA", "LumB"])          # sorted; used everywhere
CPTAC_WSI = REPO / ".scratch/cptac_validation/results/predictions/ensemble_predictions.csv"
TCGA_OOF = REPO / ".scratch/results/pam50_final_s1/split_*_results.pkl"


def norm(p):
    return p / p.sum(1, keepdims=True)


def tcga_cnv():
    X = pd.read_csv(REPO / ".datasets/cnv/tcga_brca_cna_arm.csv", index_col=0)
    y = (pd.read_csv(REPO / "tools/data/tcga_brca_pam50_labels.csv")
         .drop_duplicates("case_id").set_index("case_id")["label"])
    i = X.index.intersection(y.index)
    X, y = X.loc[i], y.loc[i]
    keep = y != "Normal"                      # CPTAC's 114-case subset has no Normal-like
    return X[keep], y[keep]


def fit_cnv(X, y):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=4000, C=0.1, class_weight="balanced"))
    clf.fit(X, y)
    assert list(clf.classes_) == list(CLASSES), clf.classes_
    return clf


def sld_em(P, prior, iters=500):
    """Saerens-Latinne-Decaestecker: estimate target priors from unlabelled probabilities."""
    pi = prior.copy()
    for _ in range(iters):
        new = norm(P * (pi / prior)).mean(0)
        if np.abs(new - pi).max() < 1e-9:
            break
        pi = new
    return pi


def report(y, models, n_boot=4000, seed=7):
    macro = lambda yy, P: roc_auc_score(yy, P, multi_class="ovr", average="macro")
    balacc = lambda yy, P: balanced_accuracy_score(yy, CLASSES[P.argmax(1)])

    rng = np.random.default_rng(seed)
    idx = []
    while len(idx) < n_boot:
        j = rng.integers(0, len(y), len(y))
        if len(np.unique(y[j])) == len(CLASSES):
            idx.append(j)

    # Score every model on every resample ONCE. Each pairwise delta is then a column
    # subtraction on shared resamples, which keeps the pairing without recomputing metrics
    # for each of the n*(n-1)/2 contrasts.
    names = list(models)
    boot = {m: np.empty((n_boot, 2)) for m in names}
    for b, j in enumerate(idx):
        yj = y[j]
        for m in names:
            Pj = models[m][j]
            boot[m][b] = (macro(yj, Pj), balacc(yj, Pj))

    rows = []
    for name, P in models.items():
        pred = CLASSES[P.argmax(1)]
        lo, hi = np.percentile(boot[name][:, 0], [2.5, 97.5])
        rows.append({
            "model": name,
            "macroAUROC": round(macro(y, P), 3),
            "95% CI": f"[{lo:.3f}, {hi:.3f}]",
            "balAcc": round(balacc(y, P), 3),
            **{f"{c}": f"{int(((pred == c) & (y == c)).sum())}/{int((y == c).sum())}"
               for c in CLASSES},
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\npaired bootstrap, {n_boot} resamples of the same {len(y)} cases")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = boot[a] - boot[b]
            out = []
            for k, lab in enumerate(("dAUROC", "dBalAcc")):
                lo, hi = np.percentile(d[:, k], [2.5, 97.5])
                # Deltas here can be negative (A worse than B), so the test is that the
                # interval excludes zero on either side — not just lo > 0.
                out.append(f"{lab} {d[:, k].mean():+.3f} [{lo:+.3f},{hi:+.3f}] "
                           f"{'sig' if (lo > 0 or hi < 0) else 'ns':3s}")
            print(f"  {a:22s} - {b:22s} " + " | ".join(out))


def external(args):
    w = pd.read_csv(CPTAC_WSI)
    wc = w.groupby("case_id").agg({**{f"p_{c}": "mean" for c in CLASSES}, "true_name": "first"})
    print(f"CPTAC WSI predictions: {len(w)} slides -> {len(wc)} cases")

    Xt, yt = tcga_cnv()
    clf = fit_cnv(Xt, yt)
    Xc = pd.read_csv(REPO / ".datasets/cnv/cptac_brca_cna_arm.csv", index_col=0)

    common = sorted(set(wc.index) & set(Xc.index))
    y = wc.loc[common, "true_name"].values
    Pw = wc.loc[common, [f"p_{c}" for c in CLASSES]].values
    Pc = clf.predict_proba(Xc.loc[common, Xt.columns])
    keep = np.isin(y, CLASSES)
    y, Pw, Pc = y[keep], Pw[keep], Pc[keep]
    print(f"external set: {len(y)} cases  {dict(pd.Series(y).value_counts())}")

    prior = np.array([(yt == c).mean() for c in CLASSES])
    print("TCGA training prior:", {c: round(p, 3) for c, p in zip(CLASSES, prior)})
    pi = sld_em(Pw, prior)
    print("SLD-EM estimated CPTAC prior:", {c: round(p, 3) for c, p in zip(CLASSES, pi)},
          "| true:", {c: round((y == c).mean(), 3) for c in CLASSES})

    Pwb = norm(Pw / prior)
    print(f"\nWSI Her2 head, max p_Her2 over {len(y)} cases: raw {Pw[:, 1].max():.4f}, "
          f"prior-balanced {Pwb[:, 1].max():.4f}; argmax selects Her2 for "
          f"{int((CLASSES[Pwb.argmax(1)] == 'Her2').sum())} cases after balancing\n")

    report(y, {
        "WSI raw": Pw,
        "WSI prior-balanced": Pwb,
        "WSI SLD-EM": norm(Pw * (pi / prior)),
        "CNV (39 arms)": Pc,
        "Fusion raw": (Pw + Pc) / 2,
        "Fusion balanced": (Pwb + Pc) / 2,
    }, n_boot=args.n_boot)


def internal(args):
    """TCGA-only head-to-head on the cases that have CLAM out-of-fold predictions."""
    rows = []
    for f in sorted(glob.glob(str(TCGA_OOF))):
        for sid, v in pickle.load(open(f, "rb")).items():
            rows.append({"case_id": "-".join(sid.split("-")[:3]), "label": int(v["label"]),
                         **{f"p{i}": p for i, p in enumerate(np.asarray(v["prob"]).ravel())}})
    w = (pd.DataFrame(rows)
         .groupby("case_id").agg({**{f"p{i}": "mean" for i in range(4)}, "label": "first"}))

    X, y = tcga_cnv()
    common = [c for c in X.index.intersection(w.index) if y[c] != "Normal"]
    X, yy, w = X.loc[common], y.loc[common], w.loc[common]
    # CLAM's integer labels are in its own class order; recover it before scoring.
    clam_order = [yy[w["label"] == i].mode()[0] for i in range(4)]
    Pw = w[[f"p{i}" for i in range(4)]].values[:, [clam_order.index(c) for c in CLASSES]]
    Pc = cross_val_predict(make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=4000, C=0.1, class_weight="balanced")),
                           X, yy, cv=StratifiedKFold(10, shuffle=True, random_state=0),
                           method="predict_proba")
    print(f"\n=== INTERNAL TCGA, {len(common)} cases with CLAM out-of-fold, both 10-fold ===")
    print(f"CLAM class order on disk: {clam_order}")
    report(yy.values, {"WSI": Pw, "CNV (39 arms)": Pc, "Fusion": (Pw + Pc) / 2},
           n_boot=args.n_boot)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--internal", action="store_true", help="also run the TCGA-only comparison")
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()

    for p in (CPTAC_WSI, REPO / ".datasets/cnv/cptac_brca_cna_arm.csv"):
        if not p.exists():
            print(f"missing {p}\nrun tools/download_cnv_mutations.py --representation arm first")
            return 1

    print("=== EXTERNAL CPTAC (TCGA-trained, nothing refit) ===")
    external(args)
    if args.internal:
        internal(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
