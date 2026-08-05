#!/usr/bin/env python3
"""Compare the fusion-operator ladder against the WSI-only model and the probability mean.

    python tools/compare_fusion_ladder.py

CLAM's ``summary.csv`` reports each fold's AUC and averages them. That is not the quantity the
rest of this thread is measured in, and the two are not interchangeable: everything in
``docs/cnv-wsi-fusion-external-validation.md`` is a *pooled* out-of-fold macro AUROC over one case
set. This script recomputes every arm that way, on the cases all arms share, so the ladder can be
put next to the equal-weight probability mean it has to beat.

The mean is the real bar. It needs no training at all -- average the WSI-only probabilities with a
39-feature logistic regression on chromosome arms -- and it reaches 0.926 internally. An operator
that trains a joint model and lands below that has not earned its complexity.

For ``film_attention`` the per-slide records also carry the conditioner's own diagnostics, so the
run can be asked a question the metrics cannot answer: did the FiLM parameters move off their
zero-initialisation at all, or did the model quietly collapse back to WSI-only?
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from pam50_arms import (CLASSES, balanced_acc, bootstrap_indices, case_of, cnv_arm, delta_ci,
                        fold_train_cases, load_tcga_arms, macro_auroc)

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / ".scratch/results"
WSI_ONLY = "pam50_final_s1"
LADDER = ["concat", "gated", "cross_attention", "film_attention", "coattn"]
DIAGNOSTICS = ["fusion_film_gamma_dev", "fusion_film_beta_abs", "fusion_tabular_logit_abs"]


def load_run(run_dir: Path):
    """Pooled per-case probabilities, the fold that produced them, and any fusion diagnostics."""
    rows = []
    for path in sorted(run_dir.glob("split_*_results.pkl")):
        fold = int(path.stem.split("_")[1])
        for slide, rec in pickle.load(open(path, "rb")).items():
            row = {"case_id": case_of(slide), "fold": fold, "label": int(rec["label"]),
                   **{f"p{i}": p for i, p in enumerate(np.asarray(rec["prob"]).ravel())}}
            for key in DIAGNOSTICS:
                if key in rec:
                    row[key] = float(rec[key])
            rows.append(row)
    frame = pd.DataFrame(rows)
    how = {**{f"p{i}": "mean" for i in range(4)}, "label": "first", "fold": "first"}
    how.update({k: "mean" for k in DIAGNOSTICS if k in frame.columns})
    return frame.groupby("case_id").agg(how)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    runs = {"WSI only": load_run(RESULTS / WSI_ONLY)}
    for mode in LADDER:
        run_dir = RESULTS / f"pam50_wsi_cnv_{mode}_s1"
        if run_dir.is_dir():
            runs[mode] = load_run(run_dir)
        else:
            print(f"  (skipping {mode}: {run_dir.relative_to(REPO)} not found)")

    X_all, y_all = load_tcga_arms()
    shared = sorted(set.intersection(*(set(r.index) for r in runs.values())) & set(X_all.index))
    y = y_all.loc[shared]
    print(f"{len(shared)} cases shared by all {len(runs)} runs\n")

    # CLAM stores its own class order; recover it from the WSI-only run and reuse it, since every
    # ladder arm was trained through the same label_dict.
    base = runs["WSI only"].loc[shared]
    order = [y[base["label"] == i].mode()[0] for i in range(len(CLASSES))]
    assert len(set(order)) == len(CLASSES), order
    perm = [order.index(c) for c in CLASSES]

    probs = {name: run.loc[shared, [f"p{i}" for i in range(4)]].values[:, perm]
             for name, run in runs.items()}

    # The bar: WSI-only averaged with a CNV logistic regression, refit per fold so it is honestly
    # out-of-fold on the same partition the CLAM runs used.
    folds = base["fold"].values
    cnv = np.zeros_like(probs["WSI only"])
    for fold in np.unique(folds):
        train = X_all.index.isin(fold_train_cases(fold) & set(X_all.index))
        held = folds == fold
        model = cnv_arm().fit(X_all[train], y_all[train])
        cnv[held] = model.predict_proba(X_all.loc[np.array(shared)[held]])
    probs["CNV only"] = cnv
    probs["probability mean"] = (probs["WSI only"] + cnv) / 2

    idx = bootstrap_indices(y.values, args.n_boot, seed=13)
    scored = {name: np.array([[macro_auroc(y.values[j], P[j]), balanced_acc(y.values[j], P[j])]
                              for j in idx]) for name, P in probs.items()}

    order_out = ["WSI only", "CNV only", "probability mean", *[m for m in LADDER if m in probs]]
    print(pd.DataFrame([
        {"arm": name,
         "macroAUROC": round(macro_auroc(y.values, probs[name]), 4),
         "AUROC 95% CI": "[{:.3f}, {:.3f}]".format(*np.percentile(scored[name][:, 0], [2.5, 97.5])),
         "balAcc": round(balanced_acc(y.values, probs[name]), 4)}
        for name in order_out]).to_string(index=False))

    print("\nvs the equal-weight probability mean, paired bootstrap:")
    for name in order_out:
        if name == "probability mean":
            continue
        cells = []
        for col, lab in enumerate(("dAUROC", "dBalAcc")):
            mean_d, lo, hi, verdict = delta_ci(scored[name][:, col],
                                               scored["probability mean"][:, col])
            cells.append(f"{lab} {mean_d:+.4f} [{lo:+.4f},{hi:+.4f}] {verdict:3s}")
        print(f"  {name:18s} " + " | ".join(cells))

    film = runs.get("film_attention")
    if film is not None and DIAGNOSTICS[0] in film.columns:
        print("\nFiLM conditioner diagnostics (zero => the second modality was ignored):")
        for key in DIAGNOSTICS:
            if key in film.columns:
                v = film.loc[shared, key]
                print(f"  {key:28s} mean {v.mean():.5f}  median {v.median():.5f}  max {v.max():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
