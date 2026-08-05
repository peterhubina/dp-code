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

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# `tools.pam50_arms`, not a bare `pam50_arms`: the bare form only resolves when Python puts this
# script's own directory on sys.path, i.e. under `python tools/evaluate_cnv_wsi_fusion.py` and
# nowhere else, so `dp-analysis` could not import this module at all. The package form works for
# both, and binds the same module object either way. `pip install -e .` is what puts the
# repository root on sys.path; the direct invocation keeps working exactly as before.
try:
    from tools.pam50_arms import (CLASSES, CPTAC_ARMS, CPTAC_WSI_PROBS, balanced_acc,
                                  bootstrap_indices, clam_column_order, cnv_arm, delta_ci,
                                  load_clam_oof, load_cptac_arms, load_cptac_wsi_probs,
                                  load_tcga_arms, macro_auroc, renormalise)
except ModuleNotFoundError as exc:  # pragma: no cover - install error, not a code path
    raise ModuleNotFoundError(
        f"{exc}. Run `pip install -e .` from the repository root once; after that this script "
        "runs from any working directory."
    ) from exc

# --- Frozen numerical constants ------------------------------------------------------------
# Every published figure in docs/cnv-wsi-fusion-external-validation.md was produced at these
# values. They are named (rather than inlined) so a test can assert them and so `dp-analysis`
# can surface them, NOT so they can be retuned: the confidence intervals in that document are a
# function of N_BOOT and BOOTSTRAP_SEED, and the internal CNV column is a function of the two
# CV settings. `stack_wsi_cnv.py` deliberately uses a *different* bootstrap seed (11); the two
# scripts are not meant to share one.
N_BOOT = 4000
BOOTSTRAP_SEED = 7
CV_FOLDS = 10
CV_SEED = 0
#: Saerens-Latinne-Decaestecker fixed-point iteration budget and convergence tolerance.
SLD_ITERS = 500
SLD_TOL = 1e-9


def sld_em(P, prior, iters=SLD_ITERS, tol=SLD_TOL):
    """Saerens-Latinne-Decaestecker: estimate target priors from unlabelled probabilities."""
    pi = prior.copy()
    for _ in range(iters):
        new = renormalise(P * (pi / prior)).mean(0)
        if np.abs(new - pi).max() < tol:
            break
        pi = new
    return pi


def report(y, models, n_boot=N_BOOT, seed=BOOTSTRAP_SEED):
    idx = bootstrap_indices(y, n_boot, seed)

    # Score every model on every resample ONCE. Each pairwise delta is then a column
    # subtraction on shared resamples, which keeps the pairing without recomputing metrics
    # for each of the n*(n-1)/2 contrasts.
    names = list(models)
    boot = {m: np.empty((n_boot, 2)) for m in names}
    for b, j in enumerate(idx):
        yj = y[j]
        for m in names:
            Pj = models[m][j]
            boot[m][b] = (macro_auroc(yj, Pj), balanced_acc(yj, Pj))

    rows = []
    for name, P in models.items():
        pred = CLASSES[P.argmax(1)]
        lo, hi = np.percentile(boot[name][:, 0], [2.5, 97.5])
        rows.append({
            "model": name,
            "macroAUROC": round(macro_auroc(y, P), 3),
            "95% CI": f"[{lo:.3f}, {hi:.3f}]",
            "balAcc": round(balanced_acc(y, P), 3),
            **{f"{c}": f"{int(((pred == c) & (y == c)).sum())}/{int((y == c).sum())}"
               for c in CLASSES},
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\npaired bootstrap, {n_boot} resamples of the same {len(y)} cases")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out = []
            for k, lab in enumerate(("dAUROC", "dBalAcc")):
                mean_d, lo, hi, verdict = delta_ci(boot[a][:, k], boot[b][:, k])
                out.append(f"{lab} {mean_d:+.3f} [{lo:+.3f},{hi:+.3f}] {verdict:3s}")
            print(f"  {a:22s} - {b:22s} " + " | ".join(out))


def external(args):
    wc, n_slides = load_cptac_wsi_probs()
    print(f"CPTAC WSI predictions: {n_slides} slides -> {len(wc)} cases")

    Xt, yt = load_tcga_arms()
    clf = cnv_arm().fit(Xt, yt)
    Xc = load_cptac_arms()

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

    Pwb = renormalise(Pw / prior)
    print(f"\nWSI Her2 head, max p_Her2 over {len(y)} cases: raw {Pw[:, 1].max():.4f}, "
          f"prior-balanced {Pwb[:, 1].max():.4f}; argmax selects Her2 for "
          f"{int((CLASSES[Pwb.argmax(1)] == 'Her2').sum())} cases after balancing\n")

    report(y, {
        "WSI raw": Pw,
        "WSI prior-balanced": Pwb,
        "WSI SLD-EM": renormalise(Pw * (pi / prior)),
        "CNV (39 arms)": Pc,
        "Fusion raw": (Pw + Pc) / 2,
        "Fusion balanced": (Pwb + Pc) / 2,
    }, n_boot=args.n_boot, seed=args.bootstrap_seed)


def internal(args):
    """TCGA-only head-to-head on the cases that have CLAM out-of-fold predictions."""
    w = load_clam_oof()

    X, y = load_tcga_arms()
    common = [c for c in X.index.intersection(w.index) if y[c] != "Normal"]
    X, yy, w = X.loc[common], y.loc[common], w.loc[common]
    # CLAM's integer labels are in its own class order; recover it before scoring.
    clam_order = clam_column_order(w, yy)
    Pw = w[[f"p{i}" for i in range(4)]].values[:, [clam_order.index(c) for c in CLASSES]]
    Pc = cross_val_predict(cnv_arm(), X, yy,
                           cv=StratifiedKFold(args.cv_folds, shuffle=True,
                                              random_state=args.cv_seed),
                           method="predict_proba")
    print(f"\n=== INTERNAL TCGA, {len(common)} cases with CLAM out-of-fold, both 10-fold ===")
    print(f"CLAM class order on disk: {clam_order}")
    report(yy.values, {"WSI": Pw, "CNV (39 arms)": Pc, "Fusion": (Pw + Pc) / 2},
           n_boot=args.n_boot, seed=args.bootstrap_seed)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--internal", action="store_true", help="also run the TCGA-only comparison")
    frozen = ap.add_argument_group(
        "frozen numerical constants",
        "Defaults are the values behind every number in "
        "docs/cnv-wsi-fusion-external-validation.md. Changing one changes the published table.")
    frozen.add_argument("--n-boot", type=int, default=N_BOOT,
                        help="paired-bootstrap resamples (default: %(default)s)")
    frozen.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED,
                        help="RNG seed for the bootstrap resamples (default: %(default)s)")
    frozen.add_argument("--cv-folds", type=int, default=CV_FOLDS,
                        help="StratifiedKFold splits for the internal CNV arm "
                             "(default: %(default)s)")
    frozen.add_argument("--cv-seed", type=int, default=CV_SEED,
                        help="StratifiedKFold random_state for the internal CNV arm "
                             "(default: %(default)s)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    for p in (CPTAC_WSI_PROBS, CPTAC_ARMS):
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
