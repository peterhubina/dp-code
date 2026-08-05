#!/usr/bin/env python3
"""Does a learned fusion rule beat an equal-weight mean of the WSI and CNV probabilities?

    python tools/stack_wsi_cnv.py

The point is to answer that cheaply, before anyone trains a joint fusion head. If a stacker
given both probability vectors cannot beat their unweighted average on 599 TCGA cases, a
jointly-trained head is unlikely to do better, and the RNA-fusion history in this project says
that outcome is live rather than hypothetical.

Fold alignment is the part that has to be right. CLAM's out-of-fold predictions come from the
10 splits in ``project/CLAM/splits/tcga_brca_subtyping_100/``; the CNV arm is therefore refit
per fold on *that fold's* training cases, so both arms are out-of-fold on the same fold. Training
the CNV arm on all of TCGA and pairing it with CLAM's out-of-fold output would leak every fold's
test cases into the CNV side.

**Those 10 splits are not a partition.** ``create_splits_seq`` draws them independently, so of
910 labelled cases 599 land in at least one test set, 311 in none, and 242 in between two and
five. Each case is therefore tagged with the *first* fold that held it out, which yields a
pseudo-partition of uneven size and quality: cases tagged fold 0 average 1.9 CLAM models behind
their probability vector, cases tagged fold 9 exactly 1.0. Two consequences, both real:

  - the nested folds are not exchangeable, so treat the internal spread as indicative;
  - ``WSI alone`` is a 1-to-5-model *ensemble* for the 242 multiply-tested cases, which flatters
    it by roughly 0.01 AUROC against a true single-model number (0.877 taking only each case's
    lowest fold, against 0.887 ensembled). The comparison is still fair to the mean, because
    every rule consumes the same ``Pw``.

Re-running the whole thing with a random stratified rule-partition instead of the fold tags
gives the same verdict, so the pseudo-partition does not drive the result.

The stacker is scored by nested cross-validation over those fold tags -- fit on nine folds'
out-of-fold pairs, evaluated on the tenth -- so no stacker ever sees the cases it is judged on.
The external number refits the whole chain on all of TCGA and applies it to CPTAC untouched.

Five combination rules, in increasing order of freedom:

  mean            fixed 0.5/0.5, the baseline to beat
  scalar          one learned weight, w*WSI + (1-w)*CNV
  per_class       one learned weight per class
  logreg_prob     multinomial logistic regression on the 8 concatenated probabilities
  logreg_logprob  the same on log probabilities, i.e. a learned log-linear opinion pool

Anything past ``scalar`` can also recalibrate, not just reweight, so a win there is not by
itself evidence that the modalities are being combined better -- see the report footer.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

# `tools.pam50_arms` rather than a bare `pam50_arms` — see the note in
# evaluate_cnv_wsi_fusion.py. The direct `python tools/stack_wsi_cnv.py` invocation is unchanged.
try:
    from tools.pam50_arms import (CLASSES, balanced_acc, bootstrap_indices, clam_column_order,
                                  cnv_arm, delta_ci, fold_train_cases, load_clam_oof,
                                  load_cptac_arms, load_cptac_wsi_probs, load_tcga_arms,
                                  macro_auroc, renormalise)
except ModuleNotFoundError as exc:  # pragma: no cover - install error, not a code path
    raise ModuleNotFoundError(
        f"{exc}. Run `pip install -e .` from the repository root once; after that this script "
        "runs from any working directory."
    ) from exc

# --- Frozen numerical constants ------------------------------------------------------------
# The values behind §7 of docs/cnv-wsi-fusion-external-validation.md. Named so a test can assert
# them, not so they can be retuned. Note the bootstrap seed is 11 here and 7 in
# evaluate_cnv_wsi_fusion.py: the two scripts report different tables and are deliberately not
# unified behind one seed.
N_BOOT = 2000
BOOTSTRAP_SEED = 11
#: The stacker (`logreg_prob` / `logreg_logprob`). Distinct from the CNV arm's own C=0.1 —
#: the stacker sees 8 probability columns, not 39 arms.
STACKER_C = 1.0
STACKER_MAX_ITER = 4000
#: Nelder-Mead settings for the `scalar` / `per_class` weight search, and the floor the
#: log-loss and the log-probability features are clipped at. `per_class` converges to a
#: boundary solution for Her2, so the tolerances decide where "the boundary" is.
NM_XATOL = 1e-4
NM_FATOL = 1e-6
NM_MAXITER = 2000
CLIP_FLOOR = 1e-9


# ------------------------------------------------------------------ combination rules
def fit_rule(kind, Pw, Pc, y, *, stacker_C=STACKER_C, stacker_max_iter=STACKER_MAX_ITER,
             nm_xatol=NM_XATOL, nm_fatol=NM_FATOL, nm_maxiter=NM_MAXITER,
             clip_floor=CLIP_FLOOR):
    if kind == "mean":
        return ("mean", None)
    if kind in ("scalar", "per_class"):
        n = 1 if kind == "scalar" else len(CLASSES)
        Y = np.stack([(y == c).astype(float) for c in CLASSES], 1)

        def loss(w):
            a = 1 / (1 + np.exp(-w))                       # keep weights in (0, 1)
            P = renormalise(a * Pw + (1 - a) * Pc)
            return -np.mean(np.sum(Y * np.log(np.clip(P, clip_floor, 1)), 1))

        best = minimize(loss, np.zeros(n), method="Nelder-Mead",
                        options={"xatol": nm_xatol, "fatol": nm_fatol, "maxiter": nm_maxiter})
        return (kind, 1 / (1 + np.exp(-best.x)))
    feats = _stack_features(kind, Pw, Pc, clip_floor)
    lr = LogisticRegression(max_iter=stacker_max_iter, C=stacker_C).fit(feats, y)
    assert list(lr.classes_) == list(CLASSES)
    return (kind, lr)


def _stack_features(kind, Pw, Pc, clip_floor=CLIP_FLOOR):
    feats = np.hstack([Pw, Pc])
    return feats if kind == "logreg_prob" else np.log(np.clip(feats, clip_floor, 1))


def apply_rule(rule, Pw, Pc, *, clip_floor=CLIP_FLOOR):
    kind, fitted = rule
    if kind == "mean":
        return (Pw + Pc) / 2
    if kind in ("scalar", "per_class"):
        return renormalise(fitted * Pw + (1 - fitted) * Pc)
    return fitted.predict_proba(_stack_features(kind, Pw, Pc, clip_floor))


RULES = ["mean", "scalar", "per_class", "logreg_prob", "logreg_logprob"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    frozen = ap.add_argument_group(
        "frozen numerical constants",
        "Defaults are the values behind §7 of "
        "docs/cnv-wsi-fusion-external-validation.md. Changing one changes the published table.")
    frozen.add_argument("--n-boot", type=int, default=N_BOOT,
                        help="paired-bootstrap resamples (default: %(default)s)")
    frozen.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED,
                        help="RNG seed for the bootstrap resamples (default: %(default)s)")
    frozen.add_argument("--stacker-C", type=float, default=STACKER_C,
                        help="inverse regularisation of the logreg stackers "
                             "(default: %(default)s)")
    frozen.add_argument("--stacker-max-iter", type=int, default=STACKER_MAX_ITER,
                        help="max_iter of the logreg stackers (default: %(default)s)")
    frozen.add_argument("--nm-xatol", type=float, default=NM_XATOL,
                        help="Nelder-Mead xatol for the weight rules (default: %(default)s)")
    frozen.add_argument("--nm-fatol", type=float, default=NM_FATOL,
                        help="Nelder-Mead fatol for the weight rules (default: %(default)s)")
    frozen.add_argument("--nm-maxiter", type=int, default=NM_MAXITER,
                        help="Nelder-Mead iteration cap (default: %(default)s)")
    frozen.add_argument("--clip-floor", type=float, default=CLIP_FLOOR,
                        help="probability floor for log-loss and log-prob features "
                             "(default: %(default)s)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    rule_kwargs = dict(stacker_C=args.stacker_C, stacker_max_iter=args.stacker_max_iter,
                       nm_xatol=args.nm_xatol, nm_fatol=args.nm_fatol,
                       nm_maxiter=args.nm_maxiter, clip_floor=args.clip_floor)

    w = load_clam_oof(with_folds=True)
    X_all, y_all_labels = load_tcga_arms()
    common = sorted(set(w.index) & set(X_all.index))
    w, X, y = w.loc[common], X_all.loc[common], y_all_labels.loc[common]
    clam_order = clam_column_order(w, y)
    Pw = w[[f"p{i}" for i in range(4)]].values[:, [clam_order.index(c) for c in CLASSES]]
    folds = w["fold"].values
    print(f"{len(common)} TCGA cases with CLAM out-of-fold predictions across "
          f"{len(np.unique(folds))} folds; CLAM class order on disk {clam_order}")

    # CNV arm, refit per CLAM fold so both arms are out-of-fold on the same fold. Train from the
    # full labelled pool, not just the 599 cases that carry CLAM predictions -- restricting to
    # `common` here would throw away ~275 usable training cases per fold for no reason.
    Pc = np.zeros_like(Pw)
    sizes = []
    for f in np.unique(folds):
        tr_cases = fold_train_cases(f) & set(X_all.index)
        te = folds == f
        eval_cases = set(X.index[te])
        assert not (eval_cases & tr_cases), \
            f"fold {f}: {len(eval_cases & tr_cases)} eval cases in its own train split"
        tr = X_all.index.isin(tr_cases)
        model = cnv_arm().fit(X_all[tr], y_all_labels[tr])
        assert list(model.classes_) == list(CLASSES)
        Pc[te] = model.predict_proba(X[te])
        sizes.append(int(tr.sum()))
    print(f"CNV arm refit per fold on {min(sizes)}-{max(sizes)} cases "
          f"(mean {np.mean(sizes):.0f}); no eval case appears in its own fold's train split\n")

    # Nested scoring: fit each rule on nine folds' pairs, score on the tenth.
    print(f"=== INTERNAL, nested over CLAM's own folds ({len(common)} cases) ===")
    rows, oof = [], {}
    for kind in RULES:
        P = np.zeros_like(Pw)
        for f in np.unique(folds):
            te = folds == f
            P[te] = apply_rule(fit_rule(kind, Pw[~te], Pc[~te], y[~te].values, **rule_kwargs),
                               Pw[te], Pc[te], clip_floor=args.clip_floor)
        oof[kind] = P
        rows.append({"rule": kind, "macroAUROC": round(macro_auroc(y.values, P), 3),
                     "balAcc": round(balanced_acc(y.values, P), 3)})
    rows.insert(0, {"rule": "WSI alone", "macroAUROC": round(macro_auroc(y.values, Pw), 3),
                    "balAcc": round(balanced_acc(y.values, Pw), 3)})
    rows.insert(1, {"rule": "CNV alone", "macroAUROC": round(macro_auroc(y.values, Pc), 3),
                    "balAcc": round(balanced_acc(y.values, Pc), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    idx = bootstrap_indices(y.values, args.n_boot, seed=args.bootstrap_seed)
    base_scores = np.array([macro_auroc(y.values[j], oof["mean"][j]) for j in idx])
    print("\nvs the equal-weight mean, paired bootstrap:")
    for kind in RULES[1:]:
        scores = np.array([macro_auroc(y.values[j], oof[kind][j]) for j in idx])
        mean_d, lo, hi, verdict = delta_ci(scores, base_scores)
        print(f"  {kind:15s} dAUROC {mean_d:+.4f} [{lo:+.4f},{hi:+.4f}] {verdict}")

    # External: refit the whole chain on all of TCGA, apply to CPTAC untouched.
    cwc, _ = load_cptac_wsi_probs()
    Xc = load_cptac_arms()
    shared_cases = sorted(set(cwc.index) & set(Xc.index))
    ye = cwc.loc[shared_cases, "true_name"].values
    Ew = cwc.loc[shared_cases, [f"p_{c}" for c in CLASSES]].values
    # Deployed CNV arm, trained on every labelled TCGA case (945), not just the 599 that happen
    # to carry CLAM out-of-fold predictions -- this is what evaluate_cnv_wsi_fusion.py reports and
    # what anyone would actually ship. The combination rules can only be fit on the 599 OOF pairs,
    # so a rule tuned against a ~451-case CNV arm is applied to a stronger one; that mismatch
    # affects the weighted rules and not `mean`, which is one more reason to prefer `mean`.
    Ec = cnv_arm().fit(X_all, y_all_labels).predict_proba(Xc.loc[shared_cases, X_all.columns])
    scored = np.isin(ye, CLASSES)
    ye, Ew, Ec = ye[scored], Ew[scored], Ec[scored]

    print(f"\n=== EXTERNAL CPTAC ({len(ye)} cases), rules fit on all TCGA out-of-fold pairs ===")
    arms = {"WSI alone": Ew, "CNV alone": Ec}
    for kind in RULES:
        arms[kind] = apply_rule(fit_rule(kind, Pw, Pc, y.values, **rule_kwargs), Ew, Ec,
                                clip_floor=args.clip_floor)

    eidx = bootstrap_indices(ye, args.n_boot, seed=args.bootstrap_seed)
    escore = {name: np.array([[macro_auroc(ye[j], P[j]), balanced_acc(ye[j], P[j])] for j in eidx])
              for name, P in arms.items()}

    ci = lambda name, col: "[{:.3f}, {:.3f}]".format(
        *np.percentile(escore[name][:, col], [2.5, 97.5]))
    print(pd.DataFrame([
        {"rule": name,
         "macroAUROC": round(macro_auroc(ye, P), 3), "AUROC 95% CI": ci(name, 0),
         "balAcc": round(balanced_acc(ye, P), 3), "balAcc 95% CI": ci(name, 1)}
        for name, P in arms.items()]).to_string(index=False))

    print("\nvs the equal-weight mean, paired bootstrap:")
    for name in ["WSI alone", "CNV alone", *RULES[1:]]:
        cells = []
        for i, lab in enumerate(("dAUROC", "dBalAcc")):
            mean_d, lo, hi, verdict = delta_ci(escore[name][:, i], escore["mean"][:, i])
            cells.append(f"{lab} {mean_d:+.4f} [{lo:+.4f},{hi:+.4f}] {verdict:3s}")
        print(f"  {name:15s} " + " | ".join(cells))

    print("\nRules past `scalar` can recalibrate as well as reweight, so a gain there is not by\n"
          "itself evidence of better modality combination. `per_class` converges to a degenerate\n"
          "Her2 weight of 1.0 (pure WSI, zero CNV) at the sigmoid boundary -- read it as a\n"
          "boundary solution, not a learned preference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
