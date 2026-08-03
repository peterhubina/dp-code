#!/usr/bin/env python3
"""Do the harmonised clinicopathological variables predict PAM50 subtype?

Asked separately of TCGA-BRCA and CPTAC-BRCA, and across the two.

The question matters because the fusion head has a tabular branch, and if that
branch carries subtype signal on its own then a WSI+clinicopath fusion gain is
not necessarily a statement about morphology. The harmonisation analysis
(docs/implementation-research/tcga-cptac-clinicopath-harmonisation.md) fixed
which variables exist in both cohorts; this script asks whether they carry the
label.

Four feature blocks, deliberately separated:

  A_harmonised   age, pN, LN+, histology, race -- the block that survives
                 harmonisation, and the only one usable for TCGA -> CPTAC
                 transfer without re-fitting.
  B_clinicopath  A + stage + pT. Internal-only: CPTAC enrolled stage IIA-IIIC by
                 protocol, so these two are eligibility-truncated there.
  C_A_plus_IHC   A + ER/PR/HER2 IHC. Reported as a CEILING, not as a usable
                 model -- IHC receptor status IS the clinical surrogate of PAM50,
                 so this block is definitionally circular.
  D_IHC_only     ER/PR/HER2 alone, to show how much of block C is just the
                 surrogate.

Metrics are macro one-vs-rest AUROC and balanced accuracy over pooled
out-of-fold predictions, repeated over several CV seeds because n is small
(TCGA ~900 cases, CPTAC 114) and a single split is not a measurement.

Usage:
    python tools/analyse_clinicopath_pam50.py
    python tools/analyse_clinicopath_pam50.py --n_repeats 10 --n_perm 500
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TCGA_CLIN = ROOT / ".scratch/harmonisation/tcga_brca_harmonised_clinicopath.csv"
CPTAC_CLIN = ROOT / ".scratch/harmonisation/cptac_brca_harmonised_clinicopath.csv"
TCGA_PAM50 = ROOT / "tools/data/tcga_brca_pam50_labels.csv"

# CLAM's ordering for tcga_brca_subtyping (main.py:348) -- kept identical so the
# WSI out-of-fold probability columns line up without a remap.
CLASSES = ["LumA", "LumB", "Basal", "Her2"]

# Fixed category vocabularies. Declared rather than inferred so the TCGA and
# CPTAC design matrices have byte-identical columns -- a category absent from
# CPTAC (race "Other", stage IV) must still occupy its column, or the fitted
# coefficients silently shift by one position at transfer time.
CATS = {
    "race": ["White", "Black", "Asian", "Other"],
    "stage": ["I", "II", "III", "IV"],
    "pT": ["T1", "T2", "T3", "T4"],
    "pN": ["N0", "N1", "N2", "N3"],
    "LNpos": ["0", "1-3", "4+"],
    "histology": ["ductal", "lobular", "mixed", "other"],
    "ER": ["Positive", "Negative"],
    "PR": ["Positive", "Negative"],
    "HER2": ["Positive", "Negative"],
}
NUMERIC = ["age"]

BLOCKS = {
    "A_harmonised": ["age", "pN", "LNpos", "histology", "race"],
    "B_clinicopath": ["age", "pN", "LNpos", "histology", "race", "stage", "pT"],
    "C_A_plus_IHC": ["age", "pN", "LNpos", "histology", "race", "ER", "PR", "HER2"],
    "D_IHC_only": ["ER", "PR", "HER2"],
}
BLOCK_NOTE = {
    "A_harmonised": "harmonised, non-leaking -- the transferable block",
    "B_clinicopath": "A + stage/pT (internal only; CPTAC stage-truncated)",
    "C_A_plus_IHC": "CEILING, circular: IHC is the clinical surrogate of PAM50",
    "D_IHC_only": "the surrogate alone",
}


# ---- data ------------------------------------------------------------------ #
def load_cohorts():
    """Both cohorts on the 4-class PAM50 target the WSI model was trained for."""
    tcga = pd.read_csv(TCGA_CLIN)
    # The harmonised table carries Xena PAM50Call_RNAseq (826 cases); the project
    # label file covers 981 and is what the CLAM model was trained on. Use the
    # latter so these numbers sit next to the WSI numbers on the same labels.
    lab = pd.read_csv(TCGA_PAM50).rename(columns={"label": "PAM50"})
    tcga = tcga.drop(columns=["PAM50"]).merge(lab, on="case_id", how="inner")
    cptac = pd.read_csv(CPTAC_CLIN)

    out = {}
    for tag, df in [("TCGA", tcga), ("CPTAC", cptac)]:
        n_all = len(df)
        n_norm = int((df["PAM50"] == "Normal").sum())
        df = df[df["PAM50"].isin(CLASSES)].reset_index(drop=True)
        print(f"{tag}: {len(df)} cases with 4-class PAM50 "
              f"(from {n_all}; dropped {n_norm} Normal-like)")
        print("   " + ", ".join(f"{c}={int((df.PAM50 == c).sum())}" for c in CLASSES))
        out[tag] = df
    return out["TCGA"], out["CPTAC"]


def design_matrix(df: pd.DataFrame, block: list) -> pd.DataFrame:
    """One-hot with an explicit 'unknown' level per categorical.

    Missingness is encoded, never dropped: complete-case filtering would cost
    TCGA 185 cases and CPTAC 30 on the LN+ column alone, and 'not recorded' is
    itself informative in these registries.
    """
    cols = {}
    for f in block:
        if f in NUMERIC:
            cols[f] = df[f].astype(float)
        else:
            v = df[f].astype("object").where(df[f].notna(), "unknown")
            for level in CATS[f] + ["unknown"]:
                cols[f"{f}={level}"] = (v == level).astype(float)
    X = pd.DataFrame(cols, index=df.index)
    # A level nobody has (CPTAC race "Other") is an all-zero column: keep it so
    # the matrices stay aligned, but it contributes nothing.
    return X


def y_vector(df):
    return df["PAM50"].map({c: i for i, c in enumerate(CLASSES)}).values


# ---- part 1: univariate ---------------------------------------------------- #
def cramers_v(table: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(table)[0]
    n = table.sum()
    r, k = table.shape
    # Bergsma-Wicher bias correction -- matters here because several cells are
    # small (Her2 n=78 in TCGA, n=14 in CPTAC) and raw V is upward-biased.
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else np.nan


def univariate(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows = []
    y = df["PAM50"]
    for f in ["age"] + list(CATS):
        sub = df[df[f].notna()]
        if len(sub) < 20 or sub["PAM50"].nunique() < 2:
            continue
        if f == "age":
            groups = [g["age"].values for _, g in sub.groupby("PAM50")]
            H, p = stats.kruskal(*groups)
            # epsilon-squared: H rescaled to [0,1], the rank-based analogue of eta^2
            n = len(sub)
            eff = (H - len(groups) + 1) / (n - len(groups))
            rows.append({"variable": f, "test": "Kruskal-Wallis", "n": n,
                         "stat": H, "p": p, "effect": max(0.0, eff),
                         "effect_name": "epsilon^2"})
        else:
            tab = pd.crosstab(sub[f], sub["PAM50"])
            tab = tab.loc[tab.sum(1) > 0, tab.sum(0) > 0]
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                continue
            chi2, p, _, _ = stats.chi2_contingency(tab.values)
            rows.append({"variable": f, "test": "chi-square", "n": int(tab.values.sum()),
                         "stat": chi2, "p": p, "effect": cramers_v(tab.values),
                         "effect_name": "Cramer's V"})
    out = pd.DataFrame(rows)
    # BH-FDR across the variables tested within this cohort
    order = np.argsort(out["p"].values)
    ranks = np.empty(len(out), int)
    ranks[order] = np.arange(1, len(out) + 1)
    q = out["p"].values * len(out) / ranks
    out["q_bh"] = np.minimum.accumulate(q[order][::-1])[::-1][np.argsort(order)]
    out["cohort"] = cohort
    return out.sort_values("p").reset_index(drop=True)


def class_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive per-subtype table -- what a univariate p-value actually means."""
    rows = []
    for c in CLASSES:
        g = df[df["PAM50"] == c]
        r = {"PAM50": c, "n": len(g), "age_median": g["age"].median()}
        for f, levels in [("histology", ["ductal", "lobular"]), ("pN", ["N0"]),
                          ("LNpos", ["0"]), ("ER", ["Positive"]),
                          ("PR", ["Positive"]), ("HER2", ["Positive"])]:
            known = g[g[f].notna()]
            for lv in levels:
                r[f"{f}={lv} %"] = (100 * (known[f] == lv).mean()) if len(known) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ---- part 2: multivariable ------------------------------------------------- #
def make_model(kind: str, seed: int, C: float = 1.0):
    if kind == "logreg":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=5000, C=C,
                                                    random_state=seed))])
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                          max_leaf_nodes=15, l2_regularization=1.0,
                                          early_stopping=False, random_state=seed)


def oof_predict(X, y, kind, seed, n_splits, C=1.0):
    """Pooled out-of-fold probabilities from one stratified CV pass."""
    P = np.zeros((len(y), len(CLASSES)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = make_model(kind, seed, C)
        m.fit(X.iloc[tr], y[tr])
        # A fold can miss a rare class; map columns back by the classes it saw.
        p = m.predict_proba(X.iloc[te])
        seen = m.classes_ if hasattr(m, "classes_") else m.named_steps["clf"].classes_
        for j, cls in enumerate(seen):
            P[te, cls] = p[:, j]
    return P


def score(y, P):
    return {
        "macro_auroc": roc_auc_score(y, P, multi_class="ovr", average="macro"),
        "weighted_auroc": roc_auc_score(y, P, multi_class="ovr", average="weighted"),
        "balanced_acc": balanced_accuracy_score(y, P.argmax(1)),
        **{f"auroc_{c}": roc_auc_score((y == i).astype(int), P[:, i])
           for i, c in enumerate(CLASSES)},
    }


def cv_block(df, block_name, kind, n_repeats, n_splits):
    X = design_matrix(df, BLOCKS[block_name])
    y = y_vector(df)
    per_repeat = [score(y, oof_predict(X, y, kind, seed=100 + r, n_splits=n_splits))
                  for r in range(n_repeats)]
    s = pd.DataFrame(per_repeat)
    row = {"block": block_name, "model": kind, "n": len(y), "n_features": X.shape[1]}
    for c in s.columns:
        row[f"{c}_mean"] = s[c].mean()
        row[f"{c}_sd"] = s[c].std(ddof=1)
    return row, X, y


def permutation_null(X, y, kind, n_perm, n_splits, seed=0):
    """Empirical null for macro AUROC under label permutation.

    The right null here, not 0.5: with 4 imbalanced classes and one-hot features,
    the pooled-OOF macro AUROC of a fitted model on random labels is not exactly
    0.5, and small-sample optimism would otherwise be read as signal.
    """
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        null.append(score(yp, oof_predict(X, yp, kind, seed=7, n_splits=n_splits))["macro_auroc"])
    return np.array(null)


def boot_ci(y, P, fn, n_boot=2000, seed=1):
    """Stratified bootstrap CI over cases."""
    rng = np.random.default_rng(seed)
    idx_by_class = [np.where(y == i)[0] for i in range(len(CLASSES))]
    vals = []
    for _ in range(n_boot):
        take = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_class])
        if len(np.unique(y[take])) < len(CLASSES):
            continue
        try:
            vals.append(fn(y[take], P[take]))
        except ValueError:
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---- part 4: cross-cohort -------------------------------------------------- #
def transfer(tcga, cptac, block_name, kind):
    Xtr = design_matrix(tcga, BLOCKS[block_name])
    Xte = design_matrix(cptac, BLOCKS[block_name])
    assert list(Xtr.columns) == list(Xte.columns), "design matrices misaligned"
    ytr, yte = y_vector(tcga), y_vector(cptac)
    m = make_model(kind, 0)
    m.fit(Xtr, ytr)
    P = m.predict_proba(Xte)
    s = score(yte, P)
    lo, hi = boot_ci(yte, P, lambda a, b: roc_auc_score(a, b, multi_class="ovr",
                                                        average="macro"))
    s.update({"block": block_name, "model": kind, "n_train": len(ytr),
              "n_test": len(yte), "macro_auroc_lo": lo, "macro_auroc_hi": hi})
    return s


# ---- part 5: incremental value over the WSI model -------------------------- #
def load_wsi_oof(results_dir: Path, k: int = 10) -> pd.DataFrame:
    """Case-level out-of-fold probabilities from the trained CLAM-MB PAM50 run."""
    import pickle
    rows = []
    for fold in range(k):
        f = results_dir / f"split_{fold}_results.pkl"
        if not f.is_file():
            return pd.DataFrame()
        for slide_id, rec in pickle.load(open(f, "rb")).items():
            rows.append({"slide_id": str(slide_id), "fold": fold,
                         "label": int(rec["label"]),
                         **{f"wsi_p{i}": float(rec["prob"].ravel()[i])
                            for i in range(len(CLASSES))}})
    df = pd.DataFrame(rows)
    df["case_id"] = df["slide_id"].str.slice(0, 12)
    pcols = [f"wsi_p{i}" for i in range(len(CLASSES))]
    # These 10 CLAM test splits are NOT a partition -- create_splits_seq.py draws an
    # independent test fraction per fold, so 260/643 slides recur across folds and the
    # union covers only the cases that were sampled into some test set at least once.
    # Every contributing prediction is still genuinely held out for the model that made
    # it, so averaging them is sound; the coverage is just narrower than 10-fold implies.
    n_folds_per_case = df.groupby("case_id")["fold"].nunique()
    print(f"   WSI out-of-fold: {len(df)} slide predictions -> {df.slide_id.nunique()} "
          f"unique slides -> {df.case_id.nunique()} unique cases")
    print(f"   cases evaluated by >1 fold model: {int((n_folds_per_case > 1).sum())} "
          f"(median {int(n_folds_per_case.median())} models/case)")
    # Mean-softmax over a case's slides, matching how the WSI model is reported.
    case = df.groupby("case_id").agg({**{c: "mean" for c in pcols},
                                      "label": "first"}).reset_index()
    case[pcols] = case[pcols].div(case[pcols].sum(1), axis=0)
    return case


def incremental(tcga, wsi, kind, n_splits=10, n_repeats=5):
    """Does clinicopath add anything on top of what the image already predicts?

    Caveat worth stating: the WSI probabilities are out-of-fold, but a stacker
    trained on other folds still sees probabilities produced by models that saw
    the held-out fold's cases in *their* training data. That biases the WSI-only
    stacker slightly upward. It biases both arms the same way, so the DELTA --
    which is the quantity of interest -- is the defensible number here, not the
    absolute level.
    """
    m = tcga.merge(wsi, on="case_id", how="inner")
    if m.empty:
        return None
    pcols = [f"wsi_p{i}" for i in range(len(CLASSES))]
    assert (m["label"].values == y_vector(m)).all(), "WSI label order disagrees with PAM50 map"
    y = y_vector(m)
    logit = pd.DataFrame(np.log(np.clip(m[pcols].values, 1e-6, 1)),
                         columns=[f"wsi_lp{i}" for i in range(len(CLASSES))], index=m.index)
    Xc = design_matrix(m, BLOCKS["A_harmonised"])

    arms = {"WSI only": logit,
            "WSI + clinicopath (A)": pd.concat([logit, Xc], axis=1),
            "clinicopath (A) only": Xc}
    # Block A costs the stacker ~20 extra columns on ~600 cases, so a negative delta
    # at one penalty could be nothing but the parameter count. Sweeping C separates
    # "the variables carry nothing" from "the model was allowed to overfit them".
    idx_by_class = [np.where(y == i)[0] for i in range(len(CLASSES))]
    out, deltas = [], []
    for C in (0.05, 0.2, 1.0):
        preds = {}
        for name, X in arms.items():
            per = []
            for r in range(n_repeats):
                P = oof_predict(X, y, kind, seed=100 + r, n_splits=n_splits, C=C)
                per.append(score(y, P))
                if r == 0:
                    preds[name] = P
            s = pd.DataFrame(per)
            out.append({"C": C, "arm": name, "n": len(y),
                        "macro_auroc_mean": s.macro_auroc.mean(),
                        "macro_auroc_sd": s.macro_auroc.std(ddof=1),
                        "balanced_acc_mean": s.balanced_acc.mean(),
                        "balanced_acc_sd": s.balanced_acc.std(ddof=1)})

        # Paired bootstrap on the delta, same resampled cases for both arms.
        rng = np.random.default_rng(1)
        a, b = preds["WSI + clinicopath (A)"], preds["WSI only"]
        d = []
        for _ in range(2000):
            take = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_class])
            if len(np.unique(y[take])) < len(CLASSES):
                continue
            f = lambda P: roc_auc_score(y[take], P[take], multi_class="ovr", average="macro")
            d.append(f(a) - f(b))
        d = np.array(d)
        deltas.append({"C": C, "delta_macro_auroc": float(np.mean(d)),
                       "ci_lo": float(np.percentile(d, 2.5)),
                       "ci_hi": float(np.percentile(d, 97.5)),
                       "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))})
    return pd.DataFrame(out), pd.DataFrame(deltas), len(y)


# ---- main ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--n_splits", type=int, default=10)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--wsi_results", default=".scratch/results/pam50_final_s1")
    ap.add_argument("--out_dir", default=".scratch/analysis/clinicopath_pam50")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tcga, cptac = load_cohorts()

    # -- 1. univariate
    print("\n" + "=" * 78 + "\n1. UNIVARIATE ASSOCIATION WITH PAM50\n" + "=" * 78)
    uni = pd.concat([univariate(tcga, "TCGA"), univariate(cptac, "CPTAC")])
    uni.to_csv(out_dir / "univariate.csv", index=False)
    for c in ["TCGA", "CPTAC"]:
        print(f"\n--- {c}")
        print(uni[uni.cohort == c][["variable", "test", "n", "stat", "p", "q_bh",
                                    "effect", "effect_name"]]
              .round({"stat": 2, "p": 5, "q_bh": 4, "effect": 3}).to_string(index=False))
    prof = pd.concat([class_profiles(tcga).assign(cohort="TCGA"),
                      class_profiles(cptac).assign(cohort="CPTAC")])
    prof.to_csv(out_dir / "class_profiles.csv", index=False)
    print("\n--- per-subtype profile")
    print(prof.round(1).to_string(index=False))

    # -- 2. multivariable CV, per cohort
    print("\n" + "=" * 78 +
          f"\n2. MULTIVARIABLE CV ({args.n_splits}-fold x {args.n_repeats} repeats)\n"
          + "=" * 78)
    cv_rows, mats = [], {}
    for cohort, df in [("TCGA", tcga), ("CPTAC", cptac)]:
        for block in BLOCKS:
            for kind in ["logreg", "hgb"]:
                row, X, y = cv_block(df, block, kind, args.n_repeats, args.n_splits)
                row["cohort"] = cohort
                cv_rows.append(row)
                mats[(cohort, block)] = (X, y)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(out_dir / "cv_metrics.csv", index=False)
    for cohort in ["TCGA", "CPTAC"]:
        print(f"\n--- {cohort} (chance macro AUROC = 0.500, chance balanced acc = 0.250)")
        s = cv[cv.cohort == cohort][["block", "model", "n", "macro_auroc_mean",
                                     "macro_auroc_sd", "balanced_acc_mean",
                                     "balanced_acc_sd"] +
                                    [f"auroc_{c}_mean" for c in CLASSES]]
        print(s.round(3).to_string(index=False))

    # -- 3. permutation null for the transferable block
    print("\n" + "=" * 78 + f"\n3. PERMUTATION NULL, block A ({args.n_perm} permutations)\n"
          + "=" * 78)
    perm_rows = []
    for cohort in ["TCGA", "CPTAC"]:
        X, y = mats[(cohort, "A_harmonised")]
        obs = score(y, oof_predict(X, y, "logreg", 7, args.n_splits))["macro_auroc"]
        null = permutation_null(X, y, "logreg", args.n_perm, args.n_splits)
        p = (1 + (null >= obs).sum()) / (1 + len(null))
        perm_rows.append({"cohort": cohort, "observed": obs, "null_mean": null.mean(),
                          "null_sd": null.std(ddof=1), "null_p95": np.percentile(null, 95),
                          "p_perm": p})
        print(f"{cohort}: observed {obs:.4f} | null {null.mean():.4f} "
              f"+/- {null.std(ddof=1):.4f} (95th pct {np.percentile(null, 95):.4f}) "
              f"| p = {p:.4f}")
    pd.DataFrame(perm_rows).to_csv(out_dir / "permutation.csv", index=False)

    # -- 4. cross-cohort transfer
    print("\n" + "=" * 78 + "\n4. TRANSFER: fit TCGA -> predict CPTAC\n" + "=" * 78)
    tr = pd.DataFrame([transfer(tcga, cptac, b, k)
                       for b in ["A_harmonised", "C_A_plus_IHC", "D_IHC_only"]
                       for k in ["logreg", "hgb"]])
    tr.to_csv(out_dir / "transfer.csv", index=False)
    print(tr[["block", "model", "n_train", "n_test", "macro_auroc", "macro_auroc_lo",
              "macro_auroc_hi", "balanced_acc"] +
             [f"auroc_{c}" for c in CLASSES]].round(3).to_string(index=False))

    # -- 5. incremental value over the WSI model
    print("\n" + "=" * 78 + "\n5. INCREMENTAL VALUE OVER THE WSI MODEL (TCGA)\n" + "=" * 78)
    wsi = load_wsi_oof(ROOT / args.wsi_results, args.n_splits)
    if wsi.empty:
        print(f"no WSI out-of-fold predictions under {args.wsi_results}; skipped")
    else:
        got = incremental(tcga, wsi, "logreg", args.n_splits, args.n_repeats)
        if got is None:
            print("no case overlap between WSI predictions and clinicopath; skipped")
        else:
            res, delta, n = got
            res.to_csv(out_dir / "incremental.csv", index=False)
            delta.to_csv(out_dir / "incremental_delta.csv", index=False)
            print(f"n = {n} cases with both WSI predictions and clinicopath\n")
            print(res.round(4).to_string(index=False))
            print("\ndelta macro AUROC (WSI + clinicopath - WSI only), paired bootstrap:")
            print(delta.round(4).to_string(index=False))

    print(f"\nArtefacts written to {out_dir}")


if __name__ == "__main__":
    main()
