#!/usr/bin/env python3
"""Score the five trained fusion operators on CPTAC against WSI alone, CNV alone and their mean.

    python tools/score_cptac_fusion_ladder.py
    python tools/score_cptac_fusion_ladder.py --out-dir /tmp/scratch

Task T1 of ``docs/implementation-research/multimodality-trained-fusion-plan.md``: the five
``pam50_wsi_cnv_*_s1`` ladder checkpoints have never been scored externally, so it is unknown
whether trained fusion inherits the WSI arm's HER2 collapse (0/14 on CPTAC) or escapes it the way
the untrained probability mean partly does (6/14). This script answers that from prediction CSVs
already on disk; it touches no slide and trains nothing except the CNV arm, which is fit on TCGA.

Nothing here is tuned, calibrated or selected on CPTAC. The three reference arms are exactly the
ones ``tools/evaluate_cnv_wsi_fusion.py --external`` and ``tools/evaluate_pam50_fusion.ipynb``
(cell 11) report, recomputed the same way from the same files, so this table's ``WSI`` / ``CNV`` /
``Mean`` rows are an anchor: they must read 0.8465 / 0.8883 / 0.9093 with Her2 recall 0/14, 12/14,
6/14 and phi(WSI, CNV) = -0.0059. If they do not, something upstream moved and the operator rows
mean nothing.

CLAUDE.md reporting rules, applied here rather than left to the reader:

1. ``CNV`` is a row of the table and every operator carries a delta against it, not only against
   the mean. Fusion's edge over CNV alone is the marginal comparison.
2. ``Mean`` -- the untrained equal-weight probability mean -- is the baseline every operator is
   contrasted with. The WSI-only arm is not the bar.
4. The protocol is named in the output: paired bootstrap, N=4000 resamples of the same 114 cases
   at seed 7, the frozen external constants of ``tools/evaluate_cnv_wsi_fusion.py``.

The diversity diagnostic is the second half of the answer. Internally the five operators make
correlated mistakes (phi 0.656 among themselves against 0.193 between the two unimodal arms),
which is why ensembling them buys nothing. Externally the two unimodal arms are essentially
independent (phi -0.006); an operator that has genuinely learned to condition H&E on copy number
should sit closer to that than to the WSI arm it was warm-started from.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# `tools.pam50_arms`, not a bare `pam50_arms`: the bare form only resolves when this script's own
# directory is on sys.path, so `dp-analysis` could not import it at all. Same rationale as
# tools/evaluate_cnv_wsi_fusion.py.
try:
    from tools.pam50_arms import (CLASSES, balanced_acc, bootstrap_indices, cnv_arm, delta_ci,
                                  load_cptac_arms, load_cptac_wsi_probs, load_tcga_arms,
                                  macro_auroc)
    from tools import evaluate_cnv_wsi_fusion, pam50_arms
except ModuleNotFoundError as exc:  # pragma: no cover - install error, not a code path
    raise ModuleNotFoundError(
        f"{exc}. Run `pip install -e .` from the repository root once; after that this script "
        "runs from any working directory."
    ) from exc

# --- Frozen numerical constants ------------------------------------------------------------
# The external half of docs/cnv-wsi-fusion-external-validation.md was produced at these values and
# so were the notebook's Table 4 rows. They are imported-in-spirit rather than literally so this
# file states its own contract, but they must equal evaluate_cnv_wsi_fusion.N_BOOT /
# BOOTSTRAP_SEED; the assertion below is what stops the two drifting apart.
N_BOOT = 4000
BOOTSTRAP_SEED = 7

if (N_BOOT, BOOTSTRAP_SEED) != (evaluate_cnv_wsi_fusion.N_BOOT,
                                evaluate_cnv_wsi_fusion.BOOTSTRAP_SEED):  # pragma: no cover
    raise AssertionError(
        f"external bootstrap constants have drifted: this module says "
        f"{(N_BOOT, BOOTSTRAP_SEED)}, tools/evaluate_cnv_wsi_fusion.py says "
        f"{(evaluate_cnv_wsi_fusion.N_BOOT, evaluate_cnv_wsi_fusion.BOOTSTRAP_SEED)}. The "
        "operator rows would no longer be comparable with the published external table.")

#: The five trained operators, in the order the ladder was run and is reported.
OPERATORS = ("concat", "gated", "cross_attention", "film_attention", "coattn")

#: Baseline WSI arm and per-operator prediction directories, relative to `--predictions-root`.
WSI_SUBDIR = "predictions"
OPERATOR_SUBDIR = "predictions_cnv_fusion_{op}"
PREDICTIONS_CSV = "ensemble_predictions.csv"

#: Reference arms, in the order they are printed. `Mean` is the baseline (rule 2) and `CNV` is
#: reported beside every fusion number (rule 1).
WSI, CNV, MEAN = "WSI", "CNV", "Mean"


def default_predictions_root() -> Path:
    """`.scratch/cptac_validation/results`, resolved the way tools/pam50_arms.py resolves it.

    Derived from `CPTAC_WSI_PROBS` rather than spelled again, so `DP_SCRATCH_ROOT` and
    `dpcode/conf/paths/default.yaml` keep working and the WSI arm this script reads is provably
    the same file every other CNV-thread script reads.
    """
    return pam50_arms.CPTAC_WSI_PROBS.parent.parent


def infer_command(op: str, root: Path) -> str:
    """The invocation that produces one operator's prediction CSV."""
    return ("python tools/cptac/infer_cptac_multimodal.py "
            f"--ckpt_dir .scratch/results/pam50_wsi_cnv_{op}_s1 --fusion_mode {op} "
            "--tabular_csv .scratch/cnv-tabular/CPTAC_BRCA_CNV_arm_4class_clam.csv "
            f"--output_dir {root / OPERATOR_SUBDIR.format(op=op)}")


def pool_slides(path: Path) -> tuple[pd.DataFrame, int]:
    """Slide-level probabilities -> per-case means, in CLASSES column order.

    Selection is BY NAME. The prediction CSVs are written in CLAM's `label_dict` order
    (`p_LumA, p_LumB, p_Basal, p_Her2`) while everything scored here is in the sorted
    `pam50_arms.CLASSES` order; positional indexing would produce a plausible, wrong table.

    Identical to `pam50_arms.load_cptac_wsi_probs()`, which is that function applied to the one
    hard-coded WSI path -- and which is called directly for that path so the anchor rows are
    produced by the same code as the published ones.
    """
    slides = pd.read_csv(path)
    pooled = slides.groupby("case_id").agg(
        {**{f"p_{c}": "mean" for c in CLASSES}, "true_name": "first"})
    return pooled, len(slides)


def load_wsi_arm(root: Path) -> tuple[pd.DataFrame, int]:
    path = root / WSI_SUBDIR / PREDICTIONS_CSV
    if path == pam50_arms.CPTAC_WSI_PROBS:
        return load_cptac_wsi_probs()
    return pool_slides(path)


def align_operator(pooled: pd.DataFrame, cases: list[str], y_all: np.ndarray, op: str,
                   path: Path) -> np.ndarray:
    """Probabilities for `cases`, in CLASSES order, with the ground truth cross-checked.

    A missing case or a disagreeing `true_name` means the operator was inferred over a different
    manifest than the WSI arm, which would make every delta below a comparison of two different
    case sets wearing one name.
    """
    missing = [c for c in cases if c not in pooled.index]
    if missing:
        raise ValueError(
            f"{path} covers {len(pooled)} cases but is missing {len(missing)} the WSI arm has "
            f"(first few: {missing[:5]}). The operator was inferred over a different manifest; "
            f"re-run:\n    {infer_command(op, path.parent.parent)}")
    aligned = pooled.loc[cases]
    disagree = aligned["true_name"].values != y_all
    if disagree.any():
        bad = np.asarray(cases)[disagree][:5]
        raise ValueError(f"{path}: ground truth disagrees with the WSI arm for "
                         f"{int(disagree.sum())} cases (first few: {list(bad)}).")
    return aligned[[f"p_{c}" for c in CLASSES]].values


def boot_scores(y: np.ndarray, models: dict, n_boot: int, seed: int) -> dict:
    """Macro AUROC and balanced accuracy per model, on ONE set of shared bootstrap resamples.

    The resample positions come from `bootstrap_indices(y, ...)`, which depends only on the label
    vector and the seed, so every contrast below is a paired difference over identical draws --
    and the draws are the same ones the published external table used, because `y` is the same
    114-case vector in the same order.
    """
    idx = bootstrap_indices(y, n_boot, seed)
    return {name: np.array([[macro_auroc(y[j], P[j]), balanced_acc(y[j], P[j])] for j in idx])
            for name, P in models.items()}


def recall_str(y: np.ndarray, P: np.ndarray, c: str) -> str:
    pred = CLASSES[P.argmax(1)]
    return f"{int(((pred == c) & (y == c)).sum())}/{int((y == c).sum())}"


def correctness(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Case-level boolean correctness at argmax, as floats, for the phi computation."""
    return (CLASSES[P.argmax(1)] == y).astype(float)


def build_rows(y: np.ndarray, models: dict, scores: dict, correct: dict) -> list[dict]:
    rows = []
    for name, P in models.items():
        lo, hi = np.percentile(scores[name][:, 0], [2.5, 97.5])
        row = {
            "model": name,
            "macro_auroc": round(float(macro_auroc(y, P)), 4),
            "auroc_lo": round(float(lo), 4),
            "auroc_hi": round(float(hi), 4),
            "bal_acc": round(float(balanced_acc(y, P)), 4),
            **{f"recall_{c}": recall_str(y, P, c) for c in CLASSES},
        }
        for ref in (MEAN, CNV):
            for k, metric in enumerate(("auroc", "balacc")):
                mean_d, lo_d, hi_d, verdict = delta_ci(scores[name][:, k], scores[ref][:, k])
                tag = f"d_{metric}_vs_{ref.lower()}"
                row[tag] = round(float(mean_d), 4)
                row[f"{tag}_lo"] = round(float(lo_d), 4)
                row[f"{tag}_hi"] = round(float(hi_d), 4)
                row[f"{tag}_sig"] = verdict
        row["phi_vs_cnv"] = round(float(np.corrcoef(correct[name], correct[CNV])[0, 1]), 4)
        rows.append(row)
    return rows


def phi_matrix(correct: dict) -> pd.DataFrame:
    names = list(correct)
    return pd.DataFrame(
        [[round(float(np.corrcoef(correct[a], correct[b])[0, 1]), 4) for b in names]
         for a in names], index=names, columns=names)


def print_report(rows: list[dict], phis: pd.DataFrame, n_cases: int, n_slides: int,
                 n_boot: int, seed: int) -> None:
    print(f"\nTable: External CPTAC, {n_cases} cases pooled from {n_slides} WSI slide rows")
    print("TCGA-trained throughout; nothing refit, tuned or thresholded on CPTAC")
    print("=" * 100)
    display = pd.DataFrame([
        {"model": r["model"],
         "AUROC": f"{r['macro_auroc']:.4f}",
         "95% CI": f"[{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}]",
         "balAcc": f"{r['bal_acc']:.4f}",
         **{c: r[f"recall_{c}"] for c in CLASSES},
         "phi vs CNV": f"{r['phi_vs_cnv']:+.4f}"}
        for r in rows])
    print(display.to_string(index=False))
    print("-" * 100)
    print(f"  paired bootstrap, {n_boot} resamples of the same {n_cases} cases, seed {seed}")
    print(f"  {'contrast':34s}{'dAUROC [95% CI]':30s}{'dBalAcc [95% CI]'}")
    for ref in (MEAN, CNV):
        for r in rows:
            if r["model"] == ref:
                continue
            cells = []
            for metric in ("auroc", "balacc"):
                tag = f"d_{metric}_vs_{ref.lower()}"
                cells.append(f"{r[tag]:+.4f} [{r[tag + '_lo']:+.4f},{r[tag + '_hi']:+.4f}] "
                             f"{r[tag + '_sig']:3s}")
            print(f"  {r['model'] + ' - ' + ref:34s}{cells[0]:30s}{cells[1]}")
        print()
    print("  the Mean block is the pre-specified baseline (CLAUDE.md rule 2); the CNV block is")
    print("  reported beside it because fusion's edge over CNV alone is the marginal one (rule 1)")
    print("=" * 100)

    print("\nError correlation phi (case-level correctness; 0 = independent mistakes)")
    print("-" * 100)
    print(phis.to_string())
    ops = [n for n in phis.index if n not in (WSI, CNV, MEAN)]
    if len(ops) > 1:
        vals = [phis.loc[a, b] for i, a in enumerate(ops) for b in ops[i + 1:]]
        print(f"\n  among the {len(ops)} operators:  mean {np.mean(vals):+.4f} "
              f"(min {min(vals):+.4f}, max {max(vals):+.4f})")
    print(f"  WSI vs CNV (the two independently trained arms): {phis.loc[WSI, CNV]:+.4f}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-root", type=Path, default=None,
                    help="directory holding predictions/ and predictions_cnv_fusion_<op>/ "
                         "(default: the resolved .scratch/cptac_validation/results)")
    ap.add_argument("--operators", nargs="*", default=list(OPERATORS),
                    help="trained operators to score (default: %(default)s)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write cptac_fusion_ladder.csv, phi_matrix.csv and "
                         "cptac_fusion_ladder.json here (default: print only)")
    frozen = ap.add_argument_group(
        "frozen numerical constants",
        "The external bootstrap settings of tools/evaluate_cnv_wsi_fusion.py. Changing one makes "
        "this table incomparable with the published external table.")
    frozen.add_argument("--n-boot", type=int, default=N_BOOT,
                        help="paired-bootstrap resamples (default: %(default)s)")
    frozen.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED,
                        help="RNG seed for the bootstrap resamples (default: %(default)s)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = args.predictions_root or default_predictions_root()

    # Everything is checked before anything is computed: fitting the CNV arm and pooling 378
    # slide rows only to die on the fifth operator would waste the run and leave a half-written
    # out-dir behind.
    inputs = {WSI: root / WSI_SUBDIR / PREDICTIONS_CSV}
    inputs.update({op: root / OPERATOR_SUBDIR.format(op=op) / PREDICTIONS_CSV
                   for op in args.operators})
    problems = []
    for name, path in inputs.items():
        if not path.exists():
            produced_by = ("dp-cptac phase=3" if name == WSI else infer_command(name, root))
            problems.append(f"missing {path}\n      CPTAC slide-level probabilities for "
                            f"`{name}`\n      produced by: {produced_by}")
    for path in (pam50_arms.CPTAC_ARMS, pam50_arms.TCGA_ARMS, pam50_arms.TCGA_LABELS):
        if not path.exists():
            problems.append(f"missing {path}\n      arm-level CNV / labels\n"
                            f"      produced by: dp-data cnv")
    if problems:
        print("score_cptac_fusion_ladder needs inputs that are not on this machine; "
              "nothing was computed:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    wsi, n_slides = load_wsi_arm(root)
    X_tcga, y_tcga = load_tcga_arms()
    X_cptac = load_cptac_arms()

    # Case set and ordering exactly as tools/evaluate_cnv_wsi_fusion.external() and notebook cell
    # 11 build them: the sorted WSI-and-CNV intersection, then non-CLASSES labels dropped. `y`
    # decides the bootstrap draws, so this order is what makes the CIs the published ones.
    cases = sorted(set(wsi.index) & set(X_cptac.index))
    y_all = wsi.loc[cases, "true_name"].values
    Pw_all = wsi.loc[cases, [f"p_{c}" for c in CLASSES]].values
    Pc_all = cnv_arm().fit(X_tcga, y_tcga).predict_proba(X_cptac.loc[cases, X_tcga.columns])
    operator_probs = {op: align_operator(pool_slides(inputs[op])[0], cases, y_all, op, inputs[op])
                      for op in args.operators}

    keep = np.isin(y_all, CLASSES)
    y = y_all[keep]
    Pw, Pc = Pw_all[keep], Pc_all[keep]
    models = {WSI: Pw, CNV: Pc, MEAN: (Pw + Pc) / 2,
              **{op: P[keep] for op, P in operator_probs.items()}}

    print(f"predictions root: {root}")
    print(f"external set: {len(y)} cases from {n_slides} WSI slide rows  "
          f"{dict(pd.Series(y).value_counts().reindex(CLASSES))}")
    print(f"CNV arm fit on {len(y_tcga)} TCGA cases x {X_tcga.shape[1]} arms, applied unrefit")
    print(f"operators scored: {', '.join(args.operators)}")

    scores = boot_scores(y, models, args.n_boot, args.bootstrap_seed)
    correct = {name: correctness(y, P) for name, P in models.items()}
    rows = build_rows(y, models, scores, correct)
    phis = phi_matrix(correct)
    print_report(rows, phis, len(y), n_slides, args.n_boot, args.bootstrap_seed)

    if args.out_dir is not None:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(rows)
        table.to_csv(out / "cptac_fusion_ladder.csv", index=False)
        phis.to_csv(out / "phi_matrix.csv")
        payload = {
            "n_cases": int(len(y)),
            "n_slides": int(n_slides),
            "n_boot": int(args.n_boot),
            "seed": int(args.bootstrap_seed),
            "predictions_root": str(root),
            "operators": list(args.operators),
            "baseline": MEAN,
            "class_order": list(CLASSES),
            "class_counts": {c: int((y == c).sum()) for c in CLASSES},
            "models": rows,
            "phi_matrix": phis.to_dict(),
        }
        (out / "cptac_fusion_ladder.json").write_text(
            json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
        print(f"\nwrote {out / 'cptac_fusion_ladder.csv'}")
        print(f"wrote {out / 'phi_matrix.csv'}")
        print(f"wrote {out / 'cptac_fusion_ladder.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
