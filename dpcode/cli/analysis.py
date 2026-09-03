"""`dp-analysis` — the CNV arm and the fusion analyses, with one configured surface.

    dp-analysis list
    dp-analysis cnv_wsi_fusion                       # TCGA -> CPTAC external validation
    dp-analysis cnv_wsi_fusion analysis.internal=true
    dp-analysis stack_wsi_cnv                        # can a learned rule beat the mean? No.
    dp-analysis cnv_controls                         # the controls the document reports
    dp-analysis make_cnv_tabular analysis.cohort=tcga
    dp-analysis compare_fusion_ladder                # subprocess; that script is user-owned
    dp-analysis cptac_fusion_ladder                  # the same ladder, externally on CPTAC

Every action is CPU-only, reads at most ~1 MB of CSVs plus the CLAM prediction pickles, and
touches no slide.

Four deliberate design points, each of which has a reason that is not style:

**Composition is programmatic, not `@hydra.main`.** The group is not in the primary config's
defaults list — `dpcode/conf/config.yaml` reserves the *package* but leaves the group unselected —
so selecting it needs `+analyses=<action>`, and `+` overrides are exactly what
:func:`dpcode.schema.reject_appended_overrides` exists to forbid. Composing here lets the entry
point select its own group while still rejecting `+`/`~` in anything the *user* typed.

**The group directory is `conf/analyses/`, the config key is `analysis`.** `.gitignore:74` is a
bare `analysis`, which matches a directory of that name at any depth, so `dpcode/conf/analysis/`
could never be committed and a fresh clone would fail to compose — the same trap DESIGN-ADDENDUM
A2 found for `wandb`, and the same remedy: `.gitignore` is left untouched and the directory is
renamed. Each option file carries `# @package analysis`, so every override is still spelled
`analysis.<key>=…` and `RootConf.analysis` is still the package that receives it.

**`--multirun` is refused.** These scripts resolve their filesystem constants once, at
`tools.pam50_arms` import time, and print a single table to stdout. Sweeping them in one process
would silently reuse the first job's paths, and sweeping them as separate processes buys nothing
over a shell loop.

**Path overrides that cannot take effect are refused too.** `tools/pam50_arms.py` resolves paths
through `dpcode.paths.resolve_paths`, which reads `dpcode/conf/paths/default.yaml` and the `DP_*`
environment variables but knows nothing about a command-line `paths.cnv_dir=...`. Rather than let
such an override look like it worked, :func:`assert_paths_reachable` compares the two and aborts,
naming the environment variable that does work.

The numbers these actions print are the thesis result. Every constant behind them lives in
`dpcode/conf/analyses/*.yaml` with the published value as its default, and per DESIGN-ADDENDUM A6
they are per-script rather than shared: `cnv_wsi_fusion` bootstraps with seed 7 and `stack_wsi_cnv`
with seed 11, and unifying those would change published confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .. import runinfo, schema
from ..paths import resolve_paths
from .config import compose_config

#: Config-group options under `dpcode/conf/analyses/`. The key is what the user types.
ACTIONS = (
    "cnv_wsi_fusion",
    "stack_wsi_cnv",
    "cnv_controls",
    "make_cnv_tabular",
    "compare_fusion_ladder",
    "cptac_fusion_ladder",
)

#: `paths.*` keys that `tools/pam50_arms.py` resolves for itself. Overriding one of these on the
#: command line cannot reach it, so :func:`assert_paths_reachable` refuses rather than pretend.
#: `analysis_dir` is deliberately absent: this module reads it from the composed config, so
#: overriding it works.
LIBRARY_RESOLVED_PATH_KEYS = (
    "repo_root",
    "cnv_dir",
    "labels_dir",
    "cptac_root",
    "dataset_csv_dir",
    "cptac_validation_dir",
    "results_root",
    "splits_root",
    "cnv_tabular_dir",
)


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #


def assert_paths_reachable(cfg: DictConfig) -> None:
    """Abort if a `paths.*` override would be silently ignored by the tools modules.

    `tools/pam50_arms.py` is plain library code: it calls `dpcode.paths.resolve_paths()` at import
    time and never sees a composed config. That is what makes `python tools/stack_wsi_cnv.py` and
    `dp-analysis stack_wsi_cnv` land on the same files — but it also means
    `dp-analysis stack_wsi_cnv paths.cnv_dir=/elsewhere` would compose cleanly, print a config
    that says `/elsewhere`, and read the original directory anyway.
    """
    library = resolve_paths()
    drifted = [
        f"paths.{key}={cfg.paths[key]!r} (the tools modules will use {library[key]!r})"
        for key in LIBRARY_RESOLVED_PATH_KEYS
        if str(cfg.paths[key]) != str(library[key])
    ]
    if drifted:
        raise ValueError(
            "These path overrides cannot reach the analysis scripts, which resolve paths at "
            "import time through dpcode.paths: " + "; ".join(drifted) + ". Set the environment "
            "variable instead (DP_REPO_ROOT / DP_DATA_ROOT / DP_SCRATCH_ROOT / DP_RESULTS_ROOT), "
            "or edit dpcode/conf/paths/default.yaml."
        )


def assert_class_order(cfg: DictConfig) -> None:
    """Pin each script's class order to the one recorded in its config.

    There are two orders in this repository and they are both correct: `tools/pam50_arms.CLASSES`
    is sorted (`Basal, Her2, LumA, LumB`) and CLAM's `label_dict` is not (`LumA, LumB, Basal,
    Her2`). `pam50_arms.clam_column_order()` is the only bridge, and it asserts the recovered map
    is a permutation. Nothing here unifies them — the config records both so that a silent
    reordering upstream (a new label table, a re-run of `pam50.R`) fails here instead of producing
    a plausible, wrong table.
    """
    if "class_order" in cfg.analysis:
        from tools import pam50_arms

        expected = list(cfg.analysis.class_order)
        actual = list(pam50_arms.CLASSES)
        if actual != expected:
            raise ValueError(
                f"analysis.class_order={expected} but tools.pam50_arms.CLASSES={actual}. "
                "Every probability matrix in the CNV thread is in the latter order."
            )
    if "clam_class_order" in cfg.analysis:
        from tools import make_cnv_tabular

        expected = list(cfg.analysis.clam_class_order)
        actual = list(make_cnv_tabular.CLASSES)
        if actual != expected:
            raise ValueError(
                f"analysis.clam_class_order={expected} but "
                f"tools.make_cnv_tabular.CLASSES={actual}. This must equal the `label_dict` "
                "order in project/CLAM/main.py for the tabular tables to line up with CLAM's "
                "integer labels."
            )


# --------------------------------------------------------------------------- #
# preconditions
# --------------------------------------------------------------------------- #


class Need(NamedTuple):
    """One input an action reads, and the command that produces it.

    `pattern` is `None` when `path` itself must exist, and a glob when `path` is a
    directory that must contain at least one match — the two cases that actually
    bite here, because a CLAM run directory that exists but holds no
    `split_*_results.pkl` is what produced the raw `KeyError: 'case_id'` these
    checks replace.
    """

    path: Path
    pattern: str | None
    what: str
    produced_by: str


def _needs(cfg: DictConfig, action: str) -> list[Need]:
    """Every input `action` reads, resolved through the same constants it will use.

    The paths come from `tools.pam50_arms`, which is where the CNV thread's
    filesystem contract is defined once, so this cannot describe a different file
    from the one the action opens a moment later.
    """
    from tools import pam50_arms as arms

    tcga_cnv = Need(arms.TCGA_ARMS, None, "arm-level CNV, TCGA (981 x 39)", "dp-data cnv")
    cptac_cnv = Need(arms.CPTAC_ARMS, None, "arm-level CNV, CPTAC (114 x 39)", "dp-data cnv")
    tcga_labels = Need(arms.TCGA_LABELS, None, "the TCGA PAM50 label table (git-tracked)",
                       "a complete checkout; `dp-data labels` re-fetches it")
    cptac_labels = Need(arms.CPTAC_LABELS, None, "the CPTAC PAM50 dataset table",
                        "dp-cptac phase=0")
    clam_manifest = Need(arms.CLAM_DATASET_CSV, None,
                         "CLAM's slide->PAM50 manifest (git-tracked, 0 writers)",
                         "a complete checkout — REPRODUCING.md B.3: never regenerate it")
    splits = Need(arms.CLAM_SPLITS, "splits_*.csv",
                  f"the {arms.SPLIT_SET} fold assignment (git-tracked, 0 writers)",
                  "a complete checkout — REPRODUCING.md B.3: never regenerate it")
    wsi_oof = Need(arms.CLAM_OOF, "split_*_results.pkl",
                   f"out-of-fold WSI probabilities from {arms.WSI_BASELINE_RUN}",
                   "dp-train experiment=pam50_wsi_final   (10 folds on one GPU)")
    cptac_wsi = Need(arms.CPTAC_WSI_PROBS, None,
                     "the external WSI arm (378 CPTAC slides -> 114 cases)",
                     "dp-cptac phase=all")

    if action == "cnv_wsi_fusion":
        needed = [cptac_wsi, cptac_cnv, tcga_cnv, tcga_labels]
        if bool(cfg.analysis.internal):
            needed.append(wsi_oof)
        return needed
    if action == "stack_wsi_cnv":
        return [wsi_oof, splits, tcga_cnv, tcga_labels, cptac_wsi, cptac_cnv]
    if action == "cnv_controls":
        return [tcga_cnv, tcga_labels, cptac_wsi, cptac_cnv, wsi_oof]
    if action == "make_cnv_tabular":
        cohort = str(cfg.analysis.cohort)
        needed = []
        if cohort in ("tcga", "both"):
            needed += [tcga_cnv, tcga_labels, clam_manifest, splits]
        if cohort in ("cptac", "both"):
            needed += [cptac_cnv, cptac_labels]
        return needed
    if action == "compare_fusion_ladder":
        # `tools/compare_fusion_ladder.py` is USER-OWNED and is not edited by this
        # refactor. Its `RESULTS` is its own literal — `Path(__file__).parent.parent
        # / ".scratch/results"` — rather than `paths.results_root`, so the check has
        # to mirror that literal or it would vouch for a directory the script never
        # opens. The five ladder arms are NOT required: the script prints
        # "(skipping <mode>: ... not found)" for each missing one and carries on.
        ladder_results = Path(str(cfg.paths.repo_root)) / ".scratch" / "results"
        return [
            Need(ladder_results / arms.WSI_BASELINE_RUN, "split_*_results.pkl",
                 "per-fold WSI-only predictions — the bar every ladder arm is read against",
                 "dp-train experiment=pam50_wsi_final   (10 folds on one GPU)"),
            splits, tcga_cnv, tcga_labels,
        ]
    if action == "cptac_fusion_ladder":
        # The external counterpart of the ladder. The WSI arm and the five operator prediction
        # CSVs are all slide-level tables written by the CPTAC inference path; the CNV arm is fit
        # here on TCGA, so `load_tcga_arms()`'s two inputs are needed as well. The CPTAC arm
        # matrix is what the CNV arm is *applied* to.
        #
        # The CPTAC tabular table is listed even though this action never opens it: it is the
        # `--tabular_csv` the five operator runs consumed, so without it those predictions cannot
        # be reproduced or extended, and naming it here is what points at
        # `dp-analysis make_cnv_tabular` instead of at a bare missing directory.
        root = (Path(str(cfg.analysis.predictions_root))
                if cfg.analysis.predictions_root is not None
                else arms.CPTAC_WSI_PROBS.parent.parent)
        needed = [
            Need(root / "predictions" / "ensemble_predictions.csv", None,
                 "the external WSI arm (378 CPTAC slides -> 114 cases)",
                 "dp-cptac phase=3"),
            Need(arms.CNV_TABULAR_DIR / "CPTAC_BRCA_CNV_arm_4class_clam.csv", None,
                 "the CPTAC tabular CNV table the operator inference was run against",
                 "dp-analysis make_cnv_tabular analysis.cohort=cptac"),
            cptac_cnv, tcga_cnv, tcga_labels,
        ]
        for op in [str(o) for o in cfg.analysis.operators]:
            needed.append(Need(
                root / f"predictions_cnv_fusion_{op}" / "ensemble_predictions.csv", None,
                f"CPTAC slide-level probabilities for the trained `{op}` operator",
                f"python tools/cptac/infer_cptac_multimodal.py --ckpt_dir "
                f".scratch/results/pam50_wsi_cnv_{op}_s1 --fusion_mode {op} --tabular_csv "
                f".scratch/cnv-tabular/CPTAC_BRCA_CNV_arm_4class_clam.csv --output_dir "
                f"{root / ('predictions_cnv_fusion_' + op)}"))
        return needed
    return []


def check_preconditions(cfg: DictConfig, action: str) -> list[str]:
    """Every missing input for `action`, described, before anything is created."""
    problems = []
    for need in _needs(cfg, action):
        if need.pattern is None:
            if not need.path.exists():
                problems.append(f"missing {need.path}\n      {need.what}\n"
                                f"      produced by: {need.produced_by}")
        elif not need.path.is_dir():
            problems.append(f"missing directory {need.path}\n      {need.what}\n"
                            f"      produced by: {need.produced_by}")
        elif not any(need.path.glob(need.pattern)):
            problems.append(f"no {need.pattern} in {need.path}\n      {need.what}\n"
                            f"      produced by: {need.produced_by}")
    return problems


# --------------------------------------------------------------------------- #
# actions
# --------------------------------------------------------------------------- #


def run_cnv_wsi_fusion(cfg: DictConfig, run_dir: Path | None = None) -> int:
    from tools import evaluate_cnv_wsi_fusion

    a = cfg.analysis
    argv = ["--n-boot", str(a.n_boot), "--bootstrap-seed", str(a.bootstrap_seed),
            "--cv-folds", str(a.cv_folds), "--cv-seed", str(a.cv_seed)]
    if a.internal:
        argv.insert(0, "--internal")
    _echo_argv("tools/evaluate_cnv_wsi_fusion.py", argv)
    return evaluate_cnv_wsi_fusion.main(argv)


def run_stack_wsi_cnv(cfg: DictConfig, run_dir: Path | None = None) -> int:
    from tools import stack_wsi_cnv

    a = cfg.analysis
    argv = ["--n-boot", str(a.n_boot), "--bootstrap-seed", str(a.bootstrap_seed),
            "--stacker-C", repr(float(a.stacker_C)),
            "--stacker-max-iter", str(a.stacker_max_iter),
            "--nm-xatol", repr(float(a.nm_xatol)), "--nm-fatol", repr(float(a.nm_fatol)),
            "--nm-maxiter", str(a.nm_maxiter), "--clip-floor", repr(float(a.clip_floor))]
    _echo_argv("tools/stack_wsi_cnv.py", argv)
    return stack_wsi_cnv.main(argv)


def run_make_cnv_tabular(cfg: DictConfig, run_dir: Path | None = None) -> int:
    from tools import make_cnv_tabular

    a = cfg.analysis
    out = a.out if a.out is not None else cfg.paths.cnv_tabular_dir
    argv = ["--cohort", str(a.cohort), "--out", str(out)]
    _echo_argv("tools/make_cnv_tabular.py", argv)
    return make_cnv_tabular.main(argv)


def run_compare_fusion_ladder(cfg: DictConfig, run_dir: Path | None = None) -> int:
    """Dispatch `tools/compare_fusion_ladder.py` as a subprocess.

    That script belongs to the thesis author and is not edited by this refactor. It already
    anchors every path on `Path(__file__)`, so it runs correctly from any working directory; it
    is launched rather than imported so its bare `from pam50_arms import …` keeps resolving the
    way it does today.
    """
    script = Path(str(cfg.paths.repo_root)) / "tools" / "compare_fusion_ladder.py"
    if not script.exists():
        raise FileNotFoundError(f"{script} does not exist")
    argv = [sys.executable, str(script), "--n-boot", str(cfg.analysis.n_boot)]
    print(f"$ {' '.join(argv)}\n", flush=True)
    # Streamed line by line through this process's `print` rather than inherited on fd 1: the run
    # directory's `output.txt` is written by a Python-level tee, and a child writing straight to
    # the file descriptor would leave the recorded output empty while the terminal looked fine.
    child = subprocess.Popen(argv, cwd=str(cfg.paths.repo_root), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert child.stdout is not None
    for line in child.stdout:
        print(line, end="")
    return child.wait()


def run_cptac_fusion_ladder(cfg: DictConfig, run_dir: Path | None = None) -> int:
    """Dispatch `tools/score_cptac_fusion_ladder.py` as a subprocess.

    A subprocess rather than an import for the same reason `compare_fusion_ladder` is one: the
    script is a standalone table producer with its own argparse contract, and dispatching it keeps
    `python tools/score_cptac_fusion_ladder.py` and this entry point provably the same run. Its
    output directory is the analysis run directory, so `cptac_fusion_ladder.csv`, `phi_matrix.csv`
    and `cptac_fusion_ladder.json` land beside `output.txt` and `config.resolved.yaml`.
    """
    script = Path(str(cfg.paths.repo_root)) / "tools" / "score_cptac_fusion_ladder.py"
    if not script.exists():
        raise FileNotFoundError(f"{script} does not exist")
    a = cfg.analysis
    argv = [sys.executable, str(script),
            "--n-boot", str(a.n_boot), "--bootstrap-seed", str(a.bootstrap_seed),
            "--operators", *[str(op) for op in a.operators]]
    if a.predictions_root is not None:
        argv += ["--predictions-root", str(a.predictions_root)]
    if run_dir is not None:
        argv += ["--out-dir", str(run_dir)]
    print(f"$ {' '.join(argv)}\n", flush=True)
    # Streamed line by line through this process's `print` rather than inherited on fd 1, so the
    # run directory's tee'd `output.txt` captures the table. See run_compare_fusion_ladder.
    child = subprocess.Popen(argv, cwd=str(cfg.paths.repo_root), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert child.stdout is not None
    for line in child.stdout:
        print(line, end="")
    return child.wait()


def _echo_argv(script: str, argv: Sequence[str]) -> None:
    print(f"$ python {script} {' '.join(argv)}\n", flush=True)


# --------------------------------------------------------------------------- #
# cnv_controls — the control numbers the headline document reports and no script produced
# --------------------------------------------------------------------------- #

#: Values as published in `docs/cnv-wsi-fusion-external-validation.md` (sections 3, 4 and 5) and
#: in the project memory note for the arm-level CNV baseline. They are here ONLY so this entry
#: point can say whether the code still produces them. Nothing reads them as an input and no
#: document is rewritten from them: a mismatch is reported, not resolved.
PUBLISHED = {
    "external_per_class_auroc": {
        "WSI raw": {"Basal": 0.972, "Her2": 0.860, "LumA": 0.861, "LumB": 0.693},
        "CNV (39 arms)": {"Basal": 0.972, "Her2": 0.871, "LumA": 0.883, "LumB": 0.827},
        "Fusion raw": {"Basal": 0.992, "Her2": 0.881, "LumA": 0.916, "LumB": 0.848},
    },
    "external_error_independence": {"phi": -0.006, "either_right": 0.912, "wsi_accuracy": 0.702},
    "internal_error_independence": {"phi": 0.269, "either_right": 0.863, "wsi_accuracy": 0.735},
    "internal_39_arms_multiseed": {"mean": 0.866, "std": 0.003},
    "aneuploidy_burden_only": 0.685,
    "regularisation_sweep": {0.01: 0.879, 0.1: 0.870, 1.0: 0.860, 10.0: 0.856},
    "leave_one_site_out": {"mean": 0.878, "std": 0.035, "n_sites": 13},
}


def run_cnv_controls(cfg: DictConfig, run_dir: Path | None = None) -> int:
    """Compute the controls `docs/cnv-wsi-fusion-external-validation.md` reports.

    Six of them had no producing script anywhere in the repository, which meant the one
    `CLAUDE.md` makes non-negotiable to report — aneuploidy burden alone — could not be checked
    or refreshed. `report()` in `evaluate_cnv_wsi_fusion.py` prints per-class *recall*, never
    per-class AUROC, so the per-class AUROC table had the same problem.

    Two protocols are in play and they are not interchangeable. The headline internal figure
    (0.866 ± 0.003) is 5-fold CV averaged over ten reseeds; the burden control, the regularisation
    sweep and the site holdout are single 5-fold runs at `single_seed`. Both are printed, and the
    one that corresponds to the published value is marked, because reading a ±0.003 multi-seed
    number next to a single-seed one is how a table quietly stops being reproducible.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from tools.pam50_arms import CLASSES, cnv_arm, load_tcga_arms

    a = cfg.analysis
    results: dict[str, Any] = {"protocol": OmegaConf.to_container(a, resolve=True)}

    X, y = load_tcga_arms()
    print(f"TCGA arm-level CNV: {X.shape[0]} non-Normal cases x {X.shape[1]} arms  "
          f"{y.value_counts().reindex(CLASSES).to_dict()}")

    def cv_auroc(features: pd.DataFrame, C: float, seed: int) -> float:
        probs = cross_val_predict(
            cnv_arm(C=C), features, y,
            cv=StratifiedKFold(a.cv_folds, shuffle=True, random_state=seed),
            method="predict_proba")
        return float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))

    seeds = list(range(a.cv_seeds))

    # --- 1. external per-class AUROC ---------------------------------------------------------
    print("\n=== 1. External per-class AUROC (CPTAC, TCGA-trained, nothing refit) ===")
    external = _external_matrices(cfg)
    rows = []
    for name, P in external["models"].items():
        row = {"model": name}
        row.update({c: round(float(roc_auc_score((external["y"] == c).astype(int), P[:, i])), 3)
                    for i, c in enumerate(CLASSES)})
        row["macro"] = round(float(roc_auc_score(external["y"], P, multi_class="ovr",
                                                 average="macro")), 3)
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    results["external_per_class_auroc"] = {r["model"]: {c: r[c] for c in CLASSES} for r in rows}
    _compare_per_class(results["external_per_class_auroc"], CLASSES)

    # --- 2. error independence ---------------------------------------------------------------
    print("\n=== 2. Error independence — the headroom a better rule would exploit ===")
    ext = _independence(external["y"], external["models"]["WSI raw"],
                        external["models"]["CNV (39 arms)"], CLASSES)
    _print_independence("external CPTAC", len(external["y"]), ext,
                        PUBLISHED["external_error_independence"])
    results["external_error_independence"] = ext

    internal = _internal_matrices(cfg)
    ins = _independence(internal["y"], internal["Pw"], internal["Pc"], CLASSES)
    _print_independence("internal TCGA", len(internal["y"]), ins,
                        PUBLISHED["internal_error_independence"])
    results["internal_error_independence"] = ins

    print(f"\n=== 3. Internal per-class AUROC ({len(internal['y'])} cases with CLAM "
          f"out-of-fold) ===")
    rows = []
    for name, P in (("WSI", internal["Pw"]), ("CNV (39 arms)", internal["Pc"]),
                    ("Fusion", (internal["Pw"] + internal["Pc"]) / 2)):
        row = {"model": name}
        row.update({c: round(float(roc_auc_score((internal["y"] == c).astype(int), P[:, i])), 3)
                    for i, c in enumerate(CLASSES)})
        row["macro"] = round(float(roc_auc_score(internal["y"], P, multi_class="ovr",
                                                 average="macro")), 3)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    results["internal_per_class_auroc"] = {r["model"]: {c: r[c] for c in CLASSES} for r in rows}

    # --- 4. the 39-arm internal baseline, multi-seed ------------------------------------------
    print(f"\n=== 4. Internal 39-arm baseline, {a.cv_folds}-fold x {len(seeds)} seeds ===")
    full = np.array([cv_auroc(X, a.cnv_C, s) for s in seeds])
    published = PUBLISHED["internal_39_arms_multiseed"]
    print(f"  macro AUROC {full.mean():.4f} +- {full.std():.4f}   "
          f"published {published['mean']} +- {published['std']}  "
          f"{_verdict(round(full.mean(), 3), published['mean'])}")
    results["internal_39_arms_multiseed"] = {"mean": float(full.mean()), "std": float(full.std()),
                                             "per_seed": full.round(6).tolist()}

    # --- 5. aneuploidy burden alone -----------------------------------------------------------
    print("\n=== 5. Aneuploidy burden alone (1 feature) — report this beside the 39-arm model ===")
    print("  CLAUDE.md makes this non-negotiable: at 0.685 it is high enough that Basal ~0.97")
    print("  reads as genome instability unless the arm *pattern* is shown to add to it.")
    burden = _burden(X, a.burden_definition, float(a.burden_threshold))
    single = cv_auroc(burden, a.cnv_C, a.single_seed)
    multi = np.array([cv_auroc(burden, a.cnv_C, s) for s in seeds])
    print(f"  definition: {a.burden_definition}"
          + (f" (threshold {a.burden_threshold})" if a.burden_definition == "frac_altered" else ""))
    print(f"  {a.cv_folds}-fold, seed {a.single_seed}      {single:.4f}   "
          f"published {PUBLISHED['aneuploidy_burden_only']}  "
          f"{_verdict(round(single, 3), PUBLISHED['aneuploidy_burden_only'])}  <- the published "
          f"protocol")
    print(f"  {a.cv_folds}-fold x {len(seeds)} seeds  {multi.mean():.4f} +- {multi.std():.4f}")
    print(f"  39 arms - burden = {full.mean() - multi.mean():+.4f} macro AUROC on the "
          f"{len(seeds)}-seed protocol, so the arm *pattern* carries signal beyond total "
          f"instability")
    results["aneuploidy_burden_only"] = {"definition": str(a.burden_definition),
                                         "threshold": float(a.burden_threshold),
                                         "single_seed": float(single),
                                         "multiseed_mean": float(multi.mean()),
                                         "multiseed_std": float(multi.std())}

    # --- 6. regularisation sweep --------------------------------------------------------------
    print("\n=== 6. Regularisation sweep — was C=0.1 cherry-picked? ===")
    sweep = {}
    for C in [float(c) for c in a.c_grid]:
        one = cv_auroc(X, C, a.single_seed)
        many = np.array([cv_auroc(X, C, s) for s in seeds])
        ref = PUBLISHED["regularisation_sweep"].get(C)
        mark = _verdict(round(one, 3), ref) if ref is not None else ""
        marker = "  <- the published model" if C == a.cnv_C else ""
        published = "     —" if ref is None else f"{ref:.3f}"
        print(f"  C={C:<6g} seed {a.single_seed}: {one:.4f}  published {published}  {mark}"
              f"   |  {len(seeds)} seeds: {many.mean():.4f} +- {many.std():.4f}{marker}")
        sweep[str(C)] = {"single_seed": float(one), "multiseed_mean": float(many.mean()),
                         "multiseed_std": float(many.std())}
    results["regularisation_sweep"] = sweep

    # --- 7. leave-one-TCGA-site-out -----------------------------------------------------------
    print("\n=== 7. Leave-one-TCGA-site-out — does site confounding inflate the internal number? "
          "===")
    site_scores, skipped = _site_holdout(X, y, CLASSES, int(a.min_site_cases), float(a.cnv_C))
    for name, score, n in site_scores:
        print(f"  {name}: n={n:<4d} macro AUROC {score:.4f}")
    for name, n, present in skipped:
        print(f"  {name}: n={n:<4d} skipped — only {present} of the four classes present")
    values = np.array([s for _, s, _ in site_scores])
    published = PUBLISHED["leave_one_site_out"]
    print(f"  {len(values)} sites with >= {a.min_site_cases} cases and all four classes: "
          f"{values.mean():.4f} +- {values.std():.4f}   published {published['mean']} +- "
          f"{published['std']} over {published['n_sites']} sites  "
          f"{_verdict(round(values.mean(), 3), published['mean'])}")
    results["leave_one_site_out"] = {"mean": float(values.mean()), "std": float(values.std()),
                                     "n_sites": len(values),
                                     "per_site": {n: float(s) for n, s, _ in site_scores}}

    _write_json(run_dir, "controls.json", results)
    return 0


def _external_matrices(cfg: DictConfig) -> dict[str, Any]:
    """CPTAC probabilities for the three arms, exactly as `evaluate_cnv_wsi_fusion.external` builds
    them: the CNV arm fit on every non-Normal TCGA case, the WSI arm read off disk."""
    from tools.pam50_arms import (CLASSES, cnv_arm, load_cptac_arms, load_cptac_wsi_probs,
                                  load_tcga_arms, renormalise)

    Xt, yt = load_tcga_arms()
    clf = cnv_arm(C=float(cfg.analysis.cnv_C)).fit(Xt, yt)
    wsi, n_slides = load_cptac_wsi_probs()
    Xc = load_cptac_arms()
    common = sorted(set(wsi.index) & set(Xc.index))
    y = wsi.loc[common, "true_name"].values
    Pw = wsi.loc[common, [f"p_{c}" for c in CLASSES]].values
    Pc = clf.predict_proba(Xc.loc[common, Xt.columns])
    keep = np.isin(y, CLASSES)
    y, Pw, Pc = y[keep], Pw[keep], Pc[keep]
    prior = np.array([(yt == c).mean() for c in CLASSES])
    Pwb = renormalise(Pw / prior)
    print(f"CPTAC: {n_slides} slides -> {len(y)} scored cases")
    return {"y": y, "models": {"WSI raw": Pw, "WSI prior-balanced": Pwb, "CNV (39 arms)": Pc,
                               "Fusion raw": (Pw + Pc) / 2, "Fusion balanced": (Pwb + Pc) / 2}}


def _internal_matrices(cfg: DictConfig) -> dict[str, Any]:
    """TCGA out-of-fold probabilities for both arms on the cases CLAM tested."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from tools.pam50_arms import (CLASSES, clam_column_order, cnv_arm, load_clam_oof,
                                  load_tcga_arms)

    a = cfg.analysis
    oof = load_clam_oof()
    X, y = load_tcga_arms()
    common = [c for c in X.index.intersection(oof.index) if y[c] != "Normal"]
    X, y, oof = X.loc[common], y.loc[common], oof.loc[common]
    order = clam_column_order(oof, y)
    Pw = oof[[f"p{i}" for i in range(4)]].values[:, [order.index(c) for c in CLASSES]]
    Pc = cross_val_predict(cnv_arm(C=float(a.cnv_C)), X, y,
                           cv=StratifiedKFold(a.oof_cv_folds, shuffle=True,
                                              random_state=a.oof_cv_seed),
                           method="predict_proba")
    return {"y": y.values, "Pw": Pw, "Pc": Pc}


def _independence(y, Pw, Pc, classes) -> dict[str, Any]:
    """Phi coefficient between the two arms' per-case correctness, plus the 2x2 table.

    Phi on a 2x2 is Pearson correlation of two binary vectors, which is what "the errors are
    independent" has to mean for the either-model-right ceiling below it to be meaningful.
    """
    right_w = classes[Pw.argmax(1)] == y
    right_c = classes[Pc.argmax(1)] == y
    n11 = int((right_w & right_c).sum())
    n10 = int((right_w & ~right_c).sum())
    n01 = int((~right_w & right_c).sum())
    n00 = int((~right_w & ~right_c).sum())
    denominator = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    phi = float((n11 * n00 - n10 * n01) / denominator) if denominator else float("nan")
    n = len(y)
    return {
        "phi": phi,
        "both_right": n11, "wsi_only": n10, "cnv_only": n01, "both_wrong": n00,
        "wsi_accuracy": float(right_w.mean()),
        "cnv_accuracy": float(right_c.mean()),
        "either_right": float((right_w | right_c).mean()),
        "cnv_rescues_share_of_all_cases": n01 / n,
        "cnv_rescues_share_of_wsi_misses": n01 / (n01 + n00) if (n01 + n00) else float("nan"),
        "wsi_rescues_share_of_all_cases": n10 / n,
        "wsi_rescues_share_of_cnv_misses": n10 / (n10 + n00) if (n10 + n00) else float("nan"),
    }


def _print_independence(label: str, n: int, got: dict[str, Any], published: dict[str, float]):
    print(f"  {label}, n={n}: phi(correctness) {got['phi']:+.3f}   published "
          f"{published['phi']:+.3f}  {_verdict(round(got['phi'], 3), published['phi'])}")
    print(f"    both right {got['both_right']}, WSI only {got['wsi_only']}, "
          f"CNV only {got['cnv_only']}, both wrong {got['both_wrong']}")
    print(f"    accuracy: WSI {got['wsi_accuracy']:.3f} (published "
          f"{published['wsi_accuracy']}), CNV {got['cnv_accuracy']:.3f}, "
          f"either-right {got['either_right']:.3f} (published {published['either_right']})")
    print(f"    CNV rescues {got['cnv_rescues_share_of_all_cases']:.1%} of all cases "
          f"({got['cnv_rescues_share_of_wsi_misses']:.1%} of WSI's misses); WSI rescues "
          f"{got['wsi_rescues_share_of_all_cases']:.1%} "
          f"({got['wsi_rescues_share_of_cnv_misses']:.1%} of CNV's)")


def _compare_per_class(got: dict[str, dict[str, float]], classes) -> None:
    mismatches = []
    for model, published in PUBLISHED["external_per_class_auroc"].items():
        for klass in classes:
            if model in got and abs(got[model][klass] - published[klass]) > 5e-4:
                mismatches.append(f"{model}/{klass}: {got[model][klass]} vs {published[klass]}")
    print("  vs the published table: " + ("all 12 cells match" if not mismatches
                                          else "DIFFERS — " + "; ".join(mismatches)))


def _burden(X: pd.DataFrame, definition: str, threshold: float) -> pd.DataFrame:
    """The single-feature aneuploidy-burden summary of the 39 arms.

    `mean_abs_log2` is the published definition (it reproduces 0.685). `frac_altered` is offered
    because "burden" is sometimes defined as a count of altered arms, and the two do not agree
    (0.685 against 0.673 at threshold 0.2) — which is exactly why the definition is a config key
    rather than an unwritten assumption.
    """
    if definition == "mean_abs_log2":
        return X.abs().mean(1).to_frame("burden")
    if definition == "frac_altered":
        return (X.abs() > threshold).mean(1).to_frame("burden")
    raise ValueError(f"unknown burden definition {definition!r}; "
                     "expected 'mean_abs_log2' or 'frac_altered'")


def _site_holdout(X, y, classes, min_cases: int, C: float):
    """Refit per TCGA tissue-source site and score that site's held-out cases.

    The site is the second barcode field (`TCGA-A2-A0T2` -> `A2`). Sites missing a class are
    reported and skipped rather than scored on a subset of classes: a 3-class macro AUROC is not
    comparable to a 4-class one, and averaging the two would be the quiet kind of wrong.
    """
    from sklearn.metrics import roc_auc_score
    from tools.pam50_arms import cnv_arm

    site = pd.Series([case.split("-")[1] for case in X.index], index=X.index)
    counts = site.value_counts()
    scored, skipped = [], []
    for name in counts[counts >= min_cases].index:
        held = site == name
        present = sorted(set(y[held]))
        if len(present) < len(classes):
            skipped.append((name, int(held.sum()), len(present)))
            continue
        model = cnv_arm(C=C).fit(X[~held], y[~held])
        auroc = float(roc_auc_score(y[held], model.predict_proba(X[held]),
                                    multi_class="ovr", average="macro"))
        scored.append((name, auroc, int(held.sum())))
    return scored, skipped


def _verdict(got: float, published: float | None, tolerance: float = 5e-4) -> str:
    if published is None:
        return ""
    return "match" if abs(got - published) <= tolerance else f"DIFFERS by {got - published:+.4f}"


# --------------------------------------------------------------------------- #
# run directory
# --------------------------------------------------------------------------- #


class _Tee:
    """Write to the real stdout and to the run directory's `output.txt` at once.

    The printed table *is* the result for these actions, so a run directory that did not contain
    it would not be self-describing. Line-buffered flushing keeps the file complete if the
    process is killed mid-bootstrap.
    """

    def __init__(self, stream, handle):
        self._stream, self._handle = stream, handle

    def write(self, text):
        self._stream.write(text)
        self._handle.write(text)
        if "\n" in text:
            self.flush()
        return len(text)

    def flush(self):
        self._stream.flush()
        self._handle.flush()

    def isatty(self):
        return self._stream.isatty()


def _run_dir(cfg: DictConfig, action: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(str(cfg.paths.analysis_dir)) / action / stamp


def _write_json(run_dir: Path | None, name: str, payload: dict) -> None:
    """Write a machine-readable copy of an action's numbers, if there is a run directory.

    `--no-run-dir` is a legitimate mode (a quick look at the table), so this is a no-op there
    rather than an error.
    """
    if run_dir is None:
        return
    target = Path(run_dir) / name
    target.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nwrote {target}")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

DISPATCH: dict[str, Callable[..., int]] = {
    "cnv_wsi_fusion": run_cnv_wsi_fusion,
    "stack_wsi_cnv": run_stack_wsi_cnv,
    "cnv_controls": run_cnv_controls,
    "make_cnv_tabular": run_make_cnv_tabular,
    "compare_fusion_ladder": run_compare_fusion_ladder,
    "cptac_fusion_ladder": run_cptac_fusion_ladder,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dp-analysis", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=(*ACTIONS, "list"),
                        help="which analysis to run, or `list` to describe them")
    parser.add_argument("overrides", nargs="*",
                        help="Hydra overrides, e.g. analysis.n_boot=200 analysis.internal=true")
    parser.add_argument("--no-run-dir", action="store_true",
                        help="do not create a run directory; print to stdout only")
    parser.add_argument("--show-config", action="store_true",
                        help="print the composed config and exit without running anything")
    return parser


def _list_actions() -> int:
    from ..paths import conf_dir

    print("dp-analysis actions (config in dpcode/conf/analyses/):\n")
    for action in ACTIONS:
        path = conf_dir() / "analyses" / f"{action}.yaml"
        # The first substantive comment line, which each option file leads with on purpose.
        # `@package` lines are skipped: they are a Hydra directive, not a description.
        first = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip("# ").strip()
            if line.startswith("#") and text and not text.startswith("@package"):
                first = text
                break
        print(f"  {action:<22s} {first}")
    print("\n  dp-analysis <action> [key=value ...]        run one")
    print("  dp-analysis <action> --show-config          see the composed config, run nothing")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    for flag in ("-m", "--multirun"):
        if flag in raw:
            print(
                f"dp-analysis does not support {flag}. These scripts resolve their filesystem "
                "constants once at import time and print one table to stdout, so a multirun "
                "would reuse the first job's paths. Run the actions in a shell loop instead.",
                file=sys.stderr)
            return 2

    # `parse_intermixed_args`, not `parse_args`: with a single positional followed
    # by `overrides` (nargs="*"), argparse fills both from the FIRST run of
    # positional tokens, so `dp-analysis cnv_wsi_fusion --no-run-dir
    # analysis.internal=true` left `analysis.internal=true` with nowhere to go and
    # died with "unrecognized arguments" — while the same tokens in the other order
    # worked. An argument-order trap in a CLI whose flags and overrides are
    # naturally typed in either order.
    args = build_parser().parse_intermixed_args(raw)
    if args.action == "list":
        return _list_actions()

    from hydra.errors import HydraException
    from omegaconf.errors import OmegaConfBaseException

    try:
        return _run(args)
    except (HydraException, OmegaConfBaseException, ValueError, FileNotFoundError,
            FileExistsError, RuntimeError) as exc:
        # A composition failure is a user error (a typo'd or mistyped override), not a crash.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    # `+analysis=<action>` is issued by this entry point, not by the user: the `analysis` group is
    # not in the primary config's defaults list. The user's own overrides are still screened.
    schema.reject_appended_overrides(args.overrides)
    cfg = compose_config([f"+analyses={args.action}", *args.overrides])
    schema.reject_appended_overrides(args.overrides,
                                     allow=bool(cfg.run.allow_config_surgery))
    assert_paths_reachable(cfg)
    assert_class_order(cfg)

    if args.show_config:
        print(OmegaConf.to_yaml(cfg, resolve=True), end="")
        return 0

    # BEFORE the run directory, deliberately. A run directory is a claim that a
    # run happened; one holding an empty `output.txt` and a `run_metadata.json`
    # saying `status: 1` is a misleading record of an analysis that never started,
    # and `.scratch/analysis/<action>/` would accumulate one per attempt. Nothing
    # has been created at this point, so aborting here leaves the tree untouched.
    problems = check_preconditions(cfg, args.action)
    if problems:
        print(f"dp-analysis {args.action} needs inputs that are not on this machine; "
              "nothing was run:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print("\nREPRODUCING.md walks the acquisition order; `dp-data verify-artifacts` "
              "checks a downloaded bundle.", file=sys.stderr)
        return 1

    if args.no_run_dir:
        return DISPATCH[args.action](cfg, None)

    run_dir = runinfo.assert_run_dir_writable(_run_dir(cfg, args.action), cfg.run.overwrite)
    runinfo.write_config_snapshot(cfg, run_dir)
    metadata = runinfo.RunMetadata(
        run_dir, run_seed=int(cfg.run.seed),
        command=["dp-analysis", args.action, *args.overrides],
        extra={"action": args.action,
               "hydra_output_dir": None,  # composed programmatically; see the module docstring
               "note": "dp-analysis composes its config programmatically, so there is no "
                       "Hydra .hydra/ directory to copy in. config.resolved.yaml is the "
                       "complete record."})
    metadata.start()
    status = 1
    try:
        with (run_dir / "output.txt").open("w", encoding="utf-8") as handle:
            tee = _Tee(sys.stdout, handle)
            original, sys.stdout = sys.stdout, tee
            try:
                status = DISPATCH[args.action](cfg, run_dir)
            finally:
                sys.stdout = original
    finally:
        metadata.finish(status)
        print(f"\nrun directory: {run_dir}")
    return status


if __name__ == "__main__":
    sys.exit(main())
