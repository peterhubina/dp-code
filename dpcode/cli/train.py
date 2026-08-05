"""`dp-train` — compose a CLAM training run, then dispatch `project/CLAM/main.py`.

    dp-train experiment=pam50_wsi_final
    dp-train experiment=pam50_wsi_cnv fusion=film_attention
    dp-train -m experiment=pam50_wsi_cnv fusion=concat,gated,cross_attention,film_attention,coattn
    dp-train experiment=pam50_wsi_cnv fusion=concat --dry-run
    dp-train --cfg job --resolve experiment=pam50_wsi_final     # Hydra's own

CLAM IS WRAPPED, NOT REWRITTEN, AND IT IS NEVER IMPORTED
`project/CLAM/main.py` guards only `main(args)`. Lines 119-433 run at import:
they build the parser, call `parse_args()` against the *importing* process's
`sys.argv`, seed torch, construct the dataset, create the results directory and
write `experiment_<exp_code>.txt`. Importing it would therefore start writing
files on the strength of `dp-train`'s own command line. So this module renders an
argv list and launches `python main.py` as a subprocess with cwd
`${paths.clam_root}` — which is also what guarantees the numerical behaviour
behind the published tables cannot drift: the training code is reached by exactly
the route the shell wrappers used.

The cwd is not cosmetic. `main.py:412-414` prefixes `--split_dir` with the literal
`splits/` and then asserts it is a directory, and the dataset CSV is resolved as
`dataset_csv/<task>.csv`; neither is overridable by any flag.

ORDER OF OPERATIONS (DESIGN-ADDENDUM A1) — the sequence, not the steps, is what
protects the completed runs:

    1. reject `+key=…` / `~key` overrides
    2. compute the CLAM run directory from `clam.results_dir`, `clam.exp_code`
       and `clam.seed` — exactly as `main.py:407` derives it
    3. `assert_run_dir_writable` — NOTHING HAS BEEN WRITTEN YET, so an abort here
       leaves a completed run untouched
    4. run CLAM's own eight cross-argument checks (`main.py:216-232`) before
       dispatch, so a bad combination fails here rather than after the metadata
       has been written
    5. write `config.resolved.yaml`, `clam_argv.json` and the opening
       `run_metadata.json`
    6. dispatch, streaming CLAM's output straight through
    7. write `metrics.json`, finalise `run_metadata.json`, copy Hydra's `.hydra/`
       in beside them

Between 1 and 2 sit the startup checks — every `paths.*` value absolute, and every
input this particular run reads present on disk. They write nothing, so they are
free to run first, and they turn the classic failure (a wrong `embeddings_dir`
discovered at the first `h5py.File`, minutes in) into a one-second abort naming the
config key.

Under `--dry-run` the input check moves to the END, after the plan has been
printed: a dry run is what README.md offers before any data has been acquired, so
it must work on an empty machine. The plan is printed either way and the missing
inputs are reported under it, exactly as `dp-cptac --dry-run` does; the exit
status is still non-zero, because "this would not run here" is part of the answer.

CLAM is launched through `shutil.which("python")` rather than `sys.executable`,
deliberately: it is the same interpreter resolution every replaced wrapper used
(`python main.py`), which is what lets a `PATH`-stubbed fake `python` capture the
dispatch in the parity harness. The resolved absolute path is recorded in
`clam_argv.json`, so the run record says which interpreter actually ran.

Hydra's own output directory stays in scratch (`hydra.run.dir` in
`dpcode/conf/config.yaml`) precisely so step 3 can be honest: Hydra creates its
output directory and writes four files into it BEFORE the task function is
called, so an overwrite guard cannot protect a directory Hydra has already opened.

Because the run directory is derived from `clam.exp_code`/`clam.seed` rather than
from `hydra.run.dir`, `--multirun` gets exactly the same contract: each arm of a
sweep writes its own complete metadata set into its own directory.

W&B: `dp-train` does not initialise W&B. CLAM does, from `clam.wandb*`, which is
what the wrappers passed. The `tracking` config group configures dpcode's own
entry points; selecting `tracking=online` here changes nothing, and this module
says so on stderr rather than letting the mismatch pass unnoticed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import hydra
from omegaconf import DictConfig, OmegaConf

from .. import clam_args, runinfo, schema
from ..paths import assert_paths_absolute, assert_paths_exist, conf_dir

__all__ = ["main"]

CONFIG_NAME = "train"
VERSION_BASE = "1.3"

#: Not a Hydra override, so it is stripped from `sys.argv` before Hydra parses it.
#: Both spellings are accepted: `run_cnv_fusion_ladder.sh` taught `--dry_run`.
DRY_RUN_FLAGS = ("--dry-run", "--dry_run")

#: Set by :func:`main` from the command line. Module state rather than an argument
#: because `@hydra.main` calls the task function itself and passes only `cfg`.
_DRY_RUN = False


def _required(key: str, why: str = "") -> str:
    """`${dp.required:key,why}` — a config value the caller must supply.

    OmegaConf's own spelling for this, `key: ???`, DOES NOT WORK in an experiment
    file: `???` in the *source* of a merge is a no-op, so it silently leaves
    `clam/base.yaml`'s value (here CLAM's `None`) in place and the run proceeds
    with the flag omitted. Verified on omegaconf 2.3.0.

    So the mandatory-value marker is an interpolation that raises when resolved,
    which happens before `dp-train` creates anything. It replaces a wrapper's
    `exit 2` — `tools/train_pam50_multimodal.sh:93-97` refused to run without
    `--pretrained_wsi_ckpt` — with a message that names the config key instead.
    """
    detail = f" — {why}" if why else ""
    raise ValueError(
        f"{key} is required by this experiment and has no default{detail}. "
        f"Supply it, e.g. `dp-train … {key}=/path/to/value`. A value containing "
        "`{fold}` must be quoted; Hydra's override grammar rejects a bare `{`:\n"
        '  dp-train … \'clam.pretrained_wsi_ckpt="/abs/s_{fold}_checkpoint.pt"\''
    )


OmegaConf.register_new_resolver("dp.required", _required, replace=True)


@hydra.main(version_base=VERSION_BASE, config_path=str(conf_dir()), config_name=CONFIG_NAME)
def _train(cfg: DictConfig) -> None:
    """Hydra task function: everything below is `_run`, with user errors made readable.

    A refused overwrite, a missing input or a rejected CLAM argument combination is
    a user error, and Hydra's default reporting prints a traceback through this
    module's internals for all three. The message is what matters; set
    `HYDRA_FULL_ERROR=1` — the variable Hydra's own footer teaches — to get the
    traceback back.
    """
    from omegaconf.errors import InterpolationResolutionError

    try:
        _run(cfg)
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        ValueError,  # includes clam_args.ClamArgumentError
        InterpolationResolutionError,  # includes ${dp.required:…}
    ) as exc:
        if os.environ.get("HYDRA_FULL_ERROR"):
            raise
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)


def _run(cfg: DictConfig) -> None:
    schema.reject_appended_overrides(allow=bool(cfg.run.allow_config_surgery))
    assert_paths_absolute(cfg)
    if not _DRY_RUN:
        _assert_inputs_exist(cfg)
    _warn_on_tracking_mismatch(cfg)

    clam_root = Path(str(cfg.paths.clam_root))
    main_py = clam_root / "main.py"
    # Needed by both paths, and by the plan: without CLAM's parser there is no
    # argv to print. A missing `main.py` means a broken checkout, not un-acquired
    # data, so it stays fatal even under `--dry-run`.
    _assert_clam_main_py(cfg, main_py)
    parser = clam_args.clam_parser(main_py)

    clam_values = OmegaConf.to_container(cfg.clam, resolve=True, throw_on_missing=True)
    argv = clam_args.render_argv(clam_values, parser)
    interpreter = shutil.which("python") or sys.executable
    command = [interpreter, "main.py", *argv]

    run_dir = runinfo.clam_run_dir(cfg.clam.results_dir, cfg.clam.exp_code, cfg.clam.seed)

    if _DRY_RUN:
        _print_plan(cfg, run_dir, clam_root, command)
        clam_args.validate_clam_args(clam_values, parser=parser, main_py=main_py)
        _report_missing_inputs(cfg)  # raises SystemExit(2) if anything is missing
        return

    runinfo.assert_run_dir_writable(run_dir, bool(cfg.run.overwrite))
    clam_args.validate_clam_args(clam_values, parser=parser, main_py=main_py)

    runinfo.write_config_snapshot(cfg, run_dir)
    runinfo.write_clam_argv(run_dir, argv, cwd=clam_root, executable=interpreter)
    hydra_dir = runinfo.hydra_output_dir()
    metadata = runinfo.RunMetadata(
        run_dir,
        run_seed=int(cfg.run.seed),
        clam_seed=int(cfg.clam.seed),
        command=list(sys.argv),
        extra={
            "entry_point": "dp-train",
            "experiment": _group_choice("experiment"),
            "fusion": str(cfg.fusion.name),
            "clam_command": " ".join(command),
            "clam_cwd": str(clam_root),
            # Hydra's job log stays where Hydra wrote it; only `.hydra/` is copied
            # into the run directory (step 7), so record where the rest lives.
            "hydra_output_dir": str(hydra_dir) if hydra_dir is not None else None,
        },
    )
    metadata.start()

    print(f"[dp-train] run directory : {run_dir}", flush=True)
    print(f"[dp-train] dispatching   : {' '.join(command)}", flush=True)
    print(f"[dp-train] working dir   : {clam_root}", flush=True)

    # `finally`, so that a Ctrl-C three hours into a 10-fold run still leaves a
    # finalised `run_metadata.json` rather than one that says "running" forever.
    # 130 is SIGINT (128 + 2), the status the shell would report for that.
    status = 130
    try:
        status = subprocess.run(command, cwd=str(clam_root)).returncode
    finally:
        metrics = runinfo.write_metrics(run_dir)
        copied = runinfo.copy_hydra_outputs(hydra_dir, run_dir)
        metadata.update(
            metrics_file=str(metrics) if metrics is not None else None,
            hydra_snapshot_copied=str(copied) if copied is not None else None,
        )
        metadata.finish(status)

    if status != 0:
        # SystemExit rather than a return value: it preserves CLAM's own exit
        # status, and under `--multirun` it stops the sweep instead of letting the
        # remaining arms run on top of a failure — which is what `set -e` did in
        # every wrapper this replaces.
        raise SystemExit(status)


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #


def _input_keys(cfg: DictConfig) -> list[str]:
    """The config keys naming inputs THIS run reads, in the order they are checked."""
    keys = ["paths.clam_root", "clam.data_root_dir", "clam.split_dir"]
    if cfg.clam.tabular_csv is not None:
        keys.append("clam.tabular_csv")
    # `prefix` is a literal instruction to group one-hot blocks by column-name
    # prefix (main.py:170-172), not a path.
    if cfg.clam.tabular_group_spec not in (None, "prefix"):
        keys.append("clam.tabular_group_spec")
    return keys


def _assert_inputs_exist(cfg: DictConfig) -> None:
    """Fail on a missing input before a run directory exists, naming the key.

    Only the inputs THIS run reads. `main.py` asserts `split_dir` itself, but only
    after it has created the run directory and written `experiment_<exp>.txt`, and
    a missing `data_root_dir` surfaces even later, at the first `h5py.File`.

    `pretrained_wsi_ckpt` is not checked: it carries a `{fold}` placeholder that
    only CLAM expands, and a missing checkpoint fails loudly at fold 0.

    NOT called under `--dry-run`; see :func:`_report_missing_inputs` for why.
    """
    assert_paths_exist(cfg, _input_keys(cfg))
    _assert_clam_main_py(cfg, Path(str(cfg.paths.clam_root)) / "main.py")


def _assert_clam_main_py(cfg: DictConfig, main_py: Path) -> None:
    """`project/CLAM/main.py` must exist — it is a tracked file, not acquired data."""
    if not main_py.is_file():
        raise FileNotFoundError(
            f"paths.clam_root={cfg.paths.clam_root} has no main.py. dp-train dispatches "
            "`python main.py` from that directory; without it there is nothing to run."
        )


def _report_missing_inputs(cfg: DictConfig) -> None:
    """`--dry-run`'s input check: report AFTER the plan, then exit non-zero.

    A dry run has to work on a machine that has downloaded nothing. README.md
    offers `dp-train --dry-run experiment=pam50_wsi_final` before any acquisition
    step, so running the same input check up front made the one command a new user
    is told to run first the one command they could not run at all.

    The plan is therefore printed unconditionally and the missing inputs are
    reported under it — the shape `dp-cptac --dry-run` already has. The exit
    status is still non-zero, and deliberately the same 2 the real run would exit
    with on the same configuration: `--dry-run` is a preflight, so "the plan is
    printed" and "this would not run here" are both true and both worth saying.
    A CI gate or a `&&` chain can rely on the status; a reader gets the plan.
    """
    try:
        assert_paths_exist(cfg, _input_keys(cfg))
    except FileNotFoundError as exc:
        sys.stdout.flush()  # keep the report under the plan, not interleaved with it
        print(
            f"\n[dry run] this configuration would NOT run on this machine.\n"
            f"[dry run] {exc}\n"
            "[dry run] The plan above is still correct; the inputs are not on disk "
            "yet. See REPRODUCING.md for how to acquire them.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


def _warn_on_tracking_mismatch(cfg: DictConfig) -> None:
    """Say so when `tracking=…` was selected but CLAM's own W&B flag is off.

    The two are deliberately separate (CLAM logs its own runs from `clam.wandb`),
    and the failure mode is silent: a user selects `tracking=online`, trains for
    five hours, and finds nothing in W&B.
    """
    if cfg.tracking.enabled and not cfg.clam.wandb:
        print(
            f"[dp-train] note: tracking={cfg.tracking.mode} configures dpcode's own "
            "entry points and does not reach CLAM. This run will not log to W&B — "
            "pass clam.wandb=true (with clam.wandb_project=… and clam.wandb_tags=[…]) "
            "if that is what you wanted.",
            file=sys.stderr,
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _group_choice(group: str) -> Any:
    """The option selected for a config group, for the run record."""
    try:
        from hydra.core.hydra_config import HydraConfig

        return HydraConfig.get().runtime.choices.get(group)
    except Exception:  # pragma: no cover - outside @hydra.main
        return None


def _print_plan(
    cfg: DictConfig, run_dir: Path, clam_root: Path, command: Sequence[str]
) -> None:
    """`--dry-run`: print what would happen and write nothing at all.

    Replaces `run_cnv_fusion_ladder.sh --dry_run`. Nothing is created here — not
    the run directory, not the metadata — so it is safe to point at a completed
    run. (Hydra still creates its own scratch output directory under
    `${paths.scratch_root}/hydra/`; that is Hydra's, and nothing reads it.)
    """
    exists = run_dir.is_dir() and (
        (run_dir / "summary.csv").exists() or any(run_dir.glob("s_*_checkpoint.pt"))
    )
    print(f"[dry run] experiment   : {_group_choice('experiment')}")
    print(f"[dry run] fusion       : {cfg.fusion.name}")
    print(f"[dry run] run directory: {run_dir}")
    if exists:
        print(
            "[dry run]              ^ ALREADY HOLDS RESULTS — a real run would abort "
            "here unless run.overwrite=true"
        )
    print(f"[dry run] working dir  : {clam_root}")
    print("[dry run] command      :")
    print(f"    {command[0]} {command[1]} \\")
    tokens = list(command[2:])
    lines = []
    current: list[str] = []
    for token in tokens:
        if token.startswith("--") and current:
            lines.append(current)
            current = []
        current.append(token)
    if current:
        lines.append(current)
    for index, line in enumerate(lines):
        tail = " \\" if index < len(lines) - 1 else ""
        print("        " + " ".join(line) + tail)
    print("[dry run] nothing was written.")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def _print_help() -> None:
    """`dp-train --help` with no experiment selected.

    Hydra's own app help COMPOSES the config to render its `$CONFIG` block, and
    `experiment` is a mandatory group (`experiment: ???` in
    `dpcode/conf/train.yaml`), so plain `--help` would abort with "You must
    specify 'experiment'" instead of printing help. Selecting a default
    experiment to make `--help` work would mean `dp-train` with no arguments
    silently running one of the published configurations, which is worse.

    So: this summary when no experiment is given, and Hydra's full help — config
    groups, the resolved config, every Hydra flag — the moment one is:
    `dp-train --help experiment=pam50_wsi_final`.
    """
    experiments = sorted(p.stem for p in (conf_dir() / "experiment").glob("*.yaml"))
    operators = sorted(p.stem for p in (conf_dir() / "fusion").glob("*.yaml"))
    print(
        f"""dp-train — compose a CLAM training run and dispatch project/CLAM/main.py.

usage: dp-train experiment=<NAME> [fusion=<OPERATOR>] [key=value ...] [--dry-run]
       dp-train -m experiment=<NAME> fusion=a,b,c          # one run per operator
       dp-train --help experiment=<NAME>                   # Hydra's full help
       dp-train --cfg job --resolve experiment=<NAME>      # the composed config

experiments (experiment=)
  {", ".join(experiments)}

fusion operators (fusion=)
  {", ".join(operators)}
  `none` omits --fusion_mode; `residual` refuses to compose (see Known gaps).

useful overrides
  clam.seed=2                repeat an experiment at another seed
  clam.k=1 clam.max_epochs=2 a wiring check rather than a real run
  clam.exp_code=NAME_smoke   write somewhere else — the run directory is
                             ${{paths.results_root}}/<exp_code>_s<seed>
  run.overwrite=true         write into a directory that already holds results.
                             Off by default: .scratch is gitignored, and the five
                             completed ladder arms cost 2h38m of GPU time.

flags handled here rather than by Hydra
  --dry-run (--dry_run)      print the run directory and the exact CLAM command,
                             write nothing, dispatch nothing

Values containing `{{fold}}` must be quoted — Hydra's override grammar rejects a
bare `{{`:
  dp-train … 'clam.pretrained_wsi_ckpt="/abs/s_{{fold}}_checkpoint.pt"'
"""
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point.

    `--dry-run` / `--dry_run` is removed from `sys.argv` before Hydra sees it:
    Hydra's parser accepts only its own flags and `key=value` overrides, and would
    reject an unknown one. Everything else — including `--cfg`, `--multirun`,
    `--hydra-help` and every override — is passed through untouched.
    """
    global _DRY_RUN

    arguments = list(sys.argv[1:] if argv is None else argv)
    kept = [a for a in arguments if a not in DRY_RUN_FLAGS]
    _DRY_RUN = len(kept) != len(arguments)

    wants_help = any(a in ("-h", "--help") for a in kept)
    has_experiment = any(a.startswith("experiment=") for a in kept)
    if wants_help and not has_experiment:
        _print_help()
        return

    sys.argv = [sys.argv[0], *kept]
    _train()


if __name__ == "__main__":
    main()
