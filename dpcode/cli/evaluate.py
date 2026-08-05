"""`dp-evaluate` — score a trained WSI + tabular fusion checkpoint directory.

Replaces `tools/evaluate_pam50_multimodal.sh`. It composes the `evaluate` config
group, renders the argv, and dispatches `project/CLAM/evaluate_multimodal.py` as
a subprocess from `project/CLAM` — exactly what the shell wrapper did, for the
same reason `dp-train` does it: the evaluator's numerical behaviour is frozen and
importing it would drag torch, matplotlib and a CUDA context into a config check.

    dp-evaluate                                   # the wrapper's defaults
    dp-evaluate --dry-run                         # print the command, run nothing
    dp-evaluate evaluate.args.ckpt_dir=/path/to/run_s1
    dp-evaluate evaluate.args.tabular_hidden_dim=64 evaluate.args.fusion_hidden_dim=32

TWO KNOWN GAPS ARE PRESERVED, NOT FIXED (DESIGN.md section 14):

* the defaults evaluate the **TCGA** test split, not CPTAC. Renaming a run does
  not change that, and swapping only `evaluate.args.tabular_csv` to a CPTAC table
  scores TCGA slides against CPTAC-shaped tabular rows;
* `fusion_mode` accepts only `auto`, `concat`, `gated` — the shell wrapper's
  list, which is narrower than `evaluate_multimodal.py`'s own
  (`auto|concat|gated|residual|cross_attention`). :data:`WRAPPER_FUSION_MODES`
  is a code constant rather than a config key so that widening it is a change
  someone has to make and justify.

What this entry point adds is that both gaps, and a third undocumented one, fail
LOUDLY and before dispatch rather than inside `load_state_dict`:

* a `film_attention` or `coattn` checkpoint directory is refused with a message
  naming the Known-gaps entry it belongs to (there is no evaluator for either);
* a checkpoint directory whose recorded architecture disagrees with the config —
  most often `tabular_hidden_dim`, 64 for every CNV ladder arm against this
  config's 256 — is reported with the exact overrides that would fix it. That
  mismatch is why the fusion ladder's printed evaluation hint has never worked
  for any of its five arms.

Both checks read the run directory's own `experiment_<exp_code>.txt`
(`project/CLAM/main.py:422`) or the `clam_argv.json` that `dp-train` writes. No
checkpoint is opened, so the check costs milliseconds and no GPU.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig

from .. import clam_args, schema
from ..paths import assert_paths_absolute, assert_paths_exist
from ..runinfo import RunMetadata, write_config_snapshot
from .config import compose_config

__all__ = [
    "EVALUATOR_RELPATH",
    "WRAPPER_FUSION_MODES",
    "UNEVALUABLE_FUSION_MODES",
    "evaluator_parser",
    "render_argv",
    "checkpoint_run_settings",
    "preflight",
    "main",
]

#: The script this entry point dispatches, relative to the repository root.
EVALUATOR_RELPATH = "project/CLAM/evaluate_multimodal.py"

#: Which `evaluate` group option is composed when none is named.
DEFAULT_OPTION = "pam50_multimodal"

#: `tools/evaluate_pam50_multimodal.sh:110` accepted exactly these three, and so
#: does this. NOT a config key: see the module docstring.
WRAPPER_FUSION_MODES = ("auto", "concat", "gated")

#: Trained by `tools/run_cnv_fusion_ladder.sh`, evaluable by nothing.
#: `evaluate_multimodal.py:70`'s `--fusion_mode` choice list omits both, and
#: under `auto`, `infer_fusion_mode` (`:116-143`) has no branch for either: a
#: `film_attention` checkpoint raises `ValueError`, and a `coattn` checkpoint is
#: misidentified as `cross_attention` — both build `self.cross_attention` — and
#: then dies in `load_state_dict(..., strict=True)`.
UNEVALUABLE_FUSION_MODES = ("film_attention", "coattn")

#: Recorded training settings that must match the evaluation config, because
#: `load_state_dict(strict=True)` compares tensor shapes. Maps the key in
#: `experiment_<exp_code>.txt` to the key under `evaluate.args`.
SHAPE_CRITICAL_KEYS = {
    "model_type": "model_type",
    "model_size": "model_size",
    "tabular_hidden_dim": "tabular_hidden_dim",
    "tabular_num_layers": "tabular_num_layers",
    "fusion_hidden_dim": "fusion_hidden_dim",
    "rna_hidden_dims": "rna_hidden_dims",
    "fusion_mode": "fusion_mode",
}

#: Existence is checked for these before dispatch, so a typo'd path costs a
#: second rather than a torch import and a model construction.
REQUIRED_INPUT_KEYS = (
    "evaluate.args.data_root_dir",
    "evaluate.args.tabular_csv",
    "evaluate.args.ckpt_dir",
    "evaluate.args.split_dir",
    "evaluate.args.dataset_csv",
)


# --------------------------------------------------------------------------- #
# the evaluator's real argument surface
# --------------------------------------------------------------------------- #


def evaluator_parser(script: str | Path | None = None) -> argparse.ArgumentParser:
    """`evaluate_multimodal.py`'s real parser, built without importing it.

    Same trick as `dpcode.clam_args.clam_parser`, one level deeper: the parser is
    built inside `def parse_args()` rather than at module level, so the AST walk
    descends into that function and executes only the statements from
    `parser = argparse.ArgumentParser(...)` up to (not including) the
    `return parser.parse_args()`. `argparse` is the only name in scope, so
    nothing else in the file — matplotlib, torch, the CLAM dataset modules — is
    touched.

    Extracting rather than re-declaring is what keeps this CLI honest: add a flag
    to the evaluator and :func:`render_argv` fails with its name, instead of the
    flag being silently unreachable through the config.
    """
    from ..paths import repo_root

    path = Path(script) if script is not None else repo_root() / EVALUATOR_RELPATH
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
        ),
        None,
    )
    if function is None:
        raise RuntimeError(f"{path}: no module-level `def parse_args()`.")

    body: list[ast.stmt] = []
    for statement in function.body:
        if _is_parse_args_return(statement):
            break
        body.append(statement)
    if not body:
        raise RuntimeError(f"{path}: `parse_args()` builds no parser before returning.")

    namespace: dict[str, Any] = {"argparse": argparse}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    parser = namespace.get("parser")
    if not isinstance(parser, argparse.ArgumentParser):
        raise RuntimeError(f"{path}: extracted slice did not produce an ArgumentParser.")
    return parser


def _is_parse_args_return(node: ast.stmt) -> bool:
    value = node.value if isinstance(node, (ast.Return, ast.Assign, ast.Expr)) else None
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "parse_args"
        and isinstance(func.value, ast.Name)
        and func.value.id == "parser"
    )


def render_argv(args_node: Any, parser: argparse.ArgumentParser | None = None) -> list[str]:
    """`evaluate.args` -> the argv list for `evaluate_multimodal.py`.

    `dpcode.clam_args.render_argv` is parser-agnostic, so the same rules apply:
    every non-`None` scalar becomes `--flag value`, every true store_true becomes
    `--flag`, and an unknown or missing key raises. A flag left at the
    evaluator's own default is emitted explicitly and parses to the same value,
    so the effective configuration still matches the shell wrapper's.
    """
    return clam_args.render_argv(args_node, parser or evaluator_parser())


# --------------------------------------------------------------------------- #
# checkpoint preflight
# --------------------------------------------------------------------------- #


def checkpoint_run_settings(ckpt_dir: str | Path) -> tuple[dict[str, Any], str | None]:
    """What the checkpoint directory says it was trained with.

    Returns `(settings, source)`; `settings` is empty and `source` is `None` when
    the directory carries no provenance at all — a hand-assembled checkpoint
    directory is legal, it just cannot be checked.

    Two sources, newest first: `clam_argv.json`, which `dp-train` writes, and
    `experiment_<exp_code>.txt`, which `project/CLAM/main.py:422-423` writes as a
    plain `print(settings)` of a dict — hence `ast.literal_eval`.
    """
    directory = Path(ckpt_dir)

    argv_json = directory / "clam_argv.json"
    if argv_json.is_file():
        try:
            payload = json.loads(argv_json.read_text(encoding="utf-8"))
            settings = _settings_from_argv(list(payload.get("argv", [])))
            if settings:
                return settings, argv_json.name
        except (OSError, ValueError):
            pass

    for experiment in sorted(directory.glob("experiment_*.txt")):
        try:
            payload = ast.literal_eval(experiment.read_text(encoding="utf-8").strip())
        except (OSError, ValueError, SyntaxError):
            continue
        if isinstance(payload, dict):
            # main.py records `--drop_out` under the name `use_drop_out`.
            if "use_drop_out" in payload:
                payload.setdefault("drop_out", payload["use_drop_out"])
            return payload, experiment.name
    return {}, None


def _settings_from_argv(argv: Sequence[str]) -> dict[str, Any]:
    """Parse a recorded CLAM argv into a settings dict, defaults included."""
    try:
        namespace = clam_args.clam_parser().parse_args(list(argv))
    except SystemExit:  # an argv from a different CLAM revision
        return {}
    return dict(vars(namespace))


def preflight(cfg: DictConfig) -> list[str]:
    """Refuse the unevaluable, warn about the mismatched. Returns the warnings.

    Raises `SystemExit` for a `film_attention` or `coattn` checkpoint directory
    and for a checkpoint directory with no `s_*_checkpoint.pt` in it. Everything
    else is reported and the run proceeds, because the evaluator itself is the
    authority on whether a state dict loads.
    """
    args_node = cfg.evaluate.args
    ckpt_dir = Path(str(args_node.ckpt_dir))

    checkpoints = sorted(ckpt_dir.glob("s_*_checkpoint.pt"))
    if not checkpoints:
        raise SystemExit(
            f"No `s_*_checkpoint.pt` in {ckpt_dir}.\n"
            "evaluate.args.ckpt_dir must name a CLAM run directory "
            "(`<results_root>/<exp_code>_s<seed>/`), not its parent."
        )

    settings, source = checkpoint_run_settings(ckpt_dir)
    if not settings:
        return [
            f"{ckpt_dir} carries no clam_argv.json and no experiment_*.txt, so the "
            "architecture it was trained with cannot be checked against this config. "
            "A shape mismatch will surface inside load_state_dict instead."
        ]

    trained_mode = settings.get("fusion_mode")
    if trained_mode in UNEVALUABLE_FUSION_MODES:
        raise SystemExit(_unevaluable_message(ckpt_dir, str(trained_mode), source))

    warnings: list[str] = []
    for recorded_key, config_key in SHAPE_CRITICAL_KEYS.items():
        if recorded_key not in settings:
            continue
        recorded = settings[recorded_key]
        configured = args_node[config_key]
        if recorded_key == "fusion_mode" and str(configured) == "auto":
            continue  # `auto` is resolved from the checkpoint keys, by design
        if recorded is None or str(recorded) == str(configured):
            continue
        warnings.append(
            f"{source} says {recorded_key}={recorded!r}, this config says "
            f"{config_key}={configured!r}. Fix with "
            f"`evaluate.args.{config_key}={recorded}`."
        )
    if warnings:
        warnings.append(
            "load_state_dict(..., strict=True) will reject a shape mismatch, so a "
            "disagreement above is a failure and not a nuance."
        )
    return warnings


def _unevaluable_message(ckpt_dir: Path, mode: str, source: str | None) -> str:
    return "\n".join(
        [
            f"{ckpt_dir} is a `{mode}` fusion run ({source}), which has no evaluation path.",
            "",
            "This is the first entry under `## Known gaps (real, blocking, unfixed)` in",
            "CLAUDE.md: \"`film_attention` and `coattn` checkpoints cannot be evaluated.\"",
            f"  * {EVALUATOR_RELPATH}:70 accepts only",
            "    auto|concat|gated|residual|cross_attention;",
            "  * under `auto`, infer_fusion_mode has no branch for either operator: a",
            "    film_attention checkpoint (film_bottleneck/film_gamma/film_beta/",
            "    tabular_head) raises ValueError, and a coattn checkpoint is",
            "    misidentified as cross_attention -- both build self.cross_attention --",
            "    and then dies in load_state_dict(..., strict=True).",
            "",
            "Closing the gap means extending infer_fusion_mode and BOTH choice lists;",
            "DESIGN.md section 14 puts that out of scope for this refactor, so the",
            "failure is raised here rather than inside torch.",
            "",
            "The five completed ladder arms are compared by `tools/compare_fusion_ladder.py`,",
            "which reads their per-fold `split_*_results.pkl` and needs no evaluator.",
        ]
    )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dp-evaluate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. evaluate.args.ckpt_dir=/path/to/run_s1",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_OPTION,
        help=f"Which `evaluate` config group option to compose (default: {DEFAULT_OPTION}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan and the exact command, then stop.",
    )
    parser.add_argument(
        "--print-argv",
        action="store_true",
        help="Print the rendered argv as JSON and stop. Implies --dry-run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from hydra.errors import HydraException
    from omegaconf.errors import OmegaConfBaseException

    args = build_parser().parse_args(argv)
    overrides = list(args.overrides)

    try:
        cfg = compose_config([f"+evaluate={args.config}", *overrides])
        schema.reject_appended_overrides(
            overrides, allow=bool(cfg.run.allow_config_surgery)
        )
        if not isinstance(cfg.evaluate, DictConfig):
            # `+evaluate=x` assigns the STRING "x" when no such group option
            # exists, because `RootConf.evaluate` is a real key. Catch that here
            # rather than three lines later as an AttributeError.
            raise RuntimeError(
                f"evaluate={args.config} did not compose a config group; got "
                f"{cfg.evaluate!r}. Is dpcode/conf/evaluate/{args.config}.yaml installed?"
            )
        assert_paths_absolute(cfg)

        mode = str(cfg.evaluate.args.fusion_mode)
        if mode not in WRAPPER_FUSION_MODES:
            # `tools/evaluate_pam50_multimodal.sh:110-113`, preserved verbatim
            # including the exit status.
            print(
                "evaluate.args.fusion_mode must be "
                + ", ".join(f"'{m}'" for m in WRAPPER_FUSION_MODES)
                + f" (got '{mode}').\n"
                f"{EVALUATOR_RELPATH} itself also accepts 'residual' and "
                "'cross_attention'; this entry point keeps the shell wrapper's "
                "narrower list on purpose (DESIGN.md section 14).",
                file=sys.stderr,
            )
            return 2

        clam_root = Path(str(cfg.paths.clam_root))
        script = Path(str(cfg.paths.repo_root)) / EVALUATOR_RELPATH
        parser = evaluator_parser(script)
        rendered = render_argv(cfg.evaluate.args, parser)

        if args.print_argv:
            print(json.dumps({"cwd": str(clam_root), "argv": rendered}, indent=2))
            return 0

        # Checkpoint first: "this operator has no evaluator" is a more useful
        # answer than "your RNA table is missing" for someone pointing this at a
        # ladder arm on a machine that never had the RNA tables.
        warnings = preflight(cfg)
        assert_paths_exist(cfg, REQUIRED_INPUT_KEYS)

        output_dir = Path(str(cfg.evaluate.args.output_dir))
        command = [sys.executable, script.name, *rendered]

        print(f"config      : evaluate={args.config}")
        print(f"checkpoints : {cfg.evaluate.args.ckpt_dir}")
        print(f"features    : {cfg.evaluate.args.data_root_dir}")
        print(f"tabular     : {cfg.evaluate.args.tabular_csv}")
        print(f"split       : {cfg.evaluate.args.split} of {cfg.evaluate.args.dataset_csv}")
        print(f"output      : {output_dir}")
        print(f"cwd         : {clam_root}")
        print("command     : " + " ".join(command))
        sys.stdout.flush()  # keep the plan above the warnings, not interleaved
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)

        if args.dry_run:
            print("\n--dry-run: nothing dispatched, nothing written.")
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        write_config_snapshot(cfg, output_dir)
        (output_dir / "eval_argv.json").write_text(
            json.dumps(
                {
                    "executable": sys.executable,
                    "script": EVALUATOR_RELPATH,
                    "argv": rendered,
                    "cwd": str(clam_root),
                    "command": " ".join(command),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        metadata = RunMetadata(
            output_dir,
            run_seed=int(cfg.run.seed),
            command=[sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
            extra={"entry_point": "dp-evaluate", "evaluate_config": args.config},
        )
        metadata.start()
        completed = subprocess.run(command, cwd=str(clam_root))
        metadata.finish(completed.returncode)
        return completed.returncode
    except (
        HydraException,
        OmegaConfBaseException,
        ValueError,
        KeyError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
    ) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except SystemExit as exc:  # the preflight refusals above
        if exc.code in (0, None):
            return 0
        print(exc.code if isinstance(exc.code, str) else f"exit {exc.code}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
