"""`dp-config` — inspect, validate and document the configuration.

Four subcommands:

    dp-config show [OVERRIDE ...]        compose and print the config
    dp-config validate [OVERRIDE ...]    compose and run every startup guard
    dp-config reference [-o PATH]        generate the config reference (markdown)
    dp-config sync-check                 ClamConf / clam-base.yaml vs CLAM's parser

`reference` writes to stdout by default and to `-o PATH` on request. It does not
choose a location: the generated document is owned by the documentation track.

`show` and `validate` compose `conf/config.yaml`, EXCEPT when an override names
the `experiment` or `fusion` group — then they compose `conf/train.yaml`, which is
what `dp-train` composes and the only primary config that carries those groups:

    dp-config validate experiment=pam50_wsi_final
    dp-config validate experiment=pam50_wsi_cnv fusion=film_attention

That is not a convenience. `validate`'s `topk` preflight fires only for
`clam.inst_loss=svm`, and the only config that sets it is
`experiment/pam50_wsi_final.yaml` — the frozen WSI baseline. Without a way to
name an experiment, the guard was unreachable. See :func:`primary_config_name`.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import sys
import typing
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

from .. import clam_args, schema
from ..paths import assert_paths_absolute, conf_dir

CONFIG_NAME = "config"
VERSION_BASE = "1.3"

#: The primary config that carries the `experiment` and `fusion` groups.
#: `config.yaml` deliberately does not: it is shared by every entry point, and a
#: group listed there would be mandatory for all of them (see the header of
#: `dpcode/conf/train.yaml`). So naming an experiment here composes what
#: `dp-train` composes, rather than a second, drifting copy of it.
TRAIN_CONFIG_NAME = "train"

#: Config groups that only exist under :data:`TRAIN_CONFIG_NAME`. An override
#: targeting one of them switches the primary config; see
#: :func:`primary_config_name`.
TRAIN_ONLY_GROUPS = ("experiment", "fusion")

#: Paths that arrive with a clone. A missing one is an error: it means the
#: checkout is broken, not that data has yet to be downloaded.
TRACKED_INPUT_KEYS = (
    "paths.repo_root",
    "paths.clam_root",
    "paths.splits_root",
    "paths.dataset_csv_dir",
    "paths.labels_dir",
)

#: Paths that only exist once data has been acquired. Reported, never fatal —
#: a fresh clone legitimately has none of them.
ACQUIRED_DATA_KEYS = (
    "paths.data_root",
    "paths.scratch_root",
    "paths.results_root",
    "paths.tcga_embeddings",
    "paths.cptac_embeddings",
    "paths.cnv_dir",
    "paths.cnv_tabular_dir",
    "paths.cptac_validation_dir",
)


def primary_config_name(overrides: Sequence[str] = ()) -> str:
    """Which primary config an override list needs: `config.yaml` or `train.yaml`.

    `experiment=…` and `fusion=…` are groups of `train.yaml` only, so composing
    `config.yaml` with either used to die with Hydra's "Could not override
    'experiment'. No match in the defaults list" — and Hydra's own suggested
    remedy, `+experiment=…`, then failed one level deeper, because every
    experiment file carries `override /fusion:` and `fusion` was not in the
    defaults list either.

    The cost of that was not cosmetic: `dp-config validate`'s `topk` preflight
    fires only when `clam.inst_loss=svm`, which no config sets except
    `experiment/pam50_wsi_final.yaml`. So the guard that exists to catch a broken
    `--inst_loss svm` dependency BEFORE a 10-fold run could not be reached by any
    documented invocation — and `future`, an undeclared transitive dependency of
    the `topk` git pin, is exactly the failure it was meant to catch.

    Switching the primary config rather than duplicating the groups keeps one
    definition of what an experiment composes to: `dp-config validate
    experiment=X` validates the same tree `dp-train experiment=X` runs.
    """
    for override in overrides:
        # Strip Hydra's override prefixes so `+experiment=…` and `~fusion` are
        # recognised too; they are rejected later, by name, as config surgery.
        key = override.lstrip("+~").split("=", 1)[0].strip()
        if key in TRAIN_ONLY_GROUPS:
            return TRAIN_CONFIG_NAME
    return CONFIG_NAME


def compose_config(
    overrides: Sequence[str] = (), *, with_hydra: bool = False
) -> DictConfig:
    """Compose the packaged config outside `@hydra.main`.

    `initialize_config_dir` with an absolute path, so the result does not depend
    on the caller's working directory — which is the whole point of the refactor.

    The primary config is `config.yaml` unless an override names a group that only
    `train.yaml` carries; see :func:`primary_config_name`.
    """
    from hydra import compose, initialize_config_dir

    config_name = primary_config_name(overrides)
    if config_name == TRAIN_CONFIG_NAME:
        # `${dp.required:…}` is registered as a side effect of importing dp-train.
        # Without it, an experiment that marks a value mandatory (the WSI+RNA arm's
        # `pretrained_wsi_ckpt`) fails here with OmegaConf's "Unsupported
        # interpolation type" instead of the message that names the key.
        from . import train as _train  # noqa: F401

    schema.register_configs()
    with initialize_config_dir(
        version_base=VERSION_BASE, config_dir=str(conf_dir()), job_name="dp-config"
    ):
        return compose(
            config_name=config_name,
            overrides=list(overrides),
            return_hydra_config=with_hydra,
        )


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #


def cmd_show(args: argparse.Namespace) -> int:
    cfg = compose_config(args.overrides)
    schema.reject_appended_overrides(
        args.overrides, allow=bool(cfg.run.allow_config_surgery)
    )
    print(OmegaConf.to_yaml(cfg, resolve=not args.no_resolve), end="")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = compose_config(args.overrides)
    problems: list[str] = []

    try:
        schema.reject_appended_overrides(
            args.overrides, allow=bool(cfg.run.allow_config_surgery)
        )
    except ValueError as exc:
        problems.append(str(exc))

    try:
        assert_paths_absolute(cfg)
    except ValueError as exc:
        problems.append(str(exc))

    missing_tracked = [
        f"{key}={OmegaConf.select(cfg, key)}"
        for key in TRACKED_INPUT_KEYS
        if not _exists(cfg, key)
    ]
    if missing_tracked:
        problems.append(
            "Tracked inputs are missing — these ship with the clone, so the checkout "
            f"is incomplete: {', '.join(missing_tracked)}"
        )

    # Only meaningful once CLAM is actually on disk; otherwise the missing-input
    # problem above is the real diagnosis and a parser read would bury it.
    drift: list[str] = []
    clam_flags = "not checked (project/CLAM/main.py not found)"
    if Path(str(cfg.paths.clam_root), "main.py").exists():
        drift = _clam_schema_drift()
        clam_flags = (
            "DRIFT" if drift else f"{len(clam_args.clam_dests())} (schema in sync)"
        )
        if drift:
            problems.append(
                "dpcode.schema.ClamConf has drifted from project/CLAM/main.py:\n  - "
                + "\n  - ".join(drift)
            )

    problems.extend(_inst_loss_dependency_problems(cfg))

    not_acquired = [key for key in ACQUIRED_DATA_KEYS if not _exists(cfg, key)]

    print(f"config       : {conf_dir() / (primary_config_name(args.overrides) + '.yaml')}")
    print(f"repo_root    : {cfg.paths.repo_root}")
    print(f"overrides    : {list(args.overrides) or '(none)'}")
    print(f"CLAM flags   : {clam_flags}")
    if not_acquired:
        print("not acquired : " + ", ".join(not_acquired))
        print("               (expected on a fresh clone; see the reproduction docs)")

    if problems:
        print()
        sys.stdout.flush()  # keep the summary above the failures, not interleaved
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print("OK")
    return 0


def cmd_reference(args: argparse.Namespace) -> int:
    text = render_reference()
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_sync_check(args: argparse.Namespace) -> int:
    drift = _clam_schema_drift()
    n_flags = len(clam_args.clam_dests())
    if drift:
        print(
            f"DRIFT between project/CLAM/main.py ({n_flags} flags), "
            "dpcode.schema.ClamConf and dpcode/conf/clam/base.yaml:",
            file=sys.stderr,
        )
        for item in drift:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print(f"in sync: {n_flags} CLAM flags in main.py, ClamConf and clam/base.yaml")
    return 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _exists(cfg: DictConfig, key: str) -> bool:
    value = OmegaConf.select(cfg, key)
    return value is not None and Path(str(value)).exists()


def _inst_loss_dependency_problems(cfg: DictConfig) -> list[str]:
    """`--inst_loss svm` needs `topk`, which is a git pin with no PyPI fallback.

    `project/CLAM/utils/core_utils.py` imports `topk.svm` only once training has
    started, so a missing `topk` surfaces minutes into a job rather than here.
    The frozen WSI baseline (`pam50_final_s1`) is an `--inst_loss svm` run, so
    this is on the path to the headline number, not a corner case.
    """
    if str(OmegaConf.select(cfg, "clam.inst_loss")) != "svm":
        return []
    try:
        import topk.svm  # noqa: F401
    except ImportError as exc:
        return [
            "clam.inst_loss=svm needs the `topk` distribution (smooth-topk), which "
            f"is not importable: {exc}. It is a git pin with no PyPI fallback — "
            "`pip install -e .` needs git and network for it. Without it CLAM "
            "fails minutes into training, in utils/core_utils.py."
        ]
    return []


def _clam_schema_drift() -> list[str]:
    """Compare CLAM's real parser against `ClamConf` and `clam/base.yaml`.

    Names, defaults and types, in both directions. This is what stops a flag added
    to `main.py` from being silently unreachable through the config.
    """
    parser = clam_args.clam_parser()
    actions = clam_args.clam_actions(parser)
    by_dest = {action.dest: action for action in actions}

    hints = typing.get_type_hints(schema.ClamConf)
    fields = {f.name: f for f in dataclasses.fields(schema.ClamConf)}
    base_yaml = OmegaConf.load(conf_dir() / "clam" / "base.yaml")
    base = OmegaConf.to_container(base_yaml, resolve=False)
    assert isinstance(base, dict)

    problems: list[str] = []

    for dest in sorted(set(by_dest) - set(fields)):
        problems.append(f"ClamConf is missing `{dest}` (present in main.py)")
    for name in sorted(set(fields) - set(by_dest)):
        problems.append(f"ClamConf has `{name}`, which main.py does not accept")
    for dest in sorted(set(by_dest) - set(base)):
        problems.append(f"clam/base.yaml is missing `{dest}`")
    for name in sorted(set(base) - set(by_dest)):
        problems.append(f"clam/base.yaml has `{name}`, which main.py does not accept")

    for dest, action in by_dest.items():
        field = fields.get(dest)
        if field is not None and field.default != action.default:
            problems.append(
                f"`{dest}`: ClamConf default {field.default!r} != main.py default "
                f"{action.default!r}"
            )
        if dest in schema.PATH_INTERPOLATION_FIELDS:
            # A5: these three are interpolations of `paths` in base.yaml, so the
            # value comparison does not apply. CLAM's real default is still
            # pinned — on ClamConf, checked just above. What base.yaml owes is
            # that it did not quietly become a literal again.
            value = base.get(dest)
            if not (isinstance(value, str) and "${paths." in value):
                problems.append(
                    f"`{dest}`: clam/base.yaml value {value!r} must interpolate the "
                    "`paths` group (DESIGN-ADDENDUM A5), otherwise CLAM's output "
                    "directory can be decoupled from dpcode's metadata directory."
                )
        elif dest in base and base[dest] != action.default:
            problems.append(
                f"`{dest}`: clam/base.yaml value {base[dest]!r} != main.py default "
                f"{action.default!r}"
            )
        expected = _expected_annotation(action)
        actual = hints.get(dest)
        if field is not None and actual != expected:
            problems.append(
                f"`{dest}`: ClamConf type {actual} != {expected} implied by main.py"
            )

    return problems


def _expected_annotation(action: argparse.Action) -> Any:
    """The dataclass annotation an argparse action implies."""
    if isinstance(action, argparse._StoreTrueAction):
        return bool
    # argparse leaves `type=None` meaning "keep the string".
    base = action.type or str
    if action.nargs in ("+", "*"):
        base = typing.List[base]  # type: ignore[valid-type]
    if action.default is None:
        return typing.Optional[base]  # type: ignore[valid-type]
    return base


def render_reference() -> str:
    """Render the configuration reference as markdown.

    Values are shown UNRESOLVED. Resolving them would bake this machine's absolute
    paths into a tracked document, which is exactly what the refactor removes.
    """
    cfg = compose_config()
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(raw, dict)

    out = io.StringIO()
    out.write("# Configuration reference\n\n")
    out.write(
        "Generated by `dp-config reference`. Do not edit by hand.\n\n"
        "Values are shown **unresolved**: `${...}` interpolations are what the config\n"
        "actually contains, and resolving them here would hard-code one machine's\n"
        "absolute paths into a tracked document. Run `dp-config show` to see the\n"
        "resolved values for the machine you are on.\n\n"
    )

    out.write("## Config groups\n\n")
    out.write("| group | options | selected by default |\n|---|---|---|\n")
    for group in sorted(p.name for p in conf_dir().iterdir() if p.is_dir()):
        options = sorted(p.stem for p in (conf_dir() / group).glob("*.yaml"))
        default = _default_option(group)
        out.write(f"| `{group}` | {', '.join(f'`{o}`' for o in options)} | `{default}` |\n")
    out.write(
        "\nGroups added by the training, analysis, evaluation and acquisition entry\n"
        "points appear once those entry points are installed.\n\n"
    )

    dataclass_docs = {
        "paths": schema.PathsConf,
        "sources": schema.SourcesConf,
        "clam": schema.ClamConf,
        "fusion": schema.FusionConf,
        "tracking": schema.TrackingConf,
        "run": schema.RunConf,
    }

    for node_name, node in raw.items():
        if not isinstance(node, dict):
            continue
        out.write(f"## `{node_name}`\n\n")
        doc = dataclass_docs.get(node_name)
        if doc is not None and doc.__doc__:
            out.write(_dedent_docstring(doc.__doc__) + "\n\n")
        if node_name == "clam":
            out.write(_clam_table())
        else:
            out.write("| key | value |\n|---|---|\n")
            for key, value in node.items():
                out.write(f"| `{node_name}.{key}` | `{value!r}` |\n")
        out.write("\n")

    return out.getvalue()


def _default_option(group: str) -> str:
    primary = OmegaConf.load(conf_dir() / f"{CONFIG_NAME}.yaml")
    for entry in primary.get("defaults", []):
        if isinstance(entry, (dict, DictConfig)) and group in entry:
            return str(entry[group])
    return "-"


def _clam_table() -> str:
    parser = clam_args.clam_parser()
    out = io.StringIO()
    out.write(
        "Every field is an argument of `project/CLAM/main.py`, extracted from that\n"
        "file's own parser. Help text and defaults are CLAM's.\n\n"
        "Three of them are NOT composed at the default shown here:\n"
        + ", ".join(f"`clam.{name}`" for name in schema.PATH_INTERPOLATION_FIELDS)
        + ".\n`clam/base.yaml` sets those to interpolations of the `paths` group, so\n"
        "CLAM's output directory cannot be decoupled from the directory dpcode writes\n"
        "run metadata into. Run `dp-config show` to see what they compose to.\n\n"
    )
    out.write("| key | type | default | choices | help |\n|---|---|---|---|---|\n")
    for action in clam_args.clam_actions(parser):
        kind = (
            "flag"
            if isinstance(action, argparse._StoreTrueAction)
            else getattr(action.type, "__name__", "str")
        )
        choices = ", ".join(f"`{c}`" for c in action.choices) if action.choices else ""
        help_text = " ".join((action.help or "").split()).replace("|", "\\|")
        out.write(
            f"| `clam.{action.dest}` | {kind} | `{action.default!r}` | {choices} | {help_text} |\n"
        )
    return out.getvalue()


def _dedent_docstring(text: str) -> str:
    lines = text.strip("\n").splitlines()
    if not lines:
        return ""
    head, *rest = lines
    indents = [len(l) - len(l.lstrip()) for l in rest if l.strip()]
    pad = min(indents) if indents else 0
    return "\n".join([head.strip(), *(l[pad:] for l in rest)]).strip()


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dp-config", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show", help="compose and print the config")
    show.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. clam.k=5")
    show.add_argument(
        "--no-resolve",
        action="store_true",
        help="print interpolations verbatim instead of resolving them",
    )
    show.set_defaults(func=cmd_show)

    validate = subparsers.add_parser("validate", help="run every startup guard")
    validate.add_argument("overrides", nargs="*", help="Hydra overrides")
    validate.set_defaults(func=cmd_validate)

    reference = subparsers.add_parser("reference", help="generate the config reference")
    reference.add_argument(
        "-o", "--output", default=None, help="write markdown here instead of stdout"
    )
    reference.set_defaults(func=cmd_reference)

    sync = subparsers.add_parser(
        "sync-check", help="check ClamConf and clam/base.yaml against CLAM's parser"
    )
    sync.set_defaults(func=cmd_sync_check)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from hydra.errors import HydraException
    from omegaconf.errors import OmegaConfBaseException

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (
        HydraException,
        OmegaConfBaseException,
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
    ) as exc:
        # A composition failure is a user error (a typo'd or mistyped override),
        # not a crash. Print the message, not a traceback through Hydra internals.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
