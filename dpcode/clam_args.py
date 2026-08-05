"""CLAM's real argparse surface, extracted without executing CLAM.

`project/CLAM/main.py` has no import-time safety. Lines 119-433 build the parser,
call `parse_args()` against the *importing* process's `sys.argv`, seed torch,
construct the dataset, create the results directory and write
`experiment_<exp_code>.txt`; only `main(args)` sits behind the `__main__` guard.
So the module cannot be imported to ask it what flags it takes.

Instead the file is parsed with `ast`, the statements between
``parser = argparse.ArgumentParser(...)`` and ``args = parser.parse_args()`` are
compiled and executed in a namespace whose only name is `argparse`, and the
resulting `ArgumentParser` is handed back. Nothing else in `main.py` runs.

The point of extracting rather than re-declaring is that the config schema then
*cannot* drift from CLAM: add a flag to `main.py` and the drift shows up as a
failing check, not as a silently unreachable option.

Parsing cleanly is not the whole contract. `main.py:214-232`, *after*
`parse_args()`, runs eight `parser.error(...)` cross-argument checks — the ones
that decide which of the six fusion operators is actually runnable. A config that
forgets `tabular_group_spec` under `coattn` parses fine, passes every parity
check, and then dies at CLAM startup. So that block is extracted too, by the same
AST route, wrapped into a callable that takes `(args, parser)`; nothing else in
`main.py` runs, and `validate_clam_args` reports CLAM's own message.

The public functions:

``clam_parser()``      the genuine parser, with CLAM's own defaults/types/choices
``validate_clam_args(ns)``  CLAM's cross-argument checks, raising
                       :class:`ClamArgumentError` with CLAM's own wording
``render_argv(cfg)``   config -> the argv list handed to `python main.py`
``parse_for_comparison(argv, cwd)``  argv -> a normalised namespace dict, which is
                       what the parity test compares between the legacy shell
                       wrappers and the Hydra entry points
"""

from __future__ import annotations

import argparse
import ast
import builtins
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .paths import repo_root

__all__ = [
    "CLAM_MAIN_RELPATH",
    "PATH_VALUED_DESTS",
    "ClamArgumentError",
    "clam_main_path",
    "clam_parser",
    "clam_actions",
    "clam_dests",
    "clam_validator",
    "validate_clam_args",
    "render_argv",
    "parse_for_comparison",
]


class ClamArgumentError(ValueError):
    """A cross-argument check from `main.py:214-232` rejected the configuration.

    The message is CLAM's own, verbatim, so a config error reads the same whether
    it is caught here before dispatch or by `main.py` after it. `ValueError` so
    that the CLI's existing error handling prints it as a user error rather than
    a traceback.
    """

CLAM_MAIN_RELPATH = "project/CLAM/main.py"

# Fields whose value is a filesystem path and must therefore be compared after
# normalisation, not as a string. Derived from the parser as well (see
# `_path_valued_dests`), but listed explicitly because the naming convention
# alone gets two of them wrong:
#
#   data_root_dir        directory of .h5 features
#   results_dir          CLAM appends `<exp_code>_s<seed>` to this
#   split_dir            NOT a path as passed: CLAM prefixes it with the literal
#                        `splits/` (main.py:412,414) and asserts it relative to
#                        CWD, so callers pass a bare name. Normalising it against
#                        the same cwd on both sides still compares correctly.
#   tabular_csv          the CNV/RNA feature table
#   tabular_group_spec   a signature CSV path, OR the literal string "prefix"
#                        (which is not a path at all; it normalises harmlessly
#                        and identically on both sides)
#   pretrained_wsi_ckpt  may contain a literal `{fold}` placeholder
#   pretrained_rna_ckpt  may contain a literal `{fold}` placeholder
PATH_VALUED_DESTS = (
    "data_root_dir",
    "results_dir",
    "split_dir",
    "tabular_csv",
    "tabular_group_spec",
    "pretrained_wsi_ckpt",
    "pretrained_rna_ckpt",
)

# `{fold}` must survive normalisation untouched. `os.path.realpath` would not
# actually mangle braces today, but swapping the placeholder for a plain
# component first means no filesystem call ever sees it, so the guarantee does
# not depend on realpath's treatment of unusual characters.
_FOLD_PLACEHOLDER = "{fold}"
_FOLD_SENTINEL = "__DP_FOLD_PLACEHOLDER__"


def clam_main_path() -> Path:
    """Absolute path of the vendored CLAM `main.py`."""
    return repo_root() / CLAM_MAIN_RELPATH


def clam_parser(main_py: str | os.PathLike[str] | None = None) -> argparse.ArgumentParser:
    """Return CLAM's real `ArgumentParser`, built without importing `main.py`."""
    path = Path(main_py) if main_py is not None else clam_main_path()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    start = _index_of(tree.body, _is_parser_construction)
    if start is None:
        raise RuntimeError(
            f"{path}: no `parser = argparse.ArgumentParser(...)` statement at module level."
        )
    end = _index_of(tree.body, _is_parse_args_call, after=start)
    if end is None:
        raise RuntimeError(
            f"{path}: no `... = parser.parse_args()` statement after the parser construction."
        )

    slice_module = ast.Module(body=tree.body[start:end], type_ignores=[])
    code = compile(slice_module, filename=str(path), mode="exec")
    # `argparse` is the only name provided. If CLAM ever adds a statement in this
    # range that needs something else, this raises NameError here rather than
    # quietly running unrelated code.
    namespace: dict[str, Any] = {"argparse": argparse}
    exec(code, namespace)  # noqa: S102 - a hand-checked slice of a tracked file

    parser = namespace.get("parser")
    if not isinstance(parser, argparse.ArgumentParser):
        raise RuntimeError(f"{path}: extracted slice did not produce an ArgumentParser.")
    return parser


def clam_actions(parser: argparse.ArgumentParser | None = None) -> list[argparse.Action]:
    """CLAM's arguments in declaration order, excluding argparse's own `-h`."""
    parser = parser or clam_parser()
    return [a for a in parser._actions if not isinstance(a, argparse._HelpAction)]


def clam_dests(parser: argparse.ArgumentParser | None = None) -> list[str]:
    """The `dest` name of every CLAM argument, in declaration order."""
    return [a.dest for a in clam_actions(parser)]


def clam_validator(
    main_py: str | os.PathLike[str] | None = None,
) -> Callable[[argparse.Namespace, Any], None]:
    """Return CLAM's cross-argument validation as a callable ``(args, parser)``.

    Extracted from `main.py:214-232` by the same AST route as the parser: the
    module-level statements that sit between `parse_args()` and the first
    function definition AND contain a `parser.error(...)` call are lifted into a
    synthetic `def`. That selection rule is what excludes `main.py:212`
    (`device = torch.device(...)`), which lives in the same range and would drag
    torch — and a CUDA context — into a config check.

    The extraction is strict in three ways, because a validator that silently
    validates nothing is worse than no validator: it raises if it finds no
    checks, if it finds fewer `parser.error` calls than the source contains in
    that range, or if the lifted code reads any name other than `args`, `parser`
    and builtins.
    """
    path = Path(main_py) if main_py is not None else clam_main_path()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    start = _index_of(tree.body, _is_parse_args_call)
    if start is None:
        raise RuntimeError(f"{path}: no `... = parser.parse_args()` statement at module level.")
    end = _index_of(
        tree.body,
        lambda node: isinstance(node, (ast.FunctionDef, ast.ClassDef)),
        after=start,
    )
    if end is None:
        end = len(tree.body)

    region = tree.body[start + 1 : end]
    checks = [stmt for stmt in region if _counts_parser_errors(stmt)]
    if not checks:
        raise RuntimeError(
            f"{path}: no `parser.error(...)` cross-argument checks found between "
            "parse_args() and the first function definition. CLAM's argument "
            "contract has moved; dpcode.clam_args must be updated to follow it."
        )
    found = sum(_counts_parser_errors(stmt) for stmt in checks)
    expected = sum(_counts_parser_errors(stmt) for stmt in region)
    if found != expected:  # pragma: no cover - defensive; the two lists agree today
        raise RuntimeError(f"{path}: {expected - found} parser.error checks were skipped.")

    unexpected = sorted(_free_names(checks) - {"args", "parser"} - set(dir(builtins)))
    if unexpected:
        raise RuntimeError(
            f"{path}: the cross-argument checks now read {unexpected}, which cannot be "
            "supplied without executing more of main.py. Update dpcode.clam_args."
        )

    function = ast.FunctionDef(
        name="_clam_validate",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="args"), ast.arg(arg="parser")],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=checks,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    module = ast.fix_missing_locations(
        ast.copy_location(ast.Module(body=[function], type_ignores=[]), checks[0])
    )
    namespace: dict[str, Any] = {}
    exec(compile(module, filename=str(path), mode="exec"), namespace)  # noqa: S102
    return namespace["_clam_validate"]


def validate_clam_args(
    namespace: Any,
    *,
    parser: argparse.ArgumentParser | None = None,
    main_py: str | os.PathLike[str] | None = None,
) -> argparse.Namespace:
    """Run CLAM's cross-argument checks; raise :class:`ClamArgumentError` on failure.

    `namespace` may be an `argparse.Namespace`, a `DictConfig`, a dataclass or a
    plain mapping. Absent keys take the parser's default, exactly as argparse
    would, so `validate_clam_args({"fusion_mode": "coattn"})` reports the missing
    `--tabular_group_spec` rather than an AttributeError. Unknown keys raise,
    for the same schema-drift reason `render_argv` does.

    Call this in the entry point BEFORE dispatch. `main.py` runs the identical
    block, but only after Hydra and the entry point have created directories and
    written metadata, and `parser.error` there exits the process with status 2.
    """
    parser = parser or clam_parser(main_py)
    dests = set(clam_dests(parser))

    if isinstance(namespace, argparse.Namespace):
        values = dict(vars(namespace))
    else:
        values = _as_mapping(namespace)
    unknown = sorted(set(values) - dests)
    if unknown:
        raise KeyError(f"Config keys that CLAM's parser does not accept: {unknown}.")

    merged = {action.dest: action.default for action in clam_actions(parser)}
    merged.update(values)
    args = argparse.Namespace(**merged)

    clam_validator(main_py)(args, _ErrorReporter())
    return args


def render_argv(clam_cfg: Any, parser: argparse.ArgumentParser | None = None) -> list[str]:
    """Render a `clam` config node into the argv list for `python main.py`.

    Every non-`None` scalar becomes ``--flag value``; every true `store_true`
    becomes ``--flag``; `nargs='+'` fields become ``--flag v1 v2 ...`` and are
    omitted when empty. That is a superset of what some legacy wrappers passed,
    on purpose: the emitted command is fully explicit, and a flag left at CLAM's
    own default parses to the same value either way, so parsed-namespace parity
    still holds.

    Strict in both directions — an unknown key or a missing key raises — because
    the whole point of the extracted parser is to catch schema drift.
    """
    parser = parser or clam_parser()
    actions = clam_actions(parser)
    values = _as_mapping(clam_cfg)

    known = {a.dest for a in actions}
    unknown = sorted(set(values) - known)
    if unknown:
        raise KeyError(
            f"Config keys that CLAM's parser does not accept: {unknown}. "
            f"Either the flag was removed from {CLAM_MAIN_RELPATH} or the key is a typo."
        )
    absent = sorted(known - set(values))
    if absent:
        raise KeyError(
            f"CLAM arguments missing from the config: {absent}. "
            f"Add them to dpcode.schema.ClamConf and dpcode/conf/clam/base.yaml."
        )

    argv: list[str] = []
    for action in actions:
        value = values[action.dest]
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(flag)
            continue
        if value is None:
            continue
        if action.nargs in ("+", "*"):
            items = list(value)
            if not items:
                continue
            argv.append(flag)
            argv.extend(str(item) for item in items)
            continue
        argv.extend([flag, str(value)])
    return argv


def parse_for_comparison(
    argv: Sequence[str],
    cwd: str | os.PathLike[str],
    parser: argparse.ArgumentParser | None = None,
) -> dict[str, Any]:
    """Parse `argv` with CLAM's parser and normalise it for equality comparison.

    `cwd` is the directory the command would actually run in — the legacy shell
    wrappers `cd project/CLAM` first and pass repo-relative paths, while other
    call sites pass absolute ones. Every path-valued field is resolved against
    `cwd` with `os.path.realpath`, so the two spellings compare equal.

    This is the acceptance comparison for the refactor: two configurations are
    "the same run" iff these dicts are equal.
    """
    parser = parser or clam_parser()
    namespace = vars(parser.parse_args(list(argv)))
    path_dests = _path_valued_dests(parser)
    cwd_str = str(cwd)
    return {
        key: _normalise_path(value, cwd_str) if key in path_dests else value
        for key, value in namespace.items()
    }


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _index_of(body: list[ast.stmt], predicate, after: int = -1) -> int | None:
    for index in range(after + 1, len(body)):
        if predicate(body[index]):
            return index
    return None


def _is_parser_construction(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
        return False
    if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
        return False
    if node.targets[0].id != "parser":
        return False
    func = node.value.func
    if isinstance(func, ast.Attribute):
        return func.attr == "ArgumentParser"
    return isinstance(func, ast.Name) and func.id == "ArgumentParser"


class _ErrorReporter:
    """Stands in for the `ArgumentParser` inside the extracted validation.

    The real `parser.error` prints usage to stderr and raises `SystemExit(2)`,
    which is right for `main.py` and useless to a library caller. Only `.error`
    is reachable from the lifted block, and the message it is given is CLAM's.
    """

    def error(self, message: str):  # noqa: D102 - mirrors argparse's signature
        raise ClamArgumentError(message)


def _counts_parser_errors(node: ast.stmt) -> int:
    """How many `parser.error(...)` calls this statement contains, at any depth."""
    return sum(
        1
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "error"
        and isinstance(child.func.value, ast.Name)
        and child.func.value.id == "parser"
    )


def _free_names(nodes: list[ast.stmt]) -> set[str]:
    """Every name READ by `nodes`. Attribute bases count; attributes do not."""
    names: set[str] = set()
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                names.add(child.id)
    return names


def _is_parse_args_call(node: ast.stmt) -> bool:
    value = node.value if isinstance(node, (ast.Assign, ast.Expr)) else None
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "parse_args"
        and isinstance(func.value, ast.Name)
        and func.value.id == "parser"
    )


def _path_valued_dests(parser: argparse.ArgumentParser) -> frozenset[str]:
    """The explicit list, validated against the parser and widened by convention.

    The explicit list is authoritative and must exist in the parser; the
    convention-based sweep (`*_dir`, `*_csv`, `*_ckpt`, `*_path`) catches a
    path-valued flag added to CLAM later without anyone updating this module.
    """
    dests = set(clam_dests(parser))
    unknown = sorted(set(PATH_VALUED_DESTS) - dests)
    if unknown:
        raise RuntimeError(
            f"PATH_VALUED_DESTS names arguments CLAM no longer has: {unknown}."
        )
    by_convention = {
        d for d in dests if d.endswith(("_dir", "_csv", "_ckpt", "_path"))
    }
    return frozenset(set(PATH_VALUED_DESTS) | by_convention)


def _normalise_path(value: Any, cwd: str) -> Any:
    if value is None:
        return None
    text = str(value)
    guarded = text.replace(_FOLD_PLACEHOLDER, _FOLD_SENTINEL)
    resolved = os.path.realpath(os.path.join(cwd, guarded))
    return resolved.replace(_FOLD_SENTINEL, _FOLD_PLACEHOLDER)


def _as_mapping(clam_cfg: Any) -> dict[str, Any]:
    if is_dataclass(clam_cfg) and not isinstance(clam_cfg, type):
        return asdict(clam_cfg)
    try:
        from omegaconf import OmegaConf
    except ImportError:  # pragma: no cover - omegaconf is a hard dependency
        OmegaConf = None  # type: ignore[assignment]
    if OmegaConf is not None and OmegaConf.is_config(clam_cfg):
        return OmegaConf.to_container(clam_cfg, resolve=True, throw_on_missing=True)  # type: ignore[return-value]
    if isinstance(clam_cfg, Mapping):
        return dict(clam_cfg)
    raise TypeError(
        f"render_argv expects a DictConfig, dataclass or Mapping, got {type(clam_cfg).__name__}."
    )
