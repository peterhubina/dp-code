"""`ClamConf` is pinned to CLAM's real parser, field for field.

This is the check that stops the config schema from drifting away from
`project/CLAM/main.py`. An argument added to CLAM but not to `ClamConf` is not a
cosmetic omission: struct mode makes it unreachable from any config file and
`render_argv` refuses to emit it, so the flag becomes silently unusable through
`dp-train` while still existing in CLAM. Conversely a field here that CLAM does
not have would be rendered into argv and rejected by argparse at dispatch, after
the run directory and its metadata had been written.

The parser is EXTRACTED, not imported: `main.py` runs 300 lines at import time
(it calls `parse_args()` against the importing process's `sys.argv`, seeds torch,
builds the dataset and writes `experiment_<exp_code>.txt`). `dpcode.clam_args`
compiles only the slice between `parser = argparse.ArgumentParser(...)` and
`args = parser.parse_args()`.

The second half of the module is DESIGN-ADDENDUM A5: three `clam` keys must be
INTERPOLATIONS of the `paths` group in `clam/base.yaml` rather than literals. If
`clam.results_dir` and `paths.results_root` were independent literals, one
override would put CLAM's checkpoints in one tree and `config.resolved.yaml` /
`run_metadata.json` / `metrics.json` in another, and the run record would name a
directory that holds nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import typing
from typing import Any

import pytest
import yaml

from conftest import REAL_REPO

from dpcode import clam_args
from dpcode.paths import conf_dir
from dpcode.schema import PATH_INTERPOLATION_FIELDS, ClamConf

BASE_YAML = conf_dir() / "clam" / "base.yaml"


@pytest.fixture(scope="module")
def parser() -> argparse.ArgumentParser:
    return clam_args.clam_parser(REAL_REPO / "project/CLAM/main.py")


@pytest.fixture(scope="module")
def schema_fields() -> dict[str, dataclasses.Field]:
    return {f.name: f for f in dataclasses.fields(ClamConf)}


def expected_annotation(action: argparse.Action) -> Any:
    """What `ClamConf` must declare for `action`, from the parser alone."""
    if isinstance(action, argparse._StoreTrueAction):
        return bool
    if action.nargs in ("+", "*"):
        inner = typing.List[action.type or str]  # type: ignore[valid-type]
        return typing.Optional[inner] if action.default is None else inner
    # `--results_dir` declares no `type=`, which argparse means as "leave the
    # string alone".
    base = action.type or str
    return typing.Optional[base] if action.default is None else base


def test_every_clam_argument_has_a_schema_field(parser, schema_fields) -> None:
    missing = [dest for dest in clam_args.clam_dests(parser) if dest not in schema_fields]
    assert not missing, (
        f"project/CLAM/main.py has arguments that dpcode.schema.ClamConf does not: "
        f"{missing}. They are unreachable from every config file until they are added "
        "to ClamConf AND to dpcode/conf/clam/base.yaml."
    )


def test_no_schema_field_is_unknown_to_clam(parser, schema_fields) -> None:
    extra = sorted(set(schema_fields) - set(clam_args.clam_dests(parser)))
    assert not extra, (
        f"dpcode.schema.ClamConf declares fields CLAM's parser does not accept: {extra}. "
        "render_argv would emit them and argparse would reject the command after the "
        "run directory had been created."
    )


def test_field_count_matches(parser, schema_fields) -> None:
    assert len(schema_fields) == len(clam_args.clam_dests(parser)) == 52


@pytest.mark.parametrize(
    "dest",
    clam_args.clam_dests(clam_args.clam_parser(REAL_REPO / "project/CLAM/main.py")),
)
def test_default_and_type_match_clam(dest: str, parser, schema_fields) -> None:
    action = next(a for a in clam_args.clam_actions(parser) if a.dest == dest)
    field = schema_fields[dest]

    assert type_name(field.type) == type_name(expected_annotation(action)), (
        f"{dest}: ClamConf says {field.type!r}, CLAM's parser implies "
        f"{expected_annotation(action)!r}"
    )

    # `from __future__ import annotations` in schema.py makes every field default
    # a plain value, so this is a direct comparison.
    assert field.default == action.default, (
        f"{dest}: ClamConf default {field.default!r} != CLAM's {action.default!r}. "
        "ClamConf records CLAM's OWN defaults; an experiment that wants a different "
        "value states it in its own config file."
    )


def type_name(annotation: Any) -> str:
    """A comparable spelling of a type.

    `dpcode/schema.py` has `from __future__ import annotations`, so
    `dataclasses.fields()` reports the SOURCE TEXT of each annotation
    (`'Optional[str]'`), while the parser side yields real objects
    (`typing.Optional[str]`). Both are reduced to the same string rather than
    resolved with `typing.get_type_hints`, which would need the module's
    namespace and would happily paper over a typo that names a different type.
    """
    text = annotation if isinstance(annotation, str) else str(annotation)
    text = text.replace("typing.", "")
    if text.startswith("<class '") and text.endswith("'>"):
        text = text[len("<class '") : -2]
    return text.replace(" ", "")


# --------------------------------------------------------------------------- #
# DESIGN-ADDENDUM A5 — the three path fields
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def base_yaml() -> dict[str, Any]:
    """`clam/base.yaml` as TEXT, unresolved.

    Read with PyYAML rather than OmegaConf on purpose: this fixture exists to
    inspect the interpolations as written, and `OmegaConf.load` would resolve
    nothing here but does apply its own scalar rules (it reads `reg: 1e-05` as a
    float where PyYAML 1.1 reads a string), which is the wrong lens for a
    "what does this file literally say" check. Value comparisons use the composed
    config instead — see `composed_clam`.
    """
    return yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def composed_clam() -> dict[str, Any]:
    """`cfg.clam` as `dp-train` sees it: schema-typed, interpolations resolved.

    This is the lens that matters for defaults — OmegaConf casts a YAML scalar to
    the type `ClamConf` declares, so what reaches CLAM is the typed value, not
    the text.
    """
    from omegaconf import OmegaConf

    from dpcode.cli.config import compose_config

    return OmegaConf.to_container(compose_config([]).clam, resolve=True)


@pytest.mark.parametrize("key", PATH_INTERPOLATION_FIELDS)
def test_path_fields_are_interpolations_not_literals(key: str, base_yaml) -> None:
    value = base_yaml[key]
    assert isinstance(value, str) and "${paths." in value, (
        f"clam/base.yaml:{key} is {value!r}. It must interpolate the `paths` group "
        "(DESIGN-ADDENDUM A5): as an independent literal, one override would split a "
        "run's checkpoints from its metadata directory, silently."
    )


def test_base_yaml_otherwise_records_clams_own_defaults(parser, composed_clam) -> None:
    """Every other key of the composed `clam` node is CLAM's default, verbatim.

    The one shared default that would change the training objective is
    `--no_inst_cluster`: two of the three trainers pass it, which makes it a
    tempting base default, and adopting it would silently retrain the WSI
    baseline under a different loss.
    """
    actions = {a.dest: a for a in clam_args.clam_actions(parser)}
    drifted = {
        key: (value, actions[key].default)
        for key, value in composed_clam.items()
        if key not in PATH_INTERPOLATION_FIELDS
        and key in actions
        and value != actions[key].default
    }
    assert not drifted, (
        "dpcode/conf/clam/base.yaml has values that are not CLAM's own defaults: "
        + ", ".join(f"{k}={v[0]!r} (CLAM: {v[1]!r})" for k, v in drifted.items())
    )


def test_base_yaml_covers_every_argument(parser, base_yaml) -> None:
    missing = sorted(set(clam_args.clam_dests(parser)) - set(base_yaml))
    assert not missing, f"dpcode/conf/clam/base.yaml is missing {missing}"
