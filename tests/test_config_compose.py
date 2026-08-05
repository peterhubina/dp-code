"""Every config-group option composes, and a typo cannot survive composition.

A config group option that does not compose is not a latent bug, it is a broken
command: `dp-train experiment=<x>` fails before it does anything. Composing all
of them here is cheap (no data, no torch) and it is the only check that covers
the options no test exercises by name.

The two negative cases are the point of using structured configs at all:

  * `clam.lrr=0.1` must die AT COMPOSITION with the key name. Untyped, it would
    be accepted, ignored, recorded in `.hydra/overrides.yaml`, logged to W&B, and
    read by nothing — the run would look configured and would not be.
  * `+clam.lrr=0.1` must ALSO be refused. It is the remedy Hydra's own error
    message suggests, it silently succeeds, and under `--multirun` a `+`-prefixed
    typo produces N identical runs with N different directory names.

`fusion=residual` is the third: it must fail at composition rather than at
dispatch, because `dp-train -m 'fusion=glob(*)'` is Hydra's taught idiom and
would otherwise create a run directory and write metadata for an operator that
cannot train (no supported trainer produces the `--pretrained_rna_ckpt` it
requires — a Known gap, documented, not fixed here).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

# Imported for its side effect: `dpcode/cli/train.py` registers the
# `${dp.required:…}` resolver, which `experiment=pam50_wsi_rna_gated` resolves.
import dpcode.cli.train  # noqa: F401
from dpcode import schema
from dpcode.cli.config import compose_config
from dpcode.paths import conf_dir

CONF = conf_dir()


def options(group: str) -> list[str]:
    return sorted(p.stem for p in (CONF / group).glob("*.yaml"))


def compose_train(overrides: list[str]):
    schema.register_configs()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF), job_name="test"):
        return compose(config_name="train", overrides=overrides)


# --------------------------------------------------------------------------- #
# every option composes
# --------------------------------------------------------------------------- #


#: Experiments that DEMAND something from the caller, and the minimum that
#: satisfies each. Both refusals are deliberate design, not gaps:
#:   pam50_wsi_cnv       `override /fusion: ???` — a missing operator would
#:                       compose a WSI-only run named `pam50_wsi_cnv_none_s1`
#:                       that looks like a ladder arm and is not one;
#:   pam50_wsi_rna_gated `${dp.required:clam.pretrained_wsi_ckpt}` — the wrapper
#:                       it replaces exited 2 without `--pretrained_wsi_ckpt`.
REQUIRED_ARGUMENTS = {
    "pam50_wsi_cnv": ["fusion=concat"],
    "pam50_wsi_rna_gated": [
        'clam.pretrained_wsi_ckpt="/abs/pam50_final_s1/s_{fold}_checkpoint.pt"'
    ],
}


@pytest.mark.parametrize("experiment", options("experiment"))
def test_every_experiment_composes(experiment: str) -> None:
    cfg = compose_train([f"experiment={experiment}", *REQUIRED_ARGUMENTS.get(experiment, [])])
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert resolved["clam"]["exp_code"], f"{experiment} sets no clam.exp_code"
    assert resolved["clam"]["task"], f"{experiment} sets no clam.task"
    assert resolved["run"]["name"] != "unnamed", f"{experiment} leaves run.name unset"


@pytest.mark.parametrize("experiment", sorted(REQUIRED_ARGUMENTS))
def test_experiments_that_demand_an_argument_refuse_without_it(experiment: str) -> None:
    from omegaconf.errors import InterpolationResolutionError

    with pytest.raises((ConfigCompositionException, InterpolationResolutionError)):
        cfg = compose_train([f"experiment={experiment}"])
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


@pytest.mark.parametrize("operator", [o for o in options("fusion") if o != "residual"])
def test_every_fusion_operator_composes(operator: str) -> None:
    cfg = compose_train(["experiment=pam50_wsi_cnv", f"fusion={operator}"])
    assert cfg.fusion.name == operator
    if operator == "none":
        # `--fusion_mode` has no `none` value: the unimodal case must OMIT the
        # flag, which `render_argv` does for a None scalar.
        assert cfg.clam.fusion_mode is None
    else:
        assert cfg.clam.fusion_mode == operator


@pytest.mark.parametrize("option", options("analyses"))
def test_every_analysis_option_composes(option: str) -> None:
    cfg = compose_config([f"+analyses={option}"])
    assert cfg.analysis.action == option


@pytest.mark.parametrize("option", options("evaluate"))
def test_every_evaluate_option_composes(option: str) -> None:
    cfg = compose_config([f"+evaluate={option}"])
    assert cfg.evaluate.args is not None


@pytest.mark.parametrize("option", options("acquire"))
def test_every_acquire_option_composes(option: str) -> None:
    cfg = compose_config([f"+acquire={option}"])
    assert cfg.acquire is not None


@pytest.mark.parametrize("option", options("cptac"))
def test_every_cptac_option_composes(option: str) -> None:
    cfg = compose_config([f"+cptac={option}"])
    assert cfg.cptac is not None


@pytest.mark.parametrize("option", options("tracking"))
def test_every_tracking_option_composes(option: str) -> None:
    # `off` must be quoted on the command line: bare `off` is a YAML 1.1 boolean.
    cfg = compose_config([f'tracking="{option}"'])
    assert cfg.tracking.mode in ("online", "offline", "disabled")


def test_the_default_composition_needs_no_arguments() -> None:
    cfg = compose_config([])
    assert cfg.tracking.enabled is False, "tracking must default to off"
    assert cfg.run.overwrite is False, "run.overwrite must default to false"
    assert cfg.run.allow_config_surgery is False


# --------------------------------------------------------------------------- #
# the negative cases
# --------------------------------------------------------------------------- #


def test_a_typo_is_rejected_at_composition() -> None:
    with pytest.raises(ConfigCompositionException) as caught:
        compose_train(["experiment=pam50_wsi_final", "clam.lrr=0.1"])
    assert "lrr" in str(caught.value) + str(caught.value.__cause__)


def test_appended_overrides_are_rejected() -> None:
    """`+key=…` composes — which is exactly why it needs its own guard."""
    cfg = compose_train(["experiment=pam50_wsi_final", "+clam.lrr=0.1"])
    assert cfg.clam.lrr == 0.1  # composed happily, and read by nothing

    with pytest.raises(ValueError) as caught:
        schema.reject_appended_overrides(["+clam.lrr=0.1"])
    assert "+clam.lrr=0.1" in str(caught.value)

    with pytest.raises(ValueError):
        schema.reject_appended_overrides(["~clam.seed"])

    # ... unless the user says so explicitly.
    schema.reject_appended_overrides(["+clam.lrr=0.1"], allow=True)


def test_residual_fusion_refuses_to_compose() -> None:
    with pytest.raises(ConfigCompositionException) as caught:
        compose_train(["experiment=pam50_wsi_cnv", "fusion=residual"])
    message = str(caught.value) + str(caught.value.__cause__)
    assert "residual_fusion_is_unavailable" in message, (
        "fusion=residual must fail at composition, before a run directory exists"
    )


def test_the_experiment_group_is_mandatory() -> None:
    """No default experiment: quietly picking one runs a published configuration."""
    with pytest.raises(ConfigCompositionException):
        compose_train([])


def test_config_reference_is_current(tmp_path: Path) -> None:
    """`docs/config-reference.md` is generated; a stale one is worse than none.

    Regenerate with `make reference`.
    """
    import subprocess
    import sys

    from conftest import REAL_REPO

    reference = REAL_REPO / "docs" / "config-reference.md"
    if not reference.exists():
        pytest.skip("docs/config-reference.md has not been generated yet (make reference)")

    regenerated = tmp_path / "config-reference.md"
    completed = subprocess.run(
        [sys.executable, "-m", "dpcode.cli.config", "reference", "-o", str(regenerated)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert regenerated.read_text(encoding="utf-8") == reference.read_text(encoding="utf-8"), (
        "docs/config-reference.md is out of date. Run `make reference`."
    )
