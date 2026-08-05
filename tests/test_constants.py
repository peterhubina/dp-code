"""Every published number's constants, pinned to their value and to the config.

Two failure modes this closes, both silent:

  * **A constant changes.** `evaluate_cnv_wsi_fusion` bootstraps 4,000 resamples
    with seed 7; `stack_wsi_cnv` uses 2,000 with seed 11. Those are DIFFERENT on
    purpose (DESIGN-ADDENDUM A6) and unifying them would move published
    confidence intervals while every test still passed. The CNV arm's logistic
    regression — `C=0.1`, `max_iter=4000`, `class_weight='balanced'` — is the
    model behind "CNV only 0.888 [0.835, 0.933]" in the headline table.
  * **The config drifts from the constant.** Moving a value into
    `dpcode/conf/analyses/*.yaml` re-defaulted is exactly the mistake the
    refactor exists to prevent: the entry point would then run a different model
    from the one the script runs, and both would look correct.

The third check is mechanical and would otherwise be found by a puzzled reader.
Hydra reads `# @package …` ONLY from the first line of a file. The group
directory is `conf/analyses/` (a bare `analysis` in `.gitignore` matches a
directory of that name at any depth, so `conf/analysis/` could never be
committed), and the directive on line 1 is the single thing that keeps the config
KEY `analysis`. Move it to line 2 and every override silently becomes
`analyses.*`, which nothing reads.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from conftest import REAL_REPO

from dpcode.paths import conf_dir

import tools.evaluate_cnv_wsi_fusion as fusion
import tools.make_cnv_tabular as make_cnv_tabular
import tools.pam50_arms as pam50_arms
import tools.stack_wsi_cnv as stack

ANALYSES_DIR = conf_dir() / "analyses"


# --------------------------------------------------------------------------- #
# the constants themselves
# --------------------------------------------------------------------------- #


def test_cnv_arm_model() -> None:
    """StandardScaler -> LogisticRegression(max_iter=4000, C=0.1, balanced)."""
    assert pam50_arms.CNV_C == 0.1
    assert pam50_arms.CNV_MAX_ITER == 4000
    assert pam50_arms.CNV_CLASS_WEIGHT == "balanced"

    estimator = pam50_arms.cnv_arm()
    logistic = estimator.steps[-1][1]
    assert logistic.C == 0.1
    assert logistic.max_iter == 4000
    assert logistic.class_weight == "balanced"


def test_external_validation_constants() -> None:
    assert fusion.N_BOOT == 4000
    assert fusion.BOOTSTRAP_SEED == 7
    assert fusion.CV_FOLDS == 10
    assert fusion.CV_SEED == 0


def test_stacker_constants() -> None:
    assert stack.N_BOOT == 2000
    assert stack.BOOTSTRAP_SEED == 11
    assert stack.STACKER_C == 1.0
    assert stack.STACKER_MAX_ITER == 4000
    assert stack.NM_XATOL == 1e-4
    assert stack.NM_FATOL == 1e-6
    assert stack.NM_MAXITER == 2000
    assert stack.CLIP_FLOOR == 1e-9


def test_the_two_bootstrap_seeds_are_different() -> None:
    """A6, stated as a test so "tidying" them into one constant fails here."""
    assert fusion.BOOTSTRAP_SEED != stack.BOOTSTRAP_SEED
    assert fusion.N_BOOT != stack.N_BOOT


def test_the_two_class_orders_are_both_recorded() -> None:
    """CLAM's label_dict order and `pam50_arms`'s sorted order are both real.

    `pam50_arms.clam_column_order()` is the only bridge between them, and it
    asserts the recovered map is a permutation. Nothing here "unifies" them: the
    orders are data, and a site rewritten to the other order would silently
    permute every reported per-class number.
    """
    assert list(pam50_arms.CLASSES) == ["Basal", "Her2", "LumA", "LumB"]
    assert make_cnv_tabular.CLASSES == ["LumA", "LumB", "Basal", "Her2"]
    assert sorted(pam50_arms.CLASSES) == sorted(make_cnv_tabular.CLASSES)


# --------------------------------------------------------------------------- #
# config defaults == module constants
# --------------------------------------------------------------------------- #

#: `(config file, config key, the value it must equal)`.
CONFIG_TO_CONSTANT = [
    ("cnv_wsi_fusion", "n_boot", fusion.N_BOOT),
    ("cnv_wsi_fusion", "bootstrap_seed", fusion.BOOTSTRAP_SEED),
    ("cnv_wsi_fusion", "cv_folds", fusion.CV_FOLDS),
    ("cnv_wsi_fusion", "cv_seed", fusion.CV_SEED),
    ("cnv_wsi_fusion", "class_order", list(pam50_arms.CLASSES)),
    ("stack_wsi_cnv", "n_boot", stack.N_BOOT),
    ("stack_wsi_cnv", "bootstrap_seed", stack.BOOTSTRAP_SEED),
    ("stack_wsi_cnv", "stacker_C", stack.STACKER_C),
    ("stack_wsi_cnv", "stacker_max_iter", stack.STACKER_MAX_ITER),
    ("stack_wsi_cnv", "nm_xatol", stack.NM_XATOL),
    ("stack_wsi_cnv", "nm_fatol", stack.NM_FATOL),
    ("stack_wsi_cnv", "nm_maxiter", stack.NM_MAXITER),
    ("stack_wsi_cnv", "clip_floor", stack.CLIP_FLOOR),
    ("stack_wsi_cnv", "class_order", list(pam50_arms.CLASSES)),
    ("cnv_controls", "cnv_C", pam50_arms.CNV_C),
    ("cnv_controls", "class_order", list(pam50_arms.CLASSES)),
    ("make_cnv_tabular", "clam_class_order", make_cnv_tabular.CLASSES),
]


@pytest.mark.parametrize(
    "option,key,expected",
    CONFIG_TO_CONSTANT,
    ids=[f"{o}.{k}" for o, k, _ in CONFIG_TO_CONSTANT],
)
def test_analysis_config_default_equals_the_module_constant(option, key, expected) -> None:
    config = OmegaConf.load(ANALYSES_DIR / f"{option}.yaml")
    value = OmegaConf.to_container(config[key], resolve=True) if OmegaConf.is_config(
        config[key]
    ) else config[key]
    assert value == expected, (
        f"analyses/{option}.yaml:{key} is {value!r} but the script uses {expected!r}. "
        "A config default that is not the script's value means `dp-analysis` and "
        "`python tools/…` produce different numbers."
    )


def test_compare_fusion_ladder_n_boot_matches_that_scripts_own_default() -> None:
    """`tools/compare_fusion_ladder.py` is USER-OWNED: read it, never edit it.

    It takes `--n-boot` as an argparse default rather than a module constant, so
    the default is extracted from the source instead of imported (importing would
    also pull pandas and its own path constants).
    """
    source = (REAL_REPO / "tools/compare_fusion_ladder.py").read_text(encoding="utf-8")
    default = _argparse_default(ast.parse(source), "--n-boot")
    configured = OmegaConf.load(ANALYSES_DIR / "compare_fusion_ladder.yaml").n_boot
    assert configured == default == 2000


def _argparse_default(tree: ast.AST, flag: str):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == flag
        ):
            for keyword in node.keywords:
                if keyword.arg == "default":
                    return ast.literal_eval(keyword.value)
    raise AssertionError(f"no add_argument({flag!r}, ..., default=...) found")


# --------------------------------------------------------------------------- #
# the @package directive
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path", sorted(ANALYSES_DIR.glob("*.yaml")), ids=lambda p: p.name
)
def test_package_directive_is_on_line_one(path: Path) -> None:
    first = path.read_text(encoding="utf-8").splitlines()[0].strip()
    assert first == "# @package analysis", (
        f"{path.name} line 1 is {first!r}. Hydra reads the @package directive only "
        "from the first line; anywhere else it is a comment, the group's package "
        "becomes `analyses`, and every `analysis.*` override silently reaches "
        "nothing. The directory is `analyses/` because `.gitignore` carries a bare "
        "`analysis` pattern (DESIGN-ADDENDUM A2)."
    )
