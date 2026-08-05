"""THE acceptance test: the Hydra entry points issue the pre-refactor commands.

A reproducibility refactor that changes what gets run has not refactored
anything, it has replaced one experiment with another. This module is what makes
that statement checkable. For each covered wrapper it runs BOTH sides —

  * the FROZEN pre-refactor wrapper from `tests/legacy_wrappers/tools/`, and
  * today's `tools/*.sh`, which is a shim over `dp-train`

— under the same `PATH`-stubbed fake `python` that records `argv`, `cwd` and the
environment of every invocation, and then asserts that the two agree on the
EFFECTIVE CONFIGURATION.

Effective configuration, not tokens. The wrappers disagree on argument order, on
relative-versus-absolute path spelling, and on which flags they pass at all
(`train_pam50_final.sh` passes `--data_root_dir ../../.datasets/...` from inside
`project/CLAM`; the ladder passes an absolute `${REPO_ROOT}/...`; `render_argv`
emits every non-None field, which is a superset of what any wrapper passed). A
token comparison would fail on cosmetics while proving nothing. So both argv
lists are parsed with the parser extracted from the real `project/CLAM/main.py`,
main.py's two post-parse mutations are applied to both namespaces
(`results_dir` gains `_s{seed}` at :407, `split_dir` gets the literal `splits/`
prefix at :412-414), every path-valued field is `realpath`-resolved against the
working directory ITS OWN side runs in, and the resulting dicts must be equal
field for field.

Four properties make the check hard to pass vacuously:

  * **both sides execute.** Composing and rendering the new side would certify
    `render_argv`, not `dp-train`.
  * **invocation counts are asserted.** `run_cnv_fusion_ladder.sh:96-99`
    `continue`s when the output directory exists and `set -euo pipefail` exits
    before python on any guard failure — either would otherwise capture nothing
    and pass.
  * **exit status is asserted** on both sides.
  * **the environment delta is compared**, not just argv. Three ER wrappers are
    driven by `SEED`/`RUNNER`/`SEEDS`, so both sides run under an `env -i`-style
    allowlist with those unset, and W&B mode is the kind of setting that is
    naturally implemented through the environment rather than through a flag.

Hermetic: a fixture repository root under `tmp_path` (see `conftest.py`), no real
data, no GPU, no network. The frozen wrappers' `REPO_ROOT="$(dirname
"${BASH_SOURCE[0]}")/.."` idiom therefore lands in the fixture, and
`train_pam50_final.sh` — which computes no REPO_ROOT and just does
`cd project/CLAM` — is given the fixture root as its working directory through an
explicit per-case table. Getting that one wrong would `realpath` into the real
330 GB store, and it is the wrapper behind the headline WSI baseline.

COVERAGE, STATED RATHER THAN IMPLIED
------------------------------------
Covered: `train_pam50_final.sh`; `run_cnv_fusion_ladder.sh` in all five
operators plus its `--wandb`, `--no_warm_start`, `--exp_suffix/--k/--max_epochs/
--seed` and `--tabular_csv` branches; `train_pam50_multimodal.sh` defaults and
its wandb/freeze/fusion_mode toggles; `train_er_ablation.sh` in each of its three
arms, its three-invocation `all` arm and its `SEED=` hook;
`train_er_novel_fusion.sh film_rna`; `evaluate_pam50_multimodal.sh` defaults and
two argument variants.

NOT covered, and why:

  * **Only each ER wrapper's DEFAULT arm.** `train_er_novel_fusion.sh` issued
    eleven invocations and only `film_rna` is ported; `train_er_multiseed.sh`
    issued about sixteen nested ones and its shim now refuses up front, so there
    is nothing to compare. Those runs are complete and on disk
    (`docs/er-prediction-results.md`), and the frozen wrapper remains the record
    of what each one was.
  * **`dp-cptac`.** Under a python stub, phase 2's `-f` guards fail, so a
    comparison would be between two early exits.
  * **`dp-evaluate` executes only on the legacy side.** It dispatches through
    `sys.executable` rather than `shutil.which("python")`, so a PATH stub cannot
    intercept it; the new side's rendered argv and cwd are read from
    `--print-argv` instead. That is a weaker check than the training cases and is
    listed here rather than glossed over.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest

from conftest import REAL_REPO, PythonStub, environment_delta

from dpcode import clam_args

#: main.py:407 and :412-414 — the two mutations argparse does not do.
_FOLD = "{fold}"
_SENTINEL = "__DP_FOLD__"


def effective(argv: Sequence[str], cwd: str, parser) -> dict[str, Any]:
    """Parse `argv` with CLAM's parser and normalise it into a comparable dict."""
    namespace = vars(parser.parse_args(list(argv)))
    namespace["results_dir"] = os.path.join(
        namespace["results_dir"], f"{namespace['exp_code']}_s{namespace['seed']}"
    )
    if namespace["split_dir"]:
        namespace["split_dir"] = os.path.join("splits", namespace["split_dir"])

    resolved: dict[str, Any] = {}
    for key, value in namespace.items():
        if key in clam_args.PATH_VALUED_DESTS and value is not None:
            guarded = str(value).replace(_FOLD, _SENTINEL)
            value = os.path.realpath(os.path.join(cwd, guarded)).replace(_SENTINEL, _FOLD)
        resolved[key] = value
    return resolved


def clam_invocations(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The captured invocations that are `python main.py …`.

    The ladder shim also runs `python -c 'from dpcode.paths import …'` to locate
    `results_root` for its skip-existing check; under the stub that returns
    nothing and the check switches itself off, which the shim says on stderr.
    That invocation is real and is counted separately.
    """
    return [r for r in records if r["argv"] and Path(r["argv"][0]).name == "main.py"]


# --------------------------------------------------------------------------- #
# the case table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Case:
    """One wrapper invocation, run on both sides.

    `args` is a callable of the fixture root because several cases must pass an
    ABSOLUTE path: the frozen wrapper resolves a relative argument against the
    FIXTURE root while the shim resolves it against the REAL repository root, so
    a relative spelling would compare two different files. Where a wrapper takes
    no path argument the two sides get identical strings.
    """

    id: str
    wrapper: str
    args: Callable[[Path], list[str]] = lambda root: []
    #: Set ONLY where the two wrappers do not accept the same spelling of the
    #: same value — see `ladder_tabular_csv`. Everywhere else both sides get
    #: byte-identical arguments, which is what makes the comparison meaningful.
    new_args: Callable[[Path], list[str]] | None = None
    env: dict[str, str] = field(default_factory=dict)
    clam_calls: int = 1
    #: Extra non-CLAM python invocations the SHIM makes (the ladder's probe).
    extra_new_calls: int = 0


_CKPT = ".scratch/results/pam50_final_s1/s_{fold}_checkpoint.pt"


def _ladder(mode: str, *extra: str) -> Callable[[Path], list[str]]:
    return lambda root: ["--modes", mode, "--no_skip_existing", *extra]


CASES: list[Case] = [
    Case("pam50_final", "train_pam50_final.sh"),
    # The five ladder operators. `--no_skip_existing` on both sides: the frozen
    # wrapper skips an existing output directory and the shim maps the flag to
    # run.overwrite=true, and without it neither side would dispatch anything.
    *[
        Case(f"ladder_{mode}", "run_cnv_fusion_ladder.sh", _ladder(mode), extra_new_calls=1)
        for mode in ("concat", "gated", "cross_attention", "film_attention", "coattn")
    ],
    Case("ladder_wandb", "run_cnv_fusion_ladder.sh", _ladder("gated", "--wandb"),
         extra_new_calls=1),
    Case("ladder_no_warm_start", "run_cnv_fusion_ladder.sh",
         _ladder("concat", "--no_warm_start"), extra_new_calls=1),
    Case("ladder_smoke_suffix", "run_cnv_fusion_ladder.sh",
         _ladder("coattn", "--k", "1", "--max_epochs", "2",
                 "--exp_suffix", "_smoke", "--seed", "4"),
         extra_new_calls=1),
    # The one case whose two sides are spelled differently, on purpose. The
    # frozen ladder builds `--tabular_csv "${REPO_ROOT}/${TABULAR_CSV}"`
    # unconditionally, so it accepts ONLY a repo-relative value (an absolute one
    # comes out doubled). The shim keeps a relative value repo-relative too, but
    # its repository root is the real clone rather than the fixture, so the two
    # are given the spelling each accepts for the same file.
    Case(
        "ladder_tabular_csv",
        "run_cnv_fusion_ladder.sh",
        lambda root: [
            "--modes", "concat", "--no_skip_existing",
            "--tabular_csv", ".scratch/cnv-tabular/CPTAC_BRCA_CNV_arm_4class_clam.csv",
        ],
        new_args=lambda root: [
            "--modes", "concat", "--no_skip_existing",
            "--tabular_csv", str(root / ".scratch/cnv-tabular/CPTAC_BRCA_CNV_arm_4class_clam.csv"),
        ],
        extra_new_calls=1,
    ),
    Case(
        "multimodal_defaults",
        "train_pam50_multimodal.sh",
        lambda root: ["--pretrained_wsi_ckpt", str(root / _CKPT)],
    ),
    Case(
        "multimodal_concat_nofreeze_nowandb",
        "train_pam50_multimodal.sh",
        lambda root: [
            "--pretrained_wsi_ckpt", str(root / _CKPT),
            "--fusion_mode", "concat", "--no_freeze_wsi_branch", "--no_wandb",
            "--k", "5", "--k_start", "1", "--k_end", "3", "--seed", "7",
            "--tabular_hidden_dim", "128", "--tabular_num_layers", "3",
            "--tabular_top_n_features", "500", "--fusion_hidden_dim", "64",
            "--max_epochs", "20",
            "--exp_code", "pam50_wsi_rna_gatedfusion_gdc",
        ],
    ),
    Case("er_wsi", "train_er_ablation.sh", lambda root: ["wsi"]),
    Case("er_rna", "train_er_ablation.sh", lambda root: ["rna"]),
    Case("er_clinpath", "train_er_ablation.sh", lambda root: ["clinpath"]),
    # Three invocations in dependency order, in one process: the count assertion
    # is the whole point of this case.
    Case("er_all", "train_er_ablation.sh", lambda root: ["all"], clam_calls=3),
    Case("er_rna_seed3", "train_er_ablation.sh", lambda root: ["rna"], env={"SEED": "3"}),
    Case("er_film_rna", "train_er_novel_fusion.sh", lambda root: ["film_rna"]),
]


@pytest.fixture(scope="session")
def clam_parser():
    """CLAM's real parser, read once from the real `main.py`.

    Explicitly from the REAL repository: `clam_args.clam_parser()` with no
    argument resolves through `repo_root()`, which the parity environment points
    at the fixture — where a stand-in `main.py` would silently define what parity
    means.
    """
    return clam_args.clam_parser(REAL_REPO / "project/CLAM/main.py")


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_wrapper_and_shim_issue_the_same_clam_command(
    case: Case, fixture_repo: Path, python_stub: PythonStub, tmp_path: Path, clam_parser
) -> None:
    arguments = case.args(fixture_repo)

    legacy_process, legacy_records, legacy_env = python_stub.run(
        ["bash", str(fixture_repo / "tools" / case.wrapper), *arguments],
        # train_pam50_final.sh is a bare `cd project/CLAM` with no REPO_ROOT, so
        # the working directory IS the fixture root for it. The others compute
        # REPO_ROOT from BASH_SOURCE and are indifferent.
        cwd=fixture_repo,
        env_extra=case.env,
    )
    assert legacy_process.returncode == 0, (
        f"frozen wrapper failed:\n{legacy_process.stdout}\n{legacy_process.stderr}"
    )

    elsewhere = tmp_path / "some-other-directory"
    elsewhere.mkdir()
    new_arguments = (case.new_args or case.args)(fixture_repo)
    new_process, new_records, new_env = python_stub.run(
        ["bash", str(REAL_REPO / "tools" / case.wrapper), *new_arguments],
        # Deliberately NOT the repository root: the console scripts are supposed
        # to work from any working directory.
        cwd=elsewhere,
        env_extra={"DP_REPO_ROOT": str(fixture_repo), **case.env},
    )
    assert new_process.returncode == 0, (
        f"shim failed:\n{new_process.stdout}\n{new_process.stderr}"
    )

    legacy_calls = clam_invocations(legacy_records)
    new_calls = clam_invocations(new_records)
    assert len(legacy_calls) == case.clam_calls, (
        f"expected {case.clam_calls} `python main.py` invocation(s) from the frozen "
        f"wrapper, captured {len(legacy_calls)} (total {len(legacy_records)}). "
        "A wrapper that exits early or `continue`s would pass this test vacuously."
    )
    assert len(new_calls) == case.clam_calls
    assert len(legacy_records) == case.clam_calls
    assert len(new_records) == case.clam_calls + case.extra_new_calls

    for index, (old, new) in enumerate(zip(legacy_calls, new_calls)):
        assert os.path.realpath(old["cwd"]) == os.path.realpath(new["cwd"]), (
            f"invocation {index}: CLAM would run in a different directory. "
            "main.py resolves `dataset_csv/<task>.csv` and the `splits/` prefix "
            "against its cwd and neither is overridable by a flag."
        )
        left = effective(old["argv"][1:], old["cwd"], clam_parser)
        right = effective(new["argv"][1:], new["cwd"], clam_parser)
        differences = {
            key: (left.get(key), right.get(key))
            for key in sorted(set(left) | set(right))
            if left.get(key) != right.get(key)
        }
        assert not differences, (
            f"invocation {index}: effective configuration differs.\n"
            + "\n".join(
                f"  {key}: frozen wrapper={old_value!r}  shim={new_value!r}"
                for key, (old_value, new_value) in differences.items()
            )
        )

        assert environment_delta(old["env"], legacy_env) == environment_delta(
            new["env"], new_env
        ), "the two sides hand CLAM different environments"


# --------------------------------------------------------------------------- #
# dp-evaluate
# --------------------------------------------------------------------------- #

#: `(id, legacy arguments, dpcode overrides)`. Paths are written against the
#: fixture root by the test, so `{root}` is substituted before use.
EVALUATE_CASES = [
    ("defaults", [], []),
    (
        "wandb_ckpt_fold",
        ["--wandb", "--ckpt_dir", ".scratch/results/x_s1", "--fold", "3"],
        [
            "evaluate.args.wandb=true",
            "evaluate.args.ckpt_dir=${paths.scratch_root}/results/x_s1",
            "evaluate.args.fold=3",
        ],
    ),
    (
        "gated_at_the_ladders_width",
        ["--fusion_mode", "gated", "--tabular_hidden_dim", "64"],
        ["evaluate.args.fusion_mode=gated", "evaluate.args.tabular_hidden_dim=64"],
    ),
]

_EVALUATE_PATH_DESTS = frozenset(
    {"data_root_dir", "tabular_csv", "ckpt_dir", "output_dir", "split_dir", "dataset_csv"}
)


@pytest.mark.parametrize(
    "case_id,legacy_args,overrides", EVALUATE_CASES, ids=[c[0] for c in EVALUATE_CASES]
)
def test_evaluate_wrapper_and_dp_evaluate_agree(
    case_id: str,
    legacy_args: list[str],
    overrides: list[str],
    fixture_repo: Path,
    python_stub: PythonStub,
) -> None:
    from dpcode.cli.evaluate import evaluator_parser

    legacy_process, records, _ = python_stub.run(
        ["bash", str(fixture_repo / "tools/evaluate_pam50_multimodal.sh"), *legacy_args],
        cwd=fixture_repo,
    )
    assert legacy_process.returncode == 0, legacy_process.stderr
    assert len(records) == 1, f"expected one python invocation, got {len(records)}"
    record = records[0]
    assert record["argv"][0] == "evaluate_multimodal.py"

    # The new side is asked for its rendered argv rather than executed: see the
    # coverage note in the module docstring.
    completed = subprocess.run(
        [sys.executable, "-m", "dpcode.cli.evaluate", "--print-argv", *overrides],
        env={**os.environ, "DP_REPO_ROOT": str(fixture_repo)},
        cwd=str(fixture_repo.parent),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    rendered = json.loads(completed.stdout)

    parser = evaluator_parser(fixture_repo / "project/CLAM/evaluate_multimodal.py")

    def normalise(argv: Sequence[str], cwd: str) -> dict[str, Any]:
        namespace = vars(parser.parse_args(list(argv)))
        return {
            key: (
                os.path.realpath(os.path.join(cwd, str(value)))
                if key in _EVALUATE_PATH_DESTS and value is not None
                else value
            )
            for key, value in namespace.items()
        }

    assert os.path.realpath(record["cwd"]) == os.path.realpath(rendered["cwd"])
    left = normalise(record["argv"][1:], record["cwd"])
    right = normalise(rendered["argv"], rendered["cwd"])
    differences = {
        key: (left.get(key), right.get(key))
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }
    assert not differences, differences
