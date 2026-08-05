"""Shared fixtures: a hermetic fixture repository and a fake `python`.

Nothing here reads `.datasets/` or `.scratch/`, opens a slide, or needs a GPU.
Every tree a test writes into is built inside pytest's `tmp_path`, deliberately:
a bare `.scratch` and a bare `.datasets` in `.gitignore` match a directory of
that name at ANY depth, so committed fixture data under `tests/` would be
excluded from a clean clone and every test that depends on it would fail there
and pass here. The only committed fixture material is
`tests/legacy_wrappers/tools/*.sh` — frozen, byte-identical copies of the
pre-refactor wrappers, which must never be edited.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pytest

#: The repository under test. `tests/` sits directly under it.
REAL_REPO = Path(__file__).resolve().parent.parent

#: Frozen pre-refactor wrappers. FROZEN means frozen: the parity test executes
#: these, so editing one silently redefines what parity is measured against.
FROZEN_WRAPPER_DIR = Path(__file__).resolve().parent / "legacy_wrappers" / "tools"

#: Environment variables allowed through to a parity subprocess. Everything else
#: is dropped (`env -i` with an allowlist), because three ER wrappers change
#: their behaviour from the environment — `SEED`, `RUNNER`, `SEEDS` — and a
#: developer shell that happens to export one would quietly compare two different
#: configurations. HOME is kept because git and Hydra both read it; PATH is
#: replaced by the caller so the stub interpreter wins.
ENV_ALLOWLIST = ("HOME", "LANG", "LC_ALL", "TERM", "TMPDIR")

#: Variables that differ between the two sides for reasons that are not
#: configuration, and are therefore excluded from the environment-delta
#: comparison. `PWD`/`OLDPWD`/`SHLVL`/`_` are set by bash on the legacy side and
#: not by `subprocess.run(cwd=...)` on the dpcode side, which changes the working
#: directory without touching the variable.
ENV_DELTA_IGNORE = frozenset({"PWD", "OLDPWD", "SHLVL", "_"})

#: The fake `python`. A Python script rather than a shell script so that the
#: interpreter is pinned absolutely by the shebang: the stub directory is FIRST
#: on PATH, so a `python3`/`python` inside a shell stub could re-enter the stub.
_STUB_TEMPLATE = """#!{interpreter}
import json, os, sys

with open(os.environ["DP_STUB_CAPTURE"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({{
        "argv": sys.argv[1:],
        "cwd": os.getcwd(),
        "env": dict(os.environ),
    }}) + "\\n")
sys.exit(0)
"""


# --------------------------------------------------------------------------- #
# the fake interpreter
# --------------------------------------------------------------------------- #


class PythonStub:
    """A directory holding an executable named `python` that records and exits 0.

    Both sides of the parity check run under the same one. It records `argv`,
    `cwd` and the full environment of every invocation, appended as JSON lines,
    because four of the frozen wrappers call python more than once and an
    assertion on the *count* is what stops a wrapper that `continue`s or exits
    early from passing vacuously.
    """

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.capture = directory / "capture.jsonl"
        directory.mkdir(parents=True, exist_ok=True)
        stub = directory / "python"
        stub.write_text(_STUB_TEMPLATE.format(interpreter=sys.executable), encoding="utf-8")
        stub.chmod(0o755)

    def reset(self) -> None:
        if self.capture.exists():
            self.capture.unlink()

    def records(self) -> list[dict[str, Any]]:
        if not self.capture.exists():
            return []
        return [
            json.loads(line)
            for line in self.capture.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        """A minimal environment: the allowlist, the stub on PATH, `extra`.

        `SEED`, `RUNNER` and `SEEDS` are absent by construction — they are not on
        the allowlist — which is the point.
        """
        env = {
            key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ
        }
        # The stub first, then the directory of the running interpreter so that
        # console scripts (`dp-train`) and real helpers stay reachable, then the
        # ordinary system directories.
        env["PATH"] = os.pathsep.join(
            [str(self.dir), str(Path(sys.executable).parent), "/usr/bin", "/bin"]
        )
        env["DP_STUB_CAPTURE"] = str(self.capture)
        env.update(extra or {})
        return env

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env_extra: Mapping[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess, list[dict[str, Any]], dict[str, str]]:
        """Run `command`, returning the process, the captured records and the env."""
        self.reset()
        env = self.env(env_extra)
        completed = subprocess.run(
            list(command), cwd=str(cwd), env=env, capture_output=True, text=True
        )
        return completed, self.records(), env


@pytest.fixture
def python_stub(tmp_path: Path) -> PythonStub:
    return PythonStub(tmp_path / "stub")


# --------------------------------------------------------------------------- #
# the hermetic fixture repository
# --------------------------------------------------------------------------- #

#: Empty stand-ins the wrappers and `dp-train` check for before dispatching.
#: `dp-train`'s `_assert_inputs_exist` refuses a missing `clam.data_root_dir`,
#: `clam.split_dir`, `clam.tabular_csv` or `clam.tabular_group_spec`, and
#: `run_cnv_fusion_ladder.sh` refuses a missing CNV table; both are checks worth
#: keeping, so the fixture satisfies them with empty files rather than the test
#: disabling them.
_FIXTURE_DIRS = (
    "project/CLAM/dataset_csv",
    "project/CLAM/splits/tcga_brca_subtyping_100",
    "project/CLAM/splits/tcga_brca_er_100",
    ".datasets/tcga-brca/embeddings",
    ".datasets/cptac-brca/embeddings",
    ".scratch/results",
    "tools/data",
)
_FIXTURE_FILES = (
    ".scratch/cnv-tabular/TCGA_BRCA_CNV_arm_4class_clam.csv",
    ".scratch/cnv-tabular/CPTAC_BRCA_CNV_arm_4class_clam.csv",
    ".scratch/cnv-tabular/chromosome_groups.csv",
    ".scratch/TCGA-BRCA-rna/TCGA_BRCA_RNA_primary_tumor_4class_clam.csv.gz",
    ".scratch/TCGA-BRCA-rna/tcga_brca_er_rna_clam.csv.gz",
    "tools/data/tcga_brca_clinicopath_clam.csv",
)


def build_fixture_repo(root: Path) -> Path:
    """Create the fixture repository root both parity sides resolve against.

    The frozen wrappers compute `REPO_ROOT="$(dirname "${BASH_SOURCE[0]}")/.."`,
    so copying them to `<root>/tools/` is what makes that idiom land here instead
    of in the real 330 GB tree. `tools/train_pam50_final.sh` computes no
    REPO_ROOT at all — it is a bare `cd project/CLAM` — which is why the parity
    test also carries a per-wrapper working-directory table.

    A REAL copy of `project/CLAM/main.py` is placed at `<root>/project/CLAM/`,
    because `dp-train` reads the parser out of `${paths.clam_root}/main.py` and a
    stub there would validate against a parser that is not CLAM's.
    """
    for relative in _FIXTURE_DIRS:
        (root / relative).mkdir(parents=True, exist_ok=True)
    for relative in _FIXTURE_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

    shutil.copy2(REAL_REPO / "project/CLAM/main.py", root / "project/CLAM/main.py")
    # `dp-evaluate` extracts the evaluator's parser from this file, and the frozen
    # wrapper dispatches it by name from inside project/CLAM.
    shutil.copy2(
        REAL_REPO / "project/CLAM/evaluate_multimodal.py",
        root / "project/CLAM/evaluate_multimodal.py",
    )
    for wrapper in FROZEN_WRAPPER_DIR.glob("*.sh"):
        shutil.copy2(wrapper, root / "tools" / wrapper.name)
    return root


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    return build_fixture_repo(tmp_path / "fixture")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def environment_delta(
    record_env: Mapping[str, str],
    launcher_env: Mapping[str, str],
    ignore: Iterable[str] = ENV_DELTA_IGNORE,
) -> dict[str, str]:
    """What a wrapper ADDED or CHANGED in the environment of its child.

    Compared between the two sides rather than inspected key by key: W&B mode in
    particular is naturally implemented through the environment, so a wrapper
    that exported `WANDB_MODE` and a Hydra config that did not would be a real
    difference in what runs, invisible in argv.
    """
    skip = set(ignore)
    return {
        key: value
        for key, value in record_env.items()
        if key not in skip and launcher_env.get(key) != value
    }
