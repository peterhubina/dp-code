"""`requirements.txt` and `pyproject.toml` must not disagree.

Two manifests are one more than anybody keeps up to date, and this repository has
had three (the root freeze, `project/CLAM/requirements.txt`, and
`project/CLAM/env.yml`) disagreeing on `timm`, on where OpenSlide comes from, and
on whether the smooth-topk pin exists at all. `project/CLAM/requirements.txt`
omits `tensorboardX` and `topk` — both imported by `project/CLAM/utils/core_utils.py`
— so installing from CLAM's own file breaks CLAM.

`requirements.txt` cannot simply be deleted in favour of `pyproject.toml`: the
Docker image installs dependencies before any source exists to install `-e .`
from. So both stay, and this module is what keeps them identical.

The named checks below are the traps. Each one is a pin that looks droppable and
is not:

  * `topk` — the git-pinned smooth-topk. Imported LAZILY at
    `utils/core_utils.py:315` by `--inst_loss svm`, i.e. by the frozen WSI
    baseline, so its absence surfaces mid-training rather than at startup.
  * `tensorboardX` — `utils/core_utils.py:298`, reached by `--log_data`, which
    all three training wrappers pass. NOT the same distribution as
    `tensorboard`, which nothing imports.
  * `openpyxl` — reached only as `pd.read_excel(engine="openpyxl")`; there is no
    `import openpyxl` anywhere, so an import scan misses it.
  * `openslide-bin` — supplies the OpenSlide C library as a wheel, which is why
    no system package is needed.
  * `opencv-python` — must be ABSENT. It and `opencv-python-headless` install the
    same `cv2` module, both used to be pinned, and install order decided the
    winner; headless is what `import cv2` resolves to and therefore what ran.
"""

from __future__ import annotations

import ast
import importlib.metadata as md
import re
from pathlib import Path

import pytest

from conftest import REAL_REPO

REQUIREMENTS = REAL_REPO / "requirements.txt"
REQUIREMENTS_DEV = REAL_REPO / "requirements-dev.txt"
PYPROJECT = REAL_REPO / "pyproject.toml"


def parse_requirements(path: Path) -> list[str]:
    """Requirement lines, comments and `-r` includes dropped."""
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        lines.append(line)
    return lines


def pyproject_list(field: str) -> list[str]:
    """Read a TOML array of strings out of `pyproject.toml` without a TOML parser.

    Python 3.10 has no `tomllib` and this repository has no TOML dependency;
    adding one to read its own manifest would be its own small irony. The arrays
    in question are plain lists of string literals, so `ast.literal_eval` on the
    bracketed block is exact.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(field)}\s*=\s*(\[.*?\])", text, re.M | re.S)
    assert match, f"{field} not found in pyproject.toml"
    return ast.literal_eval(match.group(1))


def distribution_name(requirement: str) -> str:
    """PEP 508 requirement -> canonical distribution name."""
    text = requirement.split(";", 1)[0].strip()
    for separator in (" @ ", "[", "==", ">=", "<=", "~=", "!=", ">", "<", "("):
        index = text.find(separator)
        if index != -1:
            text = text[:index]
    return re.sub(r"[-_.]+", "-", text.strip()).lower()


def test_requirements_and_pyproject_declare_the_same_set() -> None:
    from_requirements = {distribution_name(r): r for r in parse_requirements(REQUIREMENTS)}
    from_pyproject = {distribution_name(r): r for r in pyproject_list("dependencies")}

    only_requirements = sorted(set(from_requirements) - set(from_pyproject))
    only_pyproject = sorted(set(from_pyproject) - set(from_requirements))
    assert not only_requirements and not only_pyproject, (
        f"requirements.txt only: {only_requirements}\n"
        f"pyproject.toml only: {only_pyproject}"
    )
    differing = {
        name: (from_requirements[name], from_pyproject[name])
        for name in from_requirements
        if from_requirements[name].replace(" ", "") != from_pyproject[name].replace(" ", "")
    }
    assert not differing, f"same distribution, different pin: {differing}"


def test_every_runtime_dependency_is_pinned_exactly() -> None:
    """`>=` in a reproduction manifest is a promise nobody can keep."""
    loose = [
        requirement
        for requirement in parse_requirements(REQUIREMENTS)
        if "==" not in requirement and " @ " not in requirement
    ]
    assert not loose, f"unpinned runtime dependencies: {loose}"


@pytest.mark.parametrize(
    "name,why",
    [
        ("topk", "--inst_loss svm imports it lazily, mid-training"),
        ("tensorboardx", "--log_data imports it; all three trainers pass --log_data"),
        ("openpyxl", "reached only via pd.read_excel(engine='openpyxl')"),
        ("openslide-bin", "supplies the OpenSlide C library, so no system package is needed"),
        ("openslide-python", "the WSI reader itself"),
        ("hydra-core", "the configuration mechanism this refactor is built on"),
        ("omegaconf", "the interpolations every path in the tree uses"),
    ],
)
def test_a_load_bearing_pin_is_present(name: str, why: str) -> None:
    declared = {distribution_name(r) for r in parse_requirements(REQUIREMENTS)}
    assert name in declared, f"{name} is missing from requirements.txt: {why}"


def test_the_opencv_double_pin_is_resolved() -> None:
    declared = {distribution_name(r) for r in parse_requirements(REQUIREMENTS)}
    assert "opencv-python-headless" in declared
    assert "opencv-python" not in declared, (
        "opencv-python and opencv-python-headless install the same `cv2`; pinning "
        "both makes install order decide which one runs, and headless is what did"
    )


def test_pillow_is_pinned_to_what_actually_ran() -> None:
    """The old freeze said 12.1.1; 12.3.0 is installed and produced the numbers."""
    pins = {distribution_name(r): r for r in parse_requirements(REQUIREMENTS)}
    assert pins["pillow"] == "pillow==12.3.0"


def test_python_upper_bound_is_declared() -> None:
    """torch 2.0.1 publishes no cp312 wheel, so `>=3.10` alone is a trap."""
    text = PYPROJECT.read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10,<3.12"' in text


def test_scikit_survival_is_declared_unpinned_in_an_extra() -> None:
    """It has never been installed here, so no version of it produced a number.

    Declaring one would be a fabricated reproduction claim. It stays in the
    `survival` extra, unpinned, and the dormant survival thread stays broken in
    the two independent ways the audit found.
    """
    survival = pyproject_list("survival")
    assert "scikit-survival" in survival, survival
    assert not any(
        entry.startswith("scikit-survival") and "==" in entry for entry in survival
    ), "scikit-survival must NOT carry a version: none was ever installed here"
    assert "scikit-survival" not in {
        distribution_name(r) for r in parse_requirements(REQUIREMENTS)
    }
    with pytest.raises(md.PackageNotFoundError):
        md.version("scikit-survival")


def test_dev_requirements_match_the_dev_extra() -> None:
    dev_extra = {distribution_name(r): r for r in pyproject_list("dev")}
    dev_file = {distribution_name(r): r for r in parse_requirements(REQUIREMENTS_DEV)}
    assert dev_file == dev_extra
    assert REQUIREMENTS_DEV.read_text(encoding="utf-8").splitlines(), "must not be empty"
    assert "-r requirements.txt" in REQUIREMENTS_DEV.read_text(encoding="utf-8")


def test_the_installed_environment_matches_the_pins() -> None:
    """The manifest describes THIS environment, or it describes nothing.

    A drifted environment is not a nuisance here: every published number came out
    of the versions below, and `run_metadata.json` records them per run. `topk`
    is compared by presence only — it is pinned by git commit, and the version it
    reports (1.0) is not derived from that commit.
    """
    mismatched = {}
    for requirement in parse_requirements(REQUIREMENTS):
        name = distribution_name(requirement)
        try:
            installed = md.version(name)
        except md.PackageNotFoundError:
            mismatched[name] = ("NOT INSTALLED", requirement)
            continue
        if "==" in requirement:
            pinned = requirement.split("==", 1)[1].strip()
            if installed != pinned:
                mismatched[name] = (installed, pinned)
    assert not mismatched, (
        "the installed environment does not match requirements.txt "
        f"(installed, pinned): {mismatched}. Reinstall with `make install`, or "
        "correct the pin if the environment is the authority."
    )
