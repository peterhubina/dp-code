"""Path resolution is independent of the working directory, and fails by name.

The refactor's premise is that a path is written down in exactly one place —
`dpcode/conf/paths/default.yaml` — and that every entry point resolves it to the
same absolute location no matter where it was launched from. That is worth
testing rather than asserting, because the pre-refactor code resolved paths four
different ways: repo-relative from the root, `../../`-relative from inside
`project/CLAM`, absolute literals, and `$PWD` after a `cd`.

The second half is the error message. A wrong path in config used to surface as a
`FileNotFoundError` inside `h5py.File`, minutes into a run, naming a `.h5` file
rather than the key that produced it — the survival config's
`embeddings_dir: .datasets/embeddings` is the standing example. `assert_paths_exist`
must abort in a second AND name the config key, because the key is what the
reader has to go and fix.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import REAL_REPO

from dpcode.paths import (
    assert_paths_absolute,
    assert_paths_exist,
    conf_dir,
    repo_root,
    resolve_paths,
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No DP_* variable leaks in from the developer's shell."""
    for name in list(os.environ):
        if name.startswith("DP_"):
            monkeypatch.delenv(name, raising=False)


def test_resolution_is_identical_from_three_working_directories(
    clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The repository root, a subdirectory of it, and somewhere else entirely."""
    elsewhere = tmp_path / "unrelated"
    elsewhere.mkdir()
    results = []
    for directory in (REAL_REPO, REAL_REPO / "project" / "CLAM", elsewhere):
        monkeypatch.chdir(directory)
        results.append(resolve_paths())
    assert results[0] == results[1] == results[2]
    assert results[0]["repo_root"] == str(REAL_REPO)


def test_every_resolved_path_is_absolute(clean_env) -> None:
    paths = resolve_paths()
    relative = {
        key: value
        for key, value in paths.items()
        if value is not None and not Path(str(value)).is_absolute()
    }
    assert not relative, relative


def test_assert_paths_absolute_rejects_a_relative_value() -> None:
    """The survival config's defect, reduced to its shape.

    `tools/config/dataset/tcga_brca_survival.yaml` sets
    `embeddings_dir: .datasets/embeddings`. Relative AND wrong; this guard
    catches the first half before anything is created, and names the key.
    """
    with pytest.raises(ValueError) as caught:
        assert_paths_absolute({"paths": {"data_root": ".datasets/embeddings"}})
    assert "paths.data_root" in str(caught.value)


def test_assert_paths_absolute_allows_none() -> None:
    """`paths.nou_root` is None unless DP_NOU_ROOT is set. That is not an error."""
    assert_paths_absolute({"paths": {"nou_root": None, "data_root": "/somewhere"}})


def test_nou_root_has_no_committed_default(clean_env) -> None:
    """The private cohort's location is never written down in the repository."""
    assert resolve_paths()["nou_root"] is None
    text = (conf_dir() / "paths" / "default.yaml").read_text(encoding="utf-8")
    assert "DP_NOU_ROOT" in text
    assert "nou" not in text.replace("nou_root", "").replace("DP_NOU_ROOT", "").lower()


# --------------------------------------------------------------------------- #
# environment overrides
# --------------------------------------------------------------------------- #


def test_dp_repo_root_moves_everything(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DP_REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("DP_DATA_ROOT", raising=False)
    monkeypatch.delenv("DP_SCRATCH_ROOT", raising=False)
    monkeypatch.delenv("DP_RESULTS_ROOT", raising=False)

    paths = resolve_paths()
    assert paths["repo_root"] == str(tmp_path)
    assert paths["data_root"] == f"{tmp_path}/.datasets"
    assert paths["scratch_root"] == f"{tmp_path}/.scratch"
    assert paths["results_root"] == f"{tmp_path}/.scratch/results"
    assert paths["clam_root"] == f"{tmp_path}/project/CLAM"
    assert repo_root() == tmp_path


def test_the_three_cluster_variables_override_independently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DP_DATA_ROOT / DP_SCRATCH_ROOT / DP_RESULTS_ROOT — the cluster install.

    Each is read on EVERY call rather than cached, which is what lets a test (and
    the parity harness) point the tree somewhere else mid-process.
    """
    monkeypatch.setenv("DP_REPO_ROOT", str(tmp_path / "clone"))
    monkeypatch.setenv("DP_DATA_ROOT", "/mnt/scratch/someone/.datasets")
    monkeypatch.setenv("DP_SCRATCH_ROOT", "/mnt/scratch/someone/dp-code")
    monkeypatch.setenv("DP_RESULTS_ROOT", "/mnt/nfs-data/runs")
    (tmp_path / "clone").mkdir()

    paths = resolve_paths()
    assert paths["tcga_embeddings"] == "/mnt/scratch/someone/.datasets/tcga-brca/embeddings"
    assert paths["cnv_tabular_dir"] == "/mnt/scratch/someone/dp-code/cnv-tabular"
    # results_root does NOT follow scratch_root once it is set explicitly.
    assert paths["results_root"] == "/mnt/nfs-data/runs"
    # ... while the tracked inputs stay with the clone, because they are tracked.
    assert paths["splits_root"] == f"{tmp_path}/clone/project/CLAM/splits"


def test_dp_repo_root_must_be_a_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DP_REPO_ROOT", str(tmp_path / "does-not-exist"))
    with pytest.raises(RuntimeError) as caught:
        repo_root()
    assert "DP_REPO_ROOT" in str(caught.value)


# --------------------------------------------------------------------------- #
# assert_paths_exist
# --------------------------------------------------------------------------- #


def test_assert_paths_exist_names_the_config_key(tmp_path: Path) -> None:
    config = {
        "clam": {"data_root_dir": str(tmp_path / "nowhere"), "tabular_csv": None},
        "paths": {"clam_root": str(tmp_path)},
    }
    with pytest.raises(FileNotFoundError) as caught:
        assert_paths_exist(config, ["paths.clam_root", "clam.data_root_dir", "clam.tabular_csv"])

    message = str(caught.value)
    assert "clam.data_root_dir" in message, "the message must name the key, not just the path"
    assert "clam.tabular_csv=<unset>" in message, "an unset key is a distinct diagnosis"
    assert "paths.clam_root" not in message, "a key that IS present must not be reported"


def test_assert_paths_exist_passes_on_a_real_tree() -> None:
    assert_paths_exist(
        {"paths": resolve_paths()},
        ["paths.repo_root", "paths.clam_root", "paths.splits_root", "paths.labels_dir"],
    )
