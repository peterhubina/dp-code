"""The overwrite guard, and what a run directory must be able to tell you.

`.scratch` is gitignored and has no backup. Five completed fusion-ladder arms and
the frozen WSI baseline live there; between them they cost hours of GPU time and
nothing in the repository can recreate them. So `assert_run_dir_writable` is not
defensive style, it is the only thing standing between a mistyped `exp_code` and
an unrecoverable loss — and the ORDER matters as much as the check: it runs
before anything is written, which is why Hydra's own output directory is kept in
scratch (Hydra creates `hydra.run.dir` and writes four files into it BEFORE the
task function is called).

The second half is `config.resolved.yaml`. Hydra's `.hydra/config.yaml` is stored
UNRESOLVED — `${oc.env:DP_REPO_ROOT,…}` and `${paths.repo_root}/…` verbatim — so
replaying it on another machine reconstructs a DIFFERENT configuration. "This run
can be re-run from its saved config" is false unless the resolved snapshot
exists, and the snapshot must be publishable: `paths.nou_root` names a private
institutional cohort and is redacted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from dpcode import runinfo


# --------------------------------------------------------------------------- #
# the overwrite guard
# --------------------------------------------------------------------------- #


def test_a_fresh_directory_is_writable(tmp_path: Path) -> None:
    target = runinfo.assert_run_dir_writable(tmp_path / "pam50_final_s1")
    assert target.is_dir()


def test_summary_csv_blocks_the_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "pam50_wsi_cnv_coattn_s1"
    run_dir.mkdir()
    (run_dir / "summary.csv").write_text(",folds,test_auc\n0,0,0.89\n", encoding="utf-8")

    with pytest.raises(FileExistsError) as caught:
        runinfo.assert_run_dir_writable(run_dir)
    message = str(caught.value)
    assert "summary.csv" in message
    assert str(run_dir) in message, "the message must name the directory it refused"
    assert "run.overwrite=true" in message, "and say what the escape hatch is"


def test_a_checkpoint_alone_blocks_the_run(tmp_path: Path) -> None:
    """A crashed run has no summary.csv and is still not disposable."""
    run_dir = tmp_path / "pam50_final_s1"
    run_dir.mkdir()
    for fold in range(4):
        (run_dir / f"s_{fold}_checkpoint.pt").touch()

    with pytest.raises(FileExistsError) as caught:
        runinfo.assert_run_dir_writable(run_dir)
    assert "s_0_checkpoint.pt" in str(caught.value)


def test_overwrite_true_is_the_only_way_through(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_s1"
    run_dir.mkdir()
    (run_dir / "summary.csv").touch()
    assert runinfo.assert_run_dir_writable(run_dir, overwrite=True) == run_dir


def test_unrelated_contents_do_not_block(tmp_path: Path) -> None:
    """Only `summary.csv` and `s_*_checkpoint.pt` mean "results are here".

    A directory holding just a stale log or a `.hydra/` copy is a failed start,
    not a completed run, and refusing it would train nobody's patience.
    """
    run_dir = tmp_path / "run_s1"
    (run_dir / ".hydra").mkdir(parents=True)
    (run_dir / "train.log").touch()
    assert runinfo.assert_run_dir_writable(run_dir) == run_dir


def test_clam_run_dir_matches_main_py_line_407(tmp_path: Path) -> None:
    """`os.path.join(results_dir, exp_code + '_s{}'.format(seed))`, and nothing else."""
    assert runinfo.clam_run_dir("/results", "pam50_final", 1) == Path(
        "/results/pam50_final_s1"
    )
    assert runinfo.clam_run_dir("/results", "pam50_wsi_cnv_coattn", 4) == Path(
        "/results/pam50_wsi_cnv_coattn_s4"
    )


# --------------------------------------------------------------------------- #
# what a run directory must contain
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_config():
    return OmegaConf.create(
        {
            "paths": {
                "repo_root": "/repo",
                "results_root": "/repo/.scratch/results",
                "nou_root": "/mnt/private/nou",
            },
            "clam": {"exp_code": "pam50_final", "seed": 1, "lr": 1e-4},
            "run": {"name": "pam50_wsi_final", "seed": 1},
        }
    )


def test_config_snapshot_is_resolved_and_redacted(tmp_path: Path, sample_config) -> None:
    target = runinfo.write_config_snapshot(sample_config, tmp_path)
    text = target.read_text(encoding="utf-8")

    assert target.name == "config.resolved.yaml"
    assert "/mnt/private/nou" not in text, (
        "paths.nou_root names a private institutional cohort and must never appear "
        "in a snapshot meant to be publishable"
    )
    assert runinfo.REDACTED_PLACEHOLDER in text
    assert "paths.nou_root" in text, "the redaction must be visible, not silent"

    body = OmegaConf.create(target.read_text(encoding="utf-8"))
    assert body.paths.repo_root == "/repo"


def test_config_snapshot_survives_an_unresolvable_key(tmp_path: Path) -> None:
    """A snapshot written after a 2h38m run must not fail the job it describes."""
    cfg = OmegaConf.create({"paths": {"repo_root": "${oc.env:DP_NOT_SET_ANYWHERE}"}})
    target = runinfo.write_config_snapshot(cfg, tmp_path)
    assert "UNRESOLVED" in target.read_text(encoding="utf-8")


def test_clam_argv_records_the_command_and_its_cwd(tmp_path: Path) -> None:
    target = runinfo.write_clam_argv(
        tmp_path, ["--task", "tcga_brca_subtyping"], cwd="/repo/project/CLAM",
        executable="/opt/venv/bin/python",
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["argv"] == ["--task", "tcga_brca_subtyping"]
    assert payload["cwd"] == "/repo/project/CLAM", (
        "the cwd is not decoration: main.py resolves `dataset_csv/<task>.csv` and "
        "the `splits/` prefix against it, and neither is overridable by a flag"
    )
    assert payload["executable"] == "/opt/venv/bin/python"


def test_metrics_json_is_summary_csv_made_machine_readable(tmp_path: Path) -> None:
    (tmp_path / "summary.csv").write_text(
        ",folds,test_auc,val_auc,test_acc,val_acc\n"
        "0,0,0.90,0.91,0.70,0.71\n"
        "1,1,0.80,0.81,0.60,0.61\n",
        encoding="utf-8",
    )
    target = runinfo.write_metrics(tmp_path)
    assert target is not None
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["n_folds"] == 2
    assert payload["summary"]["test_auc"]["mean"] == pytest.approx(0.85)
    # Population sd, matching numpy.std's default, which is what CLAM reports.
    assert payload["summary"]["test_auc"]["std"] == pytest.approx(0.05)
    assert "population" in payload["std_convention"]


def test_metrics_json_is_absent_rather_than_wrong(tmp_path: Path) -> None:
    assert runinfo.write_metrics(tmp_path) is None


def test_metrics_json_falls_back_to_a_partial_summary(tmp_path: Path) -> None:
    """`--k_start/--k_end` make CLAM write `summary_partial_<start>_<end>.csv`."""
    (tmp_path / "summary_partial_1_3.csv").write_text(
        ",folds,test_auc\n0,1,0.88\n1,2,0.86\n", encoding="utf-8"
    )
    target = runinfo.write_metrics(tmp_path)
    assert target is not None
    assert json.loads(target.read_text(encoding="utf-8"))["source"] == "summary_partial_1_3.csv"


def test_run_metadata_start_then_finish(tmp_path: Path) -> None:
    metadata = runinfo.RunMetadata(
        tmp_path, run_seed=1, clam_seed=1, command=["dp-train", "experiment=x"],
        extra={"entry_point": "dp-train"},
    )
    metadata.start()
    opening = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert opening["status"] == "running", (
        "the opening write must exist on its own: a run killed at hour three still "
        "has to say what it was doing and on which commit"
    )
    for key in ("git", "environment", "dependencies", "determinism", "frozen_internals"):
        assert key in opening, f"run_metadata.json is missing {key}"
    assert opening["command_line"] == "dp-train experiment=x"
    assert opening["entry_point"] == "dp-train"

    metadata.update(metrics_file="metrics.json")
    metadata.finish(0)
    final = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert final["status"] == "completed"
    assert final["exit_status"] == 0
    assert final["duration_seconds"] is not None
    assert final["metrics_file"] == "metrics.json"


def test_run_metadata_records_a_failure_as_a_failure(tmp_path: Path) -> None:
    metadata = runinfo.RunMetadata(tmp_path)
    metadata.start()
    metadata.finish(2)
    final = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert final["status"] == "failed"
    assert final["exit_status"] == 2


def test_frozen_internals_name_the_two_hardcoded_training_decisions() -> None:
    """Label smoothing and the early-stopping monitor are outside argparse.

    They change results, they have no flag, and a reader of a run directory has
    to be able to learn they exist. Recorded, never "exposed" — a flag would let
    a later run change them and still look comparable to the published numbers.
    """
    internals = runinfo.FROZEN_INTERNALS
    assert internals["bag_loss_ce_label_smoothing"]["value"] == 0.1
    assert internals["early_stopping_stop_epoch"]["value"] == 5
    assert internals["early_stopping_monitor"]["value"] == "-auc"
    for entry in internals.values():
        assert entry.get("site") or entry.get("sites"), "each must name its source line"
