"""The synthetic end-to-end run: `make smoke`.

This is the check a stranger runs BEFORE downloading 98 GB of features and
requesting access to two gated HuggingFace repositories. It builds a complete
miniature of the real inputs — 1536-dim `.h5` feature files, a 4-class
`dataset_csv`, one split, and a 39-column arm-level CNV table — and runs
`dp-train` through the real `project/CLAM/main.py` for one fold and two epochs.

It is a REAL training run, and that is deliberate: a mocked one would prove that
the mock works. It is not a GPU job (`CUDA_VISIBLE_DEVICES=""` forces CPU, which
also makes it hermetic on a machine with two 3090s in use) and it retrains
nothing — every path it touches is inside pytest's `tmp_path`, and the run
directory it writes is named `smoke_*`, so it cannot collide with
`pam50_final_s1` or any of the five completed ladder arms.

What it actually establishes, in order of what breaks most often:

  1. the six console scripts are installed and `dp-train` runs from any directory;
  2. `${paths.*}` resolves against `DP_REPO_ROOT`, so the whole tree can be moved;
  3. CLAM's `.h5` contract is what `dpcode` thinks it is — including the FLAT
     `{data_dir}/{slide}.h5` fallback that the real feature store uses, rather
     than the `{data_dir}/h5_files/{slide}.h5` layout upstream CLAM assumes;
  4. `--inst_loss svm` imports `topk.svm` (the git-pinned smooth-topk), which is
     imported LAZILY at `utils/core_utils.py:315` and therefore fails
     mid-training rather than at startup when the pin is missing;
  5. `--log_data` imports `tensorboardX`, the other pin that looks droppable and
     is not;
  6. the run directory comes out self-describing: checkpoint, `summary.csv`,
     `config.resolved.yaml`, `run_metadata.json`, `clam_argv.json`,
     `metrics.json` and a copy of Hydra's `.hydra/`;
  7. the multimodal path works end to end: a 39-feature tabular table is read,
     a per-fold transform is fitted and saved, and the fusion head trains.

`project/CLAM` cannot simply be pointed at: `main.py` reads
`dataset_csv/<task>.csv` relative to its own working directory and that name is
not overridable by any flag. So the fixture builds a synthetic CLAM root that
SYMLINKS the real code directories and carries its own `dataset_csv/` and
`splits/`. Nothing in the real tree is written to.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REAL_REPO

pytestmark = pytest.mark.smoke

#: UNI2-h features are 1536-dim, everywhere.
EMBED_DIM = 1536

#: CLAM's `label_dict` order for `--task tcga_brca_subtyping`.
CLASSES = ["LumA", "LumB", "Basal", "Her2"]

#: Arm-level CNV: 39 chromosome arms, acrocentric p-arms excluded by
#: construction. Only the COUNT matters to the pipeline; the names are the real
#: ones so a failure reads like the real thing.
ARMS = [
    f"{chromosome}{arm}"
    for chromosome in [*(str(i) for i in range(1, 23))]
    for arm in ("p", "q")
    if f"{chromosome}{arm}" not in {"13p", "14p", "15p", "21p", "22p"}
]

#: Real CLAM subdirectories the synthetic root needs on `sys.path`. `main.py`
#: imports `utils.*` and `dataset_modules.*` as top-level modules, which resolve
#: only because CLAM is run with its own directory as the working directory.
CLAM_CODE_DIRS = ("models", "utils", "wsi_core", "dataset_modules", "vis_utils")


def build_synthetic_cohort(root: Path, n_slides: int = 16, seed: int = 0) -> dict:
    """Write a complete miniature cohort under `root` and return its manifest."""
    rng = np.random.default_rng(seed)
    import h5py
    import pandas as pd

    clam_root = root / "project" / "CLAM"
    features_dir = root / ".datasets" / "tcga-brca" / "embeddings"
    splits_dir = clam_root / "splits" / "tcga_brca_subtyping_100"
    for directory in (clam_root / "dataset_csv", features_dir, splits_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # --- the synthetic CLAM root -------------------------------------------
    shutil.copy2(REAL_REPO / "project/CLAM/main.py", clam_root / "main.py")
    for name in CLAM_CODE_DIRS:
        source = REAL_REPO / "project/CLAM" / name
        if source.is_dir():
            (clam_root / name).symlink_to(source, target_is_directory=True)

    # --- slides, cases and labels ------------------------------------------
    rows = []
    for index in range(n_slides):
        label = CLASSES[index % len(CLASSES)]
        case_id = f"SYNTH-{index:02d}"
        slide_id = f"{case_id}-DX1"
        rows.append({"case_id": case_id, "slide_id": slide_id, "label": label})

        # A class-dependent mean, so the model has something learnable and the
        # run does not depend on chance to finish.
        offset = CLASSES.index(label) * 0.05
        patches = int(rng.integers(8, 20))
        features = (rng.standard_normal((patches, EMBED_DIM)) * 0.1 + offset).astype(np.float32)
        # FLAT layout: dataset_generic.py:348-352 tries `{data_dir}/h5_files/…`
        # first and falls back to `{data_dir}/{slide}.h5`, which is the form the
        # real 1126-file store uses.
        with h5py.File(features_dir / f"{slide_id}.h5", "w") as handle:
            handle.create_dataset("features", data=features)
            handle.create_dataset(
                "coords", data=rng.integers(0, 4096, size=(patches, 2)).astype(np.int64)
            )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(clam_root / "dataset_csv" / "tcga_brca_subtyping.csv", index=False)

    # --- one split, stratified, patient-level -------------------------------
    # Stratified by hand, because every class must appear in EVERY part: CLAM
    # computes per-class AUC on val and test, and `--weighted_sample` divides by
    # the per-class training count (`utils/utils.py:156`), so a class missing
    # from train is a ZeroDivisionError rather than a warning.
    train, validation, test = [], [], []
    for label in CLASSES:
        slides = manifest.loc[manifest["label"] == label, "slide_id"].tolist()
        assert len(slides) >= 4, "each class needs a train pair plus a val and a test slide"
        validation.append(slides[0])
        test.append(slides[1])
        train.extend(slides[2:])
    width = max(len(train), len(validation), len(test))
    pd.DataFrame(
        {
            "train": train + [None] * (width - len(train)),
            "val": validation + [None] * (width - len(validation)),
            "test": test + [None] * (width - len(test)),
        }
    ).to_csv(splits_dir / "splits_0.csv")

    # --- the second modality: 39 arms per case ------------------------------
    cnv_dir = root / ".scratch" / "cnv-tabular"
    cnv_dir.mkdir(parents=True, exist_ok=True)
    cnv = manifest[["case_id", "label"]].copy()
    for arm in ARMS:
        cnv[arm] = rng.standard_normal(len(manifest)).astype(np.float32).round(4)
    cnv.to_csv(cnv_dir / "TCGA_BRCA_CNV_arm_4class_clam.csv", index=False)

    return {
        "root": root,
        "clam_root": clam_root,
        "results_root": root / ".scratch" / "results",
        "n_slides": n_slides,
        "n_arms": len(ARMS),
    }


@pytest.fixture(scope="module")
def cohort(tmp_path_factory) -> dict:
    return build_synthetic_cohort(tmp_path_factory.mktemp("smoke") / "repo")


def run_dp_train(cohort: dict, overrides: list[str]) -> subprocess.CompletedProcess:
    """Run `dp-train` from a directory that is neither the repo nor the cohort."""
    executable = shutil.which("dp-train")
    command = [executable] if executable else [sys.executable, "-m", "dpcode.cli.train"]
    environment = {
        **os.environ,
        "DP_REPO_ROOT": str(cohort["root"]),
        # CPU only. The smoke test must not need a GPU, and on this machine it
        # must not take one that a real run is using.
        "CUDA_VISIBLE_DEVICES": "",
        # Nothing may reach the network or a credential.
        "WANDB_MODE": "disabled",
    }
    for name in ("DP_DATA_ROOT", "DP_SCRATCH_ROOT", "DP_RESULTS_ROOT"):
        environment.pop(name, None)
    return subprocess.run(
        [*command, *overrides],
        cwd=str(Path(sys.prefix)),
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def assert_run_directory_is_self_describing(run_dir: Path) -> dict:
    """The six files that make a run replayable, plus Hydra's own copy."""
    expected = [
        "s_0_checkpoint.pt",
        "summary.csv",
        "config.resolved.yaml",
        "run_metadata.json",
        "clam_argv.json",
        "metrics.json",
    ]
    missing = [name for name in expected if not (run_dir / name).exists()]
    assert not missing, f"{run_dir} is missing {missing}; it holds {sorted(p.name for p in run_dir.iterdir())}"
    assert (run_dir / ".hydra" / "config.yaml").exists(), (
        "Hydra's own (unresolved) config was not copied in beside the resolved one"
    )

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["exit_status"] == 0
    # Git provenance is collected for `DP_REPO_ROOT`, which here is a synthetic
    # cohort in `tmp_path` and is not a work tree — so the field must SAY that
    # rather than be absent or, worse, report the commit of some other checkout.
    assert metadata["git"]["available"] is False
    assert "not a git work tree" in metadata["git"]["reason"]
    assert metadata["determinism"] is not None
    assert metadata["frozen_internals"]["bag_loss_ce_label_smoothing"]["value"] == 0.1

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_folds"] == 1
    assert "test_auc" in metrics["metrics"]
    return metrics


def test_smoke_wsi_only(cohort: dict) -> None:
    """`experiment=pam50_wsi_final`, one fold, two epochs, on synthetic slides."""
    completed = run_dp_train(
        cohort,
        [
            "experiment=pam50_wsi_final",
            "clam.exp_code=smoke_wsi",
            "clam.k=1",
            "clam.max_epochs=2",
            "clam.wandb=false",
        ],
    )
    assert completed.returncode == 0, (
        f"dp-train failed ({completed.returncode})\n"
        f"--- stdout ---\n{completed.stdout[-4000:]}\n"
        f"--- stderr ---\n{completed.stderr[-4000:]}"
    )

    run_dir = cohort["results_root"] / "smoke_wsi_s1"
    metrics = assert_run_directory_is_self_describing(run_dir)
    assert metrics["folds"][0]["fold"] == 0

    argv = json.loads((run_dir / "clam_argv.json").read_text(encoding="utf-8"))
    assert argv["cwd"] == str(cohort["clam_root"])
    assert "--inst_loss" in argv["argv"], (
        "the frozen baseline trains with --inst_loss svm, which is what pulls in "
        "the git-pinned smooth-topk dependency"
    )

    # Nothing escaped the temporary tree.
    assert not (REAL_REPO / ".scratch" / "results" / "smoke_wsi_s1").exists()


def test_smoke_wsi_plus_cnv_fusion(cohort: dict) -> None:
    """The ladder's shape: `experiment=pam50_wsi_cnv fusion=concat`.

    `clam.pretrained_wsi_ckpt=null` replaces the warm start — the real ladder
    initialises its WSI branch from `pam50_final_s1`, which does not exist in a
    synthetic cohort, and the point here is the fusion wiring rather than the
    training protocol.
    """
    completed = run_dp_train(
        cohort,
        [
            "experiment=pam50_wsi_cnv",
            "fusion=concat",
            "clam.exp_code=smoke_cnv",
            "clam.k=1",
            "clam.max_epochs=2",
            "clam.pretrained_wsi_ckpt=null",
        ],
    )
    assert completed.returncode == 0, (
        f"dp-train failed ({completed.returncode})\n"
        f"--- stdout ---\n{completed.stdout[-4000:]}\n"
        f"--- stderr ---\n{completed.stderr[-4000:]}"
    )

    run_dir = cohort["results_root"] / "smoke_cnv_s1"
    assert_run_directory_is_self_describing(run_dir)

    transform = run_dir / "s_0_tabular_transform.json"
    assert transform.exists(), (
        "the per-fold tabular transform is fitted on the TRAINING fold only and "
        "saved next to the checkpoint; without it the checkpoint cannot be applied"
    )
    saved = json.loads(transform.read_text(encoding="utf-8"))
    assert len(saved["selected_feature_names"]) == cohort["n_arms"] == 39


def test_smoke_refuses_to_overwrite_its_own_run(cohort: dict) -> None:
    """The guard that protects the five unrecoverable ladder arms, on a real run.

    Ordering, not just existence: `dp-train` computes the CLAM run directory and
    checks it BEFORE writing anything, so the refusal leaves the completed run
    exactly as it was.
    """
    run_dir = cohort["results_root"] / "smoke_wsi_s1"
    assert (run_dir / "summary.csv").exists(), "run test_smoke_wsi_only first"
    before = (run_dir / "summary.csv").read_bytes()

    completed = run_dp_train(
        cohort,
        [
            "experiment=pam50_wsi_final",
            "clam.exp_code=smoke_wsi",
            "clam.k=1",
            "clam.max_epochs=2",
            "clam.wandb=false",
        ],
    )
    assert completed.returncode != 0
    assert "already contains results" in completed.stderr
    assert (run_dir / "summary.csv").read_bytes() == before


def test_multirun_produces_the_same_layout_per_arm(cohort: dict) -> None:
    """`-m` gets exactly the contract a single run gets (DESIGN-ADDENDUM A1).

    This is only true because the run directory is derived from `clam.exp_code`
    and `clam.seed` rather than from Hydra's output directory. Hydra IGNORES
    `hydra.run.dir` under `--multirun` and uses `hydra.sweep.dir`/`subdir`
    instead, so a layout keyed off Hydra's directory would silently change shape
    the moment the ladder is swept — which is how the ladder is meant to be run:

        dp-train -m experiment=pam50_wsi_cnv fusion=concat,gated,...
    """
    completed = run_dp_train(
        cohort,
        [
            "-m",
            "experiment=pam50_wsi_cnv",
            "fusion=concat,gated",
            "clam.exp_code=smoke_sweep_${fusion.name}",
            "clam.k=1",
            "clam.max_epochs=1",
            "clam.pretrained_wsi_ckpt=null",
        ],
    )
    assert completed.returncode == 0, (
        f"dp-train --multirun failed\n{completed.stdout[-3000:]}\n{completed.stderr[-3000:]}"
    )

    for operator in ("concat", "gated"):
        assert_run_directory_is_self_describing(
            cohort["results_root"] / f"smoke_sweep_{operator}_s1"
        )

    # Hydra's own output stays in scratch, never in a results directory: it is
    # created and written to BEFORE the task function runs, so an overwrite guard
    # inside the task cannot protect a directory Hydra has already opened.
    assert (cohort["root"] / ".scratch" / "multirun").is_dir()
    assert not (cohort["results_root"] / "smoke_sweep_concat_s1" / "train.log").exists()


def test_dry_run_writes_nothing(cohort: dict) -> None:
    completed = run_dp_train(
        cohort,
        [
            "experiment=pam50_wsi_final",
            "clam.exp_code=smoke_never_written",
            "clam.k=1",
            "--dry-run",
        ],
    )
    assert completed.returncode == 0, completed.stderr
    assert "[dry run] command" in completed.stdout
    assert not (cohort["results_root"] / "smoke_never_written_s1").exists()
