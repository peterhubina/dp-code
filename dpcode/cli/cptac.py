"""`dp-cptac` — the CPTAC-BRCA external-validation pipeline, phases 0-4.

Replaces phases 1-4 of `tools/cptac/run_pipeline.sh` and adds phase 0, the step
that script never had. Phases 5-8 (the dormant RNA branch) stay in the shell
script.

    dp-cptac phase=0        cohort metadata: wsi_manifest.csv, rna_manifest.csv,
                            clinical/*.csv and cohort.csv, with no bulk transfer
    dp-cptac phase=features the GATED 16 GB UNI2-h feature archive -> 34 GB
    dp-cptac phase=1        provenance audit of the feature store
    dp-cptac phase=2        coverage reconciliation + the CLAM manifest
    dp-cptac phase=3        frozen-weight 10-fold inference (needs the
                            TCGA-trained checkpoints; about a minute on a GPU)
    dp-cptac phase=4        slide- and case-level metrics
    dp-cptac phase=1,2,3,4  several, always executed in canonical order
    dp-cptac phase=all      everything above, in order

`phase=X` is shorthand for the Hydra override `cptac.phase=X`; both spellings
work and both end up in the composed config.

WHY PHASE 0 EXISTS. `cohort.csv` is required by phase 2 and has exactly one
producer: `tools/download_cptac.py` invoked with more than one modality
(`if len(wants) > 1`, download_cptac.py:506). `--modality clinical` — what that
script's docstring used to recommend as the place to start — provably cannot
write it. Meanwhile `run_pipeline.sh` began with the 16 GB gated feature download
and only then ran the two phases that need the manifests, so the failure arrived
after the download instead of before it. Every phase here declares what it reads
and what it writes, and ALL preconditions are checked before ANY phase runs.

Nothing here is ever fitted, tuned, calibrated or thresholded on CPTAC: phase 3
loads TCGA-trained checkpoints and runs them frozen.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig

from .. import schema
from ..paths import assert_paths_absolute
from ..runinfo import RunMetadata, write_config_snapshot
from .config import compose_config

__all__ = [
    "PHASE_ORDER",
    "PHASE_ALIASES",
    "Step",
    "resolve_phases",
    "build_steps",
    "check_preconditions",
    "main",
]

DEFAULT_OPTION = "default"

#: Canonical execution order. Selections are sorted into it, never run as typed.
PHASE_ORDER = ("0", "features", "1", "2", "3", "4")

#: Every accepted spelling. The numbers are `run_pipeline.sh`'s own phase labels;
#: `2a` is what that script calls phase 2.
PHASE_ALIASES = {
    "0": "0",
    "metadata": "0",
    "features": "features",
    "embeddings": "features",
    "1": "1",
    "audit": "1",
    "2": "2",
    "2a": "2",
    "manifest": "2",
    "3": "3",
    "infer": "3",
    "4": "4",
    "summarise": "4",
    "summarize": "4",
}


@dataclass
class Step:
    """One phase: what it runs, what it needs on disk, what it leaves behind."""

    key: str
    title: str
    command: list[str]
    #: `(path, explanation)` pairs, checked before anything runs — unless an
    #: earlier SELECTED step declares the same path in `produces`.
    requires: list[tuple[Path, str]] = field(default_factory=list)
    produces: list[Path] = field(default_factory=list)
    #: `(directory, glob, explanation)`: the directory must exist and match.
    glob_requires: list[tuple[Path, str, str]] = field(default_factory=list)


def resolve_phases(value: Any) -> list[str]:
    """`"all"`, `"0"`, `"1,2,3,4"`, `[1, 2]` -> canonical, ordered, deduplicated."""
    if isinstance(value, str):
        tokens = [token for token in value.replace(" ", "").split(",") if token]
    elif isinstance(value, (list, tuple)) or hasattr(value, "__iter__"):
        tokens = [str(item) for item in value]  # a YAML/override list
    else:
        tokens = [str(value)]
    if not tokens:
        raise ValueError("cptac.phase is empty; pass e.g. phase=0 or phase=all.")

    selected: set[str] = set()
    for token in tokens:
        lowered = token.lower()
        if lowered == "all":
            selected.update(PHASE_ORDER)
            continue
        if lowered not in PHASE_ALIASES:
            raise ValueError(
                f"Unknown phase {token!r}. Valid: "
                + ", ".join(sorted(set(PHASE_ALIASES) | {"all"}))
            )
        selected.add(PHASE_ALIASES[lowered])
    return [phase for phase in PHASE_ORDER if phase in selected]


def build_steps(cfg: DictConfig, phases: Sequence[str]) -> list[Step]:
    """Turn the config into one :class:`Step` per selected phase."""
    node = cfg.cptac
    tools = Path(str(cfg.paths.repo_root)) / "tools"
    python = sys.executable
    from_phase_0 = "written by `dp-cptac phase=0`"
    from_features = "written by `dp-cptac phase=features`"

    metadata = node.metadata
    steps: dict[str, Step] = {}
    steps["0"] = Step(
        key="0",
        title="cohort metadata (both manifests + cohort.csv; no bulk transfer)",
        command=[
            python,
            str(tools / "download_cptac.py"),
            "--modality", str(metadata.modality),
            "--collection", str(metadata.collection),
            "--output", str(metadata.output),
            "--workers", str(metadata.workers),
            *(["--cohort-only"] if bool(metadata.cohort_only) else []),
            *(["--dry-run"] if bool(metadata.dry_run) else []),
            *(["--limit", str(metadata.limit)] if int(metadata.limit) else []),
        ],
        produces=[
            Path(str(metadata.output)) / "wsi_manifest.csv",
            Path(str(metadata.output)) / "rna_manifest.csv",
            Path(str(metadata.output)) / "cohort.csv",
        ],
    )

    features = node.features
    steps["features"] = Step(
        key="features",
        title="GATED UNI2-h feature archive (16 GB -> 34 GB)",
        command=[
            python,
            str(tools / "download_embeddings.py"),
            "--cohort", str(features.cohort),
            "--output_dir", str(features.output_dir),
            "--expected_h5", str(features.expected_h5),
        ],
        # The whole point: this fails BEFORE the 16 GB download, not after it.
        requires=[
            (Path(str(node.manifest.wsi_manifest)), from_phase_0),
            (Path(str(node.manifest.cohort)), from_phase_0),
        ],
        produces=[Path(str(features.output_dir))],
    )

    audit = node.audit
    steps["1"] = Step(
        key="1",
        title="phase 1 - feature provenance audit",
        command=[
            python,
            str(tools / "cptac" / "audit_feature_provenance.py"),
            "--feature_dir", str(audit.feature_dir),
            "--wsi_manifest", str(audit.wsi_manifest),
            "--out_csv", str(audit.out_csv),
        ],
        requires=[(Path(str(audit.wsi_manifest)), from_phase_0)],
        glob_requires=[(Path(str(audit.feature_dir)), "*.h5", from_features)],
        produces=[Path(str(audit.out_csv))],
    )

    manifest = node.manifest
    steps["2"] = Step(
        key="2",
        title="phase 2a - coverage reconciliation + CLAM manifest",
        command=[
            python,
            str(tools / "cptac" / "prepare_cptac_manifest.py"),
            "--feature_dir", str(manifest.feature_dir),
            "--wsi_manifest", str(manifest.wsi_manifest),
            "--cohort", str(manifest.cohort),
            "--dataset_csv", str(manifest.dataset_csv),
            "--coverage_csv", str(manifest.coverage_csv),
        ],
        requires=[
            (Path(str(manifest.wsi_manifest)), from_phase_0),
            (Path(str(manifest.cohort)), from_phase_0),
        ],
        glob_requires=[(Path(str(manifest.feature_dir)), "*.h5", from_features)],
        produces=[Path(str(manifest.dataset_csv)), Path(str(manifest.coverage_csv))],
    )

    infer = node.infer
    steps["3"] = Step(
        key="3",
        title="phase 3 - frozen-weight 10-fold inference",
        command=[
            python,
            str(tools / "cptac" / "infer_cptac_pam50.py"),
            "--feature_dir", str(infer.feature_dir),
            "--dataset_csv", str(infer.dataset_csv),
            "--ckpt_dir", str(infer.ckpt_dir),
            "--output_dir", str(infer.output_dir),
            "--n_folds", str(infer.n_folds),
            "--n_classes", str(infer.n_classes),
            "--embed_dim", str(infer.embed_dim),
            "--model_size", str(infer.model_size),
            "--dropout", str(infer.dropout),
        ],
        requires=[(Path(str(infer.dataset_csv)), "written by `dp-cptac phase=2`")],
        glob_requires=[
            (Path(str(infer.feature_dir)), "*.h5", from_features),
            (
                Path(str(infer.ckpt_dir)),
                "s_*_checkpoint.pt",
                "the TCGA-trained WSI arm, from `dp-train experiment=pam50_wsi_final`",
            ),
        ],
        produces=[Path(str(infer.output_dir)) / "ensemble_predictions.csv"],
    )

    summarise = node.summarise
    steps["4"] = Step(
        key="4",
        title="phase 4 - slide- and case-level metrics",
        command=[
            python,
            str(tools / "cptac" / "summarise_predictions.py"),
            str(summarise.results_dir),
        ],
        requires=[
            (
                Path(str(summarise.results_dir)) / "ensemble_predictions.csv",
                "written by `dp-cptac phase=3`",
            )
        ],
    )

    return [steps[phase] for phase in phases]


def check_preconditions(steps: Sequence[Step]) -> list[str]:
    """Every missing input, for every selected phase, before any of them runs.

    A requirement is satisfied if it exists on disk OR an earlier selected step
    declares it in `produces` — which is what makes `phase=all` legal on an empty
    machine while `phase=2` alone is not.
    """
    upcoming: set[str] = set()
    problems: list[str] = []
    for step in steps:
        for path, note in step.requires:
            if str(path) in upcoming or path.exists():
                continue
            problems.append(f"phase {step.key}: missing {path}  ({note})")
        for directory, pattern, note in step.glob_requires:
            if str(directory) in upcoming:
                continue
            if not directory.is_dir():
                problems.append(
                    f"phase {step.key}: missing directory {directory}  ({note})"
                )
            elif not any(directory.glob(pattern)):
                problems.append(f"phase {step.key}: no {pattern} in {directory}  ({note})")
        upcoming.update(str(path) for path in step.produces)
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dp-cptac",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="`phase=0` / `phase=1,2,3,4` / `phase=all`, plus any Hydra override "
        "(e.g. `cptac.infer.ckpt_dir=...`).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_OPTION,
        help=f"Which `cptac` config group option to compose (default: {DEFAULT_OPTION}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and every command, then stop. Distinct from "
        "`cptac.metadata.dry_run`, which is phase 0's own metadata-only flag.",
    )
    return parser


def _expand_phase_shorthand(overrides: Sequence[str]) -> list[str]:
    """`phase=0` -> `cptac.phase=0`; everything else passes through untouched.

    The bare spelling is what the pipeline documentation uses. Expanding it into
    a real override rather than handling it out of band keeps the selection
    inside the composed config, where a config snapshot records it.

    A comma-separated value is quoted on the way through: unquoted,
    `cptac.phase=1,2,3,4` is ambiguous to Hydra's override grammar (list? sweep?)
    and raises. Quoting picks the string reading, which is what
    :func:`resolve_phases` wants. `cptac.phase=...` typed in full is left alone,
    so Hydra's own message about quoting still reaches whoever typed it.
    """
    expanded = []
    for override in overrides:
        if not override.startswith("phase="):
            expanded.append(override)
            continue
        value = override[len("phase=") :]
        expanded.append(f"cptac.phase='{value}'" if "," in value else f"cptac.phase={value}")
    return expanded


def main(argv: Sequence[str] | None = None) -> int:
    from hydra.errors import HydraException
    from omegaconf.errors import OmegaConfBaseException

    args = build_parser().parse_args(argv)
    overrides = _expand_phase_shorthand(args.overrides)

    try:
        cfg = compose_config([f"+cptac={args.config}", *overrides])
        schema.reject_appended_overrides(
            overrides, allow=bool(cfg.run.allow_config_surgery)
        )
        if not isinstance(cfg.cptac, DictConfig):
            # `+cptac=x` assigns the STRING "x" when no such group option exists,
            # because `RootConf.cptac` is a real key.
            raise RuntimeError(
                f"cptac={args.config} did not compose a config group; got "
                f"{cfg.cptac!r}. Is dpcode/conf/cptac/{args.config}.yaml installed?"
            )
        assert_paths_absolute(cfg)

        phases = resolve_phases(cfg.cptac.phase)
        steps = build_steps(cfg, phases)
        repo = Path(str(cfg.paths.repo_root))

        print(f"config : cptac={args.config}")
        print(f"phases : {', '.join(phases)}")
        print(f"cwd    : {repo}")
        for step in steps:
            print(f"\n[{step.key}] {step.title}")
            print("  " + " ".join(step.command))

        problems = check_preconditions(steps)
        if problems:
            print()
            sys.stdout.flush()
            print("Missing inputs; nothing was run:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            print(
                "\nThe usual cause is that phase 0 has never run. It writes both "
                "manifests and cohort.csv and transfers no slide:\n"
                "    dp-cptac phase=0",
                file=sys.stderr,
            )
            return 1

        if args.dry_run:
            print("\n--dry-run: nothing dispatched.")
            return 0

        # Phase 3 is the only phase whose output is a NUMBER rather than a
        # manifest: `ensemble_predictions.csv` is the external WSI arm of the
        # headline table, and it is not re-derivable from public sources the way
        # phase 0's manifests are (those are byte-identical on every re-run). So
        # it gets the config snapshot and run metadata; the others do not.
        metadata = None
        if "3" in phases:
            predictions = Path(str(cfg.cptac.infer.output_dir))
            predictions.mkdir(parents=True, exist_ok=True)
            write_config_snapshot(cfg, predictions)
            metadata = RunMetadata(
                predictions,
                run_seed=int(cfg.run.seed),
                command=[sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
                extra={"entry_point": "dp-cptac", "phases": list(phases)},
            )
            metadata.start()

        for step in steps:
            print(f"\n=== [{step.key}] {step.title} ===", flush=True)
            completed = subprocess.run(step.command, cwd=str(repo))
            if completed.returncode != 0:
                if metadata is not None:
                    metadata.finish(completed.returncode)
                print(
                    f"phase {step.key} failed with exit status {completed.returncode}; "
                    "later phases were not run.",
                    file=sys.stderr,
                )
                return completed.returncode
        if metadata is not None:
            metadata.finish(0)
        return 0
    except (
        HydraException,
        OmegaConfBaseException,
        ValueError,
        KeyError,
        FileNotFoundError,
        RuntimeError,
    ) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
