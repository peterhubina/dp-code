"""Make a run directory self-describing, and refuse to clobber one.

A CLAM run directory holds checkpoints, per-fold predictions and `summary.csv`,
and Hydra adds `.hydra/{config,hydra,overrides}.yaml`. That is not enough to
replay the run, for one specific reason: **Hydra's `.hydra/config.yaml` is stored
UNRESOLVED**, with `${oc.env:DP_REPO_ROOT,…}` and `${paths.repo_root}/…` verbatim.
Six months later, on a machine where `DP_REPO_ROOT` differs, replaying that file
reconstructs a *different* configuration, and nothing on disk records which
features were actually read. Any claim that a run can be re-run from its saved
config is false unless a resolved snapshot exists — so `config.resolved.yaml` is
mandatory here, not polish.

Four files, written next to whatever CLAM writes:

    config.resolved.yaml   OmegaConf.to_yaml(cfg, resolve=True)
    run_metadata.json      git, environment, seeds, command, timing, exit status,
                           and the `frozen_internals` block below
    clam_argv.json         the exact argv handed to main.py, plus its cwd
    metrics.json           summary.csv, machine-readable

Plus :func:`assert_run_dir_writable`, which exists because `.scratch` is
gitignored: five completed ladder arms live there, they cost 2h38m of GPU time,
and nothing else in the repository can recreate them.
"""

from __future__ import annotations

import csv
import getpass
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .determinism import describe_determinism

__all__ = [
    "CONFIG_SNAPSHOT_NAME",
    "METADATA_NAME",
    "CLAM_ARGV_NAME",
    "METRICS_NAME",
    "FROZEN_INTERNALS",
    "assert_run_dir_writable",
    "write_config_snapshot",
    "write_clam_argv",
    "write_metrics",
    "collect_git_info",
    "collect_environment",
    "dependency_versions",
    "RunMetadata",
]

CONFIG_SNAPSHOT_NAME = "config.resolved.yaml"
METADATA_NAME = "run_metadata.json"
CLAM_ARGV_NAME = "clam_argv.json"
METRICS_NAME = "metrics.json"

#: Distribution name declared in pyproject.toml. `dependency_versions` reads this
#: distribution's own requirements, so the pinned list has exactly one home.
DIST_NAME = "dp-code"

#: Result-affecting parameters that are hardcoded OUTSIDE argparse and stay that
#: way. They are documented rather than exposed, because exposing them would let
#: a future run change them and still look comparable to the published numbers.
FROZEN_INTERNALS = {
    "bag_loss_ce_label_smoothing": {
        "value": 0.1,
        "site": "project/CLAM/utils/core_utils.py:320",
        "note": (
            "nn.CrossEntropyLoss(label_smoothing=0.1) whenever --bag_loss ce. "
            "Upstream CLAM uses no label smoothing. There is no flag for it."
        ),
    },
    "early_stopping_stop_epoch": {
        "value": 5,
        "site": "project/CLAM/utils/core_utils.py:425",
        "note": (
            "EarlyStopping(patience=args.patience, stop_epoch=5). The class default "
            "is stop_epoch=50; the call site overrides it and there is no flag."
        ),
    },
    "early_stopping_monitor": {
        "value": "-auc",
        "sites": [
            "project/CLAM/utils/core_utils.py:425",
            "project/CLAM/utils/core_utils.py:711",
            "project/CLAM/utils/core_utils.py:799",
        ],
        "note": (
            "monitor_value = -auc if np.isfinite(auc) else val_loss, so the saved "
            "checkpoint maximises AUC. Upstream CLAM monitors val_loss."
        ),
    },
}

#: A directory containing either of these is a completed or partial run.
_RESULT_MARKER_FILE = "summary.csv"
_RESULT_MARKER_GLOB = "s_*_checkpoint.pt"


def assert_run_dir_writable(run_dir: str | os.PathLike[str], overwrite: bool = False) -> Path:
    """Abort if `run_dir` already holds results, unless `overwrite` is true.

    Checks for `summary.csv` and for any `s_*_checkpoint.pt`. Returns the
    directory as a `Path` (created if absent) so callers can chain.
    """
    path = Path(run_dir)
    if not overwrite and path.is_dir():
        existing = []
        if (path / _RESULT_MARKER_FILE).exists():
            existing.append(_RESULT_MARKER_FILE)
        checkpoints = sorted(p.name for p in path.glob(_RESULT_MARKER_GLOB))
        existing.extend(checkpoints[:3])
        if existing:
            more = "" if len(checkpoints) <= 3 else f" (+{len(checkpoints) - 3} more checkpoints)"
            raise FileExistsError(
                f"{path} already contains results: {', '.join(existing)}{more}. "
                "Refusing to overwrite — this tree is gitignored and unrecoverable. "
                "Choose a different run name, or pass run.overwrite=true."
            )
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_config_snapshot(cfg: Any, run_dir: str | os.PathLike[str]) -> Path:
    """Write `config.resolved.yaml` — the fully resolved config. Mandatory."""
    from omegaconf import OmegaConf

    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / CONFIG_SNAPSHOT_NAME
    target.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return target


def write_clam_argv(
    run_dir: str | os.PathLike[str],
    argv: Sequence[str],
    cwd: str | os.PathLike[str],
    *,
    executable: str | None = None,
) -> Path:
    """Write `clam_argv.json` — the exact argv handed to CLAM, plus its cwd.

    The cwd is not decoration: `main.py` resolves `dataset_csv/<task>.csv` and the
    `splits/` prefix relative to it, and neither is overridable by any flag.
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / CLAM_ARGV_NAME
    payload = {
        "executable": executable or sys.executable,
        "script": "main.py",
        "argv": list(argv),
        "cwd": str(cwd),
        "command": " ".join([executable or sys.executable, "main.py", *argv]),
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def write_metrics(run_dir: str | os.PathLike[str]) -> Path | None:
    """Convert CLAM's `summary.csv` into `metrics.json`.

    Falls back to `summary_partial_<start>_<end>.csv`, which is what CLAM writes
    when `--k_start`/`--k_end` restrict the fold range. Returns `None` when there
    is no summary at all (a crashed or not-yet-finished run), so a caller can
    still finalise its metadata.

    `std` is the POPULATION standard deviation, matching `numpy.std`'s default,
    which is what CLAM itself reports to W&B.
    """
    out = Path(run_dir)
    source = out / "summary.csv"
    if not source.exists():
        partials = sorted(out.glob("summary_partial_*.csv"))
        if not partials:
            return None
        source = partials[0]

    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    metric_names = [
        name
        for name in rows[0]
        if name not in ("", "folds") and name is not None
    ]

    folds: list[dict[str, Any]] = []
    for row in rows:
        entry: dict[str, Any] = {"fold": _to_number(row.get("folds"))}
        for name in metric_names:
            entry[name] = _to_number(row.get(name))
        folds.append(entry)

    summary_stats: dict[str, dict[str, float | None]] = {}
    for name in metric_names:
        values = [f[name] for f in folds if isinstance(f[name], (int, float))]
        summary_stats[name] = {
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None),
            "n": len(values),
        }

    payload = {
        "source": source.name,
        "n_folds": len(folds),
        "metrics": metric_names,
        "folds": folds,
        "summary": summary_stats,
        "std_convention": "population (ddof=0), matching numpy.std",
    }
    target = out / METRICS_NAME
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def collect_git_info(repo: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """git SHA, branch, dirty flag, dirty paths and remote — or why they are absent."""
    from .paths import repo_root

    root = Path(repo) if repo is not None else repo_root()
    if shutil.which("git") is None:
        return {"available": False, "reason": "git executable not found"}

    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    sha = run("rev-parse", "HEAD")
    if sha is None:
        return {"available": False, "reason": f"{root} is not a git work tree"}

    porcelain = run("status", "--porcelain") or ""
    dirty_paths = [line[3:] for line in porcelain.splitlines() if line]
    return {
        "available": True,
        "root": str(root),
        "sha": sha,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": run("describe", "--always", "--dirty"),
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
        "remote": run("config", "--get", "remote.origin.url"),
    }


def collect_environment() -> dict[str, Any]:
    """Interpreter, platform, host and accelerator facts.

    torch is imported only if it is ALREADY loaded. Training entry points dispatch
    CLAM as a subprocess and never import torch themselves, and paying five
    seconds to import it just to read a version string would tax every analysis
    run. The packaged version is read from distribution metadata instead, and GPU
    facts come from `nvidia-smi`.
    """
    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "user": _current_user(),
        "cwd": os.getcwd(),
    }
    env["torch"] = _torch_info()
    env["gpus"] = _nvidia_smi_gpus()
    return env


def dependency_versions() -> dict[str, str | None]:
    """Installed version of every distribution `dp-code` declares as a dependency.

    Reads this distribution's own requirements so the pinned list lives only in
    `pyproject.toml`. Falls back to nothing (an empty dict with a `_note`) when
    `dp-code` is not installed — an uninstalled checkout has no pin list to read.
    """
    import importlib.metadata as md

    try:
        requires = md.requires(DIST_NAME) or []
    except md.PackageNotFoundError:
        return {"_note": f"{DIST_NAME} is not installed; run `pip install -e .`"}

    versions: dict[str, str | None] = {}
    for requirement in requires:
        name = _requirement_name(requirement)
        if not name:
            continue
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return dict(sorted(versions.items()))


class RunMetadata:
    """Write `run_metadata.json` at the start of a run and finalise it at the end.

    Two writes on purpose: the first one exists even if the process is killed, so
    a half-finished directory still says what it was trying to do and on which
    commit. The second adds wall-clock end, duration and exit status.
    """

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        *,
        run_seed: int | None = None,
        clam_seed: int | None = None,
        command: Sequence[str] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / METADATA_NAME
        self._run_seed = run_seed
        self._clam_seed = clam_seed
        self._command = list(command) if command is not None else list(sys.argv)
        self._extra = dict(extra or {})
        self._started_at: datetime | None = None
        self._monotonic: float | None = None
        self._payload: dict[str, Any] = {}

    def start(self) -> Path:
        self._started_at = datetime.now(timezone.utc)
        self._monotonic = time.monotonic()
        self._payload = {
            "status": "running",
            "started_at": self._started_at.isoformat(),
            "ended_at": None,
            "duration_seconds": None,
            "exit_status": None,
            "run_dir": str(self.run_dir),
            "command": self._command,
            "command_line": " ".join(self._command),
            "git": collect_git_info(),
            "environment": collect_environment(),
            "dependencies": dependency_versions(),
            "determinism": describe_determinism(self._run_seed, self._clam_seed),
            "frozen_internals": FROZEN_INTERNALS,
        }
        self._payload.update(self._extra)
        return self._write()

    def finish(self, exit_status: int) -> Path:
        if not self._payload:
            self.start()
        ended = datetime.now(timezone.utc)
        self._payload["ended_at"] = ended.isoformat()
        self._payload["exit_status"] = exit_status
        self._payload["status"] = "completed" if exit_status == 0 else "failed"
        if self._monotonic is not None:
            self._payload["duration_seconds"] = round(time.monotonic() - self._monotonic, 3)
        return self._write()

    def update(self, **fields: Any) -> None:
        """Record extra facts (fold count, dispatched pid, …) before `finish`."""
        self._payload.update(fields)

    def _write(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._payload, indent=2) + "\n", encoding="utf-8")
        return self.path


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _to_number(value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() and "." not in str(value) else number


def _current_user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - no passwd entry in some containers
        return os.environ.get("USER")


def _torch_info() -> dict[str, Any]:
    import importlib.metadata as md

    info: dict[str, Any] = {"imported_in_this_process": "torch" in sys.modules}
    try:
        info["packaged_version"] = md.version("torch")
    except md.PackageNotFoundError:
        info["packaged_version"] = None

    torch = sys.modules.get("torch")
    if torch is not None:
        info["version"] = getattr(torch, "__version__", None)
        version_mod = getattr(torch, "version", None)
        info["cuda"] = getattr(version_mod, "cuda", None)
        backends = getattr(torch, "backends", None)
        cudnn = getattr(backends, "cudnn", None) if backends is not None else None
        info["cudnn"] = cudnn.version() if cudnn is not None else None
    return info


def _nvidia_smi_gpus() -> list[dict[str, str]] | dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "reason": str(exc)}

    gpus = []
    for line in completed.stdout.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 4:
            continue
        gpus.append(
            {
                "index": fields[0],
                "name": fields[1],
                "driver_version": fields[2],
                "memory_total": fields[3],
            }
        )
    return gpus


def _requirement_name(requirement: str) -> str | None:
    """Extract the distribution name from a PEP 508 requirement string."""
    text = requirement.split(";", 1)[0].strip()
    for separator in (" @ ", "[", "==", ">=", "<=", "~=", "!=", ">", "<", "("):
        index = text.find(separator)
        if index != -1:
            text = text[:index]
    name = text.strip()
    return name or None
