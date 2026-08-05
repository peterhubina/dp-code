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

Four files, written next to whatever CLAM writes, plus a copy of Hydra's own:

    config.resolved.yaml   the resolved config, redacted (see below)
    run_metadata.json      git, environment, seeds, command, timing, exit status,
                           and the `frozen_internals` block below
    clam_argv.json         the exact argv handed to main.py, plus its cwd
    metrics.json           summary.csv, machine-readable
    .hydra/                copied in from Hydra's own scratch output directory

The order a training entry point must use, and why (DESIGN-ADDENDUM A1):

    1. reject `+`/`~` overrides
    2. run_dir = clam_run_dir(cfg.clam.results_dir, exp_code, seed)
    3. assert_run_dir_writable(run_dir, cfg.run.overwrite)   <- nothing written yet
    4. write_config_snapshot / write_clam_argv / RunMetadata.start
    5. dispatch CLAM as a subprocess
    6. write_metrics / RunMetadata.finish / copy_hydra_outputs

Hydra's own `hydra.run.dir` must NOT be the run directory: Hydra creates it and
writes `.hydra/` and the job log before the task function is ever called, so a
guard inside the task function fires after the damage. Hence step 6 copies
`.hydra/` in rather than Hydra writing it there.

Two properties of `config.resolved.yaml` are non-negotiable:

  * it is **redacted**. `resolve=True` expands every `${oc.env:...}`, and this
    file is meant to be publishable alongside the thesis. `paths.nou_root` names
    a private institutional cohort and is replaced by a placeholder; so is any
    credential-shaped key, which is a backstop — credentials are read from the
    environment at the call site and must never enter the config tree.
  * it **cannot abort a run**. An `oc.env` interpolation with no default raises
    `InterpolationResolutionError` when its variable is unset; a snapshot written
    after a 2h38m ladder arm must not turn that into a failed job. Resolution is
    therefore per-key, and unresolvable keys are recorded in the file instead of
    raised.

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
    "HYDRA_SUBDIR",
    "FROZEN_INTERNALS",
    "REDACTED_KEYS",
    "REDACTED_PLACEHOLDER",
    "assert_run_dir_writable",
    "clam_run_dir",
    "hydra_output_dir",
    "copy_hydra_outputs",
    "resolved_container",
    "redact",
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
HYDRA_SUBDIR = ".hydra"

#: Dotted config keys whose VALUE never appears in a published artifact.
#: `paths.nou_root` points at the private institutional cohort; naming its
#: location in a snapshot meant for publication describes that cohort's
#: internals, which the project's standing rules forbid.
REDACTED_KEYS = ("paths.nou_root",)

#: Backstop only. No credential is a config key today and none may become one —
#: `HF_TOKEN` and the W&B API key are read from the environment and `~/.netrc` at
#: the call site. If one ever leaks into the tree, the snapshot redacts it rather
#: than publishing it, and says so in its header.
CREDENTIAL_KEY_SUBSTRINGS = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "credential",
)

REDACTED_PLACEHOLDER = "<redacted>"

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
            "project/CLAM/utils/core_utils.py:712",
            "project/CLAM/utils/core_utils.py:800",
        ],
        "note": (
            "The two early_stopping(...) calls are at :712 (unimodal) and :800 "
            "(multimodal); each is preceded by "
            "`monitor_value = -auc if np.isfinite(auc) else val_loss` on the line "
            "above (:711, :799). The saved checkpoint therefore maximises AUC. "
            "Upstream CLAM monitors val_loss."
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

    It also refuses to proceed when Hydra's own output directory is inside
    `run_dir`. `dpcode/conf/config.yaml` keeps it in scratch, but a command-line
    `hydra.run.dir=...` can still aim it here, and Hydra writes `.hydra/` and the
    job log before any task code runs — so by the time this fires the provenance
    of a completed run may already have been overwritten. Detecting it late is
    still better than a silent retrain on top of it.
    """
    path = Path(run_dir)

    hydra_dir = hydra_output_dir()
    if hydra_dir is not None and _is_within(hydra_dir, path):
        raise FileExistsError(
            f"Hydra's output directory ({hydra_dir}) is inside the run directory "
            f"({path}). Hydra creates and writes into it BEFORE this check runs, so "
            "it must stay in scratch — see hydra.run.dir in dpcode/conf/config.yaml. "
            "Remove the hydra.run.dir override."
        )

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


def clam_run_dir(
    results_dir: str | os.PathLike[str], exp_code: Any, seed: Any
) -> Path:
    """The directory CLAM will write into, derived exactly as CLAM derives it.

    `main.py:407` is `os.path.join(args.results_dir, str(args.exp_code) + '_s{}'
    .format(args.seed))`. This function is that line and nothing else, so the
    entry point's overwrite guard and metadata land in the directory CLAM will
    actually use rather than one that merely looks like it.
    """
    return Path(results_dir) / f"{exp_code}_s{seed}"


def hydra_output_dir() -> Path | None:
    """Hydra's own output directory for the running job, or `None` outside Hydra.

    `HydraConfig.get().runtime.output_dir` is the documented accessor and is
    correct under `--multirun` too, where `hydra.run.dir` is ignored and the
    per-job directory comes from `hydra.sweep.dir`/`subdir`.
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(str(HydraConfig.get().runtime.output_dir))
    except Exception:
        return None


def copy_hydra_outputs(
    source: str | os.PathLike[str] | None,
    run_dir: str | os.PathLike[str],
) -> Path | None:
    """Copy Hydra's `.hydra/` from its scratch output directory into `run_dir`.

    Step 6 of the ordering in the module docstring. `source` is usually
    :func:`hydra_output_dir`; pass `None` and this is a no-op.

    Returns the destination, or `None` if there was nothing to copy. It does not
    raise: this runs after the training subprocess has exited, and a bookkeeping
    failure must not fail a completed run. The caller records the returned value
    (or its absence) in `run_metadata.json`.
    """
    if source is None:
        return None
    origin = Path(source) / HYDRA_SUBDIR
    if not origin.is_dir():
        return None
    destination = Path(run_dir) / HYDRA_SUBDIR
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(origin, destination, dirs_exist_ok=True)
    except OSError:
        return None
    return destination


def resolved_container(cfg: Any) -> tuple[Any, list[str]]:
    """Resolve `cfg` into plain containers without ever raising.

    Returns `(container, problems)`. Each key that cannot be resolved — an
    `oc.env` with no default and an unset variable, a `MISSING` value, a custom
    resolver that fails — becomes a `<unresolved: ...>` marker in the container
    and one line in `problems`.

    The bulk path (`OmegaConf.to_container(resolve=True)`) is tried first and is
    what runs in practice; the per-key walk is the fallback that keeps a snapshot
    from aborting a finished run.
    """
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException

    try:
        return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False), []
    except OmegaConfBaseException:
        problems: list[str] = []
        return _resolve_leniently(cfg, problems, ""), problems


def redact(container: Any) -> tuple[Any, list[str]]:
    """Blank out private and credential-shaped values. Returns `(container, notes)`.

    Two rules: the exact dotted keys in :data:`REDACTED_KEYS`, and any key whose
    name contains one of :data:`CREDENTIAL_KEY_SUBSTRINGS` at any depth. A value
    that is already `None` is left alone — an unset optional key reveals nothing
    and blanking it would imply something is there.
    """
    notes: list[str] = []

    def walk(node: Any, prefix: str) -> Any:
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                dotted = f"{prefix}{key}"
                name = str(key).lower()
                sensitive = dotted in REDACTED_KEYS or any(
                    token in name for token in CREDENTIAL_KEY_SUBSTRINGS
                )
                if sensitive and value is not None:
                    notes.append(dotted)
                    out[key] = REDACTED_PLACEHOLDER
                else:
                    out[key] = walk(value, f"{dotted}.")
            return out
        if isinstance(node, list):
            return [walk(item, prefix) for item in node]
        return node

    return walk(container, ""), notes


def write_config_snapshot(cfg: Any, run_dir: str | os.PathLike[str]) -> Path:
    """Write `config.resolved.yaml` — the resolved, redacted config. Mandatory.

    Mandatory because Hydra's `.hydra/config.yaml` is stored UNRESOLVED: replaying
    it on another machine, where `DP_REPO_ROOT` differs, reconstructs a different
    configuration. This file is the one that says which paths were actually read.

    It never raises for a config-content reason — see :func:`resolved_container`
    and the module docstring — and it is written BEFORE dispatch so that even a
    resolution problem costs zero GPU time.

    Dumped with `yaml.safe_dump` rather than `OmegaConf.to_yaml`, because the
    container may legitimately contain literal `${...}` text for a key that could
    not be resolved, and round-tripping that through `OmegaConf.create` would turn
    it back into an interpolation.
    """
    import yaml

    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / CONFIG_SNAPSHOT_NAME

    container, problems = resolved_container(cfg)
    container, redacted = redact(container)

    header = [
        f"# {CONFIG_SNAPSHOT_NAME} — the composed config, resolved on this machine.",
        "# Hydra's own .hydra/config.yaml is stored UNRESOLVED; this file is what",
        "# records the values a run actually used.",
    ]
    if redacted:
        header.append(
            "# Redacted (private cohort / credential-shaped keys): "
            + ", ".join(sorted(set(redacted)))
        )
    for problem in problems:
        header.append(f"# UNRESOLVED {problem}")

    body = yaml.safe_dump(
        container, sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    target.write_text("\n".join(header) + "\n" + body, encoding="utf-8")
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


def _is_within(candidate: Path, parent: Path) -> bool:
    """True if `candidate` is `parent` or lives under it, symlinks resolved.

    `is_relative_to` rather than a string prefix, so `/a/bc` is not "inside"
    `/a/b`.
    """
    try:
        return Path(os.path.realpath(candidate)).is_relative_to(
            Path(os.path.realpath(parent))
        )
    except (OSError, ValueError):  # pragma: no cover - unreadable path components
        return False


def _resolve_leniently(node: Any, problems: list[str], prefix: str) -> Any:
    """Per-key resolution: one bad interpolation costs one key, not the file."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(node, DictConfig):
        out: dict[Any, Any] = {}
        for key in list(node.keys()):
            dotted = f"{prefix}{key}"
            try:
                value = node[key]
            except Exception as exc:
                problems.append(f"{dotted}: {type(exc).__name__}: {_one_line(exc)}")
                out[key] = f"<unresolved: {type(exc).__name__}>"
                continue
            out[key] = _resolve_leniently(value, problems, f"{dotted}.")
        return out
    if isinstance(node, ListConfig):
        items = []
        for index in range(len(node)):
            dotted = f"{prefix.rstrip('.')}[{index}]"
            try:
                value = node[index]
            except Exception as exc:
                problems.append(f"{dotted}: {type(exc).__name__}: {_one_line(exc)}")
                items.append(f"<unresolved: {type(exc).__name__}>")
                continue
            items.append(_resolve_leniently(value, problems, f"{dotted}."))
        return items
    return node


def _one_line(exc: BaseException) -> str:
    """OmegaConf errors are multi-line; a YAML comment is not."""
    return " ".join(str(exc).split())


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
