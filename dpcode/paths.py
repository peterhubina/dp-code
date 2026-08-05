"""Repository-root resolution and path validation.

This module is the single place that answers "where is the repository?". It is
deliberately importable without Hydra, without OmegaConf and without torch, so
plain library code (``tools/pam50_arms.py`` and friends) can call
:func:`repo_root` / :func:`resolve_paths` and land on exactly the same locations
a Hydra-composed config would.

Nothing here caches. ``DP_REPO_ROOT`` is read on every call, because tests and
the parity harness point it at a fixture tree mid-process.

Three entry points matter:

``repo_root()``
    The absolute repository root. ``DP_REPO_ROOT`` wins; otherwise the root is
    inferred from where this package is installed.

``resolve_paths()``
    The fully resolved contents of ``dpcode/conf/paths/default.yaml`` as a plain
    dict, for code that is not running under Hydra.

``register_resolvers()``
    Registers the ``dp.repo_root`` OmegaConf resolver used by that YAML. Safe to
    call any number of times.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "REPO_ROOT_ENV",
    "REPO_MARKERS",
    "repo_root",
    "conf_dir",
    "register_resolvers",
    "resolve_paths",
    "assert_paths_absolute",
    "assert_paths_exist",
]

#: Environment variable that overrides root inference. Set it when the package is
#: installed non-editably, or when pointing the test harness at a fixture tree.
REPO_ROOT_ENV = "DP_REPO_ROOT"

#: Files that must exist for an *inferred* root to be believed. ``main.py`` is the
#: load-bearing one: every training entry point dispatches to it.
REPO_MARKERS = ("pyproject.toml", "project/CLAM/main.py")

#: Name of the custom OmegaConf resolver, used as ``${dp.repo_root:}`` in YAML.
RESOLVER_NAME = "dp.repo_root"

_PATHS_YAML = Path(__file__).resolve().parent / "conf" / "paths" / "default.yaml"


def repo_root() -> Path:
    """Return the absolute repository root.

    ``$DP_REPO_ROOT`` takes precedence and is trusted as given (only "is a
    directory" is checked) — the parity harness points it at a fixture tree that
    deliberately does not carry the markers below.

    Without it, the root is the directory that contains this package, which is
    correct for ``pip install -e .`` and wrong for a non-editable install; the
    markers catch the latter and the error says what to do about it.
    """
    override = os.environ.get(REPO_ROOT_ENV)
    if override:
        root = Path(override).expanduser()
        if not root.is_dir():
            raise RuntimeError(
                f"{REPO_ROOT_ENV}={override!r} does not point at a directory."
            )
        return root.resolve()

    candidate = Path(__file__).resolve().parent.parent
    missing = [m for m in REPO_MARKERS if not (candidate / m).exists()]
    if missing:
        raise RuntimeError(
            f"Cannot locate the dp-code repository root. Inferred {candidate} from the "
            f"installed package, but it is missing {missing}. Either reinstall with "
            f"`pip install -e .` from a clone, or set {REPO_ROOT_ENV} to the clone."
        )
    return candidate


def conf_dir() -> Path:
    """Absolute path of the packaged Hydra config root (``dpcode/conf``)."""
    return Path(__file__).resolve().parent / "conf"


def register_resolvers() -> None:
    """Register the ``dp.repo_root`` OmegaConf resolver.

    Idempotent by ``replace=True``: pytest imports modules repeatedly and a
    duplicate registration would otherwise raise ``ValueError``.

    ``use_cache`` is left off on purpose — the resolver reads the environment and
    a cached value would survive a test that changes ``DP_REPO_ROOT``.
    """
    from omegaconf import OmegaConf  # local: keeps module import stdlib-only

    # ``*_`` because ``${dp.repo_root:}`` passes one empty-string argument.
    OmegaConf.register_new_resolver(
        RESOLVER_NAME, lambda *_: str(repo_root()), replace=True
    )


def resolve_paths() -> dict[str, Any]:
    """Return ``dpcode/conf/paths/default.yaml`` fully resolved, as a plain dict.

    For library code that never sees a Hydra config. Values are strings, except
    ``nou_root``, which is ``None`` unless ``DP_NOU_ROOT`` is set.
    """
    from omegaconf import OmegaConf

    register_resolvers()
    cfg = OmegaConf.load(_PATHS_YAML)
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[return-value]


def _paths_node(cfg: Any) -> tuple[str, Mapping[str, Any]]:
    """Accept either a root config (with a ``paths`` node) or the node itself."""
    from omegaconf import OmegaConf

    if OmegaConf.is_config(cfg) and "paths" in cfg:
        return "paths.", cfg.paths
    if isinstance(cfg, Mapping) and "paths" in cfg:
        return "paths.", cfg["paths"]
    return "", cfg


def assert_paths_absolute(cfg: Any) -> None:
    """Fail if any ``paths.*`` value is a relative path.

    A relative path in config is one ``hydra.job.chdir=true`` away from silently
    resolving inside a run's output directory. ``None`` is allowed — it is how an
    unset optional root (``nou_root``) is spelled.

    The message names the *config key*, not just the offending path, because the
    key is what the reader has to go and fix.
    """
    from omegaconf import OmegaConf

    prefix, node = _paths_node(cfg)
    is_config = OmegaConf.is_config(node)
    offenders = []
    for key in list(node.keys()):
        # A MISSING value would raise on access; name it rather than crash.
        if is_config and OmegaConf.is_missing(node, key):
            offenders.append(f"{prefix}{key}=<MISSING>")
            continue
        value = node[key]
        if value is None:
            continue
        if not Path(str(value)).is_absolute():
            offenders.append(f"{prefix}{key}={value!r}")
    if offenders:
        raise ValueError(
            "Non-absolute or unset path(s) in config; they will break under "
            "hydra.job.chdir=true: " + ", ".join(sorted(offenders))
        )


def assert_paths_exist(cfg: Any, keys: Iterable[str]) -> None:
    """Fail if any of the dotted config `keys` names a path that does not exist.

    `keys` are full dotted keys into `cfg` (``"paths.tcga_embeddings"``,
    ``"clam.tabular_csv"``), so the error can name the key an entry point actually
    reads rather than dumping every path in the tree.

    This is the check that turns the survival config's
    ``embeddings_dir: .datasets/embeddings`` from a mid-run ``h5py`` failure into
    a first-second abort naming the key.
    """
    from omegaconf import OmegaConf

    missing = []
    for key in keys:
        if OmegaConf.is_config(cfg):
            value = OmegaConf.select(cfg, key, default=None)
        else:
            value = _select_plain(cfg, key)
        if value is None:
            missing.append(f"{key}=<unset>")
        elif not Path(str(value)).exists():
            missing.append(f"{key}={value}")
    if missing:
        raise FileNotFoundError(
            "Config points at paths that do not exist: " + ", ".join(missing)
        )


def _select_plain(mapping: Any, dotted: str) -> Any:
    node = mapping
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node
