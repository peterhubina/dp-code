"""The one sanctioned way to hand a Hydra config to Weights & Biases.

Verified on the installed wandb 0.15.3:

    run = wandb.init(mode="disabled",
                     config=OmegaConf.create({"seed": 1, "model": {"name": "clam_mb"}}))
    dict(run.config)   # -> {}

Passing a `DictConfig` straight to `wandb.init(config=...)` logs an **empty**
config. No exception, no warning — every hyperparameter is lost, and the run
record looks fine. Every call site must go through :func:`wandb_config`.

`resolve=True` matters as much as the conversion itself: without it W&B stores
the literal string `"${oc.env:DP_REPO_ROOT,...}"`, which records nothing about the
machine that actually ran. `throw_on_missing=True` turns an unfilled `MISSING`
into a loud failure at init instead of a `'???'` string in the run record.

Do not reach for `config_exclude_keys` / `config_include_keys`: they exist on
0.15.3 but were removed in wandb 0.20+, and using them would block any future SDK
upgrade. Trim the config before passing it instead.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

__all__ = ["wandb_config"]


def wandb_config(cfg: DictConfig) -> dict[str, Any]:
    """Convert a Hydra config into the plain dict `wandb.init(config=...)` needs."""
    if not OmegaConf.is_config(cfg):
        raise TypeError(
            "wandb_config expects an OmegaConf config; got "
            f"{type(cfg).__name__}. Passing a plain dict straight to wandb.init is "
            "fine, but a DictConfig is not — it logs an empty config silently."
        )
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[return-value]
