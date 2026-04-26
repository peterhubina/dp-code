"""W&B Bayesian sweep wrapper for Hydra-configured RNA training.

Usage:
    cd /workspace/dp-code/project/CLAM
    wandb sweep sweep_config_rna.yaml
    wandb agent <entity/project/sweep_id>

Each trial runs one fold by default for fast model selection. After selecting
the best hyperparameters, run full CV with train_rna_hydra.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import wandb
from omegaconf import OmegaConf

from train_rna import run_experiment
from train_rna_hydra import args_from_cfg


def _prepare_wandb_env(clam_dir: Path) -> Path:
    workspace_root = clam_dir.parents[1]
    wandb_root = workspace_root / ".scratch" / "rna_wandb"
    for env_name, env_path in {
        "WANDB_DIR": wandb_root,
        "WANDB_CONFIG_DIR": wandb_root / "config",
        "WANDB_CACHE_DIR": wandb_root / "cache",
        "WANDB_DATA_DIR": wandb_root / "data",
    }.items():
        env_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(env_path))
    os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
    return wandb_root


def _apply_sweep_config(cfg, sweep_config) -> None:
    for key, value in dict(sweep_config).items():
        OmegaConf.update(cfg, key, value, merge=True)


def run() -> None:
    clam_dir = Path(__file__).resolve().parent
    wandb_root = _prepare_wandb_env(clam_dir)

    run = wandb.init(dir=str(wandb_root))
    cfg = OmegaConf.load(clam_dir / "configs" / "rna" / "default.yaml")
    _apply_sweep_config(cfg, wandb.config)

    OmegaConf.update(cfg, "splits.k_start", 0, merge=True)
    OmegaConf.update(cfg, "splits.k_end", 1, merge=True)
    OmegaConf.update(cfg, "wandb.enabled", True, merge=True)
    OmegaConf.update(cfg, "wandb.project", run.project, merge=True)
    OmegaConf.update(cfg, "wandb.tags", ["rna", "tcga-brca", "bayes-sweep"], merge=True)
    OmegaConf.update(cfg, "output.exp_code", f"tcga_brca_rna_sweep_{run.id}", merge=True)
    OmegaConf.update(cfg, "output.results_dir", "../../.scratch/rna_sweep_results", merge=True)

    args = args_from_cfg(cfg)
    try:
        run_experiment(args, clam_dir=clam_dir)
    finally:
        wandb.finish()


if __name__ == "__main__":
    run()
