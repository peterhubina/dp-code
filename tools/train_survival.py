"""
Training script for discrete-time survival prediction with AMIL.

Uses Hydra for configuration management and W&B for experiment tracking.
Runs 5-fold cross-validation and reports mean +/- std c-index.

Usage:
    python tools/train_survival.py
    python tools/train_survival.py exp.name=my_exp exp.ver=v2
    python tools/train_survival.py training.learning_rate=0.001 training.max_epochs=30
    python tools/train_survival.py wandb.enabled=false
"""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from project.survival.dataset import SurvivalDataset
from project.survival.experiment import SurvivalExperiment
from project.survival.splits import generate_stratified_splits, save_splits

log = logging.getLogger(__name__)


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="config", config_name="survival")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))

    # Seed
    seed_everything(cfg.dataset.seed)

    # Prepare labels + bins
    labels_df, bin_edges = SurvivalDataset.prepare_labels(
        labels_csv=cfg.dataset.labels_csv,
        embeddings_dir=cfg.dataset.embeddings_dir,
        n_bins=cfg.dataset.n_bins,
        time_col=cfg.dataset.time_col,
        status_col=cfg.dataset.status_col,
    )

    # Generate or load splits
    splits_dir = Path(cfg.dataset.splits_dir)
    if not splits_dir.exists():
        splits = generate_stratified_splits(labels_df, cfg.dataset.n_folds, cfg.dataset.seed)
        save_splits(splits, splits_dir)

    # Run k-fold CV
    fold_results = []
    for fold in range(cfg.dataset.n_folds):
        log.info(f"{'='*60}")
        log.info(f"FOLD {fold}/{cfg.dataset.n_folds - 1}")
        log.info(f"{'='*60}")

        # Init W&B run per fold
        wandb_run = None
        if cfg.wandb.enabled:
            run_name = f"{cfg.exp.name}_{cfg.exp.ver}_fold{fold}"
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                group=cfg.wandb.group or f"{cfg.exp.name}_{cfg.exp.ver}",
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True,
            )
            wandb_run.config.update({"fold": fold, "bin_edges": bin_edges.tolist()})

        experiment = SurvivalExperiment(cfg, labels_df, fold, wandb_run=wandb_run)
        val_cindex = experiment.train()
        fold_results.append(val_cindex)

        # Log fold summary to W&B
        if wandb_run is not None:
            wandb_run.summary["best_val_cindex"] = val_cindex
            wandb_run.finish()

    # Summary
    log.info(f"{'='*60}")
    log.info("CROSS-VALIDATION RESULTS")
    log.info(f"{'='*60}")
    for i, ci in enumerate(fold_results):
        log.info(f"  Fold {i}: c-index = {ci:.4f}")
    mean_ci = np.mean(fold_results)
    std_ci = np.std(fold_results)
    log.info(f"  Mean C-Index: {mean_ci:.4f} +/- {std_ci:.4f}")

    # Save summary
    summary_path = Path(cfg.exp.base_dir) / cfg.exp.name / cfg.exp.ver / "summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(f"Experiment: {cfg.exp.name}/{cfg.exp.ver}\n")
        f.write(f"Model: AMIL_Surv (input_dim={cfg.model.input_dim})\n")
        f.write(f"Loss: {cfg.loss.type} (alpha={cfg.loss.alpha})\n")
        f.write(f"LR: {cfg.training.learning_rate}, WD: {cfg.training.weight_decay}\n")
        f.write(f"Epochs: {cfg.training.max_epochs}, GC: {cfg.training.gc}\n\n")
        for i, ci in enumerate(fold_results):
            f.write(f"Fold {i}: {ci:.4f}\n")
        f.write(f"\nMean: {mean_ci:.4f} +/- {std_ci:.4f}\n")

    log.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
