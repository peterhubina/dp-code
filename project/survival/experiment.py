"""
Survival experiment manager.

Orchestrates model creation, data loading, training, and logging for a single
fold of cross-validation. Integrates with W&B for experiment tracking.
"""

import logging
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

from project.base.logging import CSVLog, ReportCompiler, LogCompose
from project.survival.dataset import SurvivalDataset, collate_survival
from project.survival.model import AMIL_Surv
from project.survival.splits import load_splits
from project.survival.trainer import SurvivalTrainer


class SurvivalExperiment:
    """Manages a single fold of survival model training.

    Args:
        cfg: Hydra DictConfig with full experiment configuration.
        labels_df: Prepared DataFrame from SurvivalDataset.prepare_labels().
        fold: Fold index (0-based).
        wandb_run: Optional wandb run for logging.
    """

    def __init__(self, cfg, labels_df, fold, wandb_run=None):
        self.cfg = cfg
        self.labels_df = labels_df
        self.fold = fold
        self.wandb_run = wandb_run

        # Experiment path
        self.experiment_path = Path(cfg.exp.base_dir) / cfg.exp.name / cfg.exp.ver / f"fold_{fold}"
        self.experiment_path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = self.experiment_path / "config.yaml"
        if not config_path.exists():
            config_path.write_text(OmegaConf.to_yaml(cfg))

        # Create model
        self.model = AMIL_Surv(
            input_dim=cfg.model.input_dim,
            n_classes=cfg.model.n_classes,
            dropout=cfg.model.dropout,
            size_arg=cfg.model.size_arg,
        )
        log.info(f"Created AMIL_Surv: input_dim={cfg.model.input_dim}, "
                 f"n_classes={cfg.model.n_classes}, size={cfg.model.size_arg}")

    def train(self):
        """Run training for this fold. Returns best validation c-index."""
        cfg = self.cfg

        # Load split
        train_ids, val_ids = load_splits(cfg.dataset.splits_dir, self.fold)
        log.info(f"Fold {self.fold}: {len(train_ids)} train, {len(val_ids)} val patients")

        # Create datasets
        train_dataset = SurvivalDataset(
            data_dir=cfg.dataset.embeddings_dir,
            labels_df=self.labels_df,
            patient_ids=train_ids,
        )
        val_dataset = SurvivalDataset(
            data_dir=cfg.dataset.embeddings_dir,
            labels_df=self.labels_df,
            patient_ids=val_ids,
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_survival,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_survival,
            pin_memory=True,
        )

        # Create trainer
        trainer = SurvivalTrainer(cfg, self.model, wandb_run=self.wandb_run)
        self.trainer = trainer

        # Setup logging
        csv_path = self.experiment_path / "training.csv"
        logs = [
            CSVLog(csv_path),
            ReportCompiler(
                filepath=self.experiment_path / "report.pdf",
                source_filepath=csv_path,
            ),
        ]
        trainer.setup(train_loader, val_loader, logs=logs)

        # Train
        best_cindex = trainer.fit()

        # Save final checkpoint
        self.save_checkpoint("last.pt")

        log.info(f"Fold {self.fold} complete. Best val c-index: {best_cindex:.4f}")
        return best_cindex

    def save_checkpoint(self, filename):
        checkpoint = {
            "model": self.model.state_dict(),
            "opt": self.trainer.opt.state_dict(),
            "fold": self.fold,
        }
        file_path = self.experiment_path / "checkpoints" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, file_path)
        log.info(f"Saved checkpoint: {file_path}")
