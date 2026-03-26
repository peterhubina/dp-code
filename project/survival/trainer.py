"""
Survival-specific training loop with gradient accumulation and c-index evaluation.

Extends the pattern from project/base/trainer.py with:
- NLL survival loss instead of cross-entropy
- Gradient accumulation (batch_size=1, effective batch = gc)
- Per-epoch concordance index computation
- Early stopping on validation c-index
- W&B logging for losses and metrics
"""

import logging
import numpy as np
import torch
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored

from project.base.logging import LogCompose
from project.base.utils import Statistics
from project.survival.losses import NLLSurvLoss, CrossEntropySurvLoss

log = logging.getLogger(__name__)


def decide_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class EarlyStopping:
    """Early stopping based on validation c-index (higher is better)."""

    def __init__(self, patience=10, warmup=5, verbose=True):
        self.patience = patience
        self.warmup = warmup
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, epoch, score):
        if epoch < self.warmup:
            return
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                log.info(f"EarlyStopping: {self.counter}/{self.patience} "
                         f"(best={self.best_score:.4f}, current={score:.4f})")
            if self.counter >= self.patience:
                self.should_stop = True


class SurvivalTrainer:
    """Training loop for discrete-time survival models.

    Args:
        cfg: Configuration object with training hyperparameters.
        model: Survival model (e.g. AMIL_Surv).
        wandb_run: Optional wandb run object for logging.
    """

    def __init__(self, cfg, model, wandb_run=None):
        self.device = torch.device(decide_device())
        self.cfg = cfg
        self.model = model.to(self.device)
        self.wandb_run = wandb_run

        # Optimizer
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        # Loss function
        loss_type = cfg.loss.type
        alpha = cfg.loss.alpha
        if loss_type == "nll_surv":
            self.loss_fn = NLLSurvLoss(alpha=alpha)
        elif loss_type == "ce_surv":
            self.loss_fn = CrossEntropySurvLoss(alpha=alpha)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Gradient accumulation
        self.gc = cfg.training.gc

        # Early stopping
        if cfg.early_stopping.enabled:
            self.early_stopping = EarlyStopping(
                patience=cfg.early_stopping.patience,
                warmup=cfg.early_stopping.warmup,
            )
        else:
            self.early_stopping = None

    def setup(self, dataloader_train, dataloader_val, logs=None):
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.log = LogCompose(logs or [])

        # Define W&B metrics so charts use epoch as x-axis
        if self.wandb_run is not None:
            import wandb
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")

    def fit(self):
        self.log.on_training_start()
        best_cindex = 0.0

        for epoch in range(self.cfg.training.max_epochs):
            stats_train = self.train_epoch(epoch)
            stats_val = self.validate_epoch(epoch)

            # Merge stats
            stats = Statistics.merge(stats_train, stats_val)

            self.log.on_epoch_complete(epoch, stats)

            # W&B logging with structured keys for proper charts
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "epoch": epoch,
                    "train/loss": stats["loss_train"],
                    "train/cindex": stats["cindex_train"],
                    "val/loss": stats["loss_val"],
                    "val/cindex": stats["cindex_val"],
                })

            val_cindex = stats.get("cindex_val", 0.5)
            log.info(
                f"Epoch {epoch:03d} | "
                f"loss_train={stats.get('loss_train', float('nan')):.4f} "
                f"cindex_train={stats.get('cindex_train', float('nan')):.4f} | "
                f"loss_val={stats.get('loss_val', float('nan')):.4f} "
                f"cindex_val={val_cindex:.4f}"
            )
            if val_cindex > best_cindex:
                best_cindex = val_cindex

            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping(epoch, val_cindex)
                if self.early_stopping.should_stop:
                    log.info(f"Early stopping at epoch {epoch}")
                    break

        self.log.on_training_stop()
        return best_cindex

    def train_epoch(self, epoch):
        self.model.train()
        stats = Statistics()
        all_risk_scores = []
        all_censorships = []
        all_event_times = []

        self.opt.zero_grad()
        with tqdm(self.dataloader_train, desc=f"Train {epoch}") as progress:
            for batch_idx, (features, label, event_time, censorship) in enumerate(progress):
                # Move to device
                features = features.to(self.device)
                label = label.to(self.device)
                censorship = censorship.to(self.device)

                # Forward
                hazards, S, Y_hat, _ = self.model(features)

                # Loss
                loss = self.loss_fn(hazards=hazards, S=S, Y=label, c=censorship)
                loss_value = loss.item()

                # Risk score: negative sum of survival probabilities
                risk = -torch.sum(S, dim=1).detach().cpu().numpy()
                all_risk_scores.append(risk)
                all_censorships.append(censorship.cpu().numpy())
                all_event_times.append(event_time.numpy())

                # Backward with gradient accumulation
                loss = loss / self.gc
                loss.backward()

                if (batch_idx + 1) % self.gc == 0 or (batch_idx + 1) == len(self.dataloader_train):
                    self.opt.step()
                    self.opt.zero_grad()

                stats.step("loss_train", loss_value)
                progress.set_postfix(stats.get())

        # Compute epoch c-index
        all_risk_scores = np.concatenate(all_risk_scores)
        all_censorships = np.concatenate(all_censorships)
        all_event_times = np.concatenate(all_event_times)

        cindex = _safe_cindex(all_censorships, all_event_times, all_risk_scores)
        stats.step("cindex_train", cindex)

        return stats

    def validate_epoch(self, epoch):
        self.model.eval()
        stats = Statistics()
        all_risk_scores = []
        all_censorships = []
        all_event_times = []

        with torch.no_grad():
            with tqdm(self.dataloader_val, desc=f"Val   {epoch}") as progress:
                for features, label, event_time, censorship in progress:
                    features = features.to(self.device)
                    label = label.to(self.device)
                    censorship = censorship.to(self.device)

                    hazards, S, Y_hat, _ = self.model(features)
                    # Use alpha=0 for validation (MCAT convention)
                    loss = self.loss_fn(hazards=hazards, S=S, Y=label, c=censorship, alpha=0)

                    risk = -torch.sum(S, dim=1).cpu().numpy()
                    all_risk_scores.append(risk)
                    all_censorships.append(censorship.cpu().numpy())
                    all_event_times.append(event_time.numpy())

                    stats.step("loss_val", loss.item())
                    progress.set_postfix(stats.get())

        all_risk_scores = np.concatenate(all_risk_scores)
        all_censorships = np.concatenate(all_censorships)
        all_event_times = np.concatenate(all_event_times)

        cindex = _safe_cindex(all_censorships, all_event_times, all_risk_scores)
        stats.step("cindex_val", cindex)

        return stats


def _safe_cindex(censorships, event_times, risk_scores):
    """Compute concordance index, handling edge cases."""
    try:
        # sksurv expects event indicator (True=event), not censorship
        event_indicator = (1 - censorships).astype(bool)
        if event_indicator.sum() == 0:
            log.warning("No events in this set, c-index undefined. Returning 0.5.")
            return 0.5
        ci = concordance_index_censored(event_indicator, event_times, risk_scores)[0]
        return ci
    except Exception as e:
        log.warning(f"c-index computation failed ({e}). Returning 0.5.")
        return 0.5
