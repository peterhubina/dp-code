"""Train an MLP classifier on cached UNI2-h features."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from project.data.feature_datamodule import FeatureDataModule
from project.models.mlp_classifier import MLPClassifier
from project.base.logging import CSVLog, ReportCompiler, ModelCheckpointer, LogCompose
from project.base.utils import Statistics
from project.UNI.uni.downstream.eval_patch_features.metrics import get_eval_metrics, print_metrics


BASE_PATH = Path(".scratch/experiments")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP on cached features")
    parser.add_argument("--name", type=str, default="insitu_vs_infiltrant")
    parser.add_argument("--ver", type=str, default="v1")
    parser.add_argument("--features_path", type=str, required=True,
                        help="Path to .pt file from extract_features.py")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def decide_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_balanced_accuracy(preds, targets, num_classes=2):
    """Compute balanced accuracy (average per-class recall)."""
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)
    for c in range(num_classes):
        mask = targets == c
        total_per_class[c] = mask.sum().float()
        if total_per_class[c] > 0:
            correct_per_class[c] = (preds[mask] == c).sum().float()
    recalls = correct_per_class / total_per_class.clamp(min=1)
    return recalls.mean().item()


class ClassifierExperiment:
    """Lightweight experiment wrapper for saving checkpoints."""

    def __init__(self, model, optimizer, experiment_path):
        self.model = model
        self.optimizer = optimizer
        self.experiment_path = Path(experiment_path)

    def save_checkpoint(self, filename):
        checkpoint = {
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict(),
        }
        file_path = self.experiment_path / "checkpoints" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, file_path.as_posix())


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    stats = Statistics()
    all_preds, all_targets = [], []

    with tqdm(dataloader, desc="Train") as progress:
        for features, labels in progress:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

            stats.step("loss_train", loss.item())
            progress.set_postfix(stats.get())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    bacc = compute_balanced_accuracy(all_preds, all_targets)
    return stats, bacc


def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    stats = Statistics()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        with tqdm(dataloader, desc="Val") as progress:
            for features, labels in progress:
                features = features.to(device)
                labels = labels.to(device)

                logits = model(features)
                loss = loss_fn(logits, labels)

                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())
                all_probs.append(probs.cpu())

                stats.step("loss_val", loss.item())
                progress.set_postfix(stats.get())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)
    bacc = compute_balanced_accuracy(all_preds, all_targets)
    return stats, bacc, all_preds, all_targets, all_probs


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(decide_device())
    print(f"Using device: {device}")

    # Setup experiment directory
    experiment_path = BASE_PATH / args.name / args.ver
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Save config
    import yaml
    config = vars(args)
    (experiment_path / "config.yaml").write_text(yaml.dump(config))
    print(f" > Experiment: {args.name}/{args.ver}")
    print(f" > Config: {config}")

    # Load data
    datamodule = FeatureDataModule(
        features_path=args.features_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Create model
    model = MLPClassifier(
        num_features=datamodule.num_features,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=2,
    ).to(device)
    print(f" > Model: {model}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=datamodule.class_weights.to(device))

    # Logging
    experiment = ClassifierExperiment(model, optimizer, experiment_path)
    log = LogCompose([
        CSVLog(experiment_path / "training.csv"),
        ReportCompiler(
            filepath=experiment_path / "report.pdf",
            source_filepath=experiment_path / "training.csv",
        ),
        ModelCheckpointer(experiment),
    ])

    # Training loop
    log.on_training_start()
    for epoch in range(args.max_epochs):
        print(f"\n--- Epoch {epoch}/{args.max_epochs - 1} ---")

        stats_train, bacc_train = train_epoch(model, datamodule.dataloader_train, optimizer, loss_fn, device)
        stats_val, bacc_val, val_preds, val_targets, val_probs = validate_epoch(model, datamodule.dataloader_val, loss_fn, device)

        # Merge stats
        merged = Statistics.merge(stats_train, stats_val)
        merged["bacc_train"] = bacc_train
        merged["bacc_val"] = bacc_val

        print(f"  loss_train={merged['loss_train']:.4f}  loss_val={merged['loss_val']:.4f}  "
              f"bacc_train={bacc_train:.4f}  bacc_val={bacc_val:.4f}")

        log.on_epoch_complete(epoch=epoch, stats=merged)

    log.on_training_stop()

    # Final evaluation on validation set
    print("\n" + "=" * 50)
    print("Final Validation Evaluation")
    print("=" * 50)

    # Use probs for class 1 for binary AUROC
    probs_class1 = val_probs[:, 1].numpy()
    eval_metrics = get_eval_metrics(
        targets_all=val_targets.numpy(),
        preds_all=val_preds.numpy(),
        probs_all=probs_class1,
    )
    print_metrics(eval_metrics)

    print(f"\n > Outputs saved to: {experiment_path}")


if __name__ == "__main__":
    main()
