"""WandB sweep wrapper for CLAM training.

Runs a single fold (fold 0) per sweep trial for fast iteration.
Once the best config is found, run full 10-fold CV with main.py.

Usage:
    cd project/CLAM
    wandb sweep sweep_config.yaml          # creates sweep, prints sweep ID
    wandb agent <sweep_id>                 # launches agent to run trials
"""
import sys
import runpy

import wandb


def run():
    wandb.init(dir="../../.scratch")
    config = wandb.config
    run_id = wandb.run.id

    sys.argv = [
        "main.py",
        "--task", "tcga_brca_subtyping",
        "--data_root_dir", "../../.datasets/embeddings",
        "--embed_dim", "1536",
        "--subtyping",
        "--exp_code", f"pam50_sweep_{run_id}",
        "--results_dir", "../../.scratch/sweep_results",
        "--max_epochs", "50",
        "--k", "10",
        "--k_start", "0",
        "--k_end", "1",
        "--early_stopping",
        "--weighted_sample",
        "--wandb",
        "--wandb_project", "clam-brca-subtyping",
        "--log_data",
        # Sweep hyperparameters
        "--lr", str(config.lr),
        "--reg", str(config.reg),
        "--drop_out", str(config.drop_out),
        "--model_type", config.model_type,
        "--model_size", config.model_size,
        "--bag_weight", str(config.bag_weight),
        "--B", str(config.B),
        "--inst_loss", config.inst_loss,
        "--patience", str(config.patience),
    ]

    runpy.run_path("main.py", run_name="__main__")


if __name__ == "__main__":
    run()
