"""WandB sweep wrapper for CTC EP classification (5-fold).

Runs fold 0 only per sweep trial for fast iteration.
Once the best config is found, run full 5-fold CV with main.py.

Usage:
    cd project/CLAM
    wandb sweep sweep_config_ctc_ep.yaml   # creates sweep, prints sweep ID
    wandb agent <sweep_id>                 # launches agent to run trials
"""
import sys
import runpy

import wandb


def run():
    wandb.init(dir="../../.scratch/nou_clam")
    config = wandb.config
    run_id = wandb.run.id

    sys.argv = [
        "main.py",
        "--task", "nou_ctc_ep",
        "--data_root_dir", "../../.scratch/nou_clam/features",
        "--embed_dim", "1536",
        "--exp_code", f"ctc_ep_sweep_{run_id}",
        "--results_dir", "../../.scratch/nou_clam/sweep_results",
        "--max_epochs", "100",
        "--k", "5",
        "--k_start", "0",
        "--k_end", "1",
        "--early_stopping",
        "--weighted_sample",
        "--wandb",
        "--wandb_project", "ctc-classification",
        "--wandb_tags", "sweep", "ep",
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
