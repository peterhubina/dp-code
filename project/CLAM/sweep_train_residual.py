"""W&B sweep wrapper for RNA-preserving residual WSI-RNA fusion.

Each trial runs fold 0 only. After selecting a configuration, run full 10-fold
CV with main.py using the same residual settings.
"""

import runpy
import sys

import wandb


def run():
    wandb.init(dir="../../.scratch")
    config = wandb.config
    run_id = wandb.run.id

    sys.argv = [
        "main.py",
        "--data_root_dir",
        "../../.datasets/tcga-brca/embeddings",
        "--task",
        "tcga_brca_subtyping",
        "--exp_code",
        f"pam50_wsi_rna_residual_sweep_{run_id}",
        "--results_dir",
        "../../.scratch/residual_sweep_results",
        "--split_dir",
        "tcga_brca_subtyping_100",
        "--model_type",
        "clam_mb",
        "--model_size",
        "big",
        "--embed_dim",
        "1536",
        "--bag_loss",
        "ce",
        "--max_epochs",
        str(config.max_epochs),
        "--k",
        "10",
        "--k_start",
        "0",
        "--k_end",
        "1",
        "--B",
        "4",
        "--weighted_sample",
        "--early_stopping",
        "--patience",
        str(config.patience),
        "--subtyping",
        "--fusion_mode",
        "residual",
        "--tabular_csv",
        "../../.scratch/TCGA-BRCA-rna/TCGA_BRCA_RNA_primary_tumor_4class_clam.csv.gz",
        "--tabular_top_n_features",
        "10000",
        "--pretrained_wsi_ckpt",
        "../../.scratch/results/pam50_final_s1/s_{fold}_checkpoint.pt",
        "--freeze_wsi_branch",
        "--pretrained_rna_ckpt",
        "results/tcga_brca_rna_matched_wsi_4class_s1/s_{fold}_checkpoint.pt",
        "--freeze_rna_branch",
        "--rna_hidden_dims",
        "1024,512",
        "--rna_dropout",
        "0.4",
        "--wandb",
        "--wandb_project",
        "clam-brca-subtyping-residual",
        "--lr",
        str(config.lr),
        "--reg",
        str(config.reg),
        "--drop_out",
        str(config.drop_out),
        "--fusion_hidden_dim",
        str(config.fusion_hidden_dim),
        "--residual_scale",
        str(config.residual_scale),
    ]

    runpy.run_path("main.py", run_name="__main__")


if __name__ == "__main__":
    run()
