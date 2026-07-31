#!/bin/bash
# Three-way ER-status ablation on TCGA-BRCA: WSI-alone vs WSI+RNA vs WSI+clinicopath.
# Same site-holdout folds, frozen pretrained WSI branch for both fusion arms.
#
# The author runs training; this script is the exact, verified command set.
# Run the WSI-alone arm FIRST -- it writes the per-fold checkpoints that both
# fusion arms load and freeze.
#
# Usage (from repo root):
#   bash tools/train_er_ablation.sh wsi       # 1) WSI-alone baseline (clam_mb)
#   bash tools/train_er_ablation.sh rna       # 2) WSI + RNA   gated fusion (frozen WSI)
#   bash tools/train_er_ablation.sh clinpath  # 3) WSI + clinicopath gated fusion (frozen WSI)
#   bash tools/train_er_ablation.sh all       # all three, in dependency order
#
# W&B project: er-brca-ablation. exp_code (== W&B group) per arm:
#   er_wsi_alone / er_wsi_rna_gated / er_wsi_clinpath_gated.
# Pass --no_wandb-style overrides by editing the WANDB block below if needed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/project/CLAM"

# ---- shared configuration -------------------------------------------------- #
DATA_ROOT="../../.datasets/tcga-brca/embeddings"
RESULTS_DIR="../../.scratch/results/er"
SPLIT_DIR="tcga_brca_er_100"                 # 10 site-holdout folds
# Seed is overridable so the multi-seed repeat can reuse this exact command set:
#   SEED=2 bash tools/train_er_ablation.sh wsi
# CLAM appends _s<seed> to the run directory (main.py:407), so exp codes are unchanged.
SEED="${SEED:-1}"
K="10"
WANDB_PROJECT="er-brca-ablation"
WSI_CKPT="${RESULTS_DIR}/er_wsi_alone_s${SEED}/s_{fold}_checkpoint.pt"

# Set RUNNER=echo to PRINT the resolved commands instead of executing them:
#   RUNNER=echo bash tools/train_er_ablation.sh all
# Without this hook a dry run silently starts real training.
RUNNER="${RUNNER:-python}"
RNA_CSV="../../.scratch/TCGA-BRCA-rna/tcga_brca_er_rna_clam.csv.gz"
CLINPATH_CSV="../../tools/data/tcga_brca_clinicopath_clam.csv"

# Base CLAM args common to every arm.
common_args() {
    printf '%s ' \
        --task tcga_brca_er \
        --data_root_dir "${DATA_ROOT}" \
        --embed_dim 1536 \
        --results_dir "${RESULTS_DIR}" \
        --split_dir "${SPLIT_DIR}" \
        --k "${K}" \
        --seed "${SEED}" \
        --max_epochs 50 \
        --early_stopping --patience 5 \
        --weighted_sample \
        --log_data \
        --model_type clam_mb --model_size big \
        --drop_out 0.5 --opt adam --lr 1e-4 --reg 2.5e-6 \
        --wandb --wandb_project "${WANDB_PROJECT}"
}

train_wsi() {
    echo ">>> Arm 1/3: WSI-alone baseline (clam_mb) -> er_wsi_alone"
    "${RUNNER}" main.py $(common_args) \
        --exp_code er_wsi_alone \
        --B 4 --bag_loss ce --inst_loss svm \
        --wandb_tags er wsi-alone clam_mb
}

train_rna() {
    echo ">>> Arm 2/3: WSI + RNA gated fusion (frozen WSI) -> er_wsi_rna_gated"
    "${RUNNER}" main.py $(common_args) \
        --exp_code er_wsi_rna_gated \
        --B 4 --bag_loss ce --no_inst_cluster \
        --tabular_csv "${RNA_CSV}" \
        --tabular_case_id_col case_id \
        --tabular_hidden_dim 256 --tabular_num_layers 2 \
        --tabular_top_n_features 10000 \
        --fusion_mode gated --fusion_hidden_dim 32 \
        --pretrained_wsi_ckpt "${WSI_CKPT}" --freeze_wsi_branch \
        --wandb_tags er wsi-rna gated frozen
}

train_clinpath() {
    echo ">>> Arm 3/3: WSI + clinicopath gated fusion (frozen WSI) -> er_wsi_clinpath_gated"
    "${RUNNER}" main.py $(common_args) \
        --exp_code er_wsi_clinpath_gated \
        --B 4 --bag_loss ce --no_inst_cluster \
        --tabular_csv "${CLINPATH_CSV}" \
        --tabular_case_id_col case_id \
        --tabular_hidden_dim 256 --tabular_num_layers 2 \
        --tabular_top_n_features 0 \
        --fusion_mode gated --fusion_hidden_dim 32 \
        --pretrained_wsi_ckpt "${WSI_CKPT}" --freeze_wsi_branch \
        --wandb_tags er wsi-clinpath gated frozen
}

case "${1:-}" in
    wsi)      train_wsi ;;
    rna)      train_rna ;;
    clinpath) train_clinpath ;;
    all)      train_wsi; train_rna; train_clinpath ;;
    *)
        echo "usage: bash tools/train_er_ablation.sh {wsi|rna|clinpath|all}" >&2
        echo "  run 'wsi' first: the fusion arms load its per-fold checkpoints." >&2
        exit 2
        ;;
esac
