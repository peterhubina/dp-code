#!/bin/bash
# Multimodal PAM50 run: WSI CLAM-MB + RNA-seq MLP late fusion.
#
# Set PRETRAINED_WSI_CKPT to warm-start from a WSI-only checkpoint.

set -euo pipefail

TABULAR_CSV="${TABULAR_CSV:-../../.scratch/TCGA-BRCA-rna/TCGA_BRCA_RNA_primary_tumor_4class_clam.csv.gz}"
: "${PRETRAINED_WSI_CKPT:?PRETRAINED_WSI_CKPT must point to a trained CLAM checkpoint}"

cd project/CLAM

python main.py \
    --task                  tcga_brca_subtyping \
    --data_root_dir         ../../.datasets/tcga-brca/embeddings \
    --embed_dim             1536 \
    --subtyping \
    --exp_code              pam50_wsi_rna_latefusion \
    --results_dir           ../../.scratch/results \
    --max_epochs            50 \
    --k                     10 \
    --early_stopping \
    --patience              5 \
    --weighted_sample \
    --log_data \
    --wandb \
    --wandb_project         clam-brca-subtyping-cv \
    --wandb_tags            wsi rna late-fusion \
    --model_type            clam_mb \
    --model_size            big \
    --B                     4 \
    --no_inst_cluster \
    --bag_loss              ce \
    --drop_out              0.5 \
    --opt                   adam \
    --lr                    0.0001 \
    --reg                   0.0000025 \
    --seed                  1 \
    --split_dir             tcga_brca_subtyping_100 \
    --tabular_csv           "${TABULAR_CSV}" \
    --tabular_case_id_col   case_id \
    --tabular_hidden_dim    256 \
    --tabular_num_layers    2 \
    --fusion_mode           concat \
    --fusion_hidden_dim     32 \
    --pretrained_wsi_ckpt   "${PRETRAINED_WSI_CKPT}" \
    --freeze_wsi_branch
