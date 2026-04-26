#!/bin/bash
# RNA-seq tabular-only baseline for TCGA-BRCA PAM50 subtype classification.

set -euo pipefail

TABULAR_CSV="${TABULAR_CSV:-../../.scratch/TCGA-BRCA-rna/TCGA_BRCA_RNA_primary_tumor_4class_clam.csv.gz}"

cd project/CLAM

python main.py \
    --task                  tcga_brca_subtyping \
    --subtyping \
    --exp_code              pam50_rna_mlp \
    --results_dir           ../../.scratch/results \
    --max_epochs            50 \
    --k                     10 \
    --early_stopping \
    --patience              5 \
    --weighted_sample \
    --log_data \
    --wandb \
    --wandb_project         clam-brca-subtyping-cv \
    --wandb_tags            rna tabular baseline \
    --model_type            tabular_mlp \
    --bag_loss              ce \
    --drop_out              0.5 \
    --opt                   adam \
    --lr                    0.0001 \
    --reg                   0.00001 \
    --seed                  1 \
    --split_dir             tcga_brca_subtyping_100 \
    --tabular_csv           "${TABULAR_CSV}" \
    --tabular_case_id_col   case_id \
    --tabular_hidden_dim    256 \
    --tabular_num_layers    2
