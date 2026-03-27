#!/bin/bash
# Full 10-fold cross-validation for PAM50 molecular subtype classification.
# Config frozen from sweep candidate "baseline-best" (selected by val_auc).
#
# Usage:
#   cd /workspace/dp-code
#   bash tools/train_pam50_final.sh

set -euo pipefail

cd project/CLAM

python main.py \
    --task          tcga_brca_subtyping \
    --data_root_dir ../../.datasets/embeddings \
    --embed_dim     1536 \
    --subtyping \
    --exp_code      pam50_final \
    --results_dir   ../../.scratch/results \
    --max_epochs    50 \
    --k             10 \
    --early_stopping \
    --patience      5 \
    --weighted_sample \
    --log_data \
    --wandb \
    --wandb_project clam-brca-subtyping \
    --wandb_tags    full-cv best-config \
    --model_type    clam_mb \
    --model_size    big \
    --B             4 \
    --bag_loss      ce \
    --bag_weight    0.5533776374353542 \
    --inst_loss     svm \
    --drop_out      0.5 \
    --opt           adam \
    --lr            0.0001007597588073064 \
    --reg           0.0000024456514744717547 \
    --seed          1 \
    --split_dir     tcga_brca_subtyping_100
