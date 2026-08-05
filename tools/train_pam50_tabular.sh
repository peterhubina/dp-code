#!/bin/bash
# RNA-seq tabular-only baseline for TCGA-BRCA PAM50 subtype classification.
#
# THIS SCRIPT DOES NOT WORK AND IS DELIBERATELY LEFT THAT WAY.
#
# `--model_type tabular_mlp` below is not in main.py's choice list
# (`['clam_sb', 'clam_mb', 'mil']`, main.py:148), so argparse exits 2 before any work happens.
# Corroborating evidence that it never ran: the `.scratch/results/pam50_rna_mlp_s1` directory its
# --exp_code would produce does not exist.
#
# It is kept, unmodified and unported to `dp-train`, because it is the documented dead end behind a
# real gap: `--fusion_mode residual` requires a matched tabular-only checkpoint via
# `--pretrained_rna_ckpt`, and this is the only tabular-only trainer in the repository. So residual
# fusion is unreachable, which is why `dp-train fusion=residual` refuses at config composition
# rather than pretending. See:
#   * "Known gaps" in CLAUDE.md
#   * dpcode/conf/fusion/residual.yaml, which explains the refusal in full
#
# Making it work would mean adding a model type to CLAM, i.e. changing the training code behind the
# published numbers. That is explicitly out of scope for the reproducibility refactor; it is a
# modelling decision, not a packaging one.
#
# Every other trainer in tools/ is now a shim over `dp-train` (see tools/train_pam50_final.sh).
# This one has nothing to shim to.

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
