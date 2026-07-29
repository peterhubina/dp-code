#!/bin/bash
# Novel-fusion chapter: FiLM-conditioned attention MIL vs the controls it needs.
#
# Every arm here is pinned to the SAME split directory, seed, embeddings and frozen
# per-fold WSI checkpoints as the published three-arm ablation in
# tools/train_er_ablation.sh, so every comparison is paired.
#
# The author runs training; this script is the exact, verified command set.
# tools/train_er_ablation.sh wsi must have been run first -- every arm below loads and
# freezes its per-fold checkpoints (er_wsi_alone_s1/s_{fold}_checkpoint.pt).
#
# Usage (from repo root):
#   bash tools/train_er_novel_fusion.sh select    # A) validation-only rank sweep, RNA only
#   bash tools/train_er_novel_fusion.sh test      # B) the eight pre-registered test arms
#   bash tools/train_er_novel_fusion.sh all       # A then B
# or a single arm by name, e.g.:
#   bash tools/train_er_novel_fusion.sh film_rna
#
# ORDER MATTERS: run 'select' first and read ONLY validation AUROC from it, then set
# FILM_RANK below to the winner. DONE 2026-07-28: rank 64 won on mean validation AUROC
# (0.9582 vs 0.9572 at rank 32 and 0.9546 at rank 16). The test folds are read once.
#
# W&B project: er-brca-ablation (same as the published arms).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/project/CLAM"

# ---- shared configuration (identical to tools/train_er_ablation.sh) ---------- #
DATA_ROOT="../../.datasets/tcga-brca/embeddings"
RESULTS_DIR="../../.scratch/results/er"
# Selection runs go elsewhere so their test predictions can never be mistaken for
# a reportable arm (see the note above select_film_rank).
SELECT_RESULTS_DIR="../../.scratch/results/er_selection"
SPLIT_DIR="tcga_brca_er_100"                 # 10 site-holdout folds
SEED="1"
K="10"
WANDB_PROJECT="er-brca-ablation"
# Selection runs go to their OWN W&B project. CLAM logs mean_test_auc on every summary run
# (main.py:105), so leaving the sweep in the reportable project would put test AUROC for the
# discarded configurations right next to the results. Mirrors the on-disk quarantine below.
SELECT_WANDB_PROJECT="er-brca-selection"
WSI_CKPT="${RESULTS_DIR}/er_wsi_alone_s1/s_{fold}_checkpoint.pt"
RNA_CSV="../../.scratch/TCGA-BRCA-rna/tcga_brca_er_rna_clam.csv.gz"
CLINPATH_CSV="../../tools/data/tcga_brca_clinicopath_clam.csv"
SIGNATURES="../MCAT/dataset_csv/signatures.csv"

# Set RUNNER=echo to print the resolved commands instead of running them:
#   RUNNER=echo bash tools/train_er_novel_fusion.sh test
RUNNER="${RUNNER:-python}"

# Mechanism settings. FILM_RANK is the ONLY tuned hyperparameter and it is selected on
# validation folds, RNA only; the winner is then applied to clinicopath UNCHANGED, because
# the chapter's claim is that one unmodified mechanism serves both modalities.
FILM_RANK="64"
# Fixed a priori (not tuned): the diagnosis showed the gated RNA arm became functionally
# unimodal, and modality dropout is the direct countermeasure. It also enables inference on
# the 47 of 1003 cases that have no transcriptome.
MODALITY_DROPOUT="0.25"

# common_args [results_dir] [wandb_project]  -- default to the reportable destinations.
common_args() {
    printf '%s ' \
        --task tcga_brca_er \
        --data_root_dir "${DATA_ROOT}" \
        --embed_dim 1536 \
        --results_dir "${1:-${RESULTS_DIR}}" \
        --split_dir "${SPLIT_DIR}" \
        --k "${K}" \
        --seed "${SEED}" \
        --max_epochs 50 \
        --early_stopping --patience 5 \
        --weighted_sample \
        --log_data \
        --model_type clam_mb --model_size big \
        --drop_out 0.5 --opt adam --lr 1e-4 --reg 2.5e-6 \
        --B 4 --bag_loss ce --no_inst_cluster \
        --pretrained_wsi_ckpt "${WSI_CKPT}" --freeze_wsi_branch \
        --wandb --wandb_project "${2:-${WANDB_PROJECT}}"
}

rna_args() {
    printf '%s ' \
        --tabular_csv "${RNA_CSV}" --tabular_case_id_col case_id \
        --tabular_hidden_dim 256 --tabular_num_layers 2 \
        --tabular_top_n_features 10000
}

clinpath_args() {
    printf '%s ' \
        --tabular_csv "${CLINPATH_CSV}" --tabular_case_id_col case_id \
        --tabular_hidden_dim 256 --tabular_num_layers 2 \
        --tabular_top_n_features 0
}

# ============================ A. SELECTION (validation only) ================== #
# Read ONLY mean validation AUROC from these runs.
#
# CLAM always evaluates the test fold at the end of a run, so these runs DO write test
# predictions whether we want them or not. They are therefore written to a SEPARATE
# results directory (SELECT_RESULTS_DIR) so they cannot be picked up by the analysis
# script, which globs the main results directory by exp_code. Do not move them across,
# and do not read their test numbers -- the test folds are reserved for the arms in B.
select_film_rank() {
    for rank in 16 32 64; do
        echo ">>> SELECT: film_attention + RNA, film_rank=${rank} (validation only)"
        "${RUNNER}" main.py $(common_args "${SELECT_RESULTS_DIR}" "${SELECT_WANDB_PROJECT}") $(rna_args) \
            --exp_code "er_sel_film_rna_r${rank}" \
            --fusion_mode film_attention --film_rank "${rank}" \
            --modality_dropout "${MODALITY_DROPOUT}" \
            --wandb_tags er selection film_attention "rank${rank}"
    done
    echo
    echo "Pick the rank with the highest mean VALIDATION AUROC, set FILM_RANK above, then run 'test'."
}

# ============================ B. TEST ARMS (read once) ======================== #

# --- primary: the novel mechanism, one unmodified configuration for both modalities --- #
film_rna() {
    echo ">>> TEST 1/8: FiLM-conditioned attention + RNA -> er_wsi_rna_film"
    "${RUNNER}" main.py $(common_args) $(rna_args) \
        --exp_code er_wsi_rna_film \
        --fusion_mode film_attention --film_rank "${FILM_RANK}" \
        --modality_dropout "${MODALITY_DROPOUT}" \
        --wandb_tags er novel-fusion wsi-rna film_attention frozen
}

film_clinpath() {
    echo ">>> TEST 2/8: FiLM-conditioned attention + clinicopath -> er_wsi_clinpath_film"
    "${RUNNER}" main.py $(common_args) $(clinpath_args) \
        --exp_code er_wsi_clinpath_film \
        --fusion_mode film_attention --film_rank "${FILM_RANK}" \
        --modality_dropout "${MODALITY_DROPOUT}" \
        --wandb_tags er novel-fusion wsi-clinpath film_attention frozen
}

# --- ablation: film_rank 0 disables attention conditioning, leaving additive-logit fusion.
#     This isolates how much of any gain comes from the FiLM conditioning itself rather
#     than from simply removing the gated arm's convex-combination bottleneck. ---------- #
delta_rna() {
    echo ">>> TEST 3/8: additive-logit fusion (film_rank 0) + RNA -> er_wsi_rna_delta"
    "${RUNNER}" main.py $(common_args) $(rna_args) \
        --exp_code er_wsi_rna_delta \
        --fusion_mode film_attention --film_rank 0 \
        --modality_dropout "${MODALITY_DROPOUT}" \
        --wandb_tags er novel-fusion wsi-rna delta ablation frozen
}

delta_clinpath() {
    echo ">>> TEST 4/8: additive-logit fusion (film_rank 0) + clinicopath -> er_wsi_clinpath_delta"
    "${RUNNER}" main.py $(common_args) $(clinpath_args) \
        --exp_code er_wsi_clinpath_delta \
        --fusion_mode film_attention --film_rank 0 \
        --modality_dropout "${MODALITY_DROPOUT}" \
        --wandb_tags er novel-fusion wsi-clinpath delta ablation frozen
}

# --- capacity control: the published gated arm re-run at a width whose fusion head EXCEEDS
#     the FiLM mechanism's, so a win cannot be attributed to parameter count. The width is
#     derived from the SELECTED film_rank, not chosen freely: at film_rank 64 the FiLM
#     mechanism is 83,714 params, so the control must be gated at fusion_hidden_dim 96
#     (93,026). At film_rank 32 (42,754) dim 64 (57,922) would have sufficed. ---------- #
GATED_CAP_DIM="96"

gatedcap_rna() {
    echo ">>> TEST 5/8: capacity control, gated dim ${GATED_CAP_DIM} + RNA -> er_wsi_rna_gatedcap"
    "${RUNNER}" main.py $(common_args) $(rna_args) \
        --exp_code er_wsi_rna_gatedcap \
        --fusion_mode gated --fusion_hidden_dim "${GATED_CAP_DIM}" \
        --wandb_tags er novel-fusion wsi-rna gated capacity-control frozen
}

gatedcap_clinpath() {
    echo ">>> TEST 6/8: capacity control, gated dim ${GATED_CAP_DIM} + clinicopath -> er_wsi_clinpath_gatedcap"
    "${RUNNER}" main.py $(common_args) $(clinpath_args) \
        --exp_code er_wsi_clinpath_gatedcap \
        --fusion_mode gated --fusion_hidden_dim "${GATED_CAP_DIM}" \
        --wandb_tags er novel-fusion wsi-clinpath gated capacity-control frozen
}

# --- adapted MCAT-style co-attention baseline: tabular tokens query the patch tokens.
#     NOT a reproduction of MCAT -- same folds, frozen branch, loss and tabular encoder as
#     every other arm, so only the fusion operator differs. RNA is tokenised by MCAT's own
#     six gene signatures plus an 'unassigned' token (so the feature set matches the other
#     arms exactly); clinicopath by its natural one-hot blocks. ----------------------- #
coattn_rna() {
    echo ">>> TEST 7/8: adapted co-attention + RNA -> er_wsi_rna_coattn"
    "${RUNNER}" main.py $(common_args) $(rna_args) \
        --exp_code er_wsi_rna_coattn \
        --fusion_mode coattn --fusion_hidden_dim 64 \
        --tabular_group_spec "${SIGNATURES}" \
        --wandb_tags er novel-fusion wsi-rna coattn baseline frozen
}

coattn_clinpath() {
    echo ">>> TEST 8/8: adapted co-attention + clinicopath -> er_wsi_clinpath_coattn"
    "${RUNNER}" main.py $(common_args) $(clinpath_args) \
        --exp_code er_wsi_clinpath_coattn \
        --fusion_mode coattn --fusion_hidden_dim 64 \
        --tabular_group_spec prefix \
        --wandb_tags er novel-fusion wsi-clinpath coattn baseline frozen
}

run_test_arms() {
    film_rna; film_clinpath
    delta_rna; delta_clinpath
    gatedcap_rna; gatedcap_clinpath
    coattn_rna; coattn_clinpath
}

case "${1:-}" in
    select)           select_film_rank ;;
    test)             run_test_arms ;;
    all)              select_film_rank; run_test_arms ;;
    film_rna)         film_rna ;;
    film_clinpath)    film_clinpath ;;
    delta_rna)        delta_rna ;;
    delta_clinpath)   delta_clinpath ;;
    gatedcap_rna)     gatedcap_rna ;;
    gatedcap_clinpath) gatedcap_clinpath ;;
    coattn_rna)       coattn_rna ;;
    coattn_clinpath)  coattn_clinpath ;;
    *)
        echo "usage: bash tools/train_er_novel_fusion.sh {select|test|all|<arm>}" >&2
        echo "  arms: film_rna film_clinpath delta_rna delta_clinpath" >&2
        echo "        gatedcap_rna gatedcap_clinpath coattn_rna coattn_clinpath" >&2
        echo "  run 'select' first and read ONLY validation AUROC from it." >&2
        exit 2
        ;;
esac
