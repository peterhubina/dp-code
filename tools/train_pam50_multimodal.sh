#!/bin/bash
# Multimodal PAM50 run: WSI CLAM-MB + RNA-seq MLP fusion.
#
# Usage:
#   cd /workspace/dp-code
#   bash tools/train_pam50_multimodal.sh \
#     --pretrained_wsi_ckpt '.scratch/results/pam50_final_s1/s_{fold}_checkpoint.pt'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_ROOT_DIR=".datasets/tcga-brca/embeddings"
TABULAR_CSV=".scratch/TCGA-BRCA-rna/TCGA_BRCA_RNA_primary_tumor_4class_clam.csv.gz"
PRETRAINED_WSI_CKPT=""
RESULTS_DIR=".scratch/results"
EXP_CODE="pam50_wsi_rna_gatedfusion"
FUSION_MODE="gated"
MAX_EPOCHS="50"
K="10"
K_START="-1"
K_END="-1"
SEED="1"
WANDB=1
WANDB_PROJECT="clam-brca-subtyping-cv"
FREEZE_WSI=1
TABULAR_HIDDEN_DIM="256"
TABULAR_NUM_LAYERS="2"
TABULAR_TOP_N_FEATURES="10000"
FUSION_HIDDEN_DIM="32"

usage() {
    cat <<'EOF'
Multimodal PAM50 run: WSI CLAM-MB + RNA-seq MLP fusion.

Usage:
  cd /workspace/dp-code
  bash tools/train_pam50_multimodal.sh \
    --pretrained_wsi_ckpt '.scratch/results/pam50_final_s1/s_{fold}_checkpoint.pt'

Options:
  --pretrained_wsi_ckpt PATH  Required. WSI checkpoint path. May include "{fold}".
  --tabular_csv PATH          RNA/tabular feature CSV.
  --data_root_dir PATH        WSI embedding directory.
  --results_dir PATH          Results root directory.
  --exp_code NAME             Experiment code.
  --fusion_mode NAME          concat or gated. Default: gated.
  --max_epochs N              Training epochs.
  --k N                       Number of folds.
  --k_start N                 First fold.
  --k_end N                   End fold, exclusive. -1 means --k.
  --seed N                    Random seed.
  --wandb                     Enable W&B logging. Default.
  --no_wandb                  Disable W&B logging.
  --wandb_project NAME        W&B project name.
  --freeze_wsi_branch         Freeze pretrained WSI branch. Default.
  --no_freeze_wsi_branch      Fine-tune WSI branch.
  --tabular_hidden_dim N      RNA encoder hidden dimension.
  --tabular_num_layers N      RNA encoder layer count.
  --tabular_top_n_features N  Training-fold RNA feature selection count.
  --fusion_hidden_dim N       Fusion hidden dimension.
  -h, --help                  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained_wsi_ckpt) PRETRAINED_WSI_CKPT="$2"; shift 2 ;;
        --tabular_csv) TABULAR_CSV="$2"; shift 2 ;;
        --data_root_dir) DATA_ROOT_DIR="$2"; shift 2 ;;
        --results_dir) RESULTS_DIR="$2"; shift 2 ;;
        --exp_code) EXP_CODE="$2"; shift 2 ;;
        --fusion_mode) FUSION_MODE="$2"; shift 2 ;;
        --max_epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --k_start) K_START="$2"; shift 2 ;;
        --k_end) K_END="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --wandb) WANDB=1; shift ;;
        --no_wandb) WANDB=0; shift ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --freeze_wsi_branch) FREEZE_WSI=1; shift ;;
        --no_freeze_wsi_branch) FREEZE_WSI=0; shift ;;
        --tabular_hidden_dim) TABULAR_HIDDEN_DIM="$2"; shift 2 ;;
        --tabular_num_layers) TABULAR_NUM_LAYERS="$2"; shift 2 ;;
        --tabular_top_n_features) TABULAR_TOP_N_FEATURES="$2"; shift 2 ;;
        --fusion_hidden_dim) FUSION_HIDDEN_DIM="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${PRETRAINED_WSI_CKPT}" ]]; then
    echo "--pretrained_wsi_ckpt is required." >&2
    usage >&2
    exit 2
fi

if [[ "${FUSION_MODE}" != "concat" && "${FUSION_MODE}" != "gated" ]]; then
    echo "--fusion_mode must be 'concat' or 'gated'." >&2
    exit 2
fi

repo_path() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf "%s" "${path}"
    else
        printf "%s/%s" "${REPO_ROOT}" "${path}"
    fi
}

WANDB_ARGS=()
if [[ "${WANDB}" == "1" ]]; then
    WANDB_ARGS+=(--wandb)
fi

FREEZE_ARGS=()
if [[ "${FREEZE_WSI}" == "1" ]]; then
    FREEZE_ARGS+=(--freeze_wsi_branch)
fi

cd "${REPO_ROOT}/project/CLAM"

python main.py \
    --task                  tcga_brca_subtyping \
    --data_root_dir         "$(repo_path "${DATA_ROOT_DIR}")" \
    --embed_dim             1536 \
    --subtyping \
    --exp_code              "${EXP_CODE}" \
    --results_dir           "$(repo_path "${RESULTS_DIR}")" \
    --max_epochs            "${MAX_EPOCHS}" \
    --k                     "${K}" \
    --k_start               "${K_START}" \
    --k_end                 "${K_END}" \
    --early_stopping \
    --patience              5 \
    --weighted_sample \
    --log_data \
    --wandb_project         "${WANDB_PROJECT}" \
    --wandb_tags            wsi rna "${FUSION_MODE}-fusion" \
    --model_type            clam_mb \
    --model_size            big \
    --B                     4 \
    --no_inst_cluster \
    --bag_loss              ce \
    --drop_out              0.5 \
    --opt                   adam \
    --lr                    0.0001 \
    --reg                   0.0000025 \
    --seed                  "${SEED}" \
    --split_dir             tcga_brca_subtyping_100 \
    --tabular_csv           "$(repo_path "${TABULAR_CSV}")" \
    --tabular_case_id_col   case_id \
    --tabular_hidden_dim    "${TABULAR_HIDDEN_DIM}" \
    --tabular_num_layers    "${TABULAR_NUM_LAYERS}" \
    --tabular_top_n_features "${TABULAR_TOP_N_FEATURES}" \
    --fusion_mode           "${FUSION_MODE}" \
    --fusion_hidden_dim     "${FUSION_HIDDEN_DIM}" \
    --pretrained_wsi_ckpt   "$(repo_path "${PRETRAINED_WSI_CKPT}")" \
    "${WANDB_ARGS[@]}" \
    "${FREEZE_ARGS[@]}"
