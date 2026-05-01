#!/bin/bash
# Documentation-ready evaluation for PAM50 WSI + RNA multimodal-fusion checkpoints.
#
# Usage:
#   cd /workspace/dp-code
#   bash tools/evaluate_pam50_multimodal.sh \
#     --ckpt_dir .scratch/results/pam50_wsi_rna_gatedfusion_s1 \
#     --output_dir .scratch/results/pam50_wsi_rna_gatedfusion_eval
#
# Add --wandb to log metrics, plots, prediction tables, and artifacts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_ROOT_DIR=".datasets/tcga-brca/embeddings"
TABULAR_CSV=".scratch/TCGA-BRCA-rna/TCGA_BRCA_RNA_primary_tumor_4class_clam.csv.gz"
CKPT_DIR=".scratch/results/pam50_wsi_rna_gatedfusion_s1"
OUTPUT_DIR=".scratch/results/pam50_wsi_rna_gatedfusion_eval"
SPLIT_DIR="project/CLAM/splits/tcga_brca_subtyping_100"
DATASET_CSV="project/CLAM/dataset_csv/tcga_brca_subtyping.csv"
SPLIT="test"
FOLD="-1"
K="10"
K_START="0"
K_END="-1"
WANDB=0
WANDB_PROJECT="clam-brca-subtyping-cv"
WANDB_RUN_NAME="pam50_wsi_rna_gatedfusion_eval"
EMBED_DIM="1536"
MODEL_TYPE="clam_mb"
MODEL_SIZE="big"
FUSION_MODE="auto"
DROP_OUT="0.5"
B="4"
TABULAR_HIDDEN_DIM="256"
TABULAR_NUM_LAYERS="2"
FUSION_HIDDEN_DIM="32"

usage() {
    cat <<'EOF'
Documentation-ready evaluation for PAM50 WSI + RNA multimodal-fusion checkpoints.

Usage:
  cd /workspace/dp-code
  bash tools/evaluate_pam50_multimodal.sh \
    --ckpt_dir .scratch/results/pam50_wsi_rna_gatedfusion_s1 \
    --output_dir .scratch/results/pam50_wsi_rna_gatedfusion_eval

Add --wandb to log metrics, plots, prediction tables, and artifacts.

Options:
  --ckpt_dir PATH          Directory with s_<fold>_checkpoint.pt files.
  --output_dir PATH        Directory to write CSV/JSON/PNG evaluation outputs.
  --tabular_csv PATH       RNA/tabular feature CSV.
  --data_root_dir PATH     WSI embedding directory.
  --split_dir PATH         CLAM split directory.
  --dataset_csv PATH       CLAM dataset CSV.
  --split NAME             Split to evaluate: train, val, or test.
  --fold N                 Single fold to evaluate. Default: all folds.
  --k N                    Number of folds.
  --k_start N              First fold when evaluating a range.
  --k_end N                End fold, exclusive. -1 means --k.
  --wandb                  Log metrics, plots, tables, and artifact to W&B.
  --wandb_project NAME     W&B project name.
  --wandb_run_name NAME    W&B run name.
  --embed_dim N            WSI embedding dimension.
  --model_type NAME        clam_sb or clam_mb.
  --model_size NAME        small or big.
  --fusion_mode NAME       auto, concat, or gated. Default: auto.
  --drop_out FLOAT         Dropout used by the trained checkpoint.
  --B N                    CLAM instance sampling parameter used by checkpoint.
  --tabular_hidden_dim N   RNA encoder hidden dimension used by checkpoint.
  --tabular_num_layers N   RNA encoder layer count used by checkpoint.
  --fusion_hidden_dim N    Fusion head hidden dimension used by checkpoint.
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt_dir) CKPT_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tabular_csv) TABULAR_CSV="$2"; shift 2 ;;
        --data_root_dir) DATA_ROOT_DIR="$2"; shift 2 ;;
        --split_dir) SPLIT_DIR="$2"; shift 2 ;;
        --dataset_csv) DATASET_CSV="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --fold) FOLD="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --k_start) K_START="$2"; shift 2 ;;
        --k_end) K_END="$2"; shift 2 ;;
        --wandb) WANDB=1; shift ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --wandb_run_name) WANDB_RUN_NAME="$2"; shift 2 ;;
        --embed_dim) EMBED_DIM="$2"; shift 2 ;;
        --model_type) MODEL_TYPE="$2"; shift 2 ;;
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --fusion_mode) FUSION_MODE="$2"; shift 2 ;;
        --drop_out) DROP_OUT="$2"; shift 2 ;;
        --B) B="$2"; shift 2 ;;
        --tabular_hidden_dim) TABULAR_HIDDEN_DIM="$2"; shift 2 ;;
        --tabular_num_layers) TABULAR_NUM_LAYERS="$2"; shift 2 ;;
        --fusion_hidden_dim) FUSION_HIDDEN_DIM="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "${FUSION_MODE}" != "auto" && "${FUSION_MODE}" != "concat" && "${FUSION_MODE}" != "gated" ]]; then
    echo "--fusion_mode must be 'auto', 'concat', or 'gated'." >&2
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

cd "${REPO_ROOT}/project/CLAM"

WANDB_ARGS=()
if [[ "${WANDB}" == "1" ]]; then
    WANDB_ARGS+=(--wandb)
fi

python evaluate_multimodal.py \
    --data_root_dir "$(repo_path "${DATA_ROOT_DIR}")" \
    --tabular_csv "$(repo_path "${TABULAR_CSV}")" \
    --ckpt_dir "$(repo_path "${CKPT_DIR}")" \
    --output_dir "$(repo_path "${OUTPUT_DIR}")" \
    --split_dir "$(repo_path "${SPLIT_DIR}")" \
    --dataset_csv "$(repo_path "${DATASET_CSV}")" \
    --split "${SPLIT}" \
    --fold "${FOLD}" \
    --k "${K}" \
    --k_start "${K_START}" \
    --k_end "${K_END}" \
    --embed_dim "${EMBED_DIM}" \
    --model_type "${MODEL_TYPE}" \
    --model_size "${MODEL_SIZE}" \
    --fusion_mode "${FUSION_MODE}" \
    --drop_out "${DROP_OUT}" \
    --B "${B}" \
    --tabular_case_id_col case_id \
    --tabular_hidden_dim "${TABULAR_HIDDEN_DIM}" \
    --tabular_num_layers "${TABULAR_NUM_LAYERS}" \
    --fusion_hidden_dim "${FUSION_HIDDEN_DIM}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    "${WANDB_ARGS[@]}"
