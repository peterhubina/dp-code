#!/bin/bash
# Multimodal PAM50 run: WSI CLAM-MB + RNA-seq MLP fusion.
#
# THIS IS NOW A SHIM. The configuration lives in
# `dpcode/conf/experiment/pam50_wsi_rna_gated.yaml`, and the equivalent command is:
#
#     dp-train experiment=pam50_wsi_rna_gated \
#         'clam.pretrained_wsi_ckpt="${paths.results_root}/pam50_final_s1/s_{fold}_checkpoint.pt"'
#
# Note the quoting: a value containing `{fold}` MUST be quoted on the command line, because Hydra's
# override grammar rejects a bare `{`. This script does that quoting for you.
#
# The wrapper's own flags are kept and mapped to Hydra overrides one for one, so
# `tools/cptac/run_pipeline.sh` phase 6 and any existing muscle memory keep working. The rendered
# argv parses to exactly the namespace this script used to pass.
#
# Usage:
#   bash tools/train_pam50_multimodal.sh \
#     --pretrained_wsi_ckpt '.scratch/results/pam50_final_s1/s_{fold}_checkpoint.pt'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DP_TRAIN=(dp-train)
command -v dp-train >/dev/null 2>&1 || DP_TRAIN=(python -m dpcode.cli.train)

PRETRAINED_WSI_CKPT=""
FUSION_MODE="gated"
WANDB=1
FREEZE_WSI=1
EXTRA=()

usage() {
    cat <<'EOF'
Multimodal PAM50 run: WSI CLAM-MB + RNA-seq MLP fusion.  (shim over `dp-train`)

Usage:
  bash tools/train_pam50_multimodal.sh \
    --pretrained_wsi_ckpt '.scratch/results/pam50_final_s1/s_{fold}_checkpoint.pt'

Options (each maps to one Hydra override; anything else is passed straight through):
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
  --dry-run                   Print the CLAM command and write nothing.
  -h, --help                  Show this help.

The equivalent Hydra command is printed before the run starts.
EOF
}

# Relative paths are resolved against the repository root, as before.
repo_path() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf "%s" "${path}"
    else
        printf "%s/%s" "${REPO_ROOT}" "${path}"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained_wsi_ckpt) PRETRAINED_WSI_CKPT="$(repo_path "$2")"; shift 2 ;;
        --tabular_csv) EXTRA+=("clam.tabular_csv=$(repo_path "$2")"); shift 2 ;;
        --data_root_dir) EXTRA+=("clam.data_root_dir=$(repo_path "$2")"); shift 2 ;;
        --results_dir) EXTRA+=("clam.results_dir=$(repo_path "$2")"); shift 2 ;;
        --exp_code) EXTRA+=("clam.exp_code=$2"); shift 2 ;;
        --fusion_mode) FUSION_MODE="$2"; shift 2 ;;
        --max_epochs) EXTRA+=("clam.max_epochs=$2"); shift 2 ;;
        --k) EXTRA+=("clam.k=$2"); shift 2 ;;
        --k_start) EXTRA+=("clam.k_start=$2"); shift 2 ;;
        --k_end) EXTRA+=("clam.k_end=$2"); shift 2 ;;
        --seed) EXTRA+=("clam.seed=$2"); shift 2 ;;
        --wandb) WANDB=1; shift ;;
        --no_wandb) WANDB=0; shift ;;
        --wandb_project) EXTRA+=("clam.wandb_project=$2"); shift 2 ;;
        --freeze_wsi_branch) FREEZE_WSI=1; shift ;;
        --no_freeze_wsi_branch) FREEZE_WSI=0; shift ;;
        --tabular_hidden_dim) EXTRA+=("clam.tabular_hidden_dim=$2"); shift 2 ;;
        --tabular_num_layers) EXTRA+=("clam.tabular_num_layers=$2"); shift 2 ;;
        --tabular_top_n_features) EXTRA+=("clam.tabular_top_n_features=$2"); shift 2 ;;
        --fusion_hidden_dim) EXTRA+=("clam.fusion_hidden_dim=$2"); shift 2 ;;
        --dry-run|--dry_run) EXTRA+=(--dry-run); shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${PRETRAINED_WSI_CKPT}" ]]; then
    echo "--pretrained_wsi_ckpt is required." >&2
    usage >&2
    exit 2
fi

# The wrapper always validated this, and it is narrower than CLAM's own choice
# list on purpose: the six operators are reachable through `dp-train fusion=…`.
if [[ "${FUSION_MODE}" != "concat" && "${FUSION_MODE}" != "gated" ]]; then
    echo "--fusion_mode must be 'concat' or 'gated'." >&2
    echo "  the other four operators: dp-train experiment=pam50_wsi_rna_gated fusion=${FUSION_MODE} …" >&2
    exit 2
fi

OVERRIDES=(
    "experiment=pam50_wsi_rna_gated"
    "fusion=${FUSION_MODE}"
    # Inner double quotes are load-bearing: they make Hydra treat the value as a
    # quoted string, so a `{fold}` placeholder survives the override grammar.
    "clam.pretrained_wsi_ckpt=\"${PRETRAINED_WSI_CKPT}\""
    "clam.wandb=$([[ "${WANDB}" == "1" ]] && echo true || echo false)"
    "clam.freeze_wsi_branch=$([[ "${FREEZE_WSI}" == "1" ]] && echo true || echo false)"
)
OVERRIDES+=("${EXTRA[@]+"${EXTRA[@]}"}")

echo "tools/train_pam50_multimodal.sh now runs:" >&2
printf '    %s' "${DP_TRAIN[*]}" >&2
printf " %q" "${OVERRIDES[@]}" >&2
echo >&2
echo >&2

exec "${DP_TRAIN[@]}" "${OVERRIDES[@]}"
