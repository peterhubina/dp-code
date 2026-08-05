#!/bin/bash
# SHIM. The evaluation entry point is now `dp-evaluate`; this script translates
# its old flags into Hydra overrides and calls it, so existing commands and
# muscle memory keep working. The frozen pre-refactor copy lives in
# tests/legacy_wrappers/tools/ and is what the parity test executes.
#
#   bash tools/evaluate_pam50_multimodal.sh --ckpt_dir .scratch/results/<run>_s1
#   dp-evaluate evaluate.args.ckpt_dir=/abs/path/<run>_s1        # equivalent
#
# THREE THINGS THIS COMMAND DOES NOT DO, all preserved deliberately
# (DESIGN.md section 14 — they are documented gaps, not oversights):
#
#   1. It evaluates the TCGA test split. The defaults are the TCGA embeddings,
#      the TCGA dataset_csv and the TCGA splits; swapping only --tabular_csv to a
#      CPTAC table (which is what the fusion ladder used to print on completion)
#      scores TCGA slides against CPTAC-shaped tabular rows. A genuine external
#      run needs --data_root_dir, --dataset_csv and --tabular_csv all pointed at
#      CPTAC.
#   2. --fusion_mode accepts only auto, concat, gated. project/CLAM's evaluator
#      also accepts residual and cross_attention; this narrower list is the one
#      this wrapper has always had.
#   3. film_attention and coattn checkpoints cannot be evaluated at all — see
#      "Known gaps" in CLAUDE.md. dp-evaluate now refuses them by name instead of
#      letting them die inside load_state_dict.
#
# And one trap that is not a gap but a mismatch: --tabular_hidden_dim defaults to
# 256 (what the RNA fusion runs trained with), while every CNV ladder arm trained
# at 64. dp-evaluate reads the checkpoint directory's own experiment_*.txt and
# prints the exact override before dispatching.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Evaluate PAM50 WSI + tabular multimodal-fusion checkpoints. Thin shim over
`dp-evaluate`; every option below maps to one `evaluate.args.*` override.

Usage:
  bash tools/evaluate_pam50_multimodal.sh \
    --ckpt_dir .scratch/results/pam50_wsi_rna_gatedfusion_s1 \
    --output_dir .scratch/results/pam50_wsi_rna_gatedfusion_eval

Options (relative paths are resolved against the repository root):
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
  --tabular_hidden_dim N   Tabular encoder hidden dimension used by checkpoint.
  --tabular_num_layers N   Tabular encoder layer count used by checkpoint.
  --fusion_hidden_dim N    Fusion head hidden dimension used by checkpoint.
  --dry_run                Print the command dp-evaluate would run, then stop.
  -h, --help               Show this help.

Anything else is passed through to dp-evaluate as a Hydra override, so
`run.seed=2` or `--config other_option` work here too.
EOF
}

# Absolute paths only: dp-evaluate runs from project/CLAM and a relative path
# would resolve there.
repo_path() {
    case "$1" in
        /*) printf '%s' "$1" ;;
        *)  printf '%s/%s' "${REPO_ROOT}" "$1" ;;
    esac
}

OVERRIDES=()
EXTRA=()
set_arg() { OVERRIDES+=("evaluate.args.$1=$2"); }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt_dir)           set_arg ckpt_dir "$(repo_path "$2")"; shift 2 ;;
        --output_dir)         set_arg output_dir "$(repo_path "$2")"; shift 2 ;;
        --tabular_csv)        set_arg tabular_csv "$(repo_path "$2")"; shift 2 ;;
        --data_root_dir)      set_arg data_root_dir "$(repo_path "$2")"; shift 2 ;;
        --split_dir)          set_arg split_dir "$(repo_path "$2")"; shift 2 ;;
        --dataset_csv)        set_arg dataset_csv "$(repo_path "$2")"; shift 2 ;;
        --split)              set_arg split "$2"; shift 2 ;;
        --fold)               set_arg fold "$2"; shift 2 ;;
        --k)                  set_arg k "$2"; shift 2 ;;
        --k_start)            set_arg k_start "$2"; shift 2 ;;
        --k_end)              set_arg k_end "$2"; shift 2 ;;
        --wandb)              set_arg wandb true; shift ;;
        --wandb_project)      set_arg wandb_project "$2"; shift 2 ;;
        --wandb_run_name)     set_arg wandb_run_name "$2"; shift 2 ;;
        --embed_dim)          set_arg embed_dim "$2"; shift 2 ;;
        --model_type)         set_arg model_type "$2"; shift 2 ;;
        --model_size)         set_arg model_size "$2"; shift 2 ;;
        --fusion_mode)        set_arg fusion_mode "$2"; shift 2 ;;
        --drop_out)           set_arg drop_out "$2"; shift 2 ;;
        --B)                  set_arg B "$2"; shift 2 ;;
        --tabular_hidden_dim) set_arg tabular_hidden_dim "$2"; shift 2 ;;
        --tabular_num_layers) set_arg tabular_num_layers "$2"; shift 2 ;;
        --fusion_hidden_dim)  set_arg fusion_hidden_dim "$2"; shift 2 ;;
        --dry_run|--dry-run)  EXTRA+=(--dry-run); shift ;;
        -h|--help)            usage; exit 0 ;;
        *)                    EXTRA+=("$1"); shift ;;
    esac
done

exec dp-evaluate "${EXTRA[@]+"${EXTRA[@]}"}" "${OVERRIDES[@]+"${OVERRIDES[@]}"}"
