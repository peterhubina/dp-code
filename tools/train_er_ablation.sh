#!/bin/bash
# Three-way ER-status ablation on TCGA-BRCA: WSI-alone vs WSI+RNA vs WSI+clinicopath.
# Same site-holdout folds, frozen pretrained WSI branch for both fusion arms.
#
# THIS IS NOW A SHIM. One experiment config per arm:
#
#   bash tools/train_er_ablation.sh wsi       ->  dp-train experiment=er_wsi_alone
#   bash tools/train_er_ablation.sh rna       ->  dp-train experiment=er_wsi_rna_gated
#   bash tools/train_er_ablation.sh clinpath  ->  dp-train experiment=er_wsi_clinpath_gated
#   bash tools/train_er_ablation.sh all       ->  all three, in dependency order
#
# Run the WSI-alone arm FIRST -- it writes the per-fold checkpoints that both fusion arms load and
# freeze. `all` preserves that order.
#
# The rendered argv parses to exactly the namespace this script used to pass, arm for arm.
#
# Env hooks, unchanged:
#   SEED=2   bash tools/train_er_ablation.sh wsi      # -> clam.seed=2. The fusion arms' checkpoint
#                                                     #    path follows clam.seed automatically.
#   RUNNER=echo bash tools/train_er_ablation.sh all   # -> dp-train --dry-run: print, run nothing.
#
# W&B project: er-brca-ablation, on for every arm. exp_code (== W&B group) per arm:
#   er_wsi_alone / er_wsi_rna_gated / er_wsi_clinpath_gated.
# To train without W&B, append `clam.wandb=false` -- the wrapper never had that switch.
#
# The ER thread is complete: docs/er-prediction-results.md,
# docs/er-external-validation-results.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DP_TRAIN=(dp-train)
command -v dp-train >/dev/null 2>&1 || DP_TRAIN=(python -m dpcode.cli.train)

SEED="${SEED:-1}"
# The wrapper's dry-run hook was `RUNNER=echo`, which printed the resolved command
# instead of executing it. `dp-train --dry-run` is the same idea and also prints
# the run directory it would write to.
RUNNER="${RUNNER:-python}"
DRY=()
[[ "${RUNNER}" != "python" ]] && DRY=(--dry-run)

run_arm() {
    local label="$1" experiment="$2"
    shift 2
    echo ">>> ${label}"
    echo "    ${DP_TRAIN[*]} experiment=${experiment} clam.seed=${SEED} ${DRY[*]-} $*"
    "${DP_TRAIN[@]}" "experiment=${experiment}" "clam.seed=${SEED}" ${DRY[@]+"${DRY[@]}"} "$@"
}

train_wsi() {
    run_arm "Arm 1/3: WSI-alone baseline (clam_mb) -> er_wsi_alone" er_wsi_alone "$@"
}

train_rna() {
    run_arm "Arm 2/3: WSI + RNA gated fusion (frozen WSI) -> er_wsi_rna_gated" er_wsi_rna_gated "$@"
}

train_clinpath() {
    run_arm "Arm 3/3: WSI + clinicopath gated fusion (frozen WSI) -> er_wsi_clinpath_gated" \
        er_wsi_clinpath_gated "$@"
}

ARM="${1:-}"
[[ $# -gt 0 ]] && shift

case "${ARM}" in
    wsi)      train_wsi "$@" ;;
    rna)      train_rna "$@" ;;
    clinpath) train_clinpath "$@" ;;
    all)      train_wsi "$@"; train_rna "$@"; train_clinpath "$@" ;;
    *)
        echo "usage: bash tools/train_er_ablation.sh {wsi|rna|clinpath|all} [hydra overrides...]" >&2
        echo "  run 'wsi' first: the fusion arms load its per-fold checkpoints." >&2
        exit 2
        ;;
esac
