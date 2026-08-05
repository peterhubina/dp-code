#!/bin/bash
# Novel-fusion chapter: FiLM-conditioned attention MIL vs the controls it needs.
#
# THIS IS NOW A SHIM, AND IT COVERS ONE OF THE ELEVEN ARMS.
#
#   bash tools/train_er_novel_fusion.sh film_rna  ->  dp-train experiment=er_wsi_rna_film
#
# WHY ONLY film_rna. This script issued eleven invocations: three validation-only `select` runs at
# film_rank 16/32/64, and eight test arms (film/delta/gatedcap/coattn x RNA/clinicopath). The
# reproducibility refactor ports each wrapper's DEFAULT invocation, and for this chapter that is
# TEST 1/8 -- the arm carrying the chapter's primary claim. The ER thread is COMPLETE: all eleven
# runs are on disk under .scratch/results/er and .scratch/results/er_selection, the analysis has
# been done (docs/er-prediction-results.md), and nothing downstream needs them retrained.
#
# The other ten arms are therefore NOT reachable through this script any more. What they were is
# recorded in two places, and neither is going away:
#   * tests/legacy_wrappers/tools/train_er_novel_fusion.sh -- this file as it was, frozen;
#   * .scratch/results/er/<exp_code>_s1/experiment_<exp_code>.txt -- the settings each run received.
# Porting one is a matter of copying dpcode/conf/experiment/er_wsi_rna_film.yaml and changing the
# tabular block and the operator; the differences are listed in that file's header.
#
# FILM_RANK 64 and MODALITY_DROPOUT 0.25 live in the experiment config now. Rank 64 was selected on
# validation folds (RNA only) on 2026-07-28 -- 0.9582 vs 0.9572 at 32 and 0.9546 at 16 -- and is
# then applied unchanged to the other modality, because the chapter's claim is that one unmodified
# mechanism serves both. Modality dropout was fixed a priori, not tuned.
#
# Env hooks, unchanged:
#   SEED=2 bash tools/train_er_novel_fusion.sh film_rna       # -> clam.seed=2
#   RUNNER=echo bash tools/train_er_novel_fusion.sh film_rna  # -> dp-train --dry-run
#
# tools/train_er_ablation.sh wsi must have been run first, at the same seed: this arm loads and
# freezes er_wsi_alone_s${SEED}/s_{fold}_checkpoint.pt.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DP_TRAIN=(dp-train)
command -v dp-train >/dev/null 2>&1 || DP_TRAIN=(python -m dpcode.cli.train)

SEED="${SEED:-1}"
RUNNER="${RUNNER:-python}"
DRY=()
[[ "${RUNNER}" != "python" ]] && DRY=(--dry-run)

#: The ten arms this shim no longer runs, and what each one was.
not_ported() {
    cat >&2 <<EOF
'$1' is not ported to dp-train. Only 'film_rna' is -- see the header of this file.

The eleven original arms and how each differed from er_wsi_rna_film:
  select (rank 16|32|64) validation-only sweep, into .scratch/results/er_selection
                         and the separate W&B project er-brca-selection
  film_rna               THE PORTED ONE: experiment=er_wsi_rna_film
  film_clinpath          clinicopath table, tabular_top_n_features 0
  delta_rna              film_rank 0 (attention conditioning off, additive-logit fusion)
  delta_clinpath         film_rank 0 + clinicopath table
  gatedcap_rna           fusion_mode gated, fusion_hidden_dim 96 (capacity control)
  gatedcap_clinpath      same, clinicopath table
  coattn_rna             fusion_mode coattn, fusion_hidden_dim 64,
                         tabular_group_spec project/MCAT/dataset_csv/signatures.csv
  coattn_clinpath        fusion_mode coattn, fusion_hidden_dim 64,
                         tabular_group_spec prefix

All eleven have already been run and analysed (docs/er-prediction-results.md). The exact command
set is frozen at tests/legacy_wrappers/tools/train_er_novel_fusion.sh, and what each run actually
received is in .scratch/results/er/<exp_code>_s1/experiment_<exp_code>.txt.
EOF
    exit 2
}

ARM="${1:-}"
[[ $# -gt 0 ]] && shift

case "${ARM}" in
    film_rna)
        echo ">>> TEST 1/8: FiLM-conditioned attention + RNA -> er_wsi_rna_film"
        echo "    ${DP_TRAIN[*]} experiment=er_wsi_rna_film clam.seed=${SEED} ${DRY[*]-} $*"
        exec "${DP_TRAIN[@]}" experiment=er_wsi_rna_film "clam.seed=${SEED}" \
            ${DRY[@]+"${DRY[@]}"} "$@"
        ;;
    select|test|all|film_clinpath|delta_rna|delta_clinpath|gatedcap_rna|gatedcap_clinpath|coattn_rna|coattn_clinpath)
        not_ported "${ARM}"
        ;;
    *)
        echo "usage: bash tools/train_er_novel_fusion.sh film_rna [hydra overrides...]" >&2
        echo "  only the primary test arm is ported; run it with --help for the rest." >&2
        exit 2
        ;;
esac
