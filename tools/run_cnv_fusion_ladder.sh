#!/bin/bash
# Fusion-operator ladder: H&E (CLAM-MB + UNI2-h) as the primary modality, arm-level CNV as the
# second, every operator on identical splits.
#
#   bash tools/run_cnv_fusion_ladder.sh --dry_run          # print the plan, run nothing
#   bash tools/run_cnv_fusion_ladder.sh --modes film_attention
#   bash tools/run_cnv_fusion_ladder.sh                    # the whole ladder
#   bash tools/run_cnv_fusion_ladder.sh --k 1 --max_epochs 2 --exp_suffix _smoke   # wiring check
#
# THIS IS NOW A SHIM over `dp-train`. The configuration lives in
# `dpcode/conf/experiment/pam50_wsi_cnv.yaml` plus `dpcode/conf/fusion/<operator>.yaml`, and the
# whole ladder in one command is:
#
#     dp-train -m experiment=pam50_wsi_cnv fusion=concat,gated,cross_attention,film_attention,coattn
#
# This script stays because the loop it runs is not just a sweep: it SKIPS an operator whose output
# directory already exists, and all five arms are already on disk in a gitignored tree that nothing
# can regenerate. `dp-train` has its own guard (it refuses a directory holding `summary.csv` or an
# `s_*_checkpoint.pt`), and this loop keeps the older, blunter check on top of it.
#
# The question is whether conditioning H&E on copy number beats simply averaging two independent
# predictions. The bar is not the WSI-only model -- it is the equal-weight probability mean, which
# already reaches 0.909 externally (docs/cnv-wsi-fusion-external-validation.md).
#
# Two operators need something extra:
#   coattn    needs a token grouping -- chromosome_groups.csv, 22 tokens over 39 arms. It is part of
#             `dpcode/conf/fusion/coattn.yaml`, so nothing has to pass it here.
#   residual  needs a matched tabular-only checkpoint via --pretrained_rna_ckpt, and there is still
#             no supported way to train one: main.py restricts --model_type to clam_sb|clam_mb|mil,
#             so tools/train_pam50_tabular.sh (which passes tabular_mlp) fails at argparse.
#             `fusion=residual` now refuses AT COMPOSITION rather than after creating a run
#             directory. `--rna_ckpt` is kept only to explain that.
#
# The second-modality-alone arm does not need CLAM at all: tools/evaluate_cnv_wsi_fusion.py already
# reports CNV-only at 0.872 internal / 0.888 external from a 39-feature logistic regression.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DP_TRAIN=(dp-train)
command -v dp-train >/dev/null 2>&1 || DP_TRAIN=(python -m dpcode.cli.train)

EXPERIMENT="pam50_wsi_cnv"
EXP_PREFIX="pam50_wsi_cnv"
MODES="concat gated cross_attention film_attention coattn"
TABULAR_CSV=""
RNA_CKPT=""
SEED=1
K=10
MAX_EPOCHS=50
EXP_SUFFIX=""
DRY_RUN=0
SKIP_EXISTING=1
WARM_START=1
WANDB=0

usage() {
    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --modes) MODES="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --max_epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --exp_suffix) EXP_SUFFIX="$2"; shift 2 ;;
        --tabular_csv) TABULAR_CSV="$2"; shift 2 ;;
        --rna_ckpt) RNA_CKPT="$2"; shift 2 ;;
        --dry_run|--dry-run) DRY_RUN=1; shift ;;
        --no_skip_existing) SKIP_EXISTING=0; shift ;;
        --no_warm_start) WARM_START=0; shift ;;
        --wandb) WANDB=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

# Where CLAM will write. Read from the config rather than hardcoded, so
# DP_RESULTS_ROOT / DP_SCRATCH_ROOT move the skip check with the results.
RESULTS_ROOT="$(python -c 'from dpcode.paths import resolve_paths; print(resolve_paths()["results_root"])' 2>/dev/null || true)"
if [[ -z "${RESULTS_ROOT}" ]]; then
    # An empty prefix would make the check below silently miss every directory, so
    # it is turned OFF and said out loud instead. The protection that matters is
    # not weakened: dp-train still refuses any run directory that already holds a
    # summary.csv or an s_*_checkpoint.pt.
    echo "note: could not resolve paths.results_root through python -- is dp-code installed" >&2
    echo "      (pip install -e .)? The skip-existing check is DISABLED for this run;" >&2
    echo "      dp-train's own overwrite guard still refuses a directory holding results." >&2
fi

for MODE in ${MODES}; do
    EXP="${EXP_PREFIX}_${MODE}${EXP_SUFFIX}"
    OUT="${RESULTS_ROOT}/${EXP}_s${SEED}"
    echo
    echo "=== ${MODE} -> ${EXP} ==="

    if [[ "${MODE}" == "residual" ]]; then
        # Kept as an explanation, not as a capability. See the header and the
        # Known gaps section of CLAUDE.md.
        echo "  residual fusion is not runnable: it needs --pretrained_rna_ckpt, and no supported" >&2
        echo "  trainer produces one (main.py rejects --model_type tabular_mlp). fusion=residual" >&2
        echo "  refuses at config composition. See dpcode/conf/fusion/residual.yaml." >&2
        [[ -n "${RNA_CKPT}" ]] && echo "  (--rna_ckpt ${RNA_CKPT} cannot change that.)" >&2
        exit 1
    fi

    # The older, blunter guard: any existing directory, not just one holding
    # results. It is the only thing that has been protecting the five completed
    # arms, so it stays on top of dp-train's own check.
    if [[ "${SKIP_EXISTING}" == "1" && -n "${RESULTS_ROOT}" && -d "${OUT}" ]]; then
        echo "  exists, skipping (${OUT})"
        continue
    fi

    OVERRIDES=(
        "experiment=${EXPERIMENT}"
        "fusion=${MODE}"
        "clam.k=${K}"
        "clam.seed=${SEED}"
        "clam.max_epochs=${MAX_EPOCHS}"
    )
    # Only when a suffix is given: without one, the experiment's own
    # `exp_code: pam50_wsi_cnv_${fusion.name}` is already exactly this string.
    [[ -n "${EXP_SUFFIX}" ]] && OVERRIDES+=("clam.exp_code=${EXP}")
    if [[ -n "${TABULAR_CSV}" ]]; then
        case "${TABULAR_CSV}" in
            /*) OVERRIDES+=("clam.tabular_csv=${TABULAR_CSV}") ;;
            *)  OVERRIDES+=("clam.tabular_csv=${REPO_ROOT}/${TABULAR_CSV}") ;;
        esac
    fi
    # --no_skip_existing used to mean "run even though the directory is there",
    # which is now dp-train's run.overwrite.
    [[ "${SKIP_EXISTING}" == "0" ]] && OVERRIDES+=("run.overwrite=true")
    [[ "${WARM_START}" == "0" ]] && OVERRIDES+=("clam.pretrained_wsi_ckpt=null")
    if [[ "${WANDB}" == "1" ]]; then
        OVERRIDES+=(
            "clam.wandb=true"
            "clam.wandb_project=clam-brca-subtyping-cv"
            "clam.wandb_tags=[wsi,cnv,${MODE}-fusion]"
        )
    fi
    [[ "${DRY_RUN}" == "1" ]] && OVERRIDES+=(--dry-run)

    echo "  ${DP_TRAIN[*]} ${OVERRIDES[*]}"
    "${DP_TRAIN[@]}" "${OVERRIDES[@]}"
done

echo
echo "ladder complete."
echo "Compare the arms:  python tools/compare_fusion_ladder.py"
echo
echo "EXTERNAL VALIDATION OF A TRAINED ARM IS BLOCKED -- see 'Known gaps' in CLAUDE.md."
echo "The command this script used to print here was wrong in three ways: it evaluated the TCGA"
echo "test split rather than CPTAC, evaluate_multimodal.py cannot load a film_attention or coattn"
echo "checkpoint at all, and its --tabular_hidden_dim default of 256 does not match the 64 these"
echo "arms trained with, so even concat and gated die on a strict=True shape mismatch."
echo
echo "The bar to clear is the equal-weight probability mean: 0.909 external macro AUROC,"
echo "0.646 balanced accuracy. WSI-only alone is 0.847 / 0.513."
