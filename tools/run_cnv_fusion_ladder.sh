#!/bin/bash
# Fusion-operator ladder: H&E (CLAM-MB + UNI2-h) as the primary modality, arm-level CNV as the
# second, every operator on identical splits, then external validation on CPTAC.
#
#   bash tools/run_cnv_fusion_ladder.sh --dry_run          # print the plan, run nothing
#   bash tools/run_cnv_fusion_ladder.sh --modes film_attention
#   bash tools/run_cnv_fusion_ladder.sh                    # the whole ladder
#   bash tools/run_cnv_fusion_ladder.sh --k 1 --max_epochs 2 --exp_suffix _smoke   # wiring check
#
# The question is whether conditioning H&E on copy number beats simply averaging two independent
# predictions. The bar is not the WSI-only model -- it is the equal-weight probability mean, which
# already reaches 0.909 externally (docs/cnv-wsi-fusion-external-validation.md). Six operators that
# all fail to clear it would be a real result, given a literature where four groups report fusion
# architectures that do not help and none report the trivial baseline.
#
# Why these runs are comparable to what is already on disk: every non-Normal case has copy number
# (910/910, checked by tools/make_cnv_tabular.py), so the ladder reuses splits/tcga_brca_subtyping_100
# and the existing WSI-only run pam50_final_s1 is a valid baseline without retraining.
#
# Two operators need something extra:
#   coattn    needs a token grouping -- chromosome_groups.csv, 22 tokens over 39 arms. Handled.
#   residual  needs a matched tabular-only checkpoint via --pretrained_rna_ckpt, and there is
#             currently no supported way to train one: main.py restricts --model_type to
#             clam_sb|clam_mb|mil, so tools/train_pam50_tabular.sh (which passes tabular_mlp) fails
#             at argparse. `residual` is therefore NOT in the default ladder. Pass it explicitly
#             once a checkpoint exists at --rna_ckpt.
#
# The second-modality-alone arm does not need CLAM at all: tools/evaluate_cnv_wsi_fusion.py already
# reports CNV-only at 0.872 internal / 0.888 external from a 39-feature logistic regression.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TABULAR_CSV=".scratch/cnv-tabular/TCGA_BRCA_CNV_arm_4class_clam.csv"
GROUP_SPEC=".scratch/cnv-tabular/chromosome_groups.csv"
RESULTS_DIR=".scratch/results"
SPLIT_DIR="tcga_brca_subtyping_100"
WSI_CKPT=".scratch/results/pam50_final_s1/s_{fold}_checkpoint.pt"
MODES="concat gated cross_attention film_attention coattn"
RNA_CKPT=""
SEED=1
K=10
MAX_EPOCHS=50
EXP_PREFIX="pam50_wsi_cnv"
EXP_SUFFIX=""
DRY_RUN=0
SKIP_EXISTING=1
WARM_START=1
WANDB=0

usage() {
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
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
        --dry_run) DRY_RUN=1; shift ;;
        --no_skip_existing) SKIP_EXISTING=0; shift ;;
        --no_warm_start) WARM_START=0; shift ;;
        --wandb) WANDB=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

for required in "${TABULAR_CSV}" "${GROUP_SPEC}"; do
    if [[ ! -f "${required}" ]]; then
        echo "missing ${required}" >&2
        echo "run: python tools/make_cnv_tabular.py" >&2
        exit 1
    fi
done

run() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '  %s\n' "$*" | sed 's/ --/ \\\n      --/g'
    else
        "$@"
    fi
}

for MODE in ${MODES}; do
    EXP="${EXP_PREFIX}_${MODE}${EXP_SUFFIX}"
    OUT="${RESULTS_DIR}/${EXP}_s${SEED}"
    echo
    echo "=== ${MODE} -> ${EXP} ==="
    if [[ "${SKIP_EXISTING}" == "1" && -d "${OUT}" ]]; then
        echo "  exists, skipping (${OUT})"
        continue
    fi

    EXTRA=()
    case "${MODE}" in
        coattn)   EXTRA+=(--tabular_group_spec "${REPO_ROOT}/${GROUP_SPEC}") ;;
        residual) if [[ -z "${RNA_CKPT}" ]]; then
                      echo "  residual needs --rna_ckpt PATH (may contain {fold}); see the header" >&2
                      exit 1
                  fi
                  EXTRA+=(--pretrained_rna_ckpt "${RNA_CKPT}" --rna_hidden_dims 64,64) ;;
        # The FiLM conditioner is the operator this ladder exists to test: the tabular vector
        # predicts an affine transform of the attention network's input, re-ranking patches
        # rather than being appended after pooling. Rank 16 for a 39-dim modality.
        film_attention) EXTRA+=(--film_rank 16 --modality_dropout 0.2) ;;
    esac
    # Warm-starting the WSI branch from the WSI-only run keeps H&E the primary modality in fact
    # and not just in framing: fusion has to improve on that model, not rediscover it.
    if [[ "${WARM_START}" == "1" ]]; then
        EXTRA+=(--pretrained_wsi_ckpt "${REPO_ROOT}/${WSI_CKPT}")
    fi
    if [[ "${WANDB}" == "1" ]]; then
        EXTRA+=(--wandb --wandb_project clam-brca-subtyping-cv
                --wandb_tags wsi cnv "${MODE}-fusion")
    fi

    ( cd project/CLAM && run python main.py \
        --task tcga_brca_subtyping --subtyping \
        --data_root_dir "${REPO_ROOT}/.datasets/tcga-brca/embeddings" \
        --embed_dim 1536 --model_type clam_mb --model_size big \
        --exp_code "${EXP}" --results_dir "${REPO_ROOT}/${RESULTS_DIR}" \
        --split_dir "${SPLIT_DIR}" --k "${K}" --seed "${SEED}" \
        --max_epochs "${MAX_EPOCHS}" --early_stopping --patience 5 \
        --bag_loss ce --no_inst_cluster --drop_out 0.5 --opt adam \
        --lr 0.0001 --reg 0.0000025 --weighted_sample --log_data \
        --B 4 \
        --tabular_csv "${REPO_ROOT}/${TABULAR_CSV}" --tabular_case_id_col case_id \
        --tabular_hidden_dim 64 --tabular_num_layers 2 --tabular_top_n_features 0 \
        --fusion_mode "${MODE}" --fusion_hidden_dim 32 \
        "${EXTRA[@]}" )
done

echo
echo "ladder complete. External validation of each arm:"
echo "  bash tools/evaluate_pam50_multimodal.sh \\"
echo "    --ckpt_dir ${RESULTS_DIR}/${EXP_PREFIX}_<mode>${EXP_SUFFIX}_s${SEED} \\"
echo "    --tabular_csv .scratch/cnv-tabular/CPTAC_BRCA_CNV_arm_4class_clam.csv \\"
echo "    --output_dir ${RESULTS_DIR}/${EXP_PREFIX}_<mode>${EXP_SUFFIX}_eval"
echo
echo "The bar to clear is the equal-weight probability mean: 0.909 external macro AUROC,"
echo "0.646 balanced accuracy. WSI-only alone is 0.847 / 0.513."
