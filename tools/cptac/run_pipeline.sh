#!/bin/bash
# CPTAC-BRCA External Validation Pipeline (PAM50)
# ===============================================
# Frozen-weights evaluation of the TCGA-BRCA CLAM-MB ensemble on CPTAC-BRCA.
#
# Phases 0-4 are now `dp-cptac`, which is what this script calls; phases 5-8 (the
# dormant RNA branch) are still shell and live below.
#
# FEATURE PROVENANCE -- what is actually held constant between the cohorts.
# Both feature stores come pre-computed from HuggingFace
# (MahmoodLab/UNI2-h-features, CPTAC/cptac_brca.tar.gz) and share the ENCODER
# (UNI2-h, 1536-dim) and the effective TILE GEOMETRY (256 px at ~20x, 0 overlap).
# That, and only that, is what Phase 1 (tools/cptac/audit_feature_provenance.py)
# verifies. They were NOT produced by the same tiling pipeline: TCGA's .h5 files
# carry Trident attributes (target_magnification=20, level0_magnification=40,
# savetodir=.../trident/20x_256px_0px_overlap) while CPTAC's carry CLAM
# create_patches_fp.py attributes (contour_fn=four_pt, custom_downsample=2.0,
# step_size, use_padding) plus a `mask` dataset recording CLAM's own tissue
# segmentation (use_otsu=False, sthresh=10, mthresh=7, a_t=1.0, seg_level=3).
# So WHICH tiles enter a bag is decided by a different tissue segmenter on each
# cohort. Encoder and geometry are held constant; tile selection is not. Do not
# restate this as "preprocessing is held constant" -- the difference is a live
# confound for any claim that a cross-cohort drop is purely domain shift.
#
# Prerequisites:
#   - `pip install -e .` (this script drives the `dp-cptac` console script)
#   - HF token with APPROVED access to MahmoodLab/UNI2-h-features (gated dataset):
#     https://huggingface.co/datasets/MahmoodLab/UNI2-h-features, then export HF_TOKEN
#   - Trained CLAM checkpoints at .scratch/results/pam50_final_s1/
#   - Nothing else: phase 0 fetches the cohort metadata itself. The old header
#     listed .datasets/cptac-brca/{wsi_manifest,cohort}.csv as prerequisites and
#     no script in this repository produced them.
#
# Usage:
#   bash tools/cptac/run_pipeline.sh                 # phases 0-4, then 5-8
#   bash tools/cptac/run_pipeline.sh --wsi-only      # stop after phase 4
#   dp-cptac phase=0                                 # just the cohort metadata
#   dp-cptac phase=3                                 # just the inference

set -e
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

WSI_ONLY=0
if [[ "${1:-}" == "--wsi-only" ]]; then
    WSI_ONLY=1
    shift
fi

echo "=============================================="
echo "CPTAC-BRCA External Validation Pipeline"
echo "=============================================="

# Phases 0 (cohort metadata) -> features (gated download) -> 1 (provenance audit)
# -> 2a (CLAM manifest) -> 3 (10-fold inference) -> 4 (metrics).
# dp-cptac checks every phase's inputs BEFORE running any of them, so a missing
# manifest costs a second instead of the 16 GB feature download.
dp-cptac phase=all "$@"

if [[ "${WSI_ONLY}" == "1" ]]; then
    echo ""
    echo "--wsi-only: stopping after phase 4."
    exit 0
fi

# ---------------------------------------------------------------- multimodal
# Everything below needs the RNA side. Tier 1 re-derives TCGA RNA from the same
# GDC STAR-Counts pipeline that produced the CPTAC files, so the two cohorts share
# a gene axis and no cross-cohort normalisation is applied. Tier 2 keeps the
# Xena-trained fusion head and maps CPTAC onto its scale with FSQN instead.
#
# DORMANT, and read the caveat before using any number it produces: PAM50 labels
# are computed from the same expression matrix the RNA branch consumes
# (project/data/pam50.R), so an RNA branch predicting PAM50 leaks the target by
# construction. See "Gotchas" in CLAUDE.md.

echo ""
echo "[Phase 5] Building the shared GDC expression tables (Tier 1)..."
python tools/rna/download_gdc_rna.py
python tools/rna/build_gdc_expression.py
python tools/rna/make_case_splits.py

echo ""
echo "[Phase 6] Retraining the RNA branch on GDC-derived TCGA RNA..."
echo "          (WSI branch stays frozen at pam50_final_s1; ~1 min RNA-only, ~45 min fusion)"
# ${REPO_ROOT}, not $PWD: bash expands the word list AFTER `cd project/CLAM`
# runs, so the original `"$PWD/.scratch/..."` became
# `project/CLAM/.scratch/...`, which does not exist. Phase 6a could not run.
( cd project/CLAM && python train_rna.py \
    --data_path "${REPO_ROOT}/.scratch/rna-gdc/TCGA_BRCA_RNA_gdc_4class_clam.csv.gz" \
    --split_dir tcga_brca_subtyping_100_case --no_auto_splits \
    --exp_code pam50_rna_only_gdc --results_dir "${REPO_ROOT}/.scratch/results" \
    --k 10 --top_n_genes 10000 --class_set 4class --seed 1 \
    --early_stopping --weighted_sample )
bash tools/train_pam50_multimodal.sh \
    --pretrained_wsi_ckpt "${REPO_ROOT}/.scratch/results/pam50_final_s1/s_{fold}_checkpoint.pt" \
    --tabular_csv "${REPO_ROOT}/.scratch/rna-gdc/TCGA_BRCA_RNA_gdc_4class_clam.csv.gz" \
    --exp_code pam50_wsi_rna_gatedfusion_gdc --no_wandb

echo ""
echo "[Phase 7] External validation: RNA-only, fusion, and the RNA-ablation control..."
python tools/cptac/infer_cptac_rna.py
python tools/cptac/infer_cptac_multimodal.py \
    --output_dir .scratch/cptac_validation/results/predictions_fusion
python tools/cptac/infer_cptac_multimodal.py --rna_ablate \
    --output_dir .scratch/cptac_validation/results/predictions_fusion_ablate

echo ""
echo "[Phase 8] Tier-2 sensitivity: FSQN onto the Xena scale..."
python tools/rna/fsqn_harmonize.py
python tools/cptac/infer_cptac_multimodal.py \
    --tabular_csv .scratch/rna-gdc/CPTAC_BRCA_RNA_fsqn_xena_clam.csv.gz \
    --ckpt_dir .scratch/results/pam50_wsi_rna_gatedfusion_s1 \
    --output_dir .scratch/cptac_validation/results/predictions_fusion_fsqn

echo ""
echo "=============================================="
echo "CPTAC pipeline complete!"
echo "Results: .scratch/cptac_validation/results/"
echo "Report:  tools/evaluate_external.ipynb"
echo "=============================================="
