#!/bin/bash
# CPTAC-BRCA External Validation Pipeline (PAM50)
# ===============================================
# Frozen-weights evaluation of the TCGA-BRCA CLAM-MB ensemble on CPTAC-BRCA.
# Features come pre-computed from HuggingFace (MahmoodLab/UNI2-h-features,
# CPTAC/cptac_brca.tar.gz) -- same Trident 20x/256px/0px + UNI2-h pipeline that
# produced the TCGA training features, so preprocessing is held constant.
#
# Prerequisites:
#   - HF token with access to MahmoodLab/UNI2-h-features (gated dataset)
#   - CPTAC cohort metadata at .datasets/cptac-brca/{wsi_manifest,cohort}.csv
#   - Trained CLAM checkpoints at .scratch/results/pam50_final_s1/
#
# Usage:
#   bash tools/cptac/run_pipeline.sh

set -e
cd "$(dirname "$0")/../.."

FEATURE_DIR=".datasets/cptac-brca/embeddings"
DATASET_CSV=".datasets/cptac-brca/cptac_brca_pam50_dataset.csv"
RESULT_DIR=".scratch/cptac_validation/results/predictions"

echo "=============================================="
echo "CPTAC-BRCA External Validation Pipeline"
echo "=============================================="

echo ""
echo "[Phase 1-2a] Downloading + extracting UNI2-h features..."
python tools/download_embeddings.py --cohort cptac-brca

echo ""
echo "[Phase 1] Auditing feature provenance against the TCGA training config..."
python tools/cptac/audit_feature_provenance.py --feature_dir "$FEATURE_DIR"

echo ""
echo "[Phase 2a] Reconciling coverage and building CLAM manifest..."
python tools/cptac/prepare_cptac_manifest.py \
    --feature_dir "$FEATURE_DIR" \
    --dataset_csv "$DATASET_CSV"

echo ""
echo "[Phase 3] Running CLAM-MB PAM50 inference (10-fold ensemble)..."
python tools/cptac/infer_cptac_pam50.py \
    --feature_dir "$FEATURE_DIR" \
    --dataset_csv "$DATASET_CSV" \
    --output_dir "$RESULT_DIR"

echo ""
echo "[Phase 4] Summarising slide- and case-level metrics..."
python tools/cptac/summarise_predictions.py "$RESULT_DIR"

# ---------------------------------------------------------------- multimodal
# Everything below needs the RNA side. Tier 1 re-derives TCGA RNA from the same
# GDC STAR-Counts pipeline that produced the CPTAC files, so the two cohorts share
# a gene axis and no cross-cohort normalisation is applied. Tier 2 keeps the
# Xena-trained fusion head and maps CPTAC onto its scale with FSQN instead.

echo ""
echo "[Phase 5] Building the shared GDC expression tables (Tier 1)..."
python tools/rna/download_gdc_rna.py
python tools/rna/build_gdc_expression.py
python tools/rna/make_case_splits.py

echo ""
echo "[Phase 6] Retraining the RNA branch on GDC-derived TCGA RNA..."
echo "          (WSI branch stays frozen at pam50_final_s1; ~1 min RNA-only, ~45 min fusion)"
( cd project/CLAM && python train_rna.py \
    --data_path "$PWD/.scratch/rna-gdc/TCGA_BRCA_RNA_gdc_4class_clam.csv.gz" \
    --split_dir tcga_brca_subtyping_100_case --no_auto_splits \
    --exp_code pam50_rna_only_gdc --results_dir "$PWD/.scratch/results" \
    --k 10 --top_n_genes 10000 --class_set 4class --seed 1 \
    --early_stopping --weighted_sample )
bash tools/train_pam50_multimodal.sh \
    --pretrained_wsi_ckpt "$PWD/.scratch/results/pam50_final_s1/s_{fold}_checkpoint.pt" \
    --tabular_csv "$PWD/.scratch/rna-gdc/TCGA_BRCA_RNA_gdc_4class_clam.csv.gz" \
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
