#!/bin/bash
# NOU External Validation Pipeline - Track A (PAM50)
# ===================================================
# End-to-end orchestration script for validating the UNI2-h + CLAM-MB
# PAM50 subtyping model on the private NOU breast cancer dataset.
#
# Labels are assigned at the patient level (majority vote across tissue
# blocks) and patches are extracted at 448px to match TCGA 20x @ 224px.
#
# Prerequisites:
#   - NOU TIFF images at .datasets/nou/crops - Copy/
#   - Annotation Excel at .datasets/nou/Anotacia_CTC_blocky.xls
#   - UNI2-h weights at .scratch/checkpoints/uni2-h/pytorch_model.bin
#   - Trained CLAM checkpoints at .scratch/results/pam50_final_s1/
#
# Usage:
#   bash tools/nou/run_pipeline.sh
#
# Each phase can also be run independently:
#   python tools/nou/prepare_nou_manifest.py
#   python tools/nou/patch_nou_crops.py --output_dir .scratch/nou_validation/patches_448
#   python tools/nou/extract_nou_features.py
#   python tools/nou/infer_nou_pam50.py

set -e
cd "$(dirname "$0")/../.."

PATCH_DIR=".scratch/nou_validation/patches_448"
FEATURE_DIR=".scratch/nou_validation/features_448"
RESULT_DIR=".scratch/nou_validation/results_448/predictions"

echo "=========================================="
echo "NOU External Validation Pipeline - Track A"
echo "=========================================="

echo ""
echo "[Phase 1] Preparing data manifest (patient-level labels)..."
python tools/nou/prepare_nou_manifest.py

echo ""
echo "[Phase 2] Patching tissue crops into 448px tiles (40x->20x equiv)..."
python tools/nou/patch_nou_crops.py --output_dir "$PATCH_DIR"

echo ""
echo "[Phase 3] Extracting UNI2-h features..."
python tools/nou/extract_nou_features.py --patch_dir "$PATCH_DIR" --output_dir "$FEATURE_DIR"

echo ""
echo "[Phase 4A] Running CLAM-MB PAM50 inference (10-fold ensemble)..."
python tools/nou/infer_nou_pam50.py \
    --feature_dir "$FEATURE_DIR/h5_files" \
    --output_dir "$RESULT_DIR"

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Results: $RESULT_DIR"
echo "=========================================="
