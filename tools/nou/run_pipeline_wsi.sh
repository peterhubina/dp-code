#!/bin/bash
# NOU External Validation Pipeline - WSI Track (PAM50)
# =====================================================
# Uses full SVS whole-slide images at 20x with 224px patches,
# matching the TCGA-BRCA feature extraction protocol exactly.
#
# Prerequisites:
#   - NOU SVS files at .datasets/nou/data/
#   - Patient-level manifest at .scratch/nou_validation/metadata/nou_slide_manifest.csv
#   - UNI2-h weights at .scratch/checkpoints/uni2-h/pytorch_model.bin
#   - Trained CLAM checkpoints at .scratch/results/pam50_final_s1/
#
# Usage:
#   bash tools/nou/run_pipeline_wsi.sh

set -e
cd "$(dirname "$0")/../.."

PATCH_DIR=".scratch/nou_validation/patches_wsi"
FEATURE_DIR=".scratch/nou_validation/features_wsi"
RESULT_DIR=".scratch/nou_validation/results_wsi/predictions"
DATASET_CSV=".scratch/nou_validation/metadata/nou_pam50_wsi_dataset.csv"

echo "============================================="
echo "NOU External Validation Pipeline - WSI Track"
echo "============================================="

echo ""
echo "[Phase 1] Manifest should already exist from crops pipeline."
echo "          Run 'python tools/nou/prepare_nou_manifest.py' first if needed."

echo ""
echo "[Phase 2-WSI] Patching WSIs into 224x224 tiles at 20x..."
python tools/nou/patch_nou_wsi.py --output_dir "$PATCH_DIR"

echo ""
echo "[Phase 3] Extracting UNI2-h features..."
python tools/nou/extract_nou_features.py --patch_dir "$PATCH_DIR" --output_dir "$FEATURE_DIR"

echo ""
echo "[Phase 4A] Running CLAM-MB PAM50 inference (10-fold ensemble)..."
python tools/nou/infer_nou_pam50.py \
    --feature_dir "$FEATURE_DIR/h5_files" \
    --dataset_csv "$DATASET_CSV" \
    --output_dir "$RESULT_DIR"

echo ""
echo "============================================="
echo "WSI Pipeline complete!"
echo "Results: $RESULT_DIR"
echo "============================================="
