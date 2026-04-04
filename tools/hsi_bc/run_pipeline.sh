#!/bin/bash
# HSI-BC External Validation Pipeline (PAM50)
# =============================================
# Processes HistologyHSI-BC-Recurrence .mrxs WSIs through
# CLAM patching + UNI2-h feature extraction + CLAM-MB inference,
# matching the TCGA-BRCA feature extraction protocol.
#
# Prerequisites:
#   - Dataset at .datasets/HistologyHSI-BC-Recurrence/
#   - UNI2-h weights: set UNI2H_CKPT_PATH env var
#     (e.g. export UNI2H_CKPT_PATH=.scratch/checkpoints/uni2-h/pytorch_model.bin)
#   - Trained CLAM checkpoints at .scratch/results/pam50_final_s1/
#
# Usage:
#   bash tools/hsi_bc/run_pipeline.sh

set -e
cd "$(dirname "$0")/../.."

WSI_DIR=".datasets/HistologyHSI-BC-Recurrence/01_01_Histological_Images"
CLAM_OUTPUT=".scratch/hsi_bc_recurrence/clam_output"
FEAT_DIR=".scratch/hsi_bc_recurrence/features"
RESULT_DIR=".scratch/hsi_bc_recurrence/results/predictions"
CLAM_CSV="project/CLAM/dataset_csv/hsi_bc_pam50.csv"

echo "============================================="
echo "HSI-BC External Validation Pipeline (PAM50)"
echo "============================================="

# -----------------------------------------------
# Phase 1: Prepare manifest & dataset CSV
# -----------------------------------------------
echo ""
echo "[Phase 1] Preparing manifest and CLAM dataset CSV..."
python tools/hsi_bc/prepare_manifest.py

# -----------------------------------------------
# Phase 2: CLAM patching (.mrxs, 256x256 @ level 0)
# -----------------------------------------------
echo ""
echo "[Phase 2] CLAM tissue segmentation + patching..."
cd project/CLAM
python create_patches_fp.py \
    --source "../../${WSI_DIR}" \
    --save_dir "../../${CLAM_OUTPUT}" \
    --patch_size 256 \
    --step_size 256 \
    --seg --patch --stitch \
    --preset tcga.csv
cd ../..

# -----------------------------------------------
# Phase 3: UNI2-h feature extraction
# -----------------------------------------------
echo ""
echo "[Phase 3] Extracting UNI2-h features..."
if [ -z "$UNI2H_CKPT_PATH" ]; then
    echo "ERROR: UNI2H_CKPT_PATH not set."
    echo "  export UNI2H_CKPT_PATH=.scratch/checkpoints/uni2-h/pytorch_model.bin"
    exit 1
fi
cd project/CLAM
python extract_features_fp.py \
    --data_h5_dir "../../${CLAM_OUTPUT}" \
    --data_slide_dir "../../${WSI_DIR}" \
    --csv_path "../../${CLAM_OUTPUT}/process_list_autogen.csv" \
    --feat_dir "../../${FEAT_DIR}" \
    --model_name uni2-h \
    --batch_size 256 \
    --slide_ext .mrxs \
    --target_patch_size 224
cd ../..

# -----------------------------------------------
# Phase 4: CLAM-MB PAM50 inference (10-fold ensemble)
# -----------------------------------------------
echo ""
echo "[Phase 4] Running CLAM-MB PAM50 inference..."
python tools/hsi_bc/infer_pam50.py \
    --feature_dir "${FEAT_DIR}/h5_files" \
    --dataset_csv "${CLAM_CSV}" \
    --output_dir "${RESULT_DIR}"

echo ""
echo "============================================="
echo "HSI-BC Pipeline complete!"
echo "Results: ${RESULT_DIR}"
echo "============================================="
