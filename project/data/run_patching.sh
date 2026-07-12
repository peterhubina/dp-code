#!/bin/bash
# Batch extract patches from all WSIs at 20x magnification

python tools/patch_wsi.py batch \
    --wsi_dir ".datasets/PKG - HistologyHSI-BC-Recurrence/01_01_Histological_Images" \
    --geojson_dir ".datasets/PKG - HistologyHSI-BC-Recurrence/01_02_Tissue_Annotations" \
    --out_dir .datasets/wsi_patches \
    --patch_size 224 \
    --level 0 \
    --overlap 0.0 \
    --min_tissue 0.5
