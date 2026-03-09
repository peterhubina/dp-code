# Histological Hyperspectral Breast Cancer Recurrence Database (HistologyHSI-BC Recurrence)
Metastasis occurs in nearly 1 out of 3 breast cancer (BC) patients and significantly reduces survival rates, particularly in cases of distant metastases. As most distant metastases develop after diagnosis (i.e., recurrence) and remain incurable, there is a critical need for prognostic biomarkers to assess recurrence risk. Multimodal data analysis has emerged as a promising approach to integrate diverse information, offering a more comprehensive perspective. This study introduces the Histology HSI-BC (hyperspectral imaging - breast cancer) Recurrence Database, the first publicly accessible multimodal database designed to advance BC distant recurrence prediction. This database provides a promising resource for studying BC recurrence prediction and personalized treatment strategies by integrating the aforementioned multimodal data.

## Dataset 

The database comprises 47 histopathological whole-slide images (WSI), 677 hyperspectral (HS) images, and clinical and demographic data from 47 BC patients, of whom 22 (47%) experienced distant recurrence over a 12-year follow-up. Histopathological slides were digitized using a whole-slide scanner and annotated by expert pathologists, while HS images were acquired with an HS camera coupled to a bright-field microscope.

More information about the dataset can be found on:

## Usage

This repository contains the following scripts:
* `main.ipynb`: provide a basic example of how to load and perform some basic preprocessing to WSI (.mrxs) and HS (ENVI format) data using Python.
* `overlay_tissue_areas.ipynb`: provide a tutorial on manipulating annotations in GeoJSON format to overlay tissue compartments on a WSI using Python.
## Dependencies

Python script requires:
   - Openslide. Python module for reading whole-slide image formats. https://openslide.org/  
   - Spectral Python (SPy). Python module for hyperspectral image processing. https://www.spectralpython.net
   - Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). https://doi.org/10.1038/s41586-020-2649-2
   - J. D. Hunter, "Matplotlib: A 2D Graphics Environment," in Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, May-June 2007, doi: 10.1109/MCSE.2007.55.
   - Virtanen, P., Gommers, R., Oliphant, T.E. et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Methods 17, 261–272 (2020). https://doi.org/10.1038/s41592-019-0686-2


## License

Copyright 2025 Laura Quintana-Quintana, Esther Sauras-Colón, Javier Santana-Nunez, Alessio Fiorin

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

## Commands

Patching:

python tools/patch_wsi.py batch --wsi_dir ".datasets/PKG - HistologyHSI-BC-Recurrence/01_01_Histological_Images" --geojson_dir ".datasets/PKG - HistologyHSI-BC-Recurrence/01_02_Tissue_Annotations" --out_dir .datasets/wsi_patches --patch_size 224 --level 0 --overlap 0.0 --min_tissue 0.5


 Step 1: Extract Features (one-time)

python tools/extract_features.py --patch_root /mnt/datasets/wsi_patches --output mnt/datasets/uni2h_insitu_vs_infiltrant.pt --assets_dir .scratch/checkpoints --batch_size 256 --num_workers 8

  python tools/extract_features.py \
      --patch_root .datasets/patches \
      --output .scratch/datasets/uni2h_insitu_vs_infiltrant.pt \
      --assets_dir .scratch/checkpoints \
      --batch_size 64 \
      --num_workers 4

  This scans all slide directories under .datasets/patches/, loads UNI2-h, and extracts 1536-dim features. Output .pt contains:
  - embeddings: [N, 1536] tensor
  - labels: [N] tensor (0 = in situ, 1 = infiltrant)
  - slide_ids: list of slide IDs (for patient-level splitting)
  - label_map: class names

  Expected output for slide 25 alone: ~8,635 samples (864 in situ, 7,771 infiltrant)

  ---
  Step 2: Train MLP Classifier

  python tools/train_classifier.py --name insitu_vs_infiltrant --ver v1 --features_path mnt/datasets/uni2h_insitu_vs_infiltrant.pt --batch_size 256 --max_epochs 50 --learning_rate 0.001 --hidden_dim 256 --dropout 0.3 --val_split 0.2

  What this does:
  - Loads cached features from Step 1
  - Patient-level split: all patches from each slide stay together (prevents spatial leakage)
  - Computes class weights: infiltrant patches are downweighted to balance training
  - Trains for 50 epochs with Adam + weighted CrossEntropyLoss
  - Tracks: loss_train, loss_val, bacc_train, bacc_val
  - Saves checkpoints, CSV metrics, and PDF report

  Output directory: .scratch/experiments/insitu_vs_infiltrant/v1/
  - config.yaml: training parameters
  - training.csv: metrics per epoch
  - report.pdf: loss/accuracy plots
  - checkpoints/: model weights

  ---
  Quick Test (Slide 25 only)

  # Extract features from slide 25
  python tools/extract_features.py \
      --patch_root .datasets/patches \
      --output .scratch/datasets/test_features.pt \
      --batch_size 32

  # Train on those features
python tools/train_classifier.py --features_path .datasets/embeddings/uni2h_insitu_vs_infiltrant.pt --max_epochs 10 --name test_clf --ver v1

  python tools/train_classifier.py \
      --features_path .scratch/datasets/test_features.pt \
      --max_epochs 10 \
      --name test_clf --ver v1

  This is faster for testing before scaling to all 47 slides.

  ---
  Expected Behavior

  1. Feature extraction: Progress bar shows ~135 batches for slide 25 (8635 ÷ 64)
  2. Training: Each epoch shows train/val loss decreasing, balanced accuracy increasing from ~50%
  3. Class imbalance handling: Weighted loss ensures the model doesn't just predict all infiltrant patches
  4. Patient-level split: Train/val split is done at slide level (e.g., slide 25 might go entirely to train, future slides split 80/20)

  ---
  Scaling to All 47 Slides

  Once you have all 47 WSIs patched in .datasets/patches/{1,2,...,47}/:

  # Single feature extraction run (processes all slides at once)
  python tools/extract_features.py \
      --patch_root .datasets/patches \
      --output .scratch/datasets/uni2h_all47_insitu_vs_infiltrant.pt \
      --batch_size 64

  # Train on full dataset
  python tools/train_classifier.py \
      --features_path .scratch/datasets/uni2h_all47_insitu_vs_infiltrant.pt \
      --max_epochs 50 \
      --name insitu_vs_infiltrant_all47 --ver v1