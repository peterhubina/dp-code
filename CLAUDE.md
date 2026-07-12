# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Project Overview

Computational-pathology ML pipeline centered on **TCGA-BRCA**. The active work is:
- **PAM50 molecular-subtype classification** (4-class) from whole-slide images, using
  UNI2-h patch features fed to CLAM multiple-instance learning (MIL).
- **Multimodal fusion**: WSI (CLAM) + RNA-seq MLP, with a gated fusion head.
- **Overall-survival prediction** with an attention-MIL (AMIL) discrete-time model.
- A **CTC cohort** thread (internal name `nou`) reusing the same patch → feature → MIL flow.

A separate, older thread uses the **HistologyHSI-BC-Recurrence** dataset (47 WSI `.mrxs`,
677 hyperspectral ENVI images, clinical data) for distant-recurrence prediction. It still
lives in the repo (`tools/hsi_bc/`, `.datasets/HistologyHSI-BC-Recurrence/`,
`.scratch/hsi_bc_recurrence/`) but is not the primary pipeline.

## Layout

```
project/
├── data/            # Active data pipeline (patching + feature datamodules)
│   ├── patch_extractor.py, geometry_utils.py, patch_dataset.py
│   ├── feature_datamodule.py, transforms.py
│   └── run_patching.sh, pam50.R
├── survival/        # AMIL survival package (dataset, model, trainer, losses, splits, evaluate)
├── loggers/         # checkpointer.py
├── CLAM/            # Vendored CLAM (git-tracked copy, own main.py — see below)
├── MCAT/            # Vendored MCAT (multimodal survival transformer, own main.py)
├── UNI/             # Vendored UNI feature extractor (NOT git-tracked; must be present locally)
└── base/            # Legacy MNIST scaffold — see "Legacy" section

tools/
├── extract_features.py         # UNI2-h → .pt feature cache (1536-dim)
├── create_clam_dataset_csv.py  # Build CLAM dataset_csv manifests
├── fetch_pam50_labels.py, fetch_tcga_labels.py, download_embeddings.py
├── train_pam50_final.sh, train_pam50_tabular.sh, train_pam50_multimodal.sh
├── evaluate_pam50_multimodal.sh, evaluate_pam50.ipynb, evaluate_external.ipynb
├── train_survival.py, eval_survival.py     # Hydra-driven; config in config/survival.yaml
├── rna/            # RNA-seq download + preprocessing (download-rna.py, prepare-rna-wsi-classification.py, eda.ipynb)
├── nou/            # CTC cohort pipeline (patch → feature → infer scripts + run_pipeline*.sh)
├── hsi_bc/         # HistologyHSI-BC recurrence pipeline (prepare_manifest, infer_pam50, run_pipeline.sh)
├── data/           # Label CSVs (tcga_brca_pam50_labels.csv, tcga_brca_os_labels.csv, tcga_brca_labels.csv)
├── preprocessing/, visualisation/  # Notebooks
└── config/         # Hydra configs (only the survival path is live — see "Hydra")
```

## Entry Points

### 1. Patch extraction (WSI → tiles)
`project/data/run_patching.sh` drives `project/data/patch_extractor.py` /
`patch_dataset.py`. Tiles are written per-slide with a `metadata.csv` index.

### 2. Feature extraction (tiles → UNI2-h embeddings)
```bash
python tools/extract_features.py \
    --patch_root .datasets/patches \
    --output .scratch/datasets/features.pt \
    --assets_dir .scratch/checkpoints \
    --batch_size 64 --num_workers 4 --encoder uni2-h
```
Scans slide dirs containing `metadata.csv`, loads UNI2-h (needs `project/UNI` present and
weights in `--assets_dir`), and writes a `.pt` with `embeddings` `[N, 1536]`, `labels`,
`slide_ids`, and `label_map`. CLAM also ships its own `extract_features_fp.py`.

### 3. CLAM MIL training (subtyping / recurrence)
Run from inside the vendored repo. Valid `--task` values are defined in
`project/CLAM/main.py` (`tcga_brca_subtyping`, `tcga_brca_recurrence`, `nou_ctc_*`, …),
each mapping to a `dataset_csv/*.csv` manifest.
```bash
cd project/CLAM
python create_splits_seq.py --task tcga_brca_subtyping --seed 1 --k 10 --val_frac 0.15 --test_frac 0.0
python main.py --task tcga_brca_subtyping --model_type clam_sb --exp_code pam50 \
    --k 10 --lr 2e-4 --drop_out 0.25 --early_stopping --weighted_sample \
    --bag_loss ce --inst_loss svm --embed_dim 1536 --log_data \
    --data_root_dir /workspace/dp-code/.datasets/embeddings \
    --results_dir /workspace/dp-code/.scratch/experiments/clam
```
Convenience wrappers: `tools/train_pam50_final.sh` (WSI-only), `tools/train_pam50_tabular.sh`
(RNA-only), `tools/train_pam50_multimodal.sh` (WSI CLAM-MB + RNA MLP gated fusion),
`tools/evaluate_pam50_multimodal.sh`.

### 4. Survival (AMIL, Hydra)
```bash
python tools/train_survival.py                       # 5-fold CV, mean±std c-index, W&B
python tools/train_survival.py exp.name=my_exp exp.ver=v2 training.max_epochs=30
python tools/train_survival.py wandb.enabled=false
python tools/eval_survival.py --exp_dir .scratch/experiments/amil_surv_baseline/v1
```
Implemented in `project/survival/` (`SurvivalExperiment`, `AMIL_Surv`, `nll_surv` loss,
stratified k-fold splits). MCAT (`project/MCAT/main.py`) is a separate vendored multimodal
survival model.

## Hydra

Only the **survival** config path is live and importable:
- `tools/config/survival.yaml` → `dataset: tcga_brca_survival`, `model: amil_surv`
  (`_target_: project.survival.model.AMIL_Surv`).
- `hydra.run.dir = .scratch/logs/${exp.name}/${exp.ver}`.
- Override with dot notation: `training.learning_rate=0.001 exp.ver=v2`.

`tools/main.py` just prints the resolved `default.yaml`. Do **not** rely on `default.yaml`
or its `dataset: prostate` / `model: timm` / `augmentation: basic` groups — their `_target_`s
point at a `toyproblem.*` module that does not exist in this repo.

## Data & Output Locations

- `.datasets/tcga-brca/` — TCGA-BRCA WSIs, patches, and `embeddings/`
- `.datasets/HistologyHSI-BC-Recurrence/` — HSI recurrence raw data
- `.datasets/nou/` — CTC cohort data
- `tools/data/*.csv` — TCGA-BRCA label tables (PAM50, OS, clinical)
- `.scratch/experiments/`, `.scratch/results/` — training runs and checkpoints
- `.scratch/splits/`, `.scratch/checkpoints/`, `.scratch/logs/`, `.scratch/TCGA-BRCA-rna/`

## Conventions

- **Patient/slide-level splitting**: keep all patches from a slide in the same fold to avoid
  spatial leakage. CLAM handles this via its split CSVs; custom scripts replicate it.
- **UNI2-h embeddings are 1536-dim** — pass `--embed_dim 1536` / `input_dim: 1536`.
- **PAM50 subtyping is 4-class**; survival uses `n_bins: 4` discrete time bins.
- **Vendored repos** (`CLAM`, `MCAT`, `UNI`) keep their own entry points, requirements, and
  conventions. Run CLAM/MCAT commands from inside their directories. `CLAM` and `MCAT` are
  committed into this repo; `UNI` is not tracked and must exist locally for feature extraction.
- W&B tracking is on by default for survival (`wandb.enabled`, project `dp-survival`) and for
  the PAM50 shell scripts; disable per-run when experimenting.

## Legacy

`project/base/` (MNIST-style `Experiment`/`Trainer`/`datamodule`/`model`) plus
`tools/train.py`, `tools/main.py`, and the `prostate`/`timm`/`basic` Hydra groups are
scaffold from an earlier template. `tools/train.py` is currently broken — it imports
`project.experiment.Experiment` and the config targets `project.trainer.Trainer`, neither of
which exists (the classes live under `project/base/`). Don't build on this path; use the
entry points above.

## Environment

```bash
pip install -r requirements.txt      # includes torch, timm, openslide, spectral, hydra-core, wandb
cd docker && ./build.sh && ./run.sh  # containerized alternative
```
CLAM and UNI have their own `requirements.txt` / `env.yml` for their extra deps.
