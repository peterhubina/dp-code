# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a digital pathology machine learning pipeline for breast cancer recurrence prediction using multimodal data. The project integrates:
- Whole-slide images (WSI) in .mrxs format
- Hyperspectral imaging (HSI) data in ENVI format (.hdr/.dat files)
- Clinical and demographic data

The dataset contains 47 WSI, 677 HS images, and clinical data from 47 BC patients (22 experienced distant recurrence over 12-year follow-up).

## Architecture

### Configuration System (Hydra)

The project uses **Hydra** for hierarchical configuration management:
- Main config: `tools/config/default.yaml`
- Config groups organized in subdirectories:
  - `dataset/`: Dataset configurations (e.g., `prostate.yaml`)
  - `model/`: Model architectures (e.g., `timm.yaml`)
  - `augmentation/`: Data augmentation pipelines (e.g., `basic.yaml`)
- Logs and outputs are written to `.scratch/logs/${exp.name}/${exp.ver}/`
- Use `hydra.main` decorator with `config_path="config"` and `config_name="default"`

### Project Structure

```
project/
├── base/           # Core framework components (legacy MNIST example code)
│   ├── experiment.py   # Experiment class managing training lifecycle
│   ├── trainer.py      # Training loop implementation with GPU support
│   ├── datamodule.py   # DataLoader setup (legacy MNIST example)
│   ├── model.py        # Simple MLP model (legacy example)
│   ├── logging.py      # Logging system (CSVLog, ReportCompiler, ModelCheckpointer)
│   └── utils.py        # Statistics tracking utilities
├── data/           # Data-related modules (new pathology pipeline)
│   └── transforms.py   # Custom transforms (placeholder)
└── loggers/        # Logging utilities (new pathology pipeline)
    └── checkpointer.py # Model checkpointing (placeholder)

tools/
├── config/         # Hydra configuration files
├── preprocessing/  # Jupyter notebooks for data preprocessing
│   ├── main.ipynb                  # Load WSI/HS data
│   └── overlay_tissue_areas.ipynb  # Overlay tissue compartments
├── train.py        # Training script (uses project.experiment.Experiment)
├── eval.py         # Evaluation script (placeholder)
└── infer.py        # Inference script (placeholder)
```

**Important:** The codebase contains legacy MNIST example code in `project/base/` that demonstrates the framework structure. New pathology-specific implementations should go in `project/data/` and use Hydra configs in `tools/config/`.

### Data Pipeline

1. **WSI Processing:** Uses OpenSlide for reading whole-slide images (.mrxs format)
2. **HSI Processing:** Uses Spectral Python (SPy) for hyperspectral data (ENVI .hdr/.dat files)
   - Each HSI sample has: calibrated, raw, darkReference, whiteReference, RGBImage, SyntheticRGBImage
3. **Annotations:** GeoJSON format for tissue compartment annotations
4. **Clinical Data:** Integrated with image data for multimodal learning

### Training Framework

The framework follows a structured experiment pattern:

1. **Experiment** (`project/base/experiment.py`):
   - Manages experiment lifecycle
   - Creates model, datamodule, trainer
   - Saves config and checkpoints to `.scratch/experiments/${name}/${ver}/`
   - Supports loading from checkpoints via `Experiment.from_folder()`

2. **Trainer** (`project/base/trainer.py`):
   - Implements training/validation loops
   - Auto-detects device (CUDA/CPU)
   - Uses tqdm for progress bars
   - Integrates with logging system via `LogCompose`

3. **Logging** (`project/base/logging.py`):
   - `CSVLog`: Writes metrics to CSV
   - `ReportCompiler`: Generates PDF reports with matplotlib plots
   - `ModelCheckpointer`: Saves model and optimizer state
   - All loggers implement common interface: `on_training_start`, `on_epoch_complete`, `on_training_stop`

### Model Instantiation via Hydra

Models are instantiated using Hydra's `_target_` directive:
- `_target_: toyproblem.models.simple_timm.SimpleTimm` creates model from config
- Supports arbitrary model architectures via import path
- Config parameters passed as kwargs to model constructor

## Common Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Docker build
cd docker && ./build.sh

# Docker run
cd docker && ./run.sh
```

### Training
```bash
# Train with default config
python tools/train.py

# Train with custom experiment name/version
python tools/train.py --name my_experiment --ver v1

# Train with custom hyperparameters
python tools/train.py --batch_size 32 --max_epochs 50 --learning_rate 0.001

# Note: Legacy train.py uses argparse. Hydra-based training would use:
# python tools/train.py exp.name=my_exp exp.ver=v1 common.batch_size=32
```

### Testing Hydra Configuration
```bash
# Print resolved config
python tools/main.py

# Override config groups
python tools/main.py dataset=prostate model=timm

# Override specific parameters
python tools/main.py common.batch_size=32 common.learning_rate=0.0001
```

### Data Preprocessing
```bash
# Launch Jupyter for preprocessing notebooks
jupyter notebook tools/preprocessing/
```

## Key Dependencies

- **PyTorch**: Deep learning framework with Lightning integration
- **OpenSlide**: WSI reading library (requires openslide-bin)
- **Spectral Python (SPy)**: Hyperspectral image processing
- **Hydra**: Configuration management (v1.3.2)
- **WandB**: Experiment tracking (optional)
- **torchvision**: Standard computer vision transforms and datasets
- **timm**: PyTorch Image Models library for pretrained backbones

## Important Patterns

### Hydra Config Override
Use dot notation to override nested configs:
```bash
python script.py common.batch_size=64 model.backbone_name=resnet50
```

### Device Management
The trainer automatically selects GPU if available via `decide_device()` in `project/base/trainer.py`. MPS (Apple Silicon) support is commented out.

### Checkpoint Format
Checkpoints contain both model and optimizer state:
```python
{
    "model": model.state_dict(),
    "opt": optimizer.state_dict()
}
```

### Dataset Paths
- Raw data: `.datasets/PKG - HistologyHSI-BC-Recurrence/`
- Processed data: `.scratch/datasets/`
- Experiments: `.scratch/experiments/${name}/${ver}/`
- Hydra logs: `.scratch/logs/${exp.name}/${exp.ver}/`

## Data Format Specifics

### HSI Data Structure
Each hyperspectral sample contains:
- `calibrated.hdr/dat`: Calibrated hyperspectral cube
- `raw.hdr/dat`: Raw measurements
- `darkReference.hdr/dat`: Dark reference for calibration
- `whiteReference.hdr/dat`: White reference for calibration
- `RGBImage.png`: Standard RGB representation
- `SyntheticRGBImage.png`: RGB synthesized from HS data

### Sample Organization
```
.datasets/PKG - HistologyHSI-BC-Recurrence/02_01_HS_Images/
├── IDC/
│   ├── HS_VNIR_90_IDC_x10_C01/
│   ├── HS_VNIR_90_IDC_x10_C02/
│   └── ...
```
