# dp-code — PAM50 subtypes from H&E slides fused with arm-level copy number

Four-class PAM50 molecular-subtype classification (LumA / LumB / Basal / Her2), **trained on
TCGA-BRCA and externally validated on CPTAC-BRCA**, fusing H&E whole-slide images with arm-level
copy-number variation.

- **Primary modality — H&E WSI.** UNI2-h patch features (1536-dim) → CLAM-MB multiple-instance
  learning, 10-fold cross-validation on TCGA-BRCA.
- **Second modality — arm-level CNV.** 39 chromosome arms, each the median gene-level log2 over that
  arm. Chosen because shallow whole-genome sequencing resolves changes at arm scale, so the assay is
  cheap and clinically reachable — and because copy number is a different assay from the one the
  PAM50 labels are computed from, so unlike RNA it cannot leak the target.
- **External cohort — CPTAC-BRCA**, 114 cases / 378 slides. Nothing is ever refit, tuned, calibrated
  or thresholded on CPTAC; fusion rules are fixed on TCGA before the external set is scored.

Master's-thesis codebase. It is research code that has been made reproducible, not a library.

## The headline result

TCGA-trained, CPTAC external, n = 114. Full write-up, every control and every caveat:
[`docs/cnv-wsi-fusion-external-validation.md`](docs/cnv-wsi-fusion-external-validation.md).

| Model | macro AUROC [95% CI] | balanced acc | Her2 recall |
|---|---|---|---|
| WSI only (CLAM-MB + UNI2-h) | 0.847 [0.791, 0.895] | 0.513 | 0/14 |
| **CNV only (39 arms, logistic regression)** | **0.888** [0.835, 0.933] | **0.716** | **12/14** |
| Fusion — equal-weight probability mean | **0.909** [0.858, 0.948] | 0.646 | 6/14 |
| Fusion — mean of prior-balanced WSI + CNV *(post hoc control)* | **0.912** | **0.740** | 10/14 |

Two things this table is required to say out loud, because leaving either out is the selective
reporting this project's own literature survey criticises:

1. **The CNV-alone arm is always reported next to fusion.** Fusion's edge over CNV alone is marginal
   — ΔAUROC +0.024 with a CI lower bound of exactly +0.000, and no significant balanced-accuracy
   difference at n = 114.
2. **The baseline to beat is the equal-weight mean, not the WSI-only model.** Five trained fusion
   operators (`concat`, `gated`, `cross_attention`, `film_attention`, `coattn`) were run as a ladder
   on identical splits; every one of them loses to the untrained average, and ensembling all five
   still loses to two independently trained unimodal models. The mechanism is error correlation:
   φ = 0.656 among the jointly trained operators against 0.193 between the two unimodal arms.

Recompute these rows (≈45 s, CPU only, no slide touched) with

```bash
dp-analysis cnv_wsi_fusion                       # TCGA -> CPTAC external
dp-analysis cnv_wsi_fusion analysis.internal=true  # adds the TCGA-only head-to-head
```

…once you have four small input files. Getting them is the subject of
**[REPRODUCING.md](REPRODUCING.md)**, and it is the first thing to read.

## Install

Python **3.10 or 3.11** (`torch==2.0.1` has no 3.12 wheels), an NVIDIA driver new enough for CUDA
11.7 if you intend to train, and `git` on the box (one dependency, `topk`, is a git commit pin with
no PyPI fallback and is required by the frozen WSI baseline's `--inst_loss svm`).

```bash
git clone https://github.com/peterhubina/dp-code.git
cd dp-code
pip install -e '.[dev]'
```

Editable is not a preference. A non-editable `pip install .` copies `project/` and `tools/` into
site-packages **without** the data directories they reach by relative path
(`project/CLAM/dataset_csv/`, `project/CLAM/splits/`, `tools/data/`), so the copies import and then
read the wrong files, or none.

Check the install before downloading anything:

```bash
dp-config validate    # paths, tracked inputs, CLAM-schema drift
dp-config validate experiment=pam50_wsi_final   # ...plus the topk dependency `--inst_loss svm` needs
dp-config sync-check  # ClamConf still matches CLAM's real 52-flag parser
```

Naming an experiment composes `conf/train.yaml`, the same tree `dp-train` runs, and is what reaches
the experiment-specific checks — `topk` is only required by `clam.inst_loss=svm`, which only the
frozen WSI baseline sets.

`dp-config validate` on a fresh clone prints a `not acquired :` line listing the data trees you do
not have yet. That is expected, not a failure.

**There is no automated test suite.** The repository ships no `pytest` suite and no smoke run, so
`dp-config validate` and `dp-config sync-check` are the whole of the automated checking available to
you. In particular, nothing verifies that the configuration reproduces the original shell wrappers —
that equivalence was established once, by comparing parsed CLAM argument namespaces for every
wrapper, but it is not re-checked on any change. Treat edits to `dpcode/conf/experiment/` and
`dpcode/conf/clam/base.yaml` accordingly: compare `dp-train --dry-run` output against the frozen
wrappers in `tests/legacy_wrappers/tools/`, which are kept byte-identical to their pre-refactor form
for exactly that purpose.

**Docker is not a supported route.** `docker/` targets one institution's cluster: the base image's
availability is unverified, `docker/run.sh` bind-mounts host paths that will not exist on your
machine, and it requires a positional GPU argument. Use the pip install.

## Entry points

Six console scripts, all runnable **from any working directory** after `pip install -e .`. Every
path and every settable parameter comes from the Hydra config tree in `dpcode/conf/`; nothing is
hard-coded and nothing is relative to where you happen to stand.

| command | what it does | example |
|---|---|---|
| `dp-train` | composes a CLAM training run and dispatches `project/CLAM/main.py` | `dp-train experiment=pam50_wsi_final` |
| `dp-evaluate` | scores a trained WSI + tabular fusion checkpoint directory | `dp-evaluate --dry-run` |
| `dp-analysis` | the CNV arm, the fusion analyses and the controls (CPU, ~1 MB of inputs) | `dp-analysis cnv_wsi_fusion` |
| `dp-data` | acquisition: features, CNV, labels, and the reproduction bundle | `dp-data cnv --dry-run` |
| `dp-cptac` | the CPTAC external-validation pipeline, phases 0–4 | `dp-cptac --dry-run phase=all` |
| `dp-config` | inspect, validate and document the configuration | `dp-config validate` |

Worked examples. The first block writes nothing at all — `--help` and `--dry-run` are the way to see
what a command would do before it does it:

```bash
dp-train --help                                   # experiments and operators, no config composed
dp-train --dry-run experiment=pam50_wsi_final     # the exact CLAM command, nothing written
dp-cptac --dry-run phase=all                      # every phase's command, in order
dp-data cnv --dry-run                             # the acquisition command, nothing fetched
dp-analysis list                                  # the five analyses, one line each
```

A dry run works before anything has been downloaded: it prints the plan either way, and then, if the
inputs are not on disk yet, names what is missing and exits non-zero. So the plan is always visible
and the exit status still answers "would this run here?".

The second block does real work — CPU-only and cheap, except the last line, which is a 10-fold
training run:

```bash
dp-config show                                    # the composed config, resolved for this machine
dp-config reference -o docs/config-reference.md   # regenerate the config reference (writes a file)
dp-analysis cnv_controls                          # every control number, next to its published value
dp-analysis compare_fusion_ladder                 # the five ladder arms vs the probability mean
dp-train -m experiment=pam50_wsi_cnv fusion=concat,gated,cross_attention,film_attention,coattn
```

Three things that will trip you up otherwise:

- **`dp-train`'s primary config is `train`, not `config`** — it is `config.yaml` plus the `fusion`
  and `experiment` groups. Hydra flags that name a config (`--cfg job`, `--help experiment=…`) work
  against `train`.
- **A value containing `{fold}` must be quoted.** Hydra's override grammar rejects a bare `{`:
  `dp-train … 'clam.pretrained_wsi_ckpt="/abs/path/s_{fold}_checkpoint.pt"'`.
- **`+key=value` overrides are refused** (`run.allow_config_surgery=true` to permit). Hydra suggests
  `+` when you typo a key; accepting it would silently add a key nothing reads.

The legacy shell wrappers (`tools/train_pam50_final.sh`, `tools/run_cnv_fusion_ladder.sh`,
`tools/evaluate_pam50_multimodal.sh`, `tools/train_er_*.sh`, `tools/cptac/run_pipeline.sh`) still
work: each is now a shim that prints and then runs the equivalent `dp-*` command. Their pre-refactor
copies are frozen under `tests/legacy_wrappers/`, and a parity test executes both sides under a
stubbed `python` and compares the parsed argparse namespaces, so the refactor cannot have moved a
hyperparameter.

## Every run describes itself

A training run writes into `${paths.results_root}/<exp_code>_s<seed>/`, beside CLAM's own outputs:

| file | why it exists |
|---|---|
| `config.resolved.yaml` | Hydra's `.hydra/config.yaml` is stored **unresolved**, so replaying it on another machine reconstructs a different configuration. This is the snapshot that actually replays. |
| `run_metadata.json` | git SHA / dirty state, interpreter, platform, GPUs, dependency versions, every seed in effect, the command line, timing, exit status, and the hyperparameters CLAM hard-codes outside argparse |
| `clam_argv.json` | the exact argv handed to `main.py`, its cwd and the interpreter that ran it |
| `metrics.json` | CLAM's `summary.csv`, machine-readable |
| `.hydra/` | Hydra's own record, copied in after the run |

A run directory that already holds `summary.csv` or an `s_*_checkpoint.pt` is **refused**
(`run.overwrite=true` to override). `.scratch/` is gitignored and unrecoverable, and the five
completed ladder arms cost 2 h 38 min of GPU time.

## Layout

```
dpcode/              the configuration and entry-point layer
├── conf/            the Hydra tree: paths, sources, clam, tracking, experiment,
│                    fusion, analyses, evaluate, acquire, cptac
├── cli/             train.py evaluate.py analysis.py data.py cptac.py config.py
├── paths.py         path resolution, with or without Hydra
├── schema.py        structured configs; the closed key set
├── clam_args.py     CLAM's real parser, extracted by AST and never imported
├── runinfo.py       run-directory self-description and the overwrite guard
├── determinism.py   records what is seeded and what is not
└── wandb_util.py

project/
├── CLAM/            vendored CLAM + this project's multimodal fork
│   ├── main.py                    tasks, --fusion_mode, the --tabular_* / --pretrained_* flags
│   ├── models/model_multimodal.py CLAMRNAFusion, TabularMLPEncoder, the six fusion operators
│   ├── dataset_csv/*.csv          per-task manifests   -- TRACKED PRIMARY INPUTS
│   └── splits/<task>_100/         fold definitions     -- TRACKED PRIMARY INPUTS
├── data/            feature_datamodule.py, patch_dataset.py, transforms.py, pam50.R
├── survival/, MCAT/ dormant
├── UNI/             untracked; needed only to tile new slides, never on the PAM50 path
└── base/, loggers/  legacy scaffold

tools/               the scripts the entry points dispatch, plus the analyses
├── pam50_arms.py, evaluate_cnv_wsi_fusion.py, stack_wsi_cnv.py, compare_fusion_ladder.py
├── download_cnv_mutations.py, make_cnv_tabular.py, download_embeddings.py, download_cptac.py
├── cptac/           download -> audit -> manifest -> inference
├── data/            label tables + reference/gene_arm_hg38.csv (the pin on the 39 features)
├── rna/, diagnostics/, nou/, hsi_bc/   dormant or ablation-only
└── *.sh             shims over dp-train / dp-evaluate / dp-cptac

tests/               parity, schema, paths, config composition, synthetic end-to-end
docs/                results, the config reference, the literature survey, parked cohorts
```

`project/CLAM/dataset_csv/tcga_brca_subtyping.csv` and `project/CLAM/splits/tcga_brca_subtyping_100/`
are **distributed primary inputs, not derived artifacts**: they define the task and the exact fold
draw behind every published number, they have no recorded derivation, and regenerating them with a
different seed invalidates the entire results chain. No entry point writes into them.

## Where to go next

| you want | read |
|---|---|
| to reproduce a number | **[REPRODUCING.md](REPRODUCING.md)** — the cheap path first, then the full one |
| the result itself, with controls | [`docs/cnv-wsi-fusion-external-validation.md`](docs/cnv-wsi-fusion-external-validation.md) |
| every config key, generated | [`docs/config-reference.md`](docs/config-reference.md) |
| working conventions, gotchas, known gaps | [`CLAUDE.md`](CLAUDE.md) |
| where this sits in the literature | [`docs/implementation-research/PAM50/README.md`](docs/implementation-research/PAM50/README.md) |
| the completed ER thread | [`docs/er-prediction-results.md`](docs/er-prediction-results.md), [`docs/er-external-validation-results.md`](docs/er-external-validation-results.md) |
| a parked cohort | [`docs/parked-cohorts/histology-hsi-bc.md`](docs/parked-cohorts/histology-hsi-bc.md) |

## Known gaps

Real, unfixed, and deliberately preserved rather than papered over. `CLAUDE.md` carries the detail.

- **`film_attention` and `coattn` checkpoints cannot be evaluated.** `evaluate_multimodal.py` has no
  branch for either operator; `dp-evaluate` refuses them up front with the reason instead of dying
  inside `load_state_dict`. The ladder is compared through `dp-analysis compare_fusion_ladder`,
  which reads the per-fold prediction pickles and needs no evaluator.
- **`dp-evaluate` defaults to the TCGA test split, not CPTAC.** Swapping only the tabular table gives
  you TCGA slides scored against CPTAC-shaped rows.
- **`residual` fusion has no trainable second branch**, so `fusion=residual` refuses at composition
  time rather than after creating a run directory.
- **The reproduction bundle has not been published.** `dp-data headline-artifacts` builds it and a
  SHA256 manifest; depositing it somewhere citable is an author decision. Until then three of the
  four inputs to the cheap path are not obtainable from a clone. See REPRODUCING.md.
- **The repository's licence is unresolved** — see `CITATION.cff`. `project/CLAM` and `project/MCAT`
  are GPLv3 and CLAM is modified in place; the root `LICENCE` is unfilled Apache-2.0. Do not assume
  either applies until the author decides.

## Data access and privacy

TCGA-BRCA and CPTAC-BRCA are public. The pre-extracted UNI2-h features for both cohorts come from
the **gated** HuggingFace dataset `MahmoodLab/UNI2-h-features`, which needs an approved access
request; the UNI2-h encoder (`MahmoodLab/UNI2-h`) is separately gated and is needed only to tile new
slides.

A private institutional cohort is referenced by configuration (`DP_NOU_ROOT`) and is **not** part of
this repository, not part of any reported result, and has no committed default path. Never point a
public W&B project at a run that touches it.

## Citing

See [`CITATION.cff`](CITATION.cff) — and note that several fields there are placeholders only the
author can fill. If you use this code, cite CLAM (Lu et al.), UNI (Chen et al.), TCGA-BRCA,
CPTAC-BRCA and cBioPortal as well; this repository is a fork and an integration of their work.
