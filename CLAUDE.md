# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this project is doing right now

**4-class PAM50 molecular-subtype classification, trained on TCGA-BRCA and externally validated on
CPTAC-BRCA, fusing H&E whole-slide images with arm-level copy-number variation (CNV).**

- **Primary modality — H&E WSI.** UNI2-h patch features (1536-dim) → CLAM-MB multiple-instance
  learning, 10-fold CV on TCGA-BRCA.
- **Second modality — arm-level CNV.** 39 chromosome arms, each the *median* gene-level log2 over
  that arm. Chosen because shallow WGS resolves arm/segment scale, so the assay is cheap and
  clinically reachable; a model over 19,755 focal GISTIC calls would not transfer to an sWGS
  setting. Acrocentric p-arms (13p, 14p, 15p, 21p, 22p) are excluded by construction.
  CNV is also **non-leaky**, which RNA was not: PAM50 labels are computed from the expression
  matrix itself (`project/data/pam50.R`), so an RNA branch predicting PAM50 leaks the target by
  construction. Copy number is a different assay and carries no such circularity.
- **External cohort — CPTAC-BRCA, n = 114 cases / 378 slides.** Nothing is ever refit, tuned, or
  thresholded on CPTAC. Fusion rules are fixed on TCGA before the external set is scored.
- **Classes: LumA / LumB / Basal / Her2.** Normal-like is dropped. CPTAC's 114-case subset contains
  no Normal-like, so the cohorts already agree.

The open question the repo is currently answering: **does conditioning H&E on copy number beat
simply averaging two independent predictions?** The baseline to beat is not the WSI-only model — it
is the equal-weight probability mean.

### The standing numbers (the bar to clear)

TCGA-trained, CPTAC external, n = 114. Full write-up and every control:
`docs/cnv-wsi-fusion-external-validation.md`.

| Model | macro AUROC [95% CI] | balanced acc | Her2 recall |
|---|---|---|---|
| WSI only (CLAM-MB + UNI2-h) | 0.847 [0.791, 0.895] | 0.513 | 0/14 |
| CNV only (39 arms, logistic regression) | 0.888 [0.835, 0.933] | 0.716 | 12/14 |
| **Fusion — equal-weight mean** | **0.909** [0.858, 0.948] | 0.646 | 6/14 |
| Fusion — mean of prior-balanced WSI + CNV | **0.912** | **0.740** | 10/14 |

Internal TCGA (599 cases with CLAM out-of-fold predictions): WSI 0.887, CNV 0.862–0.872,
mean-fusion 0.922–0.926.

**Status as of 2026-08-05: the fusion-operator ladder is written but has not been run.** Check
`ls .scratch/results/pam50_wsi_cnv_*` before assuming any ladder result exists.

### Reporting rules that are not negotiable here

1. **Report the CNV-alone arm every time fusion is reported.** Fusion's edge over CNV alone is
   marginal (ΔAUROC +0.024, CI lower bound exactly +0.000; balanced accuracy not significant).
   Omitting it reproduces the selective reporting the literature survey criticises.
2. **The equal-weight mean is the baseline**, not the WSI-only model. Operators that fail to clear
   it are a publishable result in a field where four groups report fusion architectures that do not
   help and none report the trivial baseline.
3. **Never tune, calibrate, or select on CPTAC.** If something is run post hoc on CPTAC (the
   prior-balancing control was), label it post hoc.

## Entry points — the CNV + WSI fusion pipeline

Run from `/workspace/dp-code` unless stated. `python` already resolves to `/opt/venv`.

### 1. CNV features (already on disk; re-run only to refresh)
```bash
python tools/download_cnv_mutations.py --what cna --representation arm --validate-arms
```
Pulls cBioPortal `brca_tcga_pan_can_atlas_2018` / `brca_cptac_2020` plus UCSC hg38
`refGene`/`cytoBand`, and writes `.datasets/cnv/{tcga,cptac}_brca_cna_arm.csv` (981×39 and 114×39)
alongside `_gistic`, `_mutations`, `_mutation_matrix`. Default `--cohort-only` keeps just the cases
that have UNI2-h features. Coverage is complete: 981/981 TCGA and 114/114 CPTAC.

### 2. Reshape CNV into CLAM's tabular contract
```bash
python tools/make_cnv_tabular.py
```
Writes `.scratch/cnv-tabular/{TCGA,CPTAC}_BRCA_CNV_arm_4class_clam.csv` (`case_id,label,<39 arms>`;
910 and 114 rows) and `chromosome_groups.csv` — 22 chromosome tokens over the 39 arms, the grouping
`--fusion_mode coattn` needs (`--tabular_group_spec prefix` would give 39 biologically empty
singleton tokens). **It exits non-zero if any case in the existing splits lacks CNV.** That check
is load-bearing: `multimodal_dataset.py` raises on a training case with no tabular row, and
complete coverage (910/910 non-Normal) is what lets the ladder reuse
`splits/tcga_brca_subtyping_100` and treat the existing `pam50_final_s1` run as a directly
comparable WSI-only baseline instead of retraining it.

### 3. The late-fusion baseline and its controls
```bash
python tools/evaluate_cnv_wsi_fusion.py              # TCGA -> CPTAC external
python tools/evaluate_cnv_wsi_fusion.py --internal   # adds the TCGA-only head-to-head
```
Reproduces every number in `docs/cnv-wsi-fusion-external-validation.md`. Touches no slides — it
reads CLAM probabilities already on disk, prints to stdout, and writes nothing. 4,000 bootstrap
resamples, seed 7.

```bash
python tools/stack_wsi_cnv.py    # can a learned rule beat the mean? Answer on disk: no.
```
Five rules of increasing freedom (fixed mean → scalar weight → per-class weights → multinomial
logistic regression on probabilities → on log-probabilities), scored by nested CV over CLAM's fold
tags, CNV arm refit per fold. **No rule beats the mean on AUROC internally or externally, and every
learned rule is significantly worse on external balanced accuracy.** That rules out *global*
reweighting, not input-conditional gating — which is what the ladder tests.

Shared plumbing for both scripts lives in `tools/pam50_arms.py`: the CNV arm is exactly
`StandardScaler → LogisticRegression(max_iter=4000, C=0.1, class_weight='balanced')`, defined once
so the two scripts cannot silently describe different models.

### 4. The fusion-operator ladder (the current experiment)
```bash
bash tools/run_cnv_fusion_ladder.sh --dry_run                                  # print the plan
bash tools/run_cnv_fusion_ladder.sh --k 1 --max_epochs 2 --exp_suffix _smoke   # wiring check
bash tools/run_cnv_fusion_ladder.sh --modes film_attention                     # one operator
bash tools/run_cnv_fusion_ladder.sh                                            # the whole ladder
```
Runs `concat gated cross_attention film_attention coattn` through `project/CLAM/main.py` on
identical splits (`tcga_brca_subtyping_100`, k=10, seed 1), CLAM-MB `--model_size big`,
`--tabular_hidden_dim 64 --tabular_top_n_features 0 --fusion_hidden_dim 32` (sized for a 39-dim
modality, unlike the RNA wrapper's 256/10000). Results land in
`.scratch/results/pam50_wsi_cnv_<mode>_s1/`; existing dirs are skipped unless `--no_skip_existing`.

Two details keep H&E primary **in fact and not just in framing**: the WSI branch is warm-started
from `--pretrained_wsi_ckpt .scratch/results/pam50_final_s1/s_{fold}_checkpoint.pt`, so fusion must
improve on that model rather than rediscover it; and `film_attention` conditions the *attention
network's input* on the tabular vector (`--film_rank 16 --modality_dropout 0.2`), re-ranking patches
instead of being appended after pooling.

`residual` is deliberately **not** in the default ladder — it needs a matched tabular-only
checkpoint via `--pretrained_rna_ckpt`, and there is no supported way to train one (see Gotchas).

### 5. External validation of a trained fusion checkpoint — **currently blocked, see Known gaps**
```bash
bash tools/evaluate_pam50_multimodal.sh \
    --ckpt_dir .scratch/results/pam50_wsi_cnv_<mode>_s1 \
    --tabular_csv .scratch/cnv-tabular/CPTAC_BRCA_CNV_arm_4class_clam.csv \
    --output_dir .scratch/results/pam50_wsi_cnv_<mode>_eval
```
This is the command the ladder prints on completion, and **as printed it does not do what it says.**
See "Known gaps" below before relying on it.

### 6. The WSI-only arms (already run; re-run only to change the baseline)
```bash
bash tools/train_pam50_final.sh          # TCGA 10-fold CLAM-MB -> .scratch/results/pam50_final_s1
bash tools/cptac/run_pipeline.sh         # download -> audit -> manifest -> CPTAC inference
```
`train_pam50_final.sh` is a frozen sweep-selected config (CLAM-MB/big, B=4, bag_loss ce,
bag_weight 0.553, inst_loss svm, dropout 0.5, lr 1.008e-4, reg 2.446e-6, k=10, seed 1). It writes
`s_{0..9}_checkpoint.pt` and `split_{0..9}_results.pkl` — consumed by `pam50_arms.py`, by the CPTAC
inference, and by the ladder's warm start.

`tools/cptac/run_pipeline.sh` phases 1–4 produce the WSI arm for the whole CNV analysis:
`.scratch/cptac_validation/results/predictions/ensemble_predictions.csv`
(378 slides → 114 cases, columns `slide_id,case_id,true_label,…,p_LumA,p_LumB,p_Basal,p_Her2`).
Phase 2 (`audit_feature_provenance.py`) is the check that CPTAC features are geometrically
comparable to TCGA's (256 px @ 20×, 0 overlap, 1536-dim). Phases 5–8 are the RNA branch and are
independent of the CNV work.

Raw CLAM invocation, if no wrapper fits:
```bash
cd project/CLAM
python main.py --task tcga_brca_subtyping --subtyping --model_type clam_mb --model_size big \
    --embed_dim 1536 --data_root_dir /workspace/dp-code/.datasets/tcga-brca/embeddings \
    --results_dir /workspace/dp-code/.scratch/results --split_dir tcga_brca_subtyping_100 --k 10
```
Valid `--task`: `tcga_brca_subtyping` (4-class, manifest `dataset_csv/tcga_brca_subtyping.csv`,
1009 slides — LumA 503, LumB 221, Basal 174, Her2 76, Normal 35 ignored), `tcga_brca_er`,
`tcga_brca_recurrence`, `nou_ctc_{ep,emt,any}`, `task_1_tumor_vs_normal`, `task_2_tumor_subtyping`.
Valid `--fusion_mode`: `concat`, `gated`, `residual`, `cross_attention`, `film_attention`, `coattn`
— all implemented in `project/CLAM/models/model_multimodal.py` (`CLAMRNAFusion`), with
`film_attention` and `coattn` the novel additions audited in
`docs/implementation-research/phase2-verification.md`.

## Known gaps (real, blocking, unfixed)

- **`film_attention` and `coattn` checkpoints cannot be evaluated.**
  `project/CLAM/evaluate_multimodal.py:70` accepts only `auto|concat|gated|residual|cross_attention`,
  and the `evaluate_pam50_multimodal.sh` wrapper narrows that to `auto|concat|gated`. Under `auto`,
  `infer_fusion_mode` has no branch for either operator: a `film_attention` checkpoint (keys
  `film_bottleneck/film_gamma/film_beta/tabular_head`) raises `ValueError`, and a `coattn`
  checkpoint is misidentified as `cross_attention` because both build `self.cross_attention`, then
  dies in `load_state_dict(..., strict=True)`. Both fail loudly rather than silently mis-evaluating,
  but two of the five ladder arms have no evaluation path until `infer_fusion_mode` and both choice
  lists are extended.
- **`evaluate_pam50_multimodal.sh` defaults to the TCGA test split**, not CPTAC
  (`--data_root_dir .datasets/tcga-brca/embeddings`,
  `--dataset_csv project/CLAM/dataset_csv/tcga_brca_subtyping.csv`). Swapping only `--tabular_csv`
  to the CPTAC table — which is what the ladder's closing hint prints — evaluates TCGA slides
  against CPTAC-shaped tabular rows. A genuine external run also needs the CPTAC embeddings and a
  CPTAC dataset_csv.
- **`residual` fusion has no trainable second branch.** It needs `--pretrained_rna_ckpt`, and
  `tools/train_pam50_tabular.sh` (the only tabular-only trainer) passes `--model_type tabular_mlp`,
  which `main.py`'s argparse rejects — the choice list is `clam_sb|clam_mb|mil`.
- **The survival Hydra config points at a path that does not exist.**
  `tools/config/dataset/tcga_brca_survival.yaml` sets `embeddings_dir: .datasets/embeddings`; the
  real store is `.datasets/tcga-brca/embeddings`. `.datasets/tcga-brca/h5_files` is a broken symlink
  to the same non-existent path.

## Gotchas and settled questions

Things that already cost a debugging session, or were measured and must not be re-litigated.

- **Two different class orders.** CLAM's `label_dict`, `make_cnv_tabular.CLASSES` and
  `infer_cptac_pam50.LABEL_MAP` are `LumA, LumB, Basal, Her2`; `tools/pam50_arms.CLASSES` is sorted
  `Basal, Her2, LumA, LumB`. `pam50_arms.clam_column_order()` exists solely to bridge them and
  asserts the recovered map is a permutation. Reorder before scoring anything across the two.
- **CPTAC patient IDs carry a leading `X`** in cBioPortal (`X01BR001` → `01BR001`). The CPTAC label
  column is `label_name`, not `label`.
- **`--embed_dim 1536` everywhere** — UNI2-h features are 1536-dim.
- **CLAM's 10 splits are drawn independently, not partitioned.** 599 of 910 cases land in at least
  one test fold and 242 in two to five, so "WSI alone" is a small ensemble flattered by roughly
  +0.01 AUROC. An audit confirmed no leakage, and re-running with a random stratified partition
  gives the same verdict — but say so when the number is reported.
- **The external Her2 collapse is domain-shift-induced, not a calibration artifact — the
  calibration hypothesis was tested and refuted.** The WSI arm calls Her2 0/14 on CPTAC, still 0/14
  after a 12× prior boost, and 0/14 under unsupervised SLD-EM (which drives the implied Her2 prior
  to 0.000 against a true 0.123). Internally the same model calls 26/51. Do not re-run prior
  rebalancing expecting a fix; remediation belongs on the imaging side (stain normalisation,
  encoder choice).
- **The old 0.974 gated WSI+RNA PAM50 number is target-leakage-inflated.** PAM50 labels were
  computed from the same expression matrix fed to the RNA branch (`project/data/pam50.R`). Do not
  quote it. Details in `docs/implementation-research/next-steps-action-plan.md`.
- **RNA scale mismatch: resolved, do not re-open.** TCGA Xena log2 RSEM vs CPTAC linear TPM broke
  silently. Both cohorts are now log2(TPM+1) on a shared 19,944-gene protein-coding axis via
  `tools/rna/download_gdc_rna.py` + `build_gdc_expression.py` → `.scratch/rna-gdc/`. Use the GDC
  tables; `fsqn_harmonize.py` is the tier-2 sensitivity arm only.
- **Cross-cohort CNV platform difference is mild** (TCGA SNP6 → GISTIC2 vs CPTAC WGS): per-arm mean
  r = 0.960, mean |Δ| = 0.041, CPTAC sd ratio 0.82. Nothing like the RNA problem, but per-cohort
  standardisation is still worth an ablation.
- **Aneuploidy burden alone reaches 0.685 macro AUROC.** Report it alongside the 39-arm model, or
  Basal ≈ 0.97 reads as pure genome instability rather than an arm *pattern*.
- **Patient/slide-level splitting** — all patches of a slide stay in one fold, and cases are never
  split across folds. CLAM handles this via its split CSVs; custom scripts must replicate it.
- **CPTAC is stage IIA–IIIC by eligibility and has no Normal-like**, the likely reason the CNV arm
  scores higher externally (0.888) than internally (0.866). Treat that as "does not degrade", not
  "improves".
- **Power**: external Her2 n = 14, LumB n = 17. Per-class external estimates are indicative.

## Positioning (from the 50-paper survey in `docs/implementation-research/PAM50/`)

- **No published multimodal PAM50 model has an external, never-trained-on, PAM50-specific
  evaluation.** That is the slot this project occupies.
- Amer et al. 2025 (arXiv:2509.03408) is the **only** WSI+CNV PAM50 work: 10-fold CV on TCGA only,
  CNV-alone 0.8284, four-modality fusion 0.9153, both internal.
- **Do not cite TANGLE, THREADS, HE2RNA, SEQUOIA or Path2Omics as PAM50-from-histology precedent** —
  verified by full-text grep, none of them evaluate PAM50.
- Gated fusion appears in exactly two PAM50 papers and loses in both; "fusion does not beat the
  strong unimodal arm" is reported independently by four groups.

## Layout

```
project/
├── CLAM/            # Vendored CLAM + this project's multimodal fork:
│   ├── main.py                    # tasks, --fusion_mode and the --tabular_* / --pretrained_* flags
│   ├── models/model_multimodal.py # CLAMRNAFusion, TabularMLPEncoder, the 6 fusion operators
│   ├── multimodal_dataset.py      # raises if a training case has no tabular row
│   ├── evaluate_multimodal.py, evaluate_late_fusion.py, evaluate_selective_ensemble.py,
│   │   evaluate_confidence_routing.py, train_rna.py, sweep_train*.py
│   ├── dataset_csv/*.csv          # per-task manifests
│   └── splits/<task>_100/splits_{k}.csv
├── data/            # feature_datamodule.py, patch_dataset.py, transforms.py, pam50.R
│                    #   (run_patching.sh is DEAD — it calls tools/patch_wsi.py, which is gone)
├── survival/        # AMIL survival package — dormant
├── MCAT/            # Vendored MCAT — dormant
├── UNI/             # Vendored UNI feature extractor — untracked, must exist locally
├── Selective-Multimodal-…/  # Untracked reference clone of Hezil et al. 2025 (has its own .git)
└── base/, loggers/  # Legacy scaffold; checkpointer.py

tools/
├── download_cnv_mutations.py, make_cnv_tabular.py, pam50_arms.py            # CNV arm
├── evaluate_cnv_wsi_fusion.py, stack_wsi_cnv.py, run_cnv_fusion_ladder.sh   # fusion + ladder
├── train_pam50_final.sh, train_pam50_multimodal.sh, evaluate_pam50_multimodal.sh
├── extract_features.py, create_clam_dataset_csv.py, download_embeddings.py
├── download_cptac.py + cptac/     # CPTAC download, provenance audit, manifest, inference
├── build_er_labels.py, make_er_*.py, train_er_*.sh, evaluate_er_ablation.py,
│   analyze_er_ablation.py          # ER thread (complete)
├── analyse_clinicopath_pam50.py    # clinicopathological baseline (null result)
├── rna/            # GDC RNA download + harmonisation (dormant modality, kept for ablations)
├── diagnostics/    # gate_probe.py, tabular_only_probes.py — fusion-gate ablations
├── data/           # Label tables: pam50 / er / os / clinicopath + TCGA-CDR
├── nou/, hsi_bc/   # Dormant cohorts (nou/patch_nou_*.py and CLAM's create_patches_fp.py are
│                   #   the only working tile extractors left)
└── config/         # Hydra configs — only the survival path is live

docs/
├── cnv-wsi-fusion-external-validation.md   # the current headline result
├── er-prediction-results.md, er-external-validation-results.md
└── implementation-research/                # 17 planning/literature reports, incl.
    ├── next-steps-action-plan.md           #   the RNA target-leakage finding
    ├── phase2-verification.md              #   audit of film_attention / coattn
    ├── novel-fusion-design.md, novelty-risk-check.md
    └── PAM50/README.md, paper-dossier.md, survey-data.json   # the 50-paper survey
```

## Data and output locations

| Path | Contents |
|---|---|
| `.datasets/tcga-brca/embeddings/` | 1126 UNI2-h `.h5` files (no WSIs or patches live here) |
| `.datasets/cptac-brca/` | 654 feature `.h5`, `wsi/`, `rna/`, `clinical/`, `cptac_brca_pam50_dataset.csv` (~98 GB) |
| `.datasets/cnv/` | Arm-level + GISTIC CNA and mutation matrices for both cohorts |
| `.datasets/nou/`, `.datasets/HistologyHSI-BC-Recurrence/` | Dormant cohorts |
| `.scratch/cnv-tabular/` | CLAM-format CNV tabular inputs + chromosome grouping |
| `.scratch/results/` | CLAM runs — `pam50_final_s1` (WSI baseline), `pam50_wsi_rna_*`, `er/`, and (pending) `pam50_wsi_cnv_*` |
| `.scratch/cptac_validation/` | CPTAC external inference, incl. `results/predictions/ensemble_predictions.csv` |
| `.scratch/rna-gdc/`, `.scratch/TCGA-BRCA-rna/` | Harmonised GDC tables / legacy Xena tables |
| `.scratch/analysis/clinicopath_pam50/`, `.scratch/harmonisation/` | Clinicopath analysis outputs |
| `.scratch/checkpoints/uni2-h/` | UNI2-h weights for feature extraction |
| `tools/data/*.csv` | Label tables (PAM50, ER, OS, clinicopathological) |

`.datasets`, `.scratch`, `wandb`, `papers`, `analysis` and `.claude` are gitignored — nothing there
is recoverable from git.

## Dormant threads (context, not current work)

Each was taken to a conclusion. Read the linked doc before proposing to revive one.

- **WSI + RNA fusion → PAM50.** Complete. Fusion (0.981) does **not** beat RNA-only (0.988) on
  CPTAC, and an ablation showed the gate decides on RNA alone. This is the field norm, not an
  anomaly. Note the leakage caveat in Gotchas.
- **ER binary prediction.** Complete. RNA fusion beats H&E alone by +0.044 AUROC (DeLong
  p = 1.6e-5); clinicopathological fusion does not (−0.002). Discrimination transports to CPTAC
  (case AUROC 0.925 external vs 0.896 internal); calibration does not.
  `docs/er-prediction-results.md`, `docs/er-external-validation-results.md`.
- **Clinicopathological → PAM50.** Complete and weak (0.66/0.58 AUROC), adds nothing over H&E.
  Using ER/PR/HER2 as features is the circularity trap. `tools/analyse_clinicopath_pam50.py`.
- **Overall survival (AMIL, Hydra).** Dead end for this cohort pair: CPTAC has 1 recurrence event
  and 2 deaths, and TCGA-BRCA histology barely predicts survival in the literature (PORPOISE
  histology-only c-index 0.560, n.s.). Code still runs — `python tools/train_survival.py`,
  `python tools/eval_survival.py --exp_dir ...`, config `tools/config/survival.yaml` — but see
  Known gaps for the broken `embeddings_dir`.
- **CTC cohort (`nou`)** and **HistologyHSI-BC recurrence.** Parked. HSI-BC was tiled at 62.3 µm FOV
  against TCGA's 128 µm, so every existing HSI-BC external number is void until it is re-tiled.

## Legacy

`project/base/`, `tools/train.py`, `tools/main.py` and the `prostate`/`timm`/`basic` Hydra groups
are scaffold from an earlier template. `tools/train.py` is broken — it imports
`project.experiment.Experiment`, which does not exist (the classes live in `project/base/`); it is
plain argparse and never loads Hydra, so `default.yaml`'s `project.trainer.Trainer` target is a
separate defect. The `toyproblem.*` targets are in `config/dataset/prostate.yaml`,
`config/model/timm.yaml` and `config/augmentation/basic.yaml`.

The root **`README.md` is not this project's README** — it is the upstream HistologyHSI-BC dataset
README plus scratch commands, and following it will mislead you.

## Environment

```bash
pip install -r requirements.txt      # torch, timm, openslide, spectral, hydra-core, wandb
cd docker && ./build.sh && ./run.sh  # containerized alternative
```
- `python` / `pip` already resolve to `/opt/venv` (first on `$PATH`) — no `source` needed. Re-check
  `command -v python` if anything looks like a missing dependency.
- Hardware: 2× RTX 3090 (24 GB each), torch 2.0.1+cu117, CUDA available.
- W&B is on by default in the PAM50 and ER shell scripts and the survival config; the CNV ladder has
  it **off** unless `--wandb` is passed. Project `clam-brca-subtyping-cv`.
- CLAM, MCAT and UNI keep their own `requirements.txt` / `env.yml` and entry points; run CLAM/MCAT
  commands from inside their directories.
