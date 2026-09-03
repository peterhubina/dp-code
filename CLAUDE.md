# CLAUDE.md

Guidance for Claude Code in this repository. Only load-bearing facts live here; detail is in the
linked docs. Two doc rules: `docs/cnv-wsi-fusion-external-validation.md` is **USER-OWNED — read,
never write**; `docs/config-reference.md` is **generated** by `dp-config reference` — never
hand-edit.

## What this project is

4-class PAM50 subtype classification (LumA/LumB/Basal/Her2; Normal-like dropped), trained on
TCGA-BRCA, externally validated on CPTAC-BRCA (114 cases / 378 slides), fusing H&E WSIs (UNI2-h
1536-dim patch features → CLAM-MB MIL) with arm-level CNV (39 arms, median gene-level log2 per
arm — the scale deployed low-pass assays report, and non-leaky, unlike RNA, whose expression
matrix produced the PAM50 labels). Nothing is ever refit, tuned, or thresholded on CPTAC.

The core question — does conditioning H&E on copy number beat averaging two independent
predictions? — is answered: **no**. Write-up: `docs/cnv-wsi-fusion-external-validation.md`.

**Standing external numbers** (TCGA→CPTAC, n=114 — the bar to clear): WSI-only macro AUROC 0.847
(Her2 recall 0/14), CNV-only 0.888 (12/14), equal-weight mean 0.909 (6/14), prior-balanced mean
0.912 (post hoc, 10/14). Internal, 599 pooled TCGA cases: WSI 0.887, CNV 0.862–0.872, mean
0.922–0.926.

**Fusion-operator ladder — run and analysed; do NOT re-run.** Five trained operators
(`.scratch/results/pam50_wsi_cnv_{concat,gated,cross_attention,film_attention,coattn}_s1`, 10
folds each, warm-started from `pam50_final_s1`, not frozen; `film_attention` conditions the
attention input) all score 0.882–0.899 pooled — every one significantly below the untrained
probability mean (0.9259). Mechanism: error correlation (φ 0.656 among operators vs ~0.2 between
the unimodal arms). Open confound: all five share the `pam50_final_s1` warm start, so
joint-training vs shared-init diversity collapse is unseparated; a `--no_warm_start` arm would
settle it. §8 of the results doc; recomputed by `dp-analysis compare_fusion_ladder`.

## Work since 2026-08-05 (newest first)

- **Her2-collapse remediation plan** —
  `docs/implementation-research/her2-collapse-remediation-plan.md`. Mechanism nailed:
  **UNI2-h feature-space geometry** (cohort of origin readable from a single patch feature at
  AUROC 1.0000; shift 5.4× the Her2 class signal; the MIL head is innocent — a logistic probe on
  slide-mean features reproduces 0/378). Aggregation, segmentation-dilution, label-mismatch and
  17q explanations refuted (CNV's 12/14 holds without 17q). Five gated, costed items:
  moment-matching at inference → CPTAC preservation-type check → Macenko re-extraction (10.7
  GPU-h) → full TCGA re-download (1.159 TB, last resort). **Nothing implemented yet** —
  `infer_cptac_pam50.py` has no `--feature_transform`.
- **SOTA comparison** — `docs/implementation-research/PAM50/sota-comparison-cnv-fusion.md`.
  20 newly verified papers; corrects several older claims (Positioning below reflects it; on any
  conflict that doc wins). Fernandez-Romero 2026 independently reproduces the external HER2
  collapse (RPD 1.000 across 13 encoders × 3 MIL heads, TCGA→CPTAC; staining + feature-space
  divergence explain 80% of RPD variance). Wagner 2026 (UNI2+CLAM, dMMR) shows the same
  external-recall collapse and deliberately omits stain normalisation as potentially harmful — a
  live counter-argument to the named remediation.
- **Supervisor report** — `tools/evaluate_pam50_fusion.ipynb` (+ PDF). Recomputes every published
  number from disk via `tools/pam50_arms.py`; a third producer of figures alongside `dp-analysis`.
  Nominates the per-CLAM-fold CNV refit as the internal protocol; adds external error
  independence φ = −0.006.

## Reporting rules (non-negotiable)

1. **Report the CNV-alone arm every time fusion is reported.** Fusion's edge over CNV alone is
   marginal (ΔAUROC +0.024, CI lower bound exactly +0.000).
2. **The equal-weight mean is the baseline**, not the WSI-only model.
3. **Never tune, calibrate, or select on CPTAC.** Anything run post hoc on CPTAC (the
   prior-balancing control was) is labelled post hoc.
4. **Name the protocol behind every number.** Headline CNV 0.866±0.003 is 5-fold × 10 reseeds;
   most §4 controls are single 5-fold at seed 0 (`dp-analysis cnv_controls` prints both). Internal
   φ: 0.269 published (StratifiedKFold(10, seed 0)) vs 0.193 (per-CLAM-fold refit) — say which.

## Commands

```bash
pip install -e '.[dev]'   # the ONLY correct install; editable is required (scripts read sibling
                          # data dirs by relative path). python/pip already resolve to /opt/venv.
dp-config validate        # paths, tracked inputs, ClamConf vs CLAM's parser; add experiment=/fusion=
dp-config sync-check      # the schema-drift check alone
```

**There is no test suite and no Makefile** — both were built and deliberately reverted (commit
`0c38c14`). `dp-config validate` / `sync-check` are the only automated checks. `tests/` holds only
the frozen pre-refactor wrappers (`tests/legacy_wrappers/tools/`) for hand-redoing the argv parity
check against `dp-train --dry-run`.

Six console scripts, runnable from any cwd; every path and parameter comes from the Hydra tree in
`dpcode/conf/`; the old `tools/*.sh` wrappers are shims over them. All support `--dry-run`.

| command | does |
|---|---|
| `dp-train` | renders argv, dispatches `project/CLAM/main.py` as a subprocess (cwd `project/CLAM`) |
| `dp-evaluate` | dispatches `evaluate_multimodal.py`; known gaps preserved but refused *before* dispatch |
| `dp-analysis` | `cnv_wsi_fusion`, `stack_wsi_cnv`, `cnv_controls`, `make_cnv_tabular`, `compare_fusion_ladder` — CPU, seconds; `dp-analysis list` |
| `dp-data` | CNV/mutation/embedding/label downloads + `headline-artifacts` / `verify-artifacts` |
| `dp-cptac` | CPTAC pipeline phases 0–4 (phase 0 = precondition checks *before* the gated 16 GB download) |
| `dp-config` | `show`, `validate`, `reference`, `sync-check` |

Reference runs (all already on disk; re-run only to change the baseline):
`dp-train experiment=pam50_wsi_final` (WSI baseline → `.scratch/results/pam50_final_s1`, ≈66 min);
`dp-train -m experiment=pam50_wsi_cnv fusion=concat,gated,cross_attention,film_attention,coattn`
(the ladder); `dp-cptac phase=features,1,2` then `phase=3,4` (→ `ensemble_predictions.csv`).

Guards that are load-bearing:

- `dp-train` **refuses** a run directory holding `summary.csv` or checkpoints. `run.overwrite=true`
  defeats that and destroys results unrecoverable from git.
- `dp-evaluate` refuses `film_attention`/`coattn` checkpoints by name and refuses (exit 2) an
  architecture mismatch against the checkpoint's `experiment_*.txt` (typically
  `tabular_hidden_dim` 64 vs 256); `--allow-arch-mismatch` overrides.
- `fusion=residual` refuses at composition time (no way to train its tabular checkpoint).
- `dp-analysis` names missing inputs — with the producing command — *before* creating a run dir;
  each run writes a self-describing dir under `.scratch/analysis/<action>/<timestamp>/`.
- `dp-analysis make_cnv_tabular` exits non-zero if any split case lacks CNV (that coverage is what
  lets the ladder reuse `splits/tcga_brca_subtyping_100` and compare against `pam50_final_s1`);
  it also writes `chromosome_groups.csv`, required by `--fusion_mode coattn`.
  `analysis.cohort=tcga` skips the gated CPTAC chain.
- `dp-data cnv` is **fatal** when the cohort-filter label table is missing (for CPTAC it comes
  from `dp-cptac phase=2`, so that chain runs first).

Frozen constants (defined once; the two bootstrap seeds differ **on purpose** — unifying them
changes published CIs): the CNV model is `tools/pam50_arms.py`'s
`StandardScaler → LogisticRegression(max_iter=4000, C=0.1, class_weight='balanced')`;
`evaluate_cnv_wsi_fusion` N_BOOT=4000/seed 7; `stack_wsi_cnv` N_BOOT=2000/seed 11;
`compare_fusion_ladder` n_boot=2000/seed 13.

## Hydra conventions that bite

- **`dp-train`'s primary config is `train`, not `config`** (`dpcode/conf/train.yaml`); the shared
  `config.yaml` serves every other entry point. `experiment=` is mandatory and has no default.
- **Quote any value containing `{fold}`** — Hydra's grammar rejects a bare `{`:
  `'clam.pretrained_wsi_ckpt="/abs/s_{fold}_checkpoint.pt"'`. CLAM expands it per fold.
- **`+key=` / `~key` overrides are refused** unless `run.allow_config_surgery=true`. Hydra suggests
  `+` on typos; accepting silently adds a key nothing reads.
- **`dpcode/conf/tracking/` and `dpcode/conf/analyses/` are named around `.gitignore` traps** (bare
  `wandb` / `analysis` patterns). Never rename them; **never modify `.gitignore`**. The `analyses`
  files carry `# @package analysis`, so overrides are still spelled `analysis.<key>=…`.
- **`tracking=` does not reach CLAM** — CLAM logs from `clam.wandb*`; `clam.wandb=false` disables.
- **Hydra's own output dir is scratch** (`.scratch/hydra/`, `.scratch/multirun/`); run dirs derive
  from `clam.exp_code`/`clam.seed` and are self-describing (`config.resolved.yaml`,
  `run_metadata.json`, `clam_argv.json`, `metrics.json` — Hydra's own `.hydra/config.yaml` is
  unresolved and does not replay elsewhere).
- **CLAM is never imported.** `main.py` lines 119–433 run at import (parse the importer's argv,
  seed torch, create dirs). It is dispatched as a subprocess; its parser is extracted by AST
  (`dpcode/clam_args.py`), and `dp-config sync-check` stops `ClamConf` drifting from it.

## Known gaps (real, blocking, unfixed)

- **`film_attention` and `coattn` checkpoints cannot be evaluated**
  (`evaluate_multimodal.py:70` choices + `infer_fusion_mode` misrouting) — which is why §8 is
  computed from per-fold `split_*_results.pkl` by `compare_fusion_ladder`, not an evaluator.
- **`dp-evaluate` defaults to the TCGA test split.** A genuine external run needs CPTAC
  embeddings and a CPTAC dataset_csv, not just a swapped tabular table.
- **The headline-artifact bundle is unpublished** — `docs/headline-artifacts.sha256` does not
  exist, so `dp-data verify-artifacts` has nothing to check and the cheap reproduction path in
  `REPRODUCING.md` is blocked. Depositing it is an author decision.
- **ER thread partially ported on purpose** — each wrapper's default arm has an experiment
  config; the shims refuse the others by name up front.
- **Survival thread doubly broken** — wrong `embeddings_dir` in
  `tools/config/dataset/tcga_brca_survival.yaml` and `scikit-survival` never installed.
- **Two published figures have no producing command** (§2 `mean` columns; §4 per-arm r=0.960) —
  computed interactively, marked in `REPRODUCING.md`.

## Gotchas and settled questions

Each cost a debugging session or was measured; do not re-litigate.

- **Two class orders.** CLAM / `make_cnv_tabular` / `infer_cptac_pam50` use `LumA,LumB,Basal,Her2`;
  `pam50_arms.CLASSES` is sorted `Basal,Her2,LumA,LumB`. Bridge with
  `pam50_arms.clam_column_order()`; reorder before scoring anything across the two.
- CPTAC patient IDs carry a leading `X` in cBioPortal (`X01BR001`→`01BR001`); the CPTAC label
  column is `label_name`, not `label`.
- `--embed_dim 1536` everywhere (UNI2-h).
- **CLAM's 10 splits are drawn independently, not partitioned** (599/910 cases reach a test fold),
  flattering "WSI alone" by ≈+0.01. Audited: no leakage, verdict stable — but say so when quoted.
- **`dataset_csv/tcga_brca_subtyping.csv` and `splits/tcga_brca_subtyping_100/` are tracked
  primary inputs** (9 readers, 0 writers; the generating invocation is recorded nowhere).
  Regenerating them invalidates `pam50_final_s1`, the ladder, CPTAC inference and the headline
  table. No entry point writes into `paths.splits_root` / `dataset_csv_dir` / `labels_dir`.
- **The ladder is a *near*-baseline comparison.** `pam50_final_s1` uses the sweep-selected
  optimiser config (incl. `inst_loss svm`, instance clustering, `weighted_sample: true`); the
  ladder arms use rounded values and `--no_inst_cluster`. Reproduced exactly, never tidied. The
  `weighted_sample` flag also falsifies any "trained under natural class frequencies" framing.
- **The external Her2 collapse is domain shift, not calibration** — 0/14 survives a 12× prior
  boost and SLD-EM (implied prior → 0.000). Do not re-run prior rebalancing; the mechanism and
  remediation ladder live in the Her2 plan above.
- **CPTAC and TCGA features are geometrically comparable, not identically preprocessed** (CLAM vs
  Trident tissue segmentation). Never write "preprocessing is held constant".
- **The 0.974 gated WSI+RNA PAM50 number is target-leakage-inflated** (labels derive from the
  same expression matrix). Do not quote it.
- **RNA scale mismatch: resolved, closed.** Both cohorts log2(TPM+1) on a shared 19,944-gene axis
  → `.scratch/rna-gdc/`; `fsqn_harmonize.py` is a sensitivity arm only.
- Cross-cohort CNV platform difference is mild (per-arm r=0.960); per-cohort standardisation is
  still worth an ablation.
- **Aneuploidy burden alone reaches 0.685** (published protocol; 0.673 under `frac_altered`).
  Report it beside the 39-arm model, or Basal ≈0.97 reads as pure genome instability.
- Patient/slide-level splitting always: CLAM's split CSVs enforce it; custom scripts must too.
- CPTAC is stage IIA–IIIC with no Normal-like — external CNV 0.888 vs 0.866 internal means "does
  not degrade", not "improves". Power: external Her2 n=14, LumB n=17 — indicative only.
- **Determinism, honestly:** seeds and folds fixed; run-to-run variance unmeasured (no tolerance
  quoted); bitwise reproducibility unachievable (`nn.MultiheadAttention`, no deterministic-algos
  flags — setting them would change published numbers). `dpcode/determinism.py` records the
  residual nondeterminism and fixes nothing; `PYTHONHASHSEED` is set too late to act and is
  recorded as observed environment, not a seed in effect.

## Positioning (corrected 2026-08; on conflict the SOTA doc wins)

- **The slot: no published *multimodal* PAM50 model has an external, never-trained-on
  evaluation.** Keep the qualifier — unimodal external PAM50 precedents exist (Borji 2026 on
  CPTAC 0.9523, Fernandez-Romero 2026, Zhang 2025 0.6515, PathLUPI 0.727).
- **Amer 2025** is the only work with CNV as the *sole* molecular modality beside H&E for PAM50
  (others — CustOmics, MRSVM, CLOVER, Liu 2022 — pair them within larger stacks). Comparable
  numbers: CNV-alone 0.8284, CNV+Image **0.8835** (not the Image+Graph 0.86 row). Their "Simple
  Ensemble" (0.9074) beats their trained operators — so "nobody reports the trivial average" is
  dead; ours is "the trivial average also wins externally, and we report CNV-alone".
- Do **not** cite TANGLE, THREADS, HE2RNA, SEQUOIA or Path2Omics as PAM50-from-histology
  precedent — full-text-verified, none evaluate PAM50. No paper predicts intrinsic subtype from
  sWGS; claim "arm scale is what deployed low-pass assays report", not an sWGS validation.

## Data and paths

- Every location is a `paths.*` key; `dp-config show` prints resolutions; `DP_REPO_ROOT` /
  `DP_DATA_ROOT` / `DP_SCRATCH_ROOT` / `DP_RESULTS_ROOT` relocate them (`.env.example`).
- `.datasets/`, `.scratch/`, `wandb/`, `papers/`, `analysis/`, `.claude/` are gitignored —
  **nothing there is recoverable**, including `pam50_final_s1`, the five ladder arms and
  `.scratch/cptac_validation/results/predictions/ensemble_predictions.csv` (the WSI arm feeding
  every CNV analysis).
- `tools/data/reference/gene_arm_hg38.csv` + `CHECKSUMS.sha256` pin the 39 arm features (the
  upstream sources are live and drift). A fresh clone is pinned; deleting both the tracked copy
  and the `.datasets/cnv/reference/` cache un-pins it and every AUROC can move.
- The `nou` cohort is **private institutional data**: no filename, case ID or listing in tracked
  config, docs or public W&B; `paths.nou_root` has no committed default (`DP_NOU_ROOT`).
- Repo map: `dpcode/` = config + entry-point layer; `project/CLAM/` = vendored CLAM plus the
  multimodal fork (`models/model_multimodal.py`, 6 fusion operators) with tracked primary inputs
  `dataset_csv/` and `splits/`; `tools/` = analysis scripts + shims + label tables; `README.md` /
  `REPRODUCING.md` = overview and empty-machine→headline path. `project/base/`, `tools/train.py`,
  `tools/main.py` and the `prostate`/`timm`/`basic` config groups are dead legacy scaffold;
  `tools/config/` is live only for the (broken) survival path.

## Dormant threads (concluded — read the linked doc before proposing revival)

WSI+RNA fusion: 0.981 loses to RNA-only 0.988 externally, gate decides on RNA alone (plus the
leakage caveat). ER: complete — RNA fusion +0.044 AUROC over H&E (p=1.6e-5), clinicopath null;
`docs/er-*.md`. Clinicopath→PAM50: weak (0.66/0.58), ER/PR/HER2-as-features is a circularity
trap. Survival: dead end (CPTAC has 2 deaths) and broken. `nou` and HSI-BC: parked — HSI-BC was
tiled at 62.3 µm FOV vs TCGA's 128 µm, so its external numbers are void
(`docs/parked-cohorts/histology-hsi-bc.md`).

## Environment

- Python 3.10/3.11 (`torch==2.0.1` has no cp312 wheels); pins: hydra-core 1.3.4, omegaconf 2.3.0,
  torch 2.0.1+cu117, wandb 0.15.3, pytest 8.4.2. `topk` is a git pin with no PyPI fallback,
  required by `--inst_loss svm` (the frozen baseline); `dp-config validate` imports it early.
- Hardware of record: 2× RTX 3090, CUDA 11.7. Docker is not a supported route.
- W&B on in `pam50_wsi_final` and the ER experiments, off in the CNV ladder (matching the old
  wrappers); project `clam-brca-subtyping-cv`.
- **Do not install CLAM's or MCAT's own requirements files** — CLAM's omits packages CLAM itself
  imports. Install the root package only.
