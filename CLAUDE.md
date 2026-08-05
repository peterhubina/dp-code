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

The question the repo was asking — **does conditioning H&E on copy number beat simply averaging two
independent predictions?** — now has an answer: **no**. See below.

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

**Status as of 2026-08-05: the fusion-operator ladder has been RUN and ANALYSED.** All five arms are
complete on disk (`.scratch/results/pam50_wsi_cnv_{concat,gated,cross_attention,film_attention,coattn}_s1`,
10 folds each, ~2h38m of GPU time, in a gitignored tree with no backup).
`tools/compare_fusion_ladder.py` — run it as `dp-analysis compare_fusion_ladder` — pools their
out-of-fold predictions over the 599 shared cases and produces §8 of the results document:

| arm | pooled macro AUROC | balanced acc | Δ vs the mean |
|---|---|---|---|
| WSI only | 0.8872 | 0.6772 | −0.039 **sig** |
| CNV only | 0.8721 | 0.6784 | −0.054 **sig** |
| **probability mean (untrained)** | **0.9259** | **0.7513** | — |
| concat / gated / cross_attention / film_attention / coattn | 0.8827 / 0.8947 / 0.8917 / 0.8818 / 0.8992 | 0.665–0.685 | all significantly below |

**Every trained operator loses to the untrained probability mean**, five operators ensembled still
lose to two independently trained unimodal models, and the mechanism is error correlation: φ = 0.656
among the jointly trained operators against 0.193 between the two unimodal arms. `film_attention`
did *not* ignore the second modality — its conditioner diagnostics moved off zero-init — and still
finished below WSI-only. The one remaining confound: all five warm-start from the same
`pam50_final_s1` checkpoint, so "joint training collapses diversity" and "shared initialisation
collapses diversity" are not yet separated. A `--no_warm_start` arm would settle it.

### Reporting rules that are not negotiable here

1. **Report the CNV-alone arm every time fusion is reported.** Fusion's edge over CNV alone is
   marginal (ΔAUROC +0.024, CI lower bound exactly +0.000; balanced accuracy not significant).
   Omitting it reproduces the selective reporting the literature survey criticises.
2. **The equal-weight mean is the baseline**, not the WSI-only model. Operators that fail to clear
   it are a publishable result in a field where four groups report fusion architectures that do not
   help and none report the trivial baseline.
3. **Never tune, calibrate, or select on CPTAC.** If something is run post hoc on CPTAC (the
   prior-balancing control was), label it post hoc.
4. **Say which protocol a control used.** The internal headline figure (0.866 ± 0.003) is 5-fold ×
   10 reseeds; the aneuploidy-burden control, the C sweep and the site holdout are single 5-fold
   runs at seed 0. `dp-analysis cnv_controls` prints both and marks the published one.

## Entry points

Six console scripts, installed by `pip install -e '.[dev]'` and runnable **from any working
directory**. Every path and every settable parameter comes from the Hydra tree in `dpcode/conf/`.
The old `tools/*.sh` wrappers still work — each is now a shim that prints and then runs the
equivalent `dp-*` command, and their pre-refactor copies are frozen in `tests/legacy_wrappers/` for
the parity check.

| command | replaces | notes |
|---|---|---|
| `dp-train` | `train_pam50_final.sh`, `train_pam50_multimodal.sh`, `run_cnv_fusion_ladder.sh`, `train_er_*.sh` | renders argv, dispatches `python main.py` with cwd `project/CLAM` |
| `dp-evaluate` | `evaluate_pam50_multimodal.sh` | dispatches `evaluate_multimodal.py`; both known gaps preserved, but they now fail loudly and *before* dispatch |
| `dp-analysis` | `evaluate_cnv_wsi_fusion.py`, `stack_wsi_cnv.py`, `make_cnv_tabular.py`, `compare_fusion_ladder.py`, plus `cnv_controls` | in-process (the ladder comparison is a subprocess); CPU only |
| `dp-data` | `download_cnv_mutations.py`, `download_embeddings.py`, `fetch_*_labels.py` | plus `headline-artifacts` / `verify-artifacts` |
| `dp-cptac` | `tools/cptac/run_pipeline.sh` phases 1–4, plus a new phase 0 | phases 5–8 (dormant RNA) stay shell |
| `dp-config` | — | `show`, `validate`, `reference`, `sync-check` |

### 0. Before anything else

```bash
pip install -e '.[dev]'
dp-config validate    # paths absolute, tracked inputs present, ClamConf vs CLAM's parser
dp-config validate experiment=pam50_wsi_final   # ...plus the topk pin `--inst_loss svm` needs
dp-config sync-check  # the schema-drift check alone
```
`validate` and `show` accept `experiment=` / `fusion=` and then compose `conf/train.yaml` — the same
tree `dp-train` runs. Naming an experiment is what reaches the experiment-specific guards: the `topk`
import check fires only for `clam.inst_loss=svm`, which only `experiment/pam50_wsi_final.yaml` sets.

**There is no test suite, no Makefile and no automated path gate** — a pytest suite, a synthetic
smoke run and a `check-paths` gate were built and then deliberately reverted (commit `0c38c14`), so
`dp-config validate` and `dp-config sync-check` are the only automated checks. Two things this costs,
worth knowing before editing config: nothing re-verifies that `dpcode/conf/experiment/*` still
reproduces the original wrappers, and nothing enforces the no-absolute-paths rule. The pre-refactor
wrappers are kept byte-identical under `tests/legacy_wrappers/tools/` so the argv comparison can be
redone by hand against `dp-train --dry-run`.

### 1. CNV features (already on disk; re-run only to refresh)
```bash
dp-data cnv             # == download_cnv_mutations.py --what cna --representation arm --validate-arms
dp-data cnv --dry-run   # print the command, run nothing
```
Pulls cBioPortal `brca_tcga_pan_can_atlas_2018` / `brca_cptac_2020` plus UCSC hg38
`refGene`/`cytoBand`, and writes `.datasets/cnv/{tcga,cptac}_brca_cna_arm.csv` (981×39 and 114×39)
alongside `_gistic`, `_mutations`, `_mutation_matrix`. The cohort filter is on by default
(`acquire.all_cases=true` opts out) and is now **fatal** when its label table is missing — for CPTAC
that table comes from `dp-cptac phase=2`, so the CPTAC chain has to run first. It used to warn on
stderr and keep every case, silently producing a CPTAC matrix that was not 114 rows.

**`tools/data/reference/gene_arm_hg38.csv` + `CHECKSUMS.sha256` are now the tracked pin for the 39
features.** The gene→arm map is derived from a *live* UCSC refGene table and a `master` branch of the
cBioPortal datahub, so it is not reproducible across dates. `gene_arm_map()` resolves it as: the
cache under `.datasets/cnv/reference/` → the tracked copy (which seeds the cache) → and only then a
live rebuild, with a loud stderr warning that the resulting features may differ from the published
ones. Either copy is checksummed against the tracked digest on every run. So a fresh clone is
pinned; deleting both copies un-pins it, and every arm median — hence every AUROC — can move.

### 2. Reshape CNV into CLAM's tabular contract
```bash
dp-analysis make_cnv_tabular                       # both cohorts (default, unchanged)
dp-analysis make_cnv_tabular analysis.cohort=tcga  # TCGA only — no gated CPTAC chain needed
```
Writes `.scratch/cnv-tabular/{TCGA,CPTAC}_BRCA_CNV_arm_4class_clam.csv` (`case_id,label,<39 arms>`;
910 and 114 rows) and `chromosome_groups.csv` — 22 chromosome tokens over the 39 arms, the grouping
`--fusion_mode coattn` needs (`--tabular_group_spec prefix` would give 39 biologically empty
singleton tokens). **It exits non-zero if any case in the existing splits lacks CNV.** That check is
load-bearing: `multimodal_dataset.py` raises on a training case with no tabular row, and complete
coverage (910/910 non-Normal) is what lets the ladder reuse `splits/tcga_brca_subtyping_100` and
treat `pam50_final_s1` as a directly comparable WSI-only baseline instead of retraining it.

`--cohort` is new: the CPTAC table sits behind the whole gated CPTAC chain, and requiring it blocked
anyone reproducing the internal half of the thesis before finishing the external half.

### 3. The late-fusion baseline, its controls, and the ladder comparison
```bash
dp-analysis list                                   # the five actions, one line each
dp-analysis cnv_wsi_fusion                         # TCGA -> CPTAC external       (~45 s)
dp-analysis cnv_wsi_fusion analysis.internal=true  # + the TCGA head-to-head      (~70 s)
dp-analysis stack_wsi_cnv                          # learned rule vs the mean: no (~45 s)
dp-analysis cnv_controls                           # §3/§4 controls vs published  (~3 s warm)
dp-analysis compare_fusion_ladder                  # §8, the five ladder arms     (~55 s)
```
All CPU, all reading ≈1 MB of CSVs plus prediction pickles; none touches a slide. Each writes a
self-describing run directory under `.scratch/analysis/<action>/<timestamp>/` (`output.txt`,
`config.resolved.yaml`, `run_metadata.json`, and a JSON of the numbers where there is one);
`--no-run-dir` prints only. Missing inputs are named — with the command that produces each — *before*
the run directory is created, so a failed attempt leaves nothing behind. Flags and overrides may be
typed in either order. Direct invocation still works:
`python tools/evaluate_cnv_wsi_fusion.py --internal`.

`cnv_controls` exists because six published figures had **no producing script**: the §3 per-class
AUROC table (`report()` prints per-class *recall*, never AUROC), the error-independence φ, and four
rows of the §4 controls table — including the aneuploidy-burden number rule 1 makes non-negotiable.
It prints each recomputed value beside its published one and says `match` or `DIFFERS`. **It never
edits a document and never rewrites a number**; a mismatch is a finding for a human.

Shared plumbing lives in `tools/pam50_arms.py`: the CNV arm is exactly
`StandardScaler → LogisticRegression(max_iter=CNV_MAX_ITER, C=CNV_C, class_weight=CNV_CLASS_WEIGHT)`
= `(4000, 0.1, 'balanced')`, defined once so the scripts cannot silently describe different models.
The other frozen constants are named module attributes too:
`evaluate_cnv_wsi_fusion.{N_BOOT, BOOTSTRAP_SEED, CV_FOLDS, CV_SEED}` = `(4000, 7, 10, 0)`;
`stack_wsi_cnv.{N_BOOT, BOOTSTRAP_SEED, STACKER_C, STACKER_MAX_ITER, NM_XATOL, NM_FATOL, NM_MAXITER,
CLIP_FLOOR}` = `(2000, 11, 1.0, 4000, 1e-4, 1e-6, 2000, 1e-9)`. The two bootstrap seeds differ **on
purpose**; unifying them would change published confidence intervals.

### 4. The fusion-operator ladder (already run — do not re-run it)
```bash
dp-train --dry-run experiment=pam50_wsi_cnv fusion=film_attention   # the plan, nothing written
dp-train -m experiment=pam50_wsi_cnv fusion=concat,gated,cross_attention,film_attention,coattn
```
CLAM-MB `--model_size big`, `--tabular_hidden_dim 64 --tabular_top_n_features 0
--fusion_hidden_dim 32` (sized for a 39-dim modality, unlike the RNA wrapper's 256/10000), on
`splits/tcga_brca_subtyping_100`, k=10, seed 1. Results land in
`.scratch/results/pam50_wsi_cnv_<mode>_s1/`.

Two details keep H&E primary **in fact and not just in framing**: the WSI branch is warm-started
from `${paths.results_root}/pam50_final_s1/s_{fold}_checkpoint.pt` and is *not* frozen, so fusion has
to improve on that model rather than rediscover it; and `film_attention` conditions the *attention
network's input* on the tabular vector (`--film_rank 16 --modality_dropout 0.2`), re-ranking patches
instead of being appended after pooling.

`dp-train` **refuses** a run directory that already holds `summary.csv` or an `s_*_checkpoint.pt`.
`run.overwrite=true` defeats that guard and destroys results that cannot be recovered from git.

`residual` is deliberately **not** in the ladder — it needs a matched tabular-only checkpoint via
`--pretrained_rna_ckpt` and there is no supported way to train one, so `fusion=residual` refuses at
*composition* time with a message pointing at Known gaps.

### 5. External validation of a trained fusion checkpoint — **still blocked, see Known gaps**
```bash
dp-evaluate --dry-run                                    # the wrapper's defaults, printed
dp-evaluate evaluate.args.ckpt_dir=/abs/path/to/run_s1
```
`dp-evaluate` preserves both known gaps rather than fixing them, but turns them from silent or late
failures into a refusal before dispatch: a `film_attention` / `coattn` checkpoint directory is
rejected by name, and an architecture mismatch between the checkpoint's own
`experiment_<exp_code>.txt` and the config (most often `tabular_hidden_dim`, 64 for every ladder arm
against this config's 256) is **refused**, exit 2, with the overrides that would fix it — every key
checked is shape-critical and the evaluator loads with `strict=True`, so proceeding could only move
the same failure into `load_state_dict`. `--allow-arch-mismatch` dispatches anyway, for a checkpoint
whose recorded settings are wrong rather than its weights. That mismatch is why the ladder's old
printed evaluation hint never worked for **any** of its five arms.

### 6. The WSI-only arms (already run; re-run only to change the baseline)
```bash
dp-train experiment=pam50_wsi_final   # TCGA 10-fold CLAM-MB -> .scratch/results/pam50_final_s1
dp-cptac --dry-run phase=all          # the whole CPTAC pipeline, printed
dp-cptac phase=0                      # cohort metadata: both manifests + cohort.csv, no bulk transfer
dp-cptac phase=features,1,2           # GATED 16 GB archive -> provenance audit -> CLAM manifest
dp-cptac phase=3,4                    # frozen-weight 10-fold inference -> metrics
```
`pam50_wsi_final` is a frozen sweep-selected config (CLAM-MB/big, B=4, bag_loss ce,
bag_weight 0.5533776374353542, inst_loss svm, dropout 0.5, lr 1.007597588073064e-4,
reg 2.4456514744717547e-6, k=10, seed 1, W&B on). It writes `s_{0..9}_checkpoint.pt` and
`split_{0..9}_results.pkl` — consumed by `pam50_arms.py`, by the CPTAC inference, and by the ladder's
warm start. ≈66 min for 10 folds on one RTX 3090, measured from directory mtimes.

Phase 3–4 output is `.scratch/cptac_validation/results/predictions/ensemble_predictions.csv`
(378 slides → 114 cases, columns `slide_id,case_id,true_label,…,p_LumA,p_LumB,p_Basal,p_Her2`), which
is the WSI arm for the entire CNV analysis.

**Phase 0 is new and fixes a real trap**: `cohort.csv` is required by phase 2 and is written only by
`download_cptac.py` with more than one modality, so `--modality clinical` — which that script's
docstring used to recommend first — provably cannot produce it, and the old shell pipeline hit the
failure *after* the 16 GB gated download. All preconditions are now checked before any phase runs.
Phase 1 (`audit_feature_provenance.py`) verifies the CPTAC features are **geometrically** comparable
to TCGA's (256 px @ 20×, 0 overlap, 1536-dim) — geometry, not tissue segmentation; see Gotchas.

Raw CLAM invocation, if no wrapper fits:
```bash
cd project/CLAM
python main.py --task tcga_brca_subtyping --subtyping --model_type clam_mb --model_size big \
    --embed_dim 1536 --data_root_dir /abs/path/.datasets/tcga-brca/embeddings \
    --results_dir /abs/path/.scratch/results --split_dir tcga_brca_subtyping_100 --k 10
```
Valid `--task`: `tcga_brca_subtyping` (4-class, manifest `dataset_csv/tcga_brca_subtyping.csv`,
1009 slides — LumA 503, LumB 221, Basal 174, Her2 76, Normal 35 ignored), `tcga_brca_er`,
`tcga_brca_recurrence`, `nou_ctc_{ep,emt,any}`, `task_1_tumor_vs_normal`, `task_2_tumor_subtyping`.
Valid `--fusion_mode`: `concat`, `gated`, `residual`, `cross_attention`, `film_attention`, `coattn`
— all implemented in `project/CLAM/models/model_multimodal.py` (`CLAMRNAFusion`), with
`film_attention` and `coattn` the novel additions audited in
`docs/implementation-research/phase2-verification.md`.

## Hydra conventions that bite

- **`dp-train`'s primary config is `train`, NOT `config`** (`dpcode/conf/train.yaml`): `config.yaml`
  plus the `fusion` and `experiment` groups, because a group listed in the shared `config.yaml`
  would become mandatory for every entry point. `experiment` has no default — `dp-train` with none
  prints Hydra's "You must specify 'experiment'", and `dp-train --help` (with no experiment) prints
  a hand-written summary because Hydra's own help composes the config.
- **A value containing `{fold}` must be quoted.** Hydra's override grammar rejects a bare `{`:
  `dp-train … 'clam.pretrained_wsi_ckpt="/abs/s_{fold}_checkpoint.pt"'`. The placeholder is expanded
  by CLAM per fold — not by the shell, not by OmegaConf.
- **`+key=…` / `~key` overrides are refused** unless `run.allow_config_surgery=true`. Hydra suggests
  `+` whenever you typo a key under struct mode, and accepting it silently adds a key nothing reads:
  recorded in `.hydra/overrides.yaml`, logged to W&B, consumed by no one.
- **Two config-group directories are named around `.gitignore` traps and must not be renamed:**
  `dpcode/conf/tracking/` (a bare `wandb` pattern matches a directory of that name at any depth) and
  `dpcode/conf/analyses/` (same for a bare `analysis`). The `analyses` option files carry
  `# @package analysis` on line 1, so the config *key* is still `analysis` and every override is
  still spelled `analysis.<key>=…`. **Do not modify `.gitignore`** — anchoring the `wandb` pattern
  would un-ignore `project/CLAM/wandb/`'s 349 committed run directories.
- **`tracking=` does not reach CLAM.** It configures dpcode's own entry points; CLAM logs from
  `clam.wandb` / `clam.wandb_project` / `clam.wandb_tags`, which is what the wrappers passed.
  `dp-train` warns on stderr when the two disagree. To train without W&B: `clam.wandb=false`.
- **Hydra's own output directory is scratch** (`.scratch/hydra/…`, `.scratch/multirun/…`) and must
  never be pointed at a results directory: Hydra creates it and writes four files *before* the task
  function runs, so an overwrite guard there would fire too late. `dp-train` derives the run
  directory from `clam.exp_code`/`clam.seed` instead — which is also why `--multirun` gets exactly
  the same contract as a single run.
- **A training run's directory is self-describing**: `config.resolved.yaml` (Hydra's own
  `.hydra/config.yaml` is stored *unresolved*, so it does not replay on another machine),
  `run_metadata.json` (git, environment, seeds, timing, and the hyperparameters CLAM hard-codes
  outside argparse), `clam_argv.json`, `metrics.json`, plus a copy of `.hydra/`.
- **CLAM is never imported.** `project/CLAM/main.py` guards only `main(args)`; lines 119–433 run at
  import and would parse the *importing* process's `sys.argv`, seed torch, build the dataset and
  create a results directory. It is dispatched as a subprocess. Its parser is extracted by AST
  (`dpcode/clam_args.py`) so a config can be validated without executing it; `dp-config sync-check`
  is what stops `ClamConf` drifting from CLAM's 52 flags.

## Known gaps (real, blocking, unfixed)

- **`film_attention` and `coattn` checkpoints cannot be evaluated.**
  `project/CLAM/evaluate_multimodal.py:70` accepts only `auto|concat|gated|residual|cross_attention`,
  and the `evaluate_pam50_multimodal.sh` wrapper narrows that to `auto|concat|gated`. Under `auto`,
  `infer_fusion_mode` has no branch for either operator: a `film_attention` checkpoint (keys
  `film_bottleneck/film_gamma/film_beta/tabular_head`) raises `ValueError`, and a `coattn` checkpoint
  is misidentified as `cross_attention` because both build `self.cross_attention`, then dies in
  `load_state_dict(..., strict=True)`. `dp-evaluate` refuses both by name before dispatch. Two of the
  five ladder arms therefore have no evaluation path — which is why §8 is computed from the per-fold
  `split_*_results.pkl` by `compare_fusion_ladder`, not by an evaluator.
- **`dp-evaluate` (and `evaluate_pam50_multimodal.sh`) default to the TCGA test split**, not CPTAC.
  Swapping only the tabular table evaluates TCGA slides against CPTAC-shaped tabular rows. A genuine
  external run also needs the CPTAC embeddings and a CPTAC dataset_csv.
- **`residual` fusion has no trainable second branch.** It needs `--pretrained_rna_ckpt`, and
  `tools/train_pam50_tabular.sh` (the only tabular-only trainer) passes `--model_type tabular_mlp`,
  which `main.py`'s argparse rejects — the choice list is `clam_sb|clam_mb|mil`. `fusion=residual`
  refuses at composition rather than after creating a run directory.
- **The headline-artifact bundle has not been published.** `dp-data headline-artifacts` assembles it
  (15 files, ~1 MB) with a SHA256 manifest and prints the three publish steps; `dp-data
  verify-artifacts` checks a download against the tracked manifest at
  `docs/headline-artifacts.sha256`, **which does not exist yet**. Until it does, three of the four
  inputs to the cheap reproduction path cannot be obtained from a clone. Depositing it is an author
  decision (where, and whether the cohort licences permit redistributing derived data).
- **The ER thread is only partially ported, deliberately.** Each ER wrapper's *default* arm has an
  experiment config (`er_wsi_alone`, `er_wsi_rna_gated`, `er_wsi_clinpath_gated`, `er_wsi_rna_film`);
  the other chapter-2 arms do not, and the shims refuse them **by name, up front** — run
  `bash tools/train_er_multiseed.sh` to see which. Exhaustive coverage of a completed thread buys
  nothing for the headline table, and failing four hours into a matrix would cost.
- **The survival thread is doubly broken.** `tools/config/dataset/tcga_brca_survival.yaml` sets
  `embeddings_dir: .datasets/embeddings`, which does not exist (the real store is
  `.datasets/tcga-brca/embeddings`), and `scikit-survival` is an optional extra that has never been
  installed here, so `tools/{train,eval}_survival.py` fail at import before the path is reached.
- **Two published figures have no producing command at all**: the `mean` and `mean on true-Her2
  cases` columns of §2, and the cross-cohort platform correlation (per-arm r = 0.960) in §4. Both
  were computed interactively. `REPRODUCING.md` marks them as such.

## Gotchas and settled questions

Things that already cost a debugging session, or were measured and must not be re-litigated.

- **Two different class orders.** CLAM's `label_dict`, `make_cnv_tabular.CLASSES` and
  `infer_cptac_pam50.LABEL_MAP` are `LumA, LumB, Basal, Her2`; `tools/pam50_arms.CLASSES` is sorted
  `Basal, Her2, LumA, LumB`. `pam50_arms.clam_column_order()` exists solely to bridge them and
  asserts the recovered map is a permutation; `dp-analysis` additionally pins both orders from config
  before running anything. Reorder before scoring anything across the two.
- **CPTAC patient IDs carry a leading `X`** in cBioPortal (`X01BR001` → `01BR001`). The CPTAC label
  column is `label_name`, not `label`.
- **`--embed_dim 1536` everywhere** — UNI2-h features are 1536-dim.
- **CLAM's 10 splits are drawn independently, not partitioned.** 599 of 910 cases land in at least
  one test fold and 242 in two to five, so "WSI alone" is a small ensemble flattered by roughly
  +0.01 AUROC. An audit confirmed no leakage, and re-running with a random stratified partition
  gives the same verdict — but say so when the number is reported.
- **`dataset_csv/tcga_brca_subtyping.csv` and `splits/tcga_brca_subtyping_100/` are distributed
  primary inputs, not derived artifacts.** 9 readers, 0 writers; the `create_splits_seq.py`
  invocation behind those folds is recorded nowhere. Regenerating them with a different draw
  invalidates `pam50_final_s1`, `ensemble_predictions.csv`, all five ladder arms and the headline
  table. `paths.splits_root` / `paths.dataset_csv_dir` / `paths.labels_dir` are declared tracked
  inputs and no entry point writes into them.
- **The ladder is a *near*-baseline comparison, not a matched one.** `pam50_final_s1` used the
  sweep-selected `lr 1.008e-4 / reg 2.446e-6 / bag_weight 0.553 / inst_loss svm` with instance
  clustering ON; the five ladder arms use rounded `lr 1e-4 / reg 2.5e-6`, CLAM's default
  `bag_weight`, no instance loss and `--no_inst_cluster`. Those three are inert on the multimodal
  path (`utils/core_utils.py` routes to `train_loop`), but the optimiser configuration genuinely
  differs. Reproduced exactly, never tidied away.
- **The external Her2 collapse is domain-shift-induced, not a calibration artifact — the
  calibration hypothesis was tested and refuted.** The WSI arm calls Her2 0/14 on CPTAC, still 0/14
  after a 12× prior boost, and 0/14 under unsupervised SLD-EM (which drives the implied Her2 prior
  to 0.000 against a true 0.123). Internally the same model calls 26/51. Do not re-run prior
  rebalancing expecting a fix; remediation belongs on the imaging side (stain normalisation,
  encoder choice).
- **CPTAC and TCGA features are geometrically comparable, not identically preprocessed.** The
  provenance audit verifies 256 px @ 20×, 0 overlap, 1536-dim. But the CPTAC `.h5` files carry CLAM
  `create_patches_fp` attributes while TCGA's carry Trident ones, so the *tissue segmentation* — and
  therefore which tiles enter each bag — differs. Never write "preprocessing is held constant"; it is
  a live confound for the domain-shift explanation of the Her2 collapse.
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
- **Aneuploidy burden alone reaches 0.685 macro AUROC** — `mean_abs_log2`, 5-fold at seed 0, which is
  the published protocol; 0.6893 ± 0.0033 over ten seeds, and 0.673 under the `frac_altered` reading
  of "burden", which is why the definition is a config key rather than an unwritten assumption.
  Report it alongside the 39-arm model, or Basal ≈ 0.97 reads as pure genome instability rather than
  an arm *pattern*.
- **Patient/slide-level splitting** — all patches of a slide stay in one fold, and cases are never
  split across folds. CLAM handles this via its split CSVs; custom scripts must replicate it.
- **CPTAC is stage IIA–IIIC by eligibility and has no Normal-like**, the likely reason the CNV arm
  scores higher externally (0.888) than internally (0.866). Treat that as "does not degrade", not
  "improves".
- **Power**: external Her2 n = 14, LumB n = 17. Per-class external estimates are indicative.
- **Determinism, stated honestly.** Seeds and fold assignments are fixed; run-to-run variance has
  **not** been measured, so no tolerance is quoted anywhere; bitwise reproducibility is neither
  claimed nor achievable, because `cross_attention` and `coattn` use `nn.MultiheadAttention` and
  neither `torch.use_deterministic_algorithms` nor `CUBLAS_WORKSPACE_CONFIG` is set. Setting either
  would change published numbers, so `dpcode/determinism.py` *records* the residual nondeterminism
  into `run_metadata.json` and fixes none of it. Note also that `main.py` sets `PYTHONHASHSEED`
  inside an already-running interpreter, where it does not affect string hashing — it is recorded as
  observed environment, never as a seed in effect.

## Positioning (from the 50-paper survey in `docs/implementation-research/PAM50/`)

- **No published multimodal PAM50 model has an external, never-trained-on, PAM50-specific
  evaluation.** That is the slot this project occupies.
- Amer et al. 2025 (arXiv:2509.03408) is the **only** WSI+CNV PAM50 work: 10-fold CV on TCGA only,
  CNV-alone 0.8284, four-modality fusion 0.9153, both internal.
- **Do not cite TANGLE, THREADS, HE2RNA, SEQUOIA or Path2Omics as PAM50-from-histology precedent** —
  verified by full-text grep, none of them evaluate PAM50.
- Gated fusion appears in exactly two PAM50 papers and loses in both; "fusion does not beat the
  strong unimodal arm" is reported independently by four groups — and none of them reports the
  trivial average, which §8 now shows beats every operator tried here.

## Layout

```
dpcode/              # the configuration and entry-point layer; nothing moved out of project/ or tools/
├── conf/            # paths sources clam tracking | experiment fusion analyses evaluate acquire cptac
│   ├── config.yaml  #   primary config for every entry point EXCEPT dp-train
│   └── train.yaml   #   dp-train's primary config: config + fusion + experiment
├── cli/             # train.py evaluate.py analysis.py data.py cptac.py config.py
├── paths.py         # resolve_paths, assert_paths_absolute, assert_paths_exist
├── schema.py        # structured configs, reject_appended_overrides
├── clam_args.py     # clam_parser (AST extraction), render_argv, validate_clam_args
├── runinfo.py       # assert_run_dir_writable, write_config_snapshot, write_metrics, RunMetadata
├── determinism.py   # records seeds and residual nondeterminism; fixes nothing
└── wandb_util.py

project/
├── CLAM/            # Vendored CLAM + this project's multimodal fork:
│   ├── main.py                    # tasks, --fusion_mode and the --tabular_* / --pretrained_* flags
│   ├── models/model_multimodal.py # CLAMRNAFusion, TabularMLPEncoder, the 6 fusion operators
│   ├── multimodal_dataset.py      # raises if a training case has no tabular row
│   ├── evaluate_multimodal.py, evaluate_late_fusion.py, evaluate_selective_ensemble.py,
│   │   evaluate_confidence_routing.py, train_rna.py, sweep_train*.py
│   ├── dataset_csv/*.csv          # per-task manifests        <- TRACKED PRIMARY INPUT
│   └── splits/<task>_100/         # fold definitions          <- TRACKED PRIMARY INPUT
├── data/            # feature_datamodule.py, patch_dataset.py, transforms.py, pam50.R
│                    #   (run_patching.sh is DEAD — it calls tools/patch_wsi.py, which is gone)
├── survival/        # AMIL survival package — dormant and broken (Known gaps)
├── MCAT/            # Vendored MCAT — dormant
├── UNI/             # Vendored UNI feature extractor — untracked; only needed to tile new slides
├── Selective-Multimodal-…/  # Untracked reference clone of Hezil et al. 2025 (has its own .git)
└── base/, loggers/  # Legacy scaffold; checkpointer.py

tools/
├── download_cnv_mutations.py, make_cnv_tabular.py, pam50_arms.py            # CNV arm
├── evaluate_cnv_wsi_fusion.py, stack_wsi_cnv.py, compare_fusion_ladder.py   # fusion + ladder
├── train_pam50_final.sh, run_cnv_fusion_ladder.sh, train_pam50_multimodal.sh,
│   evaluate_pam50_multimodal.sh, train_er_*.sh   # ALL SHIMS over dp-train / dp-evaluate now
├── extract_features.py, create_clam_dataset_csv.py, download_embeddings.py
├── download_cptac.py + cptac/     # CPTAC download, provenance audit, manifest, inference
├── build_er_labels.py, make_er_*.py, evaluate_er_ablation.py, analyze_er_ablation.py  # ER (complete)
├── analyse_clinicopath_pam50.py   # clinicopathological baseline (null result)
├── data/           # Label tables + reference/gene_arm_hg38.csv + CHECKSUMS.sha256 (the 39-arm pin)
├── rna/            # GDC RNA download + harmonisation (dormant modality, kept for ablations)
├── diagnostics/    # gate_probe.py, tabular_only_probes.py — fusion-gate ablations
├── nou/, hsi_bc/   # Dormant cohorts (nou/patch_nou_*.py and CLAM's create_patches_fp.py are
│                   #   the only working tile extractors left)
└── config/         # Legacy Hydra configs — only the (broken) survival path is live

tests/
├── legacy_wrappers/tools/*.sh   # frozen pre-refactor wrappers, executed by the parity check
└── parity, schema, paths, config-composition and synthetic end-to-end tests

docs/
├── cnv-wsi-fusion-external-validation.md   # the headline result — USER-OWNED; read, never write
├── config-reference.md                     # GENERATED by `dp-config reference`; never hand-edit
├── parked-cohorts/histology-hsi-bc.md      # the upstream HSI-BC README + its Apache attribution
├── er-prediction-results.md, er-external-validation-results.md
└── implementation-research/                # 17 planning/literature reports, incl.
    ├── next-steps-action-plan.md           #   the RNA target-leakage finding
    ├── phase2-verification.md              #   audit of film_attention / coattn
    ├── novel-fusion-design.md, novelty-risk-check.md
    └── PAM50/README.md, paper-dossier.md, survey-data.json   # the 50-paper survey

README.md            # what the project is, the headline result, the six entry points
REPRODUCING.md       # empty machine -> headline table: the cheap path, then the deep one
CITATION.cff         # verifiable metadata + the unresolved licence question
```

## Data and output locations

| Path | Contents |
|---|---|
| `.datasets/tcga-brca/embeddings/` | 1126 UNI2-h `.h5` files (no WSIs or patches live here) |
| `.datasets/cptac-brca/` | 653 feature `.h5`, `wsi/` (read by no script), `rna/`, `clinical/`, `cptac_brca_pam50_dataset.csv` |
| `.datasets/cnv/` | Arm-level + GISTIC CNA and mutation matrices for both cohorts; `reference/gene_arm_hg38.csv` is the *cache* of the tracked pin |
| `.datasets/nou/`, `.datasets/HistologyHSI-BC-Recurrence/` | Dormant cohorts |
| `.scratch/cnv-tabular/` | CLAM-format CNV tabular inputs + chromosome grouping |
| `.scratch/results/` | CLAM runs — `pam50_final_s1` (WSI baseline), the five `pam50_wsi_cnv_*` ladder arms, `pam50_wsi_rna_*`, `er/` |
| `.scratch/analysis/<action>/<timestamp>/` | `dp-analysis` run directories: `output.txt`, `config.resolved.yaml`, `run_metadata.json`, results JSON |
| `.scratch/hydra/`, `.scratch/multirun/` | Hydra's own output; nothing load-bearing lives there |
| `.scratch/headline-artifacts/` | `dp-data headline-artifacts` output (bundle + `MANIFEST.sha256`) |
| `.scratch/cptac_validation/` | CPTAC external inference, incl. `results/predictions/ensemble_predictions.csv` |
| `.scratch/rna-gdc/`, `.scratch/TCGA-BRCA-rna/` | Harmonised GDC tables / legacy Xena tables |
| `.scratch/analysis/clinicopath_pam50/`, `.scratch/harmonisation/` | Clinicopath outputs (the harmonised CSVs have no producer) |
| `.scratch/checkpoints/uni2-h/` | UNI2-h encoder weights — used only by the parked cohorts' tiling path |
| `tools/data/*.csv` | Label tables (PAM50, ER, OS, clinicopathological) |

`.datasets`, `.scratch`, `wandb`, `papers`, `analysis` and `.claude` are gitignored — **nothing there
is recoverable from git**, including the five ladder runs and `pam50_final_s1`.

Every location above is a `paths.*` key. `dp-config show` prints what they resolve to, and
`DP_REPO_ROOT` / `DP_DATA_ROOT` / `DP_SCRATCH_ROOT` / `DP_RESULTS_ROOT` relocate them
(see `.env.example`).

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
  histology-only c-index 0.560, n.s.). Doubly broken — see Known gaps.
- **CTC cohort (`nou`)** and **HistologyHSI-BC recurrence.** Parked. HSI-BC was tiled at 62.3 µm FOV
  against TCGA's 128 µm, so every existing HSI-BC external number is void until it is re-tiled
  (`docs/parked-cohorts/histology-hsi-bc.md`). The `nou` cohort is **private institutional data**: no
  filename, case identifier or directory listing from it belongs in tracked config, documentation or
  a public W&B project, and `paths.nou_root` has no committed default (`DP_NOU_ROOT`).

## Legacy

`project/base/`, `tools/train.py`, `tools/main.py` and the `prostate`/`timm`/`basic` Hydra groups
under `tools/config/` are scaffold from an earlier template. `tools/train.py` is broken — it imports
`project.experiment.Experiment`, which does not exist (the classes live in `project/base/`); it is
plain argparse and never loads Hydra, so `default.yaml`'s `project.trainer.Trainer` target is a
separate defect. The `toyproblem.*` targets are in `tools/config/dataset/prostate.yaml`,
`tools/config/model/timm.yaml` and `tools/config/augmentation/basic.yaml`. None of it is on a live
path.

The root `README.md` is now this project's README. It used to be the upstream HistologyHSI-BC dataset
README plus a dump of stale commands, every one of which called a script that no longer exists; the
dataset material it carried is preserved at `docs/parked-cohorts/histology-hsi-bc.md`.

## Environment

```bash
pip install -e '.[dev]'    # the only correct install
```
- Python **3.10 or 3.11** (`torch==2.0.1` has no cp312 wheels). `python`/`pip` already resolve to
  `/opt/venv` here — no `source` needed. Re-check `command -v python` if anything looks like a
  missing dependency.
- Editable is not a preference: a non-editable install copies `project/` and `tools/` into
  site-packages **without** the sibling data directories they read by relative path
  (`project/CLAM/dataset_csv/`, `project/CLAM/splits/`, `tools/data/`).
- Pinned: hydra-core 1.3.4, omegaconf 2.3.0, torch 2.0.1+cu117, wandb 0.15.3, pytest 8.4.2.
  `topk` (smooth-topk) is a git commit pin with no PyPI fallback and is required by
  `--inst_loss svm`, i.e. by the frozen WSI baseline; `dp-config validate` imports it so that
  failure arrives at validate time rather than minutes into training.
- Hardware of record: 2× RTX 3090 (24 GB each), CUDA 11.7.
- **Docker is not a supported route**: the base image's availability is unverified, `docker/run.sh`
  requires a positional GPU argument and bind-mounts one cluster's host paths.
- W&B is on in `experiment=pam50_wsi_final` and the ER experiments and off in the CNV ladder, exactly
  as in the wrappers they replace. Project `clam-brca-subtyping-cv`; `clam.wandb=false` turns it off.
- CLAM and MCAT keep their own `requirements.txt` / `env.yml`. **Do not install them** — CLAM's copy
  omits `tensorboardX` and `topk`, both of which CLAM itself imports. Install the root package.
