# Reproducing the results

From an empty machine to the headline table in
[`docs/cnv-wsi-fusion-external-validation.md`](docs/cnv-wsi-fusion-external-validation.md).

There are two paths and they are very different in cost:

| | inputs | GPU | gated downloads | wall clock |
|---|---|---|---|---|
| **A — the cheap path** (the external half of the headline table, and the internal head-to-head) | 4 files, ≈340 KB (plus ≈430 KB for `--internal`) | none | none | about a minute |
| **B — the deep path** (rebuild every input from source) | ≈105 GB | 1× 24 GB card | 1 HuggingFace repo, approval required | ≈4 h GPU + download time |

Read path A first even if you intend to run path B: it tells you what the ≈340 KB actually is, and
it is the honest measure of how much of this project's evidence is portable.

---

## 0. What "reproducible" means here

Stated precisely, because the difference matters when a number does not match:

- **Seeds and fold assignments are fixed.** `project/CLAM/main.py` seeds `random`, NumPy and torch
  per fold, `cudnn.deterministic=True`, `cudnn.benchmark=False`, and the ten cross-validation folds
  are tracked CSV files rather than a runtime draw.
- **Run-to-run variance has not been measured.** No tolerance is quoted anywhere in this repository,
  because nobody has run the same configuration twice and compared. Do not invent one.
- **Bitwise reproducibility is neither claimed nor achievable here.** Two fusion operators
  (`cross_attention`, `coattn`) use `nn.MultiheadAttention`, and neither
  `torch.use_deterministic_algorithms` nor `CUBLAS_WORKSPACE_CONFIG` is set. Setting either would
  change the numerical behaviour behind every published number, so this refactor did not.
- The CPU analyses (`dp-analysis`) are a different case: their inputs are files on disk and their
  seeds are literals in `dpcode/conf/analyses/*.yaml`. When this document was written, every point
  estimate, every per-class recall, every significance verdict and every control printed by
  `dp-analysis cnv_wsi_fusion`, `stack_wsi_cnv`, `cnv_controls` and `compare_fusion_ladder` matched
  its published value exactly — including all twelve cells of the per-class AUROC table and all four
  control rows. **One residual discrepancy:** the *external* bootstrap confidence intervals printed
  by `cnv_wsi_fusion` differ from the §1/§3 tables in the third decimal (`[0.793, 0.897]` where the
  document says `[0.791, 0.895]`, and similarly ≤0.003 elsewhere in those two tables), consistent
  with those cells having been produced by a different bootstrap draw. §5's internal CIs and every
  CI in §7 match exactly. The discrepancy is recorded here rather than resolved by editing anything.
- `run_metadata.json` in every run directory records the seeds in effect **and** the residual sources
  of nondeterminism. It does not fix them.

---

## Path A — the cheap path

### A.1 What it needs

`tools/evaluate_cnv_wsi_fusion.py` — which is what `dp-analysis cnv_wsi_fusion` runs — reads exactly
these files. It never opens a slide, never loads torch, and never touches a GPU.

| file | size | in a fresh clone? | what it is |
|---|---|---|---|
| `tools/data/tcga_brca_pam50_labels.csv` | 18 KB | **yes, tracked** | 981 PAM50 calls from cBioPortal |
| `.datasets/cnv/tcga_brca_cna_arm.csv` | 247 KB | no | 981 × 39 arm medians — the CNV arm is *fit* on TCGA |
| `.datasets/cnv/cptac_brca_cna_arm.csv` | 29 KB | no | 114 × 39, the external input |
| `.scratch/cptac_validation/results/predictions/ensemble_predictions.csv` | 45 KB | no | 378 slides → 114 cases, the WSI arm's frozen 10-fold ensemble output |

`analysis.internal=true` additionally reads `.scratch/results/pam50_final_s1/split_{0..9}_results.pkl`
(≈430 KB, pure NumPy, no torch needed) and the tracked
`project/CLAM/splits/tcga_brca_subtyping_100/`.

### A.2 The distribution problem, stated plainly

**Three of those four files are gitignored and there is no release mechanism today.** No LFS, no
`.gitattributes`, no Zenodo deposit, no GitHub release. So on a fresh clone, path A cannot be run
at all yet — the code is portable, the inputs are not.

What exists is the machinery to fix that:

```bash
dp-data headline-artifacts --dry-run   # what would be bundled, from your machine
dp-data headline-artifacts             # copies them + writes MANIFEST.sha256 + a README
```

Verified output of the dry run on the machine of record: **15 files, 1044 KB** — the four files
above, the ten fold pickles, and `gene_arm_hg38.csv` (the pin on the 39-feature space, also tracked
at `tools/data/reference/`). The bundle's layout matches a default clone, so installing it is
`cp -a <bundle>/. <clone>/`.

A downloaded bundle is checked against a **tracked** manifest:

```bash
dp-data verify-artifacts --bundle <dir>
```

```
No tracked manifest at <your clone>/docs/headline-artifacts.sha256.
That file is written by the publish step of `dp-data headline-artifacts` (step 3) and is what lets
a download be checked against a checksum that arrived with the clone. It has not been published yet.
```

That is the current state (the real message prints an absolute path; it is abbreviated here so this
document contains no machine-specific paths). **The author has not published the bundle**, so
`docs/headline-artifacts.sha256` does not exist and there is no download URL.

With no tracked manifest, `verify-artifacts` falls back to the bundle's **own** `MANIFEST.sha256`
and says so — that checks the bundle is internally consistent, **not** that it is the right bundle.
Against a freshly built bundle it prints `all 15 files verified`.

> **Download URL: _`<PLACEHOLDER — the bundle has not been deposited anywhere yet>`_**
>
> Publishing it takes three author decisions no script can make: where (Zenodo gives a citable DOI, a
> GitHub release does not, institutional storage may not outlive the thesis), whether the licence
> terms of TCGA-BRCA and CPTAC-BRCA permit redistributing these derived files (arm-level medians and
> model outputs — not primary patient data), and then committing the manifest to
> `docs/headline-artifacts.sha256`. `dp-data headline-artifacts` prints those three steps when it
> finishes.

Until then, the only way to obtain the three untracked files is path B, or an email to the author.

### A.3 Running it

```bash
pip install -e .
dp-analysis cnv_wsi_fusion                          # external, ~45 s
dp-analysis cnv_wsi_fusion analysis.internal=true   # + internal head-to-head, ~70 s
```

Expected output, abridged, from a verification run on the machine of record (see section 0 for the
one place these numbers and the published tables differ):

```
=== EXTERNAL CPTAC (TCGA-trained, nothing refit) ===
CPTAC WSI predictions: 378 slides -> 114 cases
external set: 114 cases  {'LumA': 56, 'Basal': 27, 'LumB': 17, 'Her2': 14}

             model  macroAUROC         95% CI  balAcc Basal  Her2  LumA  LumB
           WSI raw       0.847 [0.793, 0.897]   0.513 24/27  0/14 52/56  4/17
WSI prior-balanced       0.865 [0.813, 0.911]   0.554 26/27  0/14 34/56 11/17
     CNV (39 arms)       0.888 [0.833, 0.935]   0.716 22/27 12/14 37/56  9/17
        Fusion raw       0.909 [0.861, 0.949]   0.646 23/27  6/14 50/56  7/17
   Fusion balanced       0.912 [0.864, 0.952]   0.740 24/27 10/14 43/56 10/17
```

```
=== INTERNAL TCGA, 599 cases with CLAM out-of-fold, both 10-fold ===
        model  macroAUROC         95% CI  balAcc   Basal  Her2    LumA   LumB
          WSI       0.887 [0.865, 0.908]   0.677  91/106 26/51 257/318 66/124
CNV (39 arms)       0.862 [0.836, 0.888]   0.662  92/106 26/51 227/318 69/124
       Fusion       0.922 [0.904, 0.938]   0.747 100/106 32/51 256/318 76/124
```

Each `dp-analysis` action writes a self-describing run directory under
`${paths.analysis_dir}/<action>/<timestamp>/` containing `output.txt`, `config.resolved.yaml`,
`run_metadata.json` and, where the action produces one, a machine-readable `*.json`. Pass
`--no-run-dir` for a look at the table without leaving a record.

**Always report the CNV-alone row.** Fusion's edge over CNV alone is ΔAUROC +0.024 with a CI lower
bound of exactly +0.000; the balanced-accuracy difference is not significant. And the baseline the
fusion operators have to beat is the equal-weight mean, not the WSI-only model.

---

## Path B — rebuild everything from source

### B.0 What you need before you start

| | |
|---|---|
| Python | 3.10 or 3.11 — `torch==2.0.1` publishes no 3.12 wheels |
| GPU | one NVIDIA card ≥ 24 GB for training; driver new enough for CUDA 11.7. The machine of record is 2× RTX 3090, torch 2.0.1+cu117 |
| Disk | ≈105 GB of downloads (66 GB TCGA features + 34 GB CPTAC features + ≈5 GB metadata/CNV), and ≈50 GB free *during* CPTAC extraction on top of that |
| Network | `git` at install time (`topk` is a git commit pin with no PyPI fallback, and `--inst_loss svm` needs it) |
| Accounts | a HuggingFace account **with approved access** to `MahmoodLab/UNI2-h-features` |
| Time | ≈66 min GPU for the WSI baseline, ≈2 h 38 min for the five-arm ladder, ≲1 min for CPTAC inference, plus the download |

Every timing above is measured from run-directory mtimes on the machine of record, not from a
stopwatch, and reflects a single card running the folds sequentially.

Each acquisition step below is marked **open** (plain HTTPS, no account: cBioPortal, UCSC, the GDC
open tier, TCIA PathDB, Zenodo) or **gated** (an approval request that can be refused). Nothing in
this project sits in the middle tier of "an account but no approval" — either you need no
credentials at all, or you need an approved HuggingFace token.

### B.1 Four ordering constraints that fail silently or strand you

These are not warnings. Each one is a hard ordering constraint; three of them used to fail *after*
an expensive step.

**1. CPTAC acquisition must start with `dp-cptac phase=0`.**
`tools/cptac/prepare_cptac_manifest.py` requires `.datasets/cptac-brca/cohort.csv`, and that file has
exactly one producer: `download_cptac.py` invoked with **more than one** modality. The route the
module docstring used to recommend first — `--modality clinical` — provably cannot write it, and the
old shell pipeline began with the 16 GB gated feature download and only then ran the phases that
need the manifests, so the failure arrived after the download. `dp-cptac phase=0` runs the one
invocation that writes both manifests **and** `cohort.csv` with no bulk transfer, and `dp-cptac`
checks every phase's preconditions before running any phase.

**2. The CNV download needs the CPTAC dataset table, and that is now a hard failure.**
`download_cnv_mutations.py` filters each cohort to the cases that have WSI features, reading
`tools/data/tcga_brca_pam50_labels.csv` (tracked) for TCGA and
`.datasets/cptac-brca/cptac_brca_pam50_dataset.csv` (written by `dp-cptac phase=2`) for CPTAC. It
used to warn on stderr and fall back to "keep every case", which *succeeded* and wrote a CPTAC matrix
with far more than 114 rows. It now exits with a message naming the missing file. The escape hatch is
explicit — `dp-data cnv acquire.all_cases=true` — and the script says plainly that it does not
reproduce the documented 981/114 shapes. (It does not change any *value*: the gene axis and the
per-case arm medians do not depend on which cases are kept, only the row set does.)

**3. A TCGA-only reader can build the tabular input without the gated CPTAC chain.**
`make_cnv_tabular` used to read the CPTAC manifest unconditionally, so the *internal* half of the
thesis could not be rebuilt before finishing the *external* half. It now takes `--cohort`:

```bash
dp-analysis make_cnv_tabular analysis.cohort=tcga     # or python tools/make_cnv_tabular.py --cohort tcga
```

The default is still `both`, so the documented invocation is unchanged.

**4. Two gated HuggingFace repositories, and approval can be refused.**

| repo | what it carries | needed for |
|---|---|---|
| `MahmoodLab/UNI2-h-features` | **gated** — pre-extracted UNI2-h features for *both* cohorts | everything on the deep path |
| `MahmoodLab/UNI2-h` | **gated** — the encoder weights | nothing on this path; only the parked cohorts, which tile raw slides |

Request access on the hub, then export `HF_TOKEN`. There is no workaround: without the features you
cannot train the WSI arm, and the encoder route (tiling 1,126 + 654 gigapixel slides yourself) costs
hours to days, needs the second gated repo *and* the untracked `project/UNI/` clone, and is never
required — both cohorts ship pre-extracted.

### B.2 The sequence

```bash
# 0. install and check, before downloading 100 GB
pip install -e '.[dev]'
dp-config validate
dp-config sync-check
```

```bash
# 1. labels — already tracked. Skip this unless you mean to refresh them; both outputs live in
#    the git-tracked tools/data/, so check `git diff` afterwards. A changed label table changes
#    every published number.
dp-data labels --dry-run          # open (cBioPortal REST; the TCGA-CDR sheet is an Elsevier CDN)
# dp-data labels                  # drop --dry-run to actually fetch
```

```bash
# 2. TCGA WSI features — GATED, 66 GB, 1126 .h5
export HF_TOKEN=hf_...
dp-data embeddings                          # acquire.cohort=tcga-brca is the default
```

```bash
# 3. CPTAC — metadata first, then the gated archive, then the two local phases
dp-cptac --dry-run phase=all                # print every command, run nothing
dp-cptac phase=0                            # open; manifests + cohort.csv, no bulk transfer
dp-cptac phase=features                     # GATED, 16 GB archive -> 34 GB, 653 .h5
dp-cptac phase=1                            # provenance audit of the feature store
dp-cptac phase=2                            # coverage + cptac_brca_pam50_dataset.csv (378/114)
```

The CPTAC `.svs` slides (68 GB with `--cohort-only`) are **never read by any script in this
repository** — only two manifest columns are. Do not download them.

```bash
# 4. copy number — open (cBioPortal datahub + UCSC hg38), ~50 MB transfer
dp-data cnv                                 # == --what cna --representation arm --validate-arms
```

```bash
# 5. reshape CNV into CLAM's tabular contract
dp-analysis make_cnv_tabular                # writes ${paths.cnv_tabular_dir}, exits non-zero on a coverage hole
```

```bash
# 6. the WSI baseline — GPU, ~66 min for 10 folds
dp-train --dry-run experiment=pam50_wsi_final    # see the exact CLAM command first
dp-train experiment=pam50_wsi_final
```

This experiment passes `--wandb` (the wrapper it replaces did, unconditionally). On a machine with
no W&B credentials, add `clam.wandb=false`, or set `WANDB_MODE=offline`. The `tracking=` config
group configures dpcode's own entry points and does **not** reach CLAM; `dp-train` prints a note on
stderr if you select one expecting it to.

```bash
# 7. CPTAC inference with the frozen TCGA-trained checkpoints — GPU, <1 min
dp-cptac phase=3
dp-cptac phase=4
```

Nothing in phases 3–4 is fitted, tuned, calibrated or thresholded on CPTAC.

```bash
# 8. the results
dp-analysis cnv_wsi_fusion
dp-analysis cnv_wsi_fusion analysis.internal=true
dp-analysis stack_wsi_cnv
dp-analysis cnv_controls
```

```bash
# 9. optional: the fusion-operator ladder — GPU, ~2 h 38 min for five arms
dp-train -m experiment=pam50_wsi_cnv fusion=concat,gated,cross_attention,film_attention,coattn
dp-analysis compare_fusion_ladder
```

The ladder warm-starts each WSI branch from `pam50_final_s1`, so step 6 must come first. `dp-train`
refuses to write into a run directory that already holds `summary.csv` or an `s_*_checkpoint.pt`;
`run.overwrite=true` defeats that guard and destroys 2 h 38 min of results that live in a gitignored
tree. `--dry-run` says so before you find out.

### B.3 Never regenerate these

| artifact | why |
|---|---|
| `project/CLAM/dataset_csv/tcga_brca_subtyping.csv` | The slide→PAM50 join that *defines* the 4-class task (1009 slides / 943 cases). **9 readers, 0 writers** — it is a distributed primary input with no recorded derivation. |
| `project/CLAM/splits/tcga_brca_subtyping_100/` | The exact fold draw behind `pam50_final_s1`, `ensemble_predictions.csv`, all five ladder arms and the whole headline table. `create_splits_seq.py` can produce *a* set of splits, but the invocation that produced *these* is recorded nowhere. A different draw invalidates every standing number. |
| `tools/data/reference/gene_arm_hg38.csv` | The Hugo-symbol → chromosome-arm map, and the only pin on the 39 features. It was derived from UCSC hg38 refGene + cytoBand, and **neither source is pinned** — refGene is a live table, so re-deriving it on a later date can move gene→arm assignments, hence every arm median, hence every AUROC. Resolution order in `download_cnv_mutations.py` is: the cache under `.datasets/cnv/reference/`, then this tracked copy (which seeds the cache), and **only then** a live rebuild — with a loud stderr warning saying the features may differ from the published ones. Either copy is checksummed against `tools/data/reference/CHECKSUMS.sha256` on every run. A fresh clone is therefore pinned; deleting both copies un-pins it. |

No entry point writes into any of them. `paths.splits_root`, `paths.dataset_csv_dir` and
`paths.labels_dir` are declared tracked inputs in `dpcode/conf/paths/default.yaml`.

### B.4 Things a reader will try, and regret

| tempting | what actually happens |
|---|---|
| `python tools/extract_features.py` to build the features yourself | hours to days for 1,780 gigapixel slides; needs the untracked `project/UNI/` **and** the second gated repo; never necessary |
| `download_cptac.py --modality wsi` | 68 GB of `.svs` that no script in this repository opens |
| `cd project/CLAM && pip install -r requirements.txt` | that file omits `tensorboardX` and `topk`, both of which CLAM itself imports; `--inst_loss svm` then dies mid-training. Install the root package instead |
| `python create_splits_seq.py …` to "regenerate" the splits | silently produces a different fold draw, invalidating every published number (B.3) |
| deleting `gene_arm_hg38.csv` (both the cache and the tracked copy) and re-running `dp-data cnv` | re-derives the gene→arm map from a live UCSC table, silently changing the 39-feature space (B.3). With either copy present you are pinned |
| `cd docker && ./run.sh` | `set -eu` plus a required positional GPU argument; the mounts are one cluster's paths |
| `python tools/train_survival.py` | `scikit-survival` is not installed (it is an optional extra), and the survival config's `embeddings_dir` points at a path that does not exist. Dormant thread |
| running anything from the root `README.md`'s old `## Commands` block | it is gone; every command in it referenced a deleted script |

---

## Which command produces which published number

Row references are to `docs/cnv-wsi-fusion-external-validation.md`.

| published | command | needs |
|---|---|---|
| §1 headline table, §3 matched-decision-rule table, §3 per-class recall, §5 internal table | `dp-analysis cnv_wsi_fusion [analysis.internal=true]` | path A inputs |
| §2 recalibration control — `max p_Her2` raw / prior-balanced, "0 cases argmaxed to Her2" | `dp-analysis cnv_wsi_fusion` | path A inputs |
| §2 columns **`mean`** and **`mean on true-Her2 cases`** | **no producer** — computed interactively; nothing in the repository prints them | — |
| §3 per-class **AUROC** table, §3 error-independence φ | `dp-analysis cnv_controls` | path A inputs |
| §4 internal 5-fold × 10 seeds, leave-one-site-out, aneuploidy-burden-only, C sweep | `dp-analysis cnv_controls` | path A inputs |
| §4 arm derivation vs TCGA's official arm calls | `dp-data cnv` (i.e. `--validate-arms`) | network |
| §4 cross-cohort platform check (per-arm r = 0.960) | **no producer** — computed interactively | — |
| §7 learned-stacker table, internal and external | `dp-analysis stack_wsi_cnv` | path A inputs |
| §8 fusion-operator ladder, model-count control, error-correlation φ, FiLM diagnostics | `dp-analysis compare_fusion_ladder` | the five ladder run directories (path B step 9) |

**Read the two protocols in `cnv_controls` separately.** The headline internal figure (0.866 ± 0.003)
is 5-fold CV averaged over **ten reseeds**; the aneuploidy-burden control, the regularisation sweep
and the site holdout are **single 5-fold runs at seed 0**. `dp-analysis cnv_controls` prints both and
marks the one that corresponds to the published value. Reading them as one protocol is exactly how
that table would quietly stop reproducing — the burden control, for instance, is 0.6854 at seed 0
(published 0.685) and 0.6893 ± 0.0033 over ten seeds.

`dp-analysis cnv_controls` compares each recomputed value against the published one and prints
`match` or `DIFFERS`. It never edits a document and never rewrites a number; a mismatch is a finding
for a human.

---

## What is not reproducible, and why

Reporting these is the point of this document. None of them is fixed by the refactor, and none is
hidden by it.

1. **The task manifest and the fold draw are distributed artifacts with no recorded derivation**
   (B.3). The *numbers* survive because both are tracked; the *method* does not.
2. **The published bundle does not exist yet** (A.2), so path A is currently unreachable from a
   clone.
3. **Two upstream sources are unpinned**: the cBioPortal datahub is fetched from `master` and UCSC
   refGene is a live table. The tracked `gene_arm_hg38.csv` plus its checksum is the mitigation, not
   a fix.
4. **The ladder's baseline is a near-baseline, not a matched one.** `pam50_final_s1` was trained with
   the sweep-selected `lr=1.008e-4`, `reg=2.446e-6`, `bag_weight=0.553`, `inst_loss=svm` and instance
   clustering **on**; the five ladder arms use rounded `lr=1e-4`, `reg=2.5e-6`, CLAM's default
   `bag_weight`, no instance loss and `--no_inst_cluster`. Those three settings are inert on the
   multimodal code path, but the optimiser configuration genuinely differs. It is reproduced exactly
   rather than tidied.
5. **CLAM's 10 splits are drawn independently, not partitioned.** 599 of 910 cases land in at least
   one test fold and 242 in two to five, so "WSI alone" is a small ensemble, flattered by roughly
   +0.01 AUROC. An audit found no leakage and a random stratified partition gives the same verdict —
   but the number should never be reported without this sentence.
6. **CPTAC and TCGA features are geometrically comparable, not identically preprocessed.**
   `dp-cptac phase=1` verifies 256 px at 20×, 0 overlap, 1536-dim — the *geometry*. The CPTAC `.h5`
   files carry CLAM `create_patches_fp` attributes while TCGA's carry Trident ones, so the tissue
   segmentation, and therefore which tiles enter each bag, differs. That is a live confound for the
   claim that the external Her2 collapse is domain-shift-induced; do not restate "preprocessing is
   held constant".
7. **Two of the five ladder arms cannot be evaluated at all.** `evaluate_multimodal.py` has no branch
   for `film_attention` or `coattn`; `dp-evaluate` refuses them up front rather than dying inside
   `load_state_dict`. `dp-analysis compare_fusion_ladder` reads the per-fold prediction pickles and
   needs no evaluator, which is why §8 exists.
8. **`dp-evaluate` defaults to the TCGA test split, not CPTAC** — deliberately preserved from the
   wrapper it replaces. Swapping only the tabular table scores TCGA slides against CPTAC-shaped rows.
   A genuine external run also needs the CPTAC feature store and a CPTAC dataset CSV.
9. **`fusion=residual` has no trainable second branch** and refuses at composition time.
10. **The ER thread is only partially ported.** Each ER wrapper's *default* arm has an experiment
    config (`er_wsi_alone`, `er_wsi_rna_gated`, `er_wsi_clinpath_gated`, `er_wsi_rna_film`); the
    remaining chapter-2 arms do not, and the shims refuse them **by name, up front**, rather than
    failing four hours into a matrix. That coverage limit is a decision, not an oversight:
    `bash tools/train_er_multiseed.sh` prints exactly which arms are unavailable and what to run
    instead.
11. **`.scratch/harmonisation/*_harmonised_clinicopath.csv` has no producer**, so the
    clinicopathological thread cannot be rebuilt from a clone.

---

## Checking an installation

| command | what it proves |
|---|---|
| `pip install -e '.[dev]'` | the only correct install (see README for why editable is not a preference) |
| `dp-config validate` | every `paths.*` value absolute, tracked inputs present, `ClamConf` in sync with CLAM's real parser |
| `dp-config validate experiment=pam50_wsi_final` | the above, plus `topk` importable — that check fires only for `inst_loss=svm`, which only this experiment sets, so it is reached only by naming an experiment |
| `dp-config sync-check` | just the schema-drift check: `in sync: 52 CLAM flags in main.py, ClamConf and clam/base.yaml` |
| `dp-config show experiment=…` | the fully resolved configuration for an experiment, before it runs |
| `dp-train --dry-run experiment=…` | the exact CLAM command an experiment would issue, without creating a run directory. Works on a machine that has downloaded nothing: the plan prints first, then any missing input is named and the exit status is non-zero |
| `dp-config reference -o docs/config-reference.md` | regenerates the config reference; it is generated, never hand-written |

`dp-config validate` on a fresh clone prints a `not acquired :` line naming the data trees you have
not downloaded. That is information, not a failure.

### What is not checked automatically

There is no `pytest` suite, no synthetic smoke run and no automated absolute-path gate in this
repository. Three consequences worth stating plainly rather than discovering later:

- **Wrapper equivalence is not re-verified on change.** That the configuration reproduces the
  original shell wrappers was established once, by executing each pre-refactor wrapper under a
  stubbed interpreter and comparing parsed CLAM argument namespaces field for field. Nothing re-runs
  that comparison. The frozen wrappers are kept byte-identical in `tests/legacy_wrappers/tools/` so
  the comparison can be redone by hand against `dp-train --dry-run`.
- **Nothing enforces the no-absolute-paths rule.** `grep -rI "/workspace/dp-code"` over tracked code
  and configuration is the manual equivalent; historical run artifacts under `results/hydra/`,
  `project/CLAM/tmp_eval/`, `project/CLAM/results/` and `docs/implementation-research/` legitimately
  contain the string, because they are records of what was run.
- **No end-to-end path is exercised cheaply.** The smallest real check is a short `dp-train` on a
  real fold, which needs the WSI features.
