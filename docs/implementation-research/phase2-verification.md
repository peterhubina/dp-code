# Phase 2 independent verification: `film_attention` and `coattn` fusion modes

Independent audit of the uncommitted working-tree change that adds two `--fusion_mode`
values to the vendored CLAM. Performed with no prior context on the change; nothing in the
change description was taken on trust.

- **Date:** 2026-07-28 (07:00–07:05 UTC)
- **Baseline:** `git HEAD = 4cda3e8` ("fusion analysis")
- **Python:** `/opt/venv`, torch 2.0.1+cu117, numpy 1.26.4, pandas 2.3.3
- **Scripts:** `/tmp/claude-0/-workspace-dp-code/5a663856-0b89-4161-94c7-b5eb22ab3d7c/scratchpad/verify/`
  (`claim1_differential.py`, `claim2_freeze.py`, `claim3_leakage.py`, `claim_groups.py`)
- **No multi-fold training was run.** Nothing in the repo was modified except this file.

### Exact content verified

All results below pertain to these file hashes, captured immediately before and immediately
after the final test run (identical both times):

| file | md5 |
|---|---|
| `project/CLAM/models/model_multimodal.py` | `01080b9643850a83bfb6a29d262af407` |
| `project/CLAM/main.py` | `34360c2b7f218f3d8e00e278b80fe7d9` |
| `project/CLAM/utils/core_utils.py` | `ebfd3858e8fe6aaca74603b24e86d516` |
| `project/CLAM/utils/tabular_groups.py` | `dfa08b709a590190902259c41b9f77d5` |
| `project/CLAM/tests/test_fusion_modes.py` | `0001c5ad150cbdab94a78ab3ce83a834` |

> **The working tree moved during this verification.** See
> [Concurrency warning](#concurrency-warning-read-this-first) — this materially limits how
> long these results remain valid.

---

## Verdict summary

| Claim | Verdict |
|---|---|
| 1. Four pre-existing fusion modes behave exactly as before | **PASS** — max abs diff `0.0` across 128 forward comparisons |
| 2. `--freeze_wsi_branch` genuinely freezes the WSI branch under the new modes | **PASS** — 16 configurations, 0 gradients, 0 value changes, steps non-vacuous |
| 3. Splits and the tabular transform (leakage controls) unchanged | **PASS** — files byte-identical to HEAD; fit-on-train-only and label check re-verified functionally |
| Extra: `utils/tabular_groups.py` correctness | **PASS** |
| Extra: author's own test script | **PASS** (46 checks) — corroborating, not sufficient |
| Extra: `git status` limited to the five claimed files | **FAIL** — four additional files are dirty/new (see §5) |

---

## Concurrency warning (read this first)

`ps` shows **three other `claude` processes** running against this repo, plus a live
`python tools/download_cptac.py --modality all --cohort-only --workers 8`. The files under
audit were being edited *while I was auditing them*:

- At 07:00 UTC `git diff HEAD --stat` reported `model_multimodal.py | 169 +-` and
  `main.py | 25 +-`.
- At 07:02 UTC the same command reported `170 +-` and `29 +-`.

Diffing the two states, the mid-audit edits were:

1. `model_multimodal.py`: `.detach()` added to the four new fusion metrics
   (`fusion_film_gamma_dev`, `fusion_film_beta_abs`, `fusion_tabular_logit_abs`,
   `fusion_coattn_max_weight`), and `self.tabular_head(encoded_tabular)` hoisted into a
   local `tabular_logits` instead of being called twice. Value-neutral for logits.
2. `main.py`: a new guard `--log_heatmaps is not supported with --fusion_mode coattn`.

Both are improvements and neither touches an original code path. **Every result in this
report was re-derived from scratch against the final hashes in the table above**, and the
hashes were re-checked after the run and were unchanged. But this report certifies *those
bytes*, not "the change" as an abstract thing. If the other session keeps editing, re-run
the four scripts.

---

## Claim 1 — the four original modes are bit-identical: **PASS**

### Method

Two independent lines of evidence.

**(a) Read the diff by eye.** `git diff HEAD -- project/CLAM/` deletes exactly six lines
across the whole vendored CLAM (`git diff HEAD -- project/CLAM/ | grep '^-'`):

```
-parser.add_argument('--fusion_mode', ... choices=['concat','gated','residual','cross_attention'] ...
-                    help='... supports concat, gated, residual and cross_attention')
-            'residual_scale': args.residual_scale}
-        if fusion_mode not in {"concat", "gated", "residual", "cross_attention"}:
-            raise ValueError("fusion_mode must be 'concat', 'gated', 'residual', or 'cross_attention'.")
-        if fusion_mode in {"gated", "residual", "cross_attention"} and fusion_hidden_dim <= 0:
```

All six are argparse metadata, the settings dict's closing brace, or constructor
validation. Not one line inside `_gated_fusion`, `_cross_attention_fusion`,
`_residual_fusion`, the `concat` branch, `_pool_wsi_features`, or the pre-existing
`forward` body was modified. The validation changes only *widen* the accepted set
(`{concat,gated,residual,cross_attention}` → `FUSION_MODES` ⊇ it) and only *add* `coattn`
to the positive-`fusion_hidden_dim` requirement. The two new `elif` branches are inserted
between `elif fusion_mode == "cross_attention"` and the `else` (residual) branch, so the
residual `else` is still reached by exactly one mode.

**(b) Git-based differential execution.** `git show HEAD:project/CLAM/models/model_multimodal.py`
was written to a temp path and loaded as a separate module via
`importlib.util.spec_from_file_location`, giving two live classes in one process. For each
configuration both classes were constructed under `torch.manual_seed(1234)` with identical
kwargs, then run on identical inputs (`torch.manual_seed` reset before each pair of forward
calls so dropout draws match).

Grid: `wsi_model_type ∈ {clam_sb, clam_mb}` × `mode ∈ {concat, gated, residual,
cross_attention}` × `dropout ∈ {0.0, 0.25}` × `freeze_wsi_branch ∈ {False, True}`
= **32 model pairs**, each evaluated in `train()` and `eval()` mode × `return_features ∈
{False, True}` = **128 forward comparisons**, each with `instance_eval=True` and a label
passed. Compared: `logits`, `y_prob`, `y_hat`, `attention`, the **key set** of the results
dict, and every tensor in the results dict. Requirement was `== 0.0`, not `allclose`.

`residual` was **not** skipped. It was constructed with its own kwargs
(`rna_hidden_dims=(64,32)`, `rna_dropout=0.3`, `residual_scale=0.2`; it builds an `RNA_MLP`
instead of the shared `TabularMLPEncoder`). One accommodation was needed: `RNA_MLP` uses
`nn.BatchNorm1d`, which cannot run in `train()` mode with the batch-of-1 that MIL supplies,
so for `residual` + train-mode the RNA sub-branch was put in `eval()` on **both** models
identically. This is a pre-existing property of `residual` (it implies residual fusion is
only trainable with `--freeze_rna_branch`, or via `pretrained_rna_ckpt` + eval BN) and is
unrelated to this change; the differential remains valid because the two sides are treated
identically.

### Results

| mode | max abs diff (logits / y_prob / y_hat / attention / all results tensors) |
|---|---|
| `concat` | **0.0** |
| `gated` | **0.0** |
| `residual` | **0.0** |
| `cross_attention` | **0.0** |

- Overall max abs diff across all 128 comparisons: **0.0**
- `attention_only=True` path, all 8 mode × backbone combinations: **0.0**
- State dicts: identical key sets in all 32 pairs (`keys_only_old == keys_only_new == []`),
  max abs parameter difference **0.0**. Example: `clam_sb|concat` = 30 parameter tensors,
  1,137,425 elements.
- Failure list: empty.

### Behavioural deltas for old modes (non-defects, disclosed)

1. The `ValueError` message for an **invalid** mode changed:
   old `fusion_mode must be 'concat', 'gated', 'residual', or 'cross_attention'.` →
   new `fusion_mode must be one of ['coattn', 'concat', 'cross_attention', 'film_attention', 'gated', 'residual'].`
   Same exception type; only the string differs.
2. `experiment_<exp_code>.txt` and the W&B config gain three keys (`film_rank`,
   `modality_dropout`, `tabular_group_spec`) on **every** run, including old-mode and
   WSI-alone runs. Recorded-metadata change only; no code reads them outside
   `_is_multimodal(args)`.

### `--fusion_mode` unset (WSI-alone) path

Checked statically and via `--help`:

- Every new `parser.error` check sits inside `if args.fusion_mode is not None:`.
- Every new `core_utils.train` statement sits inside `if _is_multimodal(args):`, and
  `_is_multimodal` is unchanged (`getattr(args,'fusion_mode',None) is not None`).
- The four new `FUSION_RESULT_KEYS` entries are only ever *looked up* in a results dict;
  a WSI-alone `CLAM_SB`/`CLAM_MB` results dict never contains them, so
  `_extract_fusion_metrics` returns the same thing as before.
- `python main.py --help` was diffed against `git show HEAD:project/CLAM/main.py --help`.
  The only differences are the three new options, the widened `--fusion_mode` choice list,
  and argparse re-wrapping the usage block (different program name). **No pre-existing
  argument, choice list, default, or help string changed.**

---

## Claim 2 — `--freeze_wsi_branch` genuinely freezes the WSI branch: **PASS**

### Method

Mirrors the real code path, whose ordering was verified in `core_utils.train`:
`_fit_multimodal_transform` (rel. line 23) → `build_tabular_groups` (53) →
`CLAMRNAFusion(...)` (60) → `model.freeze_wsi_branch()` (87) → `get_optim()` (126). Note
`utils/utils.py:get_optim` uses `filter(lambda p: p.requires_grad, model.parameters())`,
and it is called *after* the freeze.

For each configuration: build → `freeze_wsi_branch()` → `model.train()` (to exercise the
`train()` override) → snapshot every `model.wsi` parameter → one forward + `backward()` +
`Adam(lr=1e-2, weight_decay=1e-3).step()`. Then assert: no `model.wsi` parameter has
`requires_grad`, none has a non-`None` `.grad`, none changed value, `model.wsi.training`
is `False`, and **at least one non-WSI parameter actually moved**.

Both optimiser constructions were tested: the real `filter(requires_grad)` one **and** a
stricter variant handing *every* parameter (including the frozen ones) to Adam with weight
decay — the frozen params must still not move.

### Results — 16 configurations, zero failures

| mode | backbone | drop | mod-drop | film_rank | filtered optim | WSI tensors | WSI elements | `requires_grad` | got grad | changed | non-WSI moved | max move |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| film_attention | clam_sb | 0.25 | 0.0 | 32 | yes | 18 | 1,056,013 | 0 | 0 | **0** | 11/15 | 0.01 |
| film_attention | clam_sb | 0.25 | 0.0 | 32 | **no** | 18 | 1,056,013 | 0 | 0 | **0** | 11/15 | 0.01 |
| film_attention | clam_sb | 0.0 | 0.3 | 32 | yes | 18 | 1,056,013 | 0 | 0 | **0** | 11/16 | 0.01 |
| film_attention | clam_sb | 0.0 | 0.0 | **0** | yes | 18 | 1,056,013 | 0 | 0 | **0** | 6/10 | 0.01 |
| film_attention | clam_mb | 0.25 | 0.0 | 32 | yes | 24 | 1,056,784 | 0 | 0 | **0** | 11/15 | 0.01 |
| film_attention | clam_mb | 0.25 | 0.0 | 32 | **no** | 24 | 1,056,784 | 0 | 0 | **0** | 11/15 | 0.01 |
| film_attention | clam_mb | 0.0 | 0.3 | 32 | yes | 24 | 1,056,784 | 0 | 0 | **0** | 11/16 | 0.01 |
| film_attention | clam_mb | 0.0 | 0.0 | **0** | yes | 24 | 1,056,784 | 0 | 0 | **0** | 6/10 | 0.01 |
| coattn | clam_sb | 0.25 | 0.0 | 32 | yes | 18 | 1,056,013 | 0 | 0 | **0** | 24/28 | 0.01 |
| coattn | clam_sb | 0.25 | 0.0 | 32 | **no** | 18 | 1,056,013 | 0 | 0 | **0** | 24/28 | 0.01 |
| coattn | clam_sb | 0.0 | 0.3 | 32 | yes | 18 | 1,056,013 | 0 | 0 | **0** | 24/29 | 0.01 |
| coattn | clam_sb | 0.0 | 0.0 | 0 | yes | 18 | 1,056,013 | 0 | 0 | **0** | 24/28 | 0.01 |
| coattn | clam_mb | 0.25 | 0.0 | 32 | yes | 24 | 1,056,784 | 0 | 0 | **0** | 24/28 | 0.01 |
| coattn | clam_mb | 0.25 | 0.0 | 32 | **no** | 24 | 1,056,784 | 0 | 0 | **0** | 24/28 | 0.01 |
| coattn | clam_mb | 0.0 | 0.3 | 32 | yes | 24 | 1,056,784 | 0 | 0 | **0** | 24/29 | 0.01 |
| coattn | clam_mb | 0.0 | 0.0 | 0 | yes | 24 | 1,056,784 | 0 | 0 | **0** | 24/28 | 0.01 |

**Non-vacuity:** in every configuration between 6 and 24 non-WSI parameters moved, by up to
`1.0e-2` (the Adam step magnitude at `lr=1e-2`). The steps were real.

`model.wsi.training == False` after `model.train()` in all 16 cases — the `train()` override
correctly keeps the frozen branch in eval, so WSI dropout stays off.

### Why the freeze holds despite gradients flowing through WSI modules

Worth recording, because it is not obvious from the code. In `_attention_level_fusion`
only the **feature extractor** (`self.wsi.attention_net[:3]`) is wrapped in
`torch.no_grad()`. The attention head (`attention_net[3]`) and, for FiLM, the bag
classifier (`self.wsi.classifiers`) are called **outside** `no_grad`, so autograd does build
a graph through them. The freeze holds solely because `freeze_wsi_branch()` sets
`requires_grad = False` on every WSI parameter, so no `.grad` is accumulated. The measured
result (0 non-`None` grads, 0 value changes, in all 16 cases including the unfiltered
optimiser) confirms this. Consequence to be aware of: the backward graph is slightly larger
than a fully-detached implementation would be — a memory cost, not a correctness bug. It
also means that if `--freeze_wsi_branch` is ever *omitted*, the WSI attention head and
classifier will train, which is the intended unfrozen behaviour but a bigger change than
for the older modes.

### Supporting property: FiLM is an exact identity at initialisation

The code comment claims that at init the FiLM logits are exactly the WSI-alone logits.
Verified (eval mode, `dropout=0.0`):

| configuration | `max abs(fusion_logits − wsi_alone_logits)` |
|---|---|
| `film_attention`, clam_sb, `film_rank=32`, init | **0.0** |
| `film_attention`, clam_mb, `film_rank=32`, init | **0.0** |
| `film_attention`, clam_mb, `film_rank=0`, init | **0.0** |
| `film_attention`, clam_mb, `force_tabular_absent=True` | **0.0** |
| `coattn`, clam_mb, `force_tabular_absent=True` | **2.2218** (not WSI-alone — see §6) |

---

## Claim 3 — splits and the tabular transform are unchanged and leakage-free: **PASS**

### 3a. The files are byte-identical to HEAD

`git show HEAD:<path> | md5sum` vs the working tree:

| file | HEAD md5 | working tree md5 | |
|---|---|---|---|
| `dataset_modules/multimodal_dataset.py` | `24f6cc02…` | `24f6cc02…` | SAME |
| `dataset_modules/rna_dataset.py` | `81c3d7a7…` | `81c3d7a7…` | SAME |
| `dataset_modules/dataset_generic.py` | `5400ae71…` | `5400ae71…` | SAME |
| `utils/utils.py` | `42dcedbd…` | `42dcedbd…` | SAME |
| `create_splits_seq.py` | `75dd4023…` | `75dd4023…` | SAME |
| `evaluate_multimodal.py` | `b60abc28…` | `b60abc28…` | SAME |

`git diff HEAD --stat -- project/CLAM/dataset_modules/` is empty and `git status` reports
nothing untracked there. **The split-reading path** (`return_splits`, `get_split_from_df`,
`get_merged_split_from_df`, `_make_split`, and `dataset_generic.py`'s CSV split loader) is
therefore unchanged, as is `create_splits_seq.py`. `utils/core_utils.py` has exactly three
hunks (`@@ -29,0 +30,4 @@`, `@@ -332,0 +337,11 @@`, `@@ -342,0 +358,3 @@`) — the
`FUSION_RESULT_KEYS` additions and the multimodal model-construction block. Neither
`_fit_multimodal_transform` nor `_save_multimodal_transform` nor `save_splits` was touched.

### 3b. Functional re-verification (not just diffing)

Synthetic cohort: 20 cases × 12 features, train fold = first 12 cases,
`tabular_top_n_features = 5`.

| check | measured | interpretation |
|---|---|---|
| variance top-N indices from `RNAFeatureTransform.fit` | `[0, 2, 4, 7, 11]` | matches a train-rows-only variance ranking exactly |
| the same selection if it had been fitted on **all 20 rows** | `[0, 2, 7, 8, 11]` | **different** — so the test can actually detect leakage |
| `max abs(transform.mean − train-only mean)` | **0.0** | standardisation mean is train-fold only |
| `max abs(transform.std − train-only std)` | **0.0** | standardisation std is train-fold only |
| `max abs(transform.mean − all-rows mean)` | **2.7914** | the two are clearly distinguishable; the check is not blind |
| val-fold row transformed vs hand-computed `(x − train_mean)/train_std` | **0.0** | val/test rows are scored with the train-fitted statistics |
| rows used to fit | 12 of 20 available | train fold only |

- **`case_id` join.** `TabularFeatureStore.fit_transform` with an unmatched training case
  raises `ValueError: 1 training cases are missing tabular features. Examples: ['NOT_A_CASE']`.
  Silent dropping of a training case is impossible.
- **Label-disagreement check.** `Generic_Multimodal_MIL_Dataset._filter_to_tabular_cases`
  was driven directly with a stub carrying (i) fully agreeing labels → **no raise**, and
  (ii) one flipped label → `ValueError: WSI and tabular labels disagree for matched cases.
  Examples: [{'case_id': 'C003', 'slide_id': 'C003.svs', ...}]`. Both directions confirmed,
  so the check is live and not trivially always-true.
- **Ordering.** `_fit_multimodal_transform(train_split, val_split, test_split, args, cur)`
  is called at `core_utils.train` relative line 23, `build_tabular_groups` at 53,
  `CLAMRNAFusion(...)` at 60. The transform is fitted on the train split **before** the
  model exists, and the new co-attention grouping reads
  `train_split.tabular_store.transform.selected_feature_names`, i.e. the already-train-fitted
  transform. It cannot see val/test data.

### Pre-existing quirk, not introduced here

`Generic_Multimodal_MIL_Dataset._make_split` hands the **same** `TabularFeatureStore` object
to the train, val and test splits, so `train_split.tabular_store is val_split.tabular_store`.
`set_tabular_transform` on any split mutates the shared store. Functionally this is fine —
all splits end up using the single train-fitted transform, which is the desired behaviour —
but it means a split object carries no independent transform state. This is identical at
HEAD and is not a consequence of the change.

---

## 4. `utils/tabular_groups.py` (new file): **PASS**

**`prefix` spec** — 18 clinicopath one-hot columns (`age`, `stage_*`, `t_*`, `n_*`, `m_*`,
`histology_*`):

- Groups: `age(1), histology(3), m(2), n(4), stage(4), t(4)`
- Indices emitted: 18; distinct indices: 18; `sorted(flat) == range(18)` → **exact
  partition**, every index exactly once, min 0, max 17. No duplicates, none out of range.

**Signature-CSV spec** — 30 features, four gene-set columns, one deliberately overlapping
(`G3` in two sets) and one matching nothing:

- Groups returned: `prolif(4), immune(5), stroma(3), unassigned(19)`
- Covers every index at least once: **true**; missing indices: **none**
- Out-of-range indices: **none**; duplicate indices within a group: **none**
- Overlap count across groups: 1 (the intentional `G3`)
- The all-non-matching column (`absent`) is correctly skipped rather than becoming an empty
  token (an empty group would build `nn.Linear(0, H)`).

**Error paths:** empty `feature_names` → `ValueError`; unreadable spec path →
`FileNotFoundError` with a helpful message.

**Minor findings (no failure):**

1. **Dead code.** The `if not indices: raise ValueError("No feature matched any group…")`
   branch is unreachable: the `unassigned` catch-all is appended before it, so `indices` is
   non-empty whenever `feature_names` is. Verified — a CSV matching nothing returns a single
   `unassigned(30)` token instead of raising. Harmless, but the guard gives false comfort:
   a wrong signature CSV degrades silently to one giant token rather than erroring.
2. **`prefix` is only sensible for one-hot blocks.** On underscore-free names (bare gene
   symbols such as `TP53`, `ESR1`) every feature becomes its own token — verified: 5 features
   → 5 tokens. With an RNA feature set of a few thousand genes this would build thousands of
   `nn.Linear(1, fusion_hidden_dim)` token encoders and a multi-thousand-token attention.
   The docstring says `prefix` is for clinicopath blocks, but nothing enforces it.
3. Numeric columns in a signature CSV are compared as `str(value)`, so a CSV whose gene
   column is read as float would match nothing and fall through to `unassigned`. Not
   triggered by the intended usage.

---

## 5. `git status` hygiene: **FAIL** (four files beyond the five claimed)

`git status --porcelain` at 07:05 UTC, HEAD `4cda3e8`:

```
 M docs/implementation-research/novel-fusion-design.md      <-- NOT in the claimed list
 M project/CLAM/main.py                                     <-- claimed
 M project/CLAM/models/model_multimodal.py                  <-- claimed
 M project/CLAM/utils/core_utils.py                         <-- claimed
 M tools/download_cptac.py                                  <-- NOT in the claimed list
?? docs/implementation-research/cptac-brca-external-cohort-findings.md  <-- NOT in the claimed list
?? project/CLAM/tests/                                      <-- claimed (tests/test_fusion_modes.py)
?? project/CLAM/utils/tabular_groups.py                     <-- claimed
?? tools/train_er_novel_fusion.sh                           <-- NOT in the claimed list
```

Assessment of the four extras:

- **`tools/download_cptac.py` (modified, 446 lines changed)** — a rewrite of the CPTAC
  external-cohort downloader (adds RNA and clinical modalities alongside WSI). Grepping its
  diff for `fusion|clam|core_utils|main.py` returns one hit, inside a comment ("fusion head
  can consume either"). It does **not** import from or affect the CLAM training path. It is
  unrelated work sitting in the same dirty tree, and a `git commit -a` would sweep it in.
  A live `python tools/download_cptac.py --modality all --cohort-only --workers 8` process
  is running.
- **`tools/train_er_novel_fusion.sh` (new)** — the driver script for the new arms
  (`--fusion_mode film_attention/coattn/gated`, `--film_rank`). Clearly part of this change
  but omitted from the file list I was given. It was last written at 07:01 UTC, i.e. during
  this audit. **I did not execute it** (it launches multi-fold training).
- **`docs/…/novel-fusion-design.md` (modified, +129/−7)** and
  **`docs/…/cptac-brca-external-cohort-findings.md` (new)** — documentation. The test
  script's docstring cites `novel-fusion-design.md §7.4` for its mutation-testing evidence;
  that file exists but was being edited during this audit, so I did not treat its contents
  as evidence.

Nothing under `project/CLAM/dataset_modules/`, `project/CLAM/models/model_clam.py`,
`project/CLAM/models/model_rna.py`, `project/CLAM/utils/utils.py`,
`project/CLAM/create_splits_seq.py` or `project/CLAM/evaluate_multimodal.py` is dirty.

---

## 6. Other findings and concerns (none breaks claims 1–3)

Ordered roughly by how much I would want them addressed.

1. **`evaluate_multimodal.py` cannot evaluate either new mode.** It was not updated. Its
   `CLAMRNAFusion(...)` call passes no `film_rank`, `modality_dropout` or
   `tabular_group_indices`, and `infer_fusion_mode()` only recognises `fusion_head.*`,
   `tabular_projection.*`, `fusion_gate.*`, `fusion_classifier.*` prefixes. Consequences:
   `coattn` raises `fusion_mode 'coattn' requires tabular_group_indices`; a
   `film_attention` checkpoint trained with `--film_rank 0` or `--modality_dropout > 0`
   will not satisfy `load_state_dict(..., strict=True)` against a default-constructed
   model. Old modes are unaffected. This is a missing-capability gap, not a regression, but
   it means the new arms currently have no offline evaluation path.
2. **`coattn` discards most of the pretrained WSI branch.** Measured by perturbing each
   frozen sub-module and reading the change in logits (clam_mb, eval, init):

   | perturbed sub-module | `film_attention` Δlogits | `coattn` Δlogits |
   |---|---|---|
   | `attention_net[0]` (fc trunk) | 2.888 | 1.008 |
   | `attention_net[3]` (CLAM attention head) | 0.363 | **0.000** |
   | `classifiers` (CLAM bag head) | 4.820 | **0.000** |

   `coattn` uses only the frozen patch-encoder trunk; the pretrained attention head and bag
   classifier contribute nothing (it substitutes its own `patch_projection` +
   `MultiheadAttention` + `image_head`). That is a legitimate design choice, but "the WSI
   branch is frozen and shared across arms" means something materially weaker for `coattn`
   than for `film_attention`, and it should be stated that way in any comparison.
3. **`coattn`'s missing-modality fallback is not WSI-alone.** With
   `force_tabular_absent=True` the tabular tokens become zeros, so `MultiheadAttention`
   softmaxes a zero query and degenerates to near-uniform mean pooling over patches. At
   init, `max abs(coattn_absent_logits − wsi_alone_logits) = 2.2218`, versus exactly `0.0`
   for `film_attention`. If missing-modality robustness is being reported per arm, these
   two are not measuring the same fallback.
4. **Both new modes have a zero-gradient first step for the tabular encoder.** Because
   `tabular_head` is zero-initialised (and for FiLM `film_gamma`/`film_beta` too), at step 0
   `∂L/∂encoded_tabular = 0`, so the whole `tabular_encoder` receives exactly zero gradient
   — measured `|grad tabular_encoder.encoder[0].weight|max`:

   | step | `film_attention` | `coattn` | `film_bottleneck` (film) |
   |---|---|---|---|
   | 0 | **0.0** | **0.0** | 0.0 |
   | 1 | 2.46e-3 | 8.79e-3 | 1.32e-6 |
   | 2 | 4.54e-3 | 1.14e-2 | 1.97e-5 |
   | 3 | 5.53e-3 | 9.06e-3 | 1.48e-5 |

   It self-resolves after one step, so this is a cold start, not a deadlock. But note that
   with `weight_decay > 0` the tabular encoder's *weights* are still decayed on that first
   step while receiving no learning signal (its zero-valued biases correctly do not move).
   The author's test documents the FiLM half of this as expected behaviour.
5. **`residual` fusion is only trainable with the RNA branch in eval.** `RNA_MLP` uses
   `nn.BatchNorm1d`, which raises `ValueError: Expected more than 1 value per channel when
   training, got input size torch.Size([1, 64])` under MIL's batch-of-1. Pre-existing at
   HEAD, unrelated to this change, but I hit it while building the differential and it is
   worth knowing that `--fusion_mode residual` implicitly requires `--freeze_rna_branch`.
6. **`--log_heatmaps` with `coattn`.** `coattn` returns MultiheadAttention weights of shape
   `[1, n_tokens, n_patches]`, whereas `core_utils.summary` expects a 2-D per-class patch
   attention. A guard (`parser.error('--log_heatmaps is not supported with --fusion_mode
   coattn')`) was added to `main.py` **during this audit**. It lives in argparse only, so a
   direct caller of `core_utils.train` could still reach the bad path. `film_attention`'s
   attention is `[n_classes, N]` / `[1, N]`, matching CLAM, so it is fine.
7. **Minor:** `_encode_tabular` always runs `self.tabular_encoder(tabular_features)` and
   then discards the result when the modality is dropped — wasted compute, no correctness
   impact. `modality_dropout` also draws from the global RNG (`torch.rand(())`) inside the
   forward, which perturbs the global RNG stream; irrelevant to the old modes because the
   `self.training and self.modality_dropout > 0.0` guard short-circuits at the default
   `0.0` (and claim 1's exact-equality result confirms it).

---

## 7. The author's own test script (corroborating only)

`cd project/CLAM && python tests/test_fusion_modes.py` → **exit 0, ALL CHECKS PASSED,
46 `[PASS]` lines, 0 `[FAIL]`.** Re-run twice, same result. It writes only to
`tempfile.gettempdir()`, not into the repo.

I did not treat this as sufficient, per instruction. Two notes:

- Its own regression test pins baseline `60a9639133dfabd335ede43feeef55cb5db3da3a` rather
  than HEAD, with a comment explaining that comparing against HEAD would pass vacuously once
  committed. I checked: `60a9639` is an ancestor of HEAD, and
  `model_multimodal.py` is **identical** at `60a9639` and at HEAD
  (`0977c51fdfbfd5ae3538d7ce86e02b71` both). So its baseline and mine are the same content.
  The pin is sound and does not smuggle in a favourable baseline.
- Its coverage is a strict subset of the independent differential in §1: it tests only
  `clam_mb`, `size_arg="big"`, `dropout=0.0`, `eval()` mode, unfrozen, and one input.
  Mine additionally covers `clam_sb`, `dropout=0.25`, `train()` mode, the frozen branch,
  `return_features=True`, `instance_eval=True`, `attention_only=True`, and the results-dict
  key set — 128 comparisons vs its 4.

---

## 8. What I did not test

Stated plainly rather than implied:

- **No multi-fold training was run** (explicitly out of scope). All evidence is unit-level:
  synthetic tensors, synthetic tabular tables, single optimiser steps. Nothing here says the
  new modes *learn well*, converge, or beat a baseline — only that they are wired correctly
  and do not disturb the existing modes.
- **No end-to-end run of `main.py` on real data.** The dataset construction path
  (`Generic_Multimodal_MIL_Dataset.__init__` reading real `.pt` bags and a real tabular CSV)
  was exercised only through its unchanged, byte-identical source plus a stubbed call to
  `_filter_to_tabular_cases`.
- **`tools/train_er_novel_fusion.sh` was not executed** and its arguments were not validated
  against the argparse surface beyond reading it.
- **No GPU/CUDA path** was exercised; everything ran on CPU.
- **`tools/download_cptac.py`'s 446-line diff was not reviewed** for correctness. I only
  established that it does not touch the CLAM training path.
- **Checkpoint save/load round-trip for the new modes was not tested** (`load_wsi_checkpoint`
  against a real pretrained CLAM checkpoint, and `strict=True` reload of a trained
  `film_attention`/`coattn` state dict). Finding §6.1 was derived by reading
  `evaluate_multimodal.py`, not by loading a real checkpoint.
