# ER-status prediction — Part A → Part B hand-off

**Prepared:** 2026-07-17 (Fable-5 orchestrator + Opus-4.8 subagents). Every number below
is verified against a tool result from the preparation session, not asserted from memory.

Part A prepared labels, splits, code, fusion tables, and the exact runnable training
commands. **The author runs all training.** Part B resumes after training, reads this file
plus W&B, and produces the ablation report (AUROC/AUPRC/F1, per-site, DeLong, calibration).

---

## 1. Outcome at a glance

- **ER label:** binary, from cBioPortal `brca_tcga` `ER_STATUS_BY_IHC` (pre-binarised
  Positive/Negative call), cross-checked against the local GDC/Xena clinical matrix — **0
  case-level disagreements**. Retained **1046 cases**; **808 ER-positive (77.2%) / 238
  ER-negative (22.8%)**.
- **WSI-alone cohort (dataset_csv):** **1003 cases / 1068 slides** (embeddings ∩ ER label,
  primary-tumour slides only). Slide-level balance 835 ER+ / 233 ER−.
- **Matched fusion N (well-powered, above the ~400 floor):**
  - **WSI + RNA:** **956 cases** (WSI ∩ RNA ∩ ER).
  - **WSI + clinicopath:** **1003 cases** (WSI ∩ clinicopath ∩ ER).
- **Splits:** 10 size-balanced **tissue-submitting-site holdout** folds + 1 leave-site-
  groups-out split. Independently re-verified: no case or slide crosses partitions, test
  sites are disjoint from train/val, both ER classes present in every partition, and the 10
  test folds are a disjoint cover of all 1003 cases.
- **Three training commands:** `tools/train_er_ablation.sh {wsi|rna|clinpath|all}` (run
  `wsi` first). Verbatim commands in §4.

---

## 2. Files produced (real paths)

| Artifact | Path | Shape |
|---|---|---|
| ER labels | `tools/data/tcga_brca_er_labels.csv` | `case_id,label` (ER-positive/ER-negative), 1046 rows |
| Clinicopath (human-readable) | `tools/data/tcga_brca_clinicopath.csv` | case_id, age, ajcc_stage, pathologic_t/n/m, histological_type; 1046 rows (no grade — see caveats) |
| Clinicopath fusion table | `tools/data/tcga_brca_clinicopath_clam.csv` | `case_id,label` + 24 numeric features (raw age + one-hot stage/T/N/M/histology, each with an `_unknown` column); 1046 rows |
| RNA fusion table | `.scratch/TCGA-BRCA-rna/tcga_brca_er_rna_clam.csv.gz` | `case_id,label,sample,sample_type_code` + 20530 gene features (ESR1 retained); 996 rows |
| CLAM dataset_csv | `project/CLAM/dataset_csv/tcga_brca_er.csv` | `case_id,slide_id,label`; 1068 slides / 1003 cases |
| 10-fold splits | `project/CLAM/splits/tcga_brca_er_100/splits_{0..9}.csv` (+ `_bool`, `_descriptor`) | site-holdout CV; only `splits_{i}.csv` is read at train time |
| Leave-site-groups-out split | `project/CLAM/splits/tcga_brca_er_lsgo/splits_0.csv` (+ `_bool`, `_descriptor`) | 201-case / 7-site held-out test |
| Ablation runner | `tools/train_er_ablation.sh` | three arms; the exact commands in §4 |

**Code changes:** `project/CLAM/main.py` gained a `tcga_brca_er` task (`n_classes=2`,
`label_dict={'ER-negative':0,'ER-positive':1}`, `csv_path='dataset_csv/tcga_brca_er.csv'`,
`patient_strat=True`). `tools/train_pam50_final.sh` data-root bug fixed
(`.datasets/embeddings` → `.datasets/tcga-brca/embeddings`). Reusable builders:
`tools/make_er_dataset_csv.py`, `tools/make_er_site_splits.py`,
`tools/build_er_labels.py`.

---

## 3. Split design (the leakage argument for the reviewer)

**Test folds hold out whole tissue-submitting sites (TSS = TCGA barcode chars 6–7, i.e.
`case_id.split('-')[1]`), per Howard 2021.** No site's cases ever span the train/test
boundary, so test performance measures generalisation to unseen submitting sites (the TSS
batch confounder is controlled).

TCGA-BRCA is dominated by a few large sites (BH=138, A2=100, E2=89 …) over a long tail of
single-case sites (38 sites total, 24 of which carry ≥1 ER-negative case). An off-the-shelf
`StratifiedGroupKFold` therefore produced wildly uneven, sometimes single-class folds (one
fold's test had 11 cases; another fold's validation had no ER-negative). The orchestrator
replaced it with a **size-balanced greedy site-packing** (largest site → least-loaded fold),
giving 10 folds of 95–138 test cases, each with both ER classes.

**Validation is an internal model-selection set, not the generalisation measurement**, so it
is carved from each fold's training pool by ER-stratified *case* sampling (~10%). It may
share sites with train (it is not the evaluation set) but never with test (it is drawn from
the test-excluded pool). This is a deliberate, documented choice: a fully nested
site-holdout for validation is infeasible at k=10 on this dominant-site cohort without
starving the minority class. The headline generalisation claim rests entirely on the
site-held-out **test** sets, which are strict.

**Per-fold test composition (cases):** fold0 138 (23−/115+, 1 site=BH), fold1 100 (24/76),
fold2 96 (23/73), fold3 96 (10/86), fold4 96 (22/74), fold5 96 (23/73), fold6 96 (19/77),
fold7 95 (25/70), fold8 95 (26/69), fold9 95 (32/63). LSGO test: 201 cases / 7 sites
(41−/160+).

---

## 4. Training commands (author runs these; WSI-alone first)

Run from the repo root. `tools/train_er_ablation.sh` wraps these exactly; the verbatim
`python main.py` invocations (run from `project/CLAM`) are:

**Arm 1 — WSI-alone baseline (writes the frozen per-fold checkpoints):**
```
cd project/CLAM
python main.py --task tcga_brca_er --data_root_dir ../../.datasets/tcga-brca/embeddings \
  --embed_dim 1536 --results_dir ../../.scratch/results/er --split_dir tcga_brca_er_100 \
  --k 10 --seed 1 --max_epochs 50 --early_stopping --patience 5 --weighted_sample --log_data \
  --model_type clam_mb --model_size big --drop_out 0.5 --opt adam --lr 1e-4 --reg 2.5e-6 \
  --exp_code er_wsi_alone --B 4 --bag_loss ce --inst_loss svm \
  --wandb --wandb_project er-brca-ablation --wandb_tags er wsi-alone clam_mb
```

**Arm 2 — WSI + RNA gated fusion (frozen WSI branch):**
```
cd project/CLAM
python main.py --task tcga_brca_er --data_root_dir ../../.datasets/tcga-brca/embeddings \
  --embed_dim 1536 --results_dir ../../.scratch/results/er --split_dir tcga_brca_er_100 \
  --k 10 --seed 1 --max_epochs 50 --early_stopping --patience 5 --weighted_sample --log_data \
  --model_type clam_mb --model_size big --drop_out 0.5 --opt adam --lr 1e-4 --reg 2.5e-6 \
  --exp_code er_wsi_rna_gated --B 4 --bag_loss ce --no_inst_cluster \
  --tabular_csv ../../.scratch/TCGA-BRCA-rna/tcga_brca_er_rna_clam.csv.gz \
  --tabular_case_id_col case_id --tabular_hidden_dim 256 --tabular_num_layers 2 \
  --tabular_top_n_features 10000 --fusion_mode gated --fusion_hidden_dim 32 \
  --pretrained_wsi_ckpt ../../.scratch/results/er/er_wsi_alone_s1/s_{fold}_checkpoint.pt \
  --freeze_wsi_branch \
  --wandb --wandb_project er-brca-ablation --wandb_tags er wsi-rna gated frozen
```

**Arm 3 — WSI + clinicopath gated fusion (frozen WSI branch):**
```
cd project/CLAM
python main.py --task tcga_brca_er --data_root_dir ../../.datasets/tcga-brca/embeddings \
  --embed_dim 1536 --results_dir ../../.scratch/results/er --split_dir tcga_brca_er_100 \
  --k 10 --seed 1 --max_epochs 50 --early_stopping --patience 5 --weighted_sample --log_data \
  --model_type clam_mb --model_size big --drop_out 0.5 --opt adam --lr 1e-4 --reg 2.5e-6 \
  --exp_code er_wsi_clinpath_gated --B 4 --bag_loss ce --no_inst_cluster \
  --tabular_csv ../../tools/data/tcga_brca_clinicopath_clam.csv \
  --tabular_case_id_col case_id --tabular_hidden_dim 256 --tabular_num_layers 2 \
  --tabular_top_n_features 0 --fusion_mode gated --fusion_hidden_dim 32 \
  --pretrained_wsi_ckpt ../../.scratch/results/er/er_wsi_alone_s1/s_{fold}_checkpoint.pt \
  --freeze_wsi_branch \
  --wandb --wandb_project er-brca-ablation --wandb_tags er wsi-clinpath gated frozen
```

Notes on the config: all three arms share the same folds (`--split_dir tcga_brca_er_100`),
seed, and WSI backbone. The fusion arms load and **freeze** the Arm-1 per-fold checkpoints
(`s_{fold}_checkpoint.pt`, `{fold}` is templated by CLAM), so only the tabular encoder +
gated fusion head train. CLAM fits feature selection (`--tabular_top_n_features`: 10000 for
RNA, 0=all for the 24 clinicopath features) **and** standardisation on each training fold
only — so the fusion tables are raw and the leakage control is automatic. Hyperparameters
(lr 1e-4, reg 2.5e-6, drop_out 0.5, B 4, model_size big) are inherited from the tuned PAM50
CLAM-MB config on the same embeddings; re-tune if desired. `--bag_weight` is left at CLAM's
default 0.7.

**Optional leave-site-groups-out run** (for a single dedicated generalisation number):
re-run each arm with `--split_dir tcga_brca_er_lsgo --k 1` and a distinct `--exp_code`
(e.g. `er_wsi_alone_lsgo`); train Arm-1 LSGO first so its `s_0_checkpoint.pt` exists for the
fusion arms. The per-site generalisation report can otherwise be built from the pooled
10-fold out-of-fold test predictions (see §5).

---

## 5. W&B + required outputs for Part B (rebuild results without re-training)

**W&B project `er-brca-ablation`; one run group per arm via `--exp_code`:**
`er_wsi_alone`, `er_wsi_rna_gated`, `er_wsi_clinpath_gated` (seed 1 ⇒ run dirs
`<exp_code>_s1`). W&B captures, per arm: per-fold `final/test_auc` and `final/val_auc`;
summary `mean_test_auc`/`std_test_auc`/`mean_val_auc`/`std_val_auc` (+ `_acc`); per-fold
`fold_{i}_test_auc`/`fold_{i}_val_auc`; and the full run **config** (model_type, fusion_mode,
tabular_csv, embed_dim, seed, split_dir, tabular_top_n_features, freeze_wsi_branch, …).

**IMPORTANT — the substrate Part B actually needs is on disk, not in W&B.** Stock CLAM does
**not** log AUPRC, F1, per-site metrics, or a per-slide prediction table to W&B. Those are
all computable offline from the per-fold results the run writes to
`.scratch/results/er/<exp_code>_s1/`:

- `split_{i}_results.pkl` — per **test** slide `{slide_id, prob (per-class array), label}`
  for fold `i`. Join `slide_id → case_id → site` via `dataset_csv/tcga_brca_er.csv` (site =
  `case_id.split('-')[1]`) and fold membership via the split files. This is everything
  needed for **AUROC/AUPRC/F1**, **per-site** breakdown (Howard generalisation), **DeLong**
  paired tests, and **calibration** — computed in Part B.
- `s_{i}_checkpoint.pt` — per-fold model (frozen WSI branch + fusion head).
- `s_{i}_tabular_transform.json` — per-fold fitted feature selection + standardisation
  (fusion arms; records exactly which features were selected).
- `summary.csv`, `fold_{i}_history.csv`, and the `splits_{i}.csv` copy CLAM saves.

**Action for the author:** keep the entire `.scratch/results/er/` tree after training —
do not rely on W&B alone. Also record (or leave discoverable) the W&B run id/URL per arm.

**Paired comparison note:** for DeLong of a fusion arm vs WSI-alone, restrict to the matched
cases (RNA arm: the 956 WSI∩RNA∩ER cases; clinicopath arm: all 1003). CLAM's fusion dataset
already filters each fold to matched cases, so the fusion `split_{i}_results.pkl` covers the
matched subset; intersect the WSI-alone predictions to the same slide_ids offline.

---

## 6. Verification performed (actual counts)

Re-derived independently this session against the written files:

- dataset_csv labels ⊆ {ER-negative, ER-positive}, no NaN/Indeterminate; 1068 slides / 1003
  cases; slide-level 835 ER+ / 233 ER−.
- 10-fold splits: every slide_id in the split files exists in the dataset_csv; train/val/test
  slide sets pairwise disjoint; no case_id crosses partitions; **test sites disjoint from
  train∪val sites** in every fold; both ER classes present in every partition; the 10 test
  folds partition all 1003 cases exactly once (disjoint cover).
- LSGO split: 201 test cases across 7 held-out sites, site-disjoint from train/val, both
  classes present.
- Fusion N intersections: WSI∩RNA∩ER = **956**; WSI∩clinicopath∩ER = **1003**.
- Clinicopath fusion table: 24 feature columns all numeric; label matches the ER label file
  per case (0 mismatches); **no ER/PR/HER2/receptor field among features** (no ER-from-ER
  leakage).
- RNA fusion table: 996 cases, label matches ER file (0 mismatches), no duplicate case_id,
  ESR1 present; parses through CLAM's `read_tabular_feature_table` (features (996, 20530)).

---

## 7. Caveats a resume agent must know

- **ER cutoff / percent-positive.** The label is the pre-binarised cBioPortal/GDC
  Positive/Negative IHC call. A percent-positive field exists only as **ordinal buckets**
  (`ER_STATUS_IHC_PERCENT_POSITIVE` in `brca_tcga`, ~470 patients: `<10%`, `90-99%`, …), not
  a raw numeric percentage. The 1%-vs-10% cutoff sensitivity analysis from the fuller plan
  therefore **was not run**; treat the provided call as the primary (≈1%-equivalent) label.
  No percentage was fabricated. The bucket field was **excluded from features** (ER-derived
  → would be leakage).
- **cBioPortal study nuance.** The locked-decision study `brca_tcga_pan_can_atlas_2018` does
  **not** expose `ER_STATUS_BY_IHC`; the ER call came from cBioPortal `brca_tcga`,
  cross-checked 0-disagreement against the local Xena matrix. Both converge to 1046 calls.
- **RNA is not target leakage.** The RNA table retains ESR1 and other ER-pathway genes **by
  design**. The ER label is IHC-derived (a protein stain), so the transcriptome is a
  legitimate, biologically independent predictor — unlike PAM50, whose label is itself
  defined from expression. Do not "fix" this; state it in the report.
- **Class imbalance.** ~77% ER-positive. `--weighted_sample` is on; report AUPRC and F1 for
  the positive class (ER-positive), not accuracy, as the headline fusion metrics.
- **Grade absent.** No usable histologic grade exists for TCGA-BRCA in `brca_tcga`, the
  pan-can study, or the local matrix (pan-can `GRADE` has 0 populated BRCA records); the
  clinicopath arm has no grade feature. Not fabricated.
- **Missing clinicopath values** are encoded as explicit `_unknown` one-hot columns
  (ajcc_stage 8 unknown, histological_type 1 unknown); no rows dropped.
- **Validation shares sites with train** by design (see §3) — this is intended, not a leak;
  the test set is the strict site-holdout.
- **Split strategy was revised by the orchestrator** away from `StratifiedGroupKFold` to
  balanced greedy site-packing (documented in §3 and in `tools/make_er_site_splits.py`),
  because the raw approach produced tiny/single-class folds on this cohort.
