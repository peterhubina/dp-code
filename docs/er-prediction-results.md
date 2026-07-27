# ER-status prediction on TCGA-BRCA — three-way fusion ablation (results)

**Question.** Does adding a second modality (RNA-seq **or** clinicopathology) to H&E whole-slide
images improve binary estrogen-receptor (ER) status prediction, under a leakage-controlled,
tissue-submitting-site-held-out evaluation?

**Answer (headline).** **RNA-seq fusion beats H&E-alone by a clear, statistically significant
margin (+0.044 AUROC, DeLong p = 1.6×10⁻⁵). Clinicopathology fusion does not (−0.002 AUROC,
p = 0.74).** The signal that helps ER prediction beyond the slide is the transcriptome, not
routine clinicopathological variables (age, stage, T/N/M, histology).

All numbers below are out-of-fold, computed at the **case level** (per-case mean of slide
probabilities) over the 10 site-holdout folds, from the saved per-fold CLAM test predictions.
Reproduce with `python tools/analyze_er_ablation.py` (writes
`.scratch/results/er/analysis/`).

---

## 1. Setup

- **Task.** Binary ER status (ER-positive vs ER-negative), IHC-derived label from cBioPortal
  `brca_tcga` (`ER_STATUS_BY_IHC`), cross-checked 0-disagreement against the local GDC/Xena
  clinical matrix. Cohort with WSI embeddings and an ER label: **1003 cases / 1068 slides**,
  **77.4% ER-positive** (776 / 227 at case level).
- **Backbone.** UNI2-h patch features (1536-dim) → CLAM attention-MIL (`clam_mb`, model_size
  big). Identical across all three arms.
- **Fusion.** Gated fusion head; the WSI branch is the pretrained per-fold WSI-alone checkpoint,
  **frozen**; only the tabular encoder + gated head train. RNA table: 20530 genes (ESR1
  retained); clinicopath table: 24 features (age + one-hot stage/T/N/M/histology, no receptor
  status). CLAM fits feature selection (top-10000 genes for RNA, all 24 for clinicopath) and
  standardisation on each **training fold only**.
- **Splits.** 10 folds holding out whole tissue-submitting sites (TSS = barcode chars 6–7),
  size-balanced greedy site-packing, ER-stratified case-level validation. No case or slide, and
  no submitting site, crosses the train/test boundary. The three arms share the same folds;
  fusion arms operate on the modality-matched subset of each fold.
- **Matched N.** WSI-alone 1003 cases; WSI+RNA 956 cases (WSI∩RNA); WSI+clinicopath 1003 cases
  (all WSI cases have clinicopath).

---

## 2. Ablation table (case-level, out-of-fold)

| Arm | N | AUROC | AUPRC (ER+) | F1 (ER+) | Balanced acc. | ECE (10-bin) |
|---|---:|---:|---:|---:|---:|---:|
| **WSI-alone** | 1003 | 0.896 | 0.961 | 0.905 | 0.826 | 0.090 |
| **WSI + RNA** | 956 | **0.941** | **0.975** | **0.958** | **0.920** | **0.049** |
| **WSI + clinicopath** | 1003 | 0.894 | 0.959 | 0.912 | 0.826 | 0.067 |

Per-fold AUROC (mean ± std over the 10 site-holdout folds), consistent with the pooled numbers:
WSI-alone 0.906 ± 0.045; WSI+RNA 0.949 ± 0.038; WSI+clinicopath 0.898 ± 0.049.

## 3. Significance — DeLong paired test (fusion vs WSI-alone, on the matched case set)

| Comparison | N matched | WSI-alone AUROC | Fusion AUROC | Δ AUROC | z | p | Bootstrap 95% CI on Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **WSI + RNA vs WSI-alone** | 956 | 0.897 | 0.941 | **+0.044** | 4.31 | **1.6×10⁻⁵** | **[+0.024, +0.065]** |
| **WSI + clinicopath vs WSI-alone** | 1003 | 0.896 | 0.894 | −0.002 | −0.33 | 0.74 | [−0.013, +0.010] |

The RNA improvement is significant and its bootstrap CI excludes zero; the clinicopath difference
is indistinguishable from zero and its CI straddles zero.

**Independent cross-check.** These numbers were produced twice, by two separately-written analysis
pipelines — `tools/evaluate_er_ablation.py` (bootstrap CIs + figures, output in
`.scratch/results/er/report/`) and `tools/analyze_er_ablation.py` (output in
`.scratch/results/er/analysis/`). They agree to four decimal places on every shared quantity
(e.g. RNA Δ 0.044235 vs 0.0442; DeLong z 4.310954 vs 4.311; clinicopath p 0.739151 vs 0.739),
which rules out an implementation artefact in the headline result.

Pooled AUROC confidence intervals (bootstrap): WSI-alone 0.896 [0.869, 0.920]; WSI+RNA
0.941 [0.919, 0.961]; WSI+clinicopath 0.894 [0.867, 0.918].

**Figures** (in `.scratch/results/er/report/`, `_case` and `_slide` variants): `fig_roc_*.png`,
`fig_pr_*.png`, `fig_calibration_*.png`, `fig_per_fold_auroc_*.png`, `fig_per_site_auroc_*.png`.

## 4. Operating point (threshold 0.5) — confusion matrices

| Arm | TN | FP | FN | TP | Sensitivity (ER+) | Specificity (ER−) |
|---|---:|---:|---:|---:|---:|---:|
| WSI-alone | 175 | 52 | 92 | 684 | 0.881 | 0.771 |
| WSI + RNA | 191 | 23 | 39 | 703 | **0.947** | **0.893** |
| WSI + clinicopath | 171 | 56 | 79 | 697 | 0.898 | 0.753 |

The RNA arm's largest gain is in **specificity for the minority ER-negative class** (0.771 →
0.893) — the clinically consequential direction, since ER-negative patients should not receive
endocrine therapy. Clinicopath fusion, if anything, slightly *lowers* ER-negative specificity.

## 5. Per-tissue-site generalization (Howard 2021)

AUROC computed within each held-out site's test fold, for sites with both ER classes and
n ≥ 10 cases (19 of 38 sites qualify; the remaining 19 are single-class or too small to score).
Full table: `.scratch/results/er/analysis/per_site_auroc.csv`.

| Site | N (−/+) | WSI-alone | WSI+RNA | WSI+clinicopath |
|---|---|---:|---:|---:|
| BH | 138 (23/115) | 0.909 | 0.935 | 0.921 |
| A2 | 100 (24/76) | 0.939 | 0.977 | 0.937 |
| E2 | 89 (22/67) | 0.899 | 0.980 | 0.910 |
| A8 | 83 (8/75) | 0.958 | 0.990 | 0.832 |
| D8 | 78 (16/62) | 0.991 | 0.995 | 0.978 |
| AR | 70 (14/56) | 0.888 | 0.948 | 0.883 |
| B6 | 47 (15/32) | 0.888 | 0.877 | 0.846 |
| AO | 47 (12/35) | 0.929 | 1.000 | 0.933 |
| AC | 47 (6/41) | 0.967 | 0.973 | 0.980 |
| C8 | 44 (18/26) | 0.814 | 0.936 | 0.823 |
| EW | 43 (12/31) | 0.989 | 1.000 | 0.978 |
| A7 | 41 (12/29) | 0.908 | 0.977 | 0.928 |
| AN | 32 (10/22) | 0.650 | 0.805 | 0.664 |
| E9 | 29 (6/23) | 0.870 | 1.000 | 0.870 |
| GM | 19 (7/12) | 0.810 | 0.667 | 0.774 |
| LL | 19 (5/14) | 0.771 | 0.846 | 0.743 |
| OL | 15 (6/9) | 0.944 | 0.972 | 0.944 |
| A1 | 12 (3/9) | 0.778 | 0.714 | 0.889 |
| S3 | 10 (2/8) | 1.000 | 1.000 | 1.000 |

RNA fusion improves AUROC at 15 of the 19 scored sites (occasionally on very small sites it is
noisier, e.g. GM n=19, A1 n=12). Clinicopath fusion tracks WSI-alone site-by-site. Performance
varies markedly across submitting sites (e.g. AN at 0.65 for WSI-alone) — the reason site
holdout matters and the motivation for the deferred CPTAC external check.

## 6. Calibration

Expected calibration error (10-bin, case level): WSI-alone 0.090, WSI+RNA **0.049**,
WSI+clinicopath 0.067. The RNA arm is both the most discriminating and the best calibrated; all
three are only moderately calibrated and would benefit from post-hoc temperature scaling before
any thresholded clinical read-out.

## 7. Verification (leakage and matched-N re-derivation)

Re-derived directly from the split files, the fusion tables, and the per-fold prediction pickles.
A **fresh-context verifier independently recomputed all five checks from the raw files, trusting
no prior number, and returned PASS on every one**:

- **(a) Split integrity — PASS.** In all 10 folds the train/val/test slide sets and case sets are
  pairwise disjoint (no case's slides straddle a split); every split slide_id exists in the
  dataset_csv; the 10 test folds partition all 1003 cases exactly once (union 1003, 0 repeats,
  0 missing). Each fold's train+val+test = 1068 slides; test sizes range 95–138, as expected
  under site holdout.
- **(b) Site holdout — PASS.** In all 10 folds the TEST sites are disjoint from both TRAIN and VAL
  sites; the leave-site-groups-out split holds out 7 sites, likewise disjoint. 0 violations.
  (Train and val do share sites, by design.)
- **(c) Per-fold transforms — PASS (no standardisation leakage).** Both fusion arms have 10/10
  *distinct* per-fold standardisation mean-vectors (RNA: mean across-fold std 2.775; clinicopath:
  0.0254), confirming train-fold-only fitting. Selected features: 10000 (RNA), 24 (clinicopath).
- **(d) Fusion N = matched intersection — PASS.** WSI cases 1003, RNA table 996, clinicopath table
  1046 ⇒ |WSI∩RNA| = 956 and |WSI∩clinicopath| = 1003. Each arm's predicted-case set equals its
  intersection by **set equality**, not merely by count.
- **(e) Label integrity — PASS.** dataset_csv labels ∈ {ER-negative, ER-positive}, no
  NaN/Indeterminate; fusion-table labels match the dataset_csv on every shared case (0 mismatches);
  all 3155 slide-level prediction entries across the 3 arms × 10 folds carry the integer label
  matching the dataset_csv via {ER-negative:0, ER-positive:1} (0 mismatches).

## 8. Caveats and limitations

- **RNA is not target leakage.** The RNA table retains ESR1 and other ER-pathway genes by design.
  The ER label is IHC-derived (a protein stain), so the transcriptome — including ESR1 mRNA — is a
  biologically independent predictor of the protein-level readout, not the label itself. This is
  exactly what distinguishes ER (a clean fusion task) from PAM50 (whose label is defined from
  expression, making RNA fusion circular).
- **Internal CV, not external validation.** WSI-alone AUROC ≈ 0.90 exceeds the ~0.80–0.82 reported
  for ER-from-H&E in the literature; those figures are typically external-cohort numbers, whereas
  this is internal site-holdout CV on strong UNI2-h foundation features. The staged CPTAC external
  validation (deferred) is where the externally-generalising number will be established.
- **ER cutoff sensitivity not run.** The label is the pre-binarised Positive/Negative IHC call;
  a percent-positive field exists only as coarse ordinal buckets, so the 1%-vs-10% cutoff
  sensitivity analysis could not be run. No percentage was fabricated.
- **Leave-site-groups-out not separately trained** (deferred by the author, 2026-07-22). The
  dedicated LSGO split exists but was not trained as its own run; per-site generalization here is
  read from the pooled 10-fold site-holdout test predictions. Training it is a one-command add-on
  (see the hand-off file).
- **Unit of analysis.** Headline numbers are case-level (per-case mean of slide probabilities).
  The slide-level analysis gives the same conclusion (RNA Δ +0.050, p ≈ 3×10⁻⁶; clinicopath null),
  so the result is invariant to the aggregation choice.
- **Class imbalance (77% ER+).** Report AUPRC/F1/balanced accuracy alongside AUROC; accuracy alone
  is misleading. RNA fusion's gain is concentrated in the minority (ER-negative) class.
- **Single seed.** All arms use seed 1. Repeating across seeds would tighten the variance estimate.

## 9. Conclusion

Under a leakage-controlled, tissue-site-held-out evaluation on TCGA-BRCA, **adding RNA-seq to H&E
significantly improves ER-status prediction (+0.044 AUROC, p = 1.6×10⁻⁵), driven by better
detection of the minority ER-negative class, while adding routine clinicopathology does not
(−0.002 AUROC, p = 0.74).** This is a positive, publishable ablation result: the multimodal
benefit for a clinically-anchored, IHC-derived label is real and specific to the molecular
modality — and it establishes the clean, non-circular fusion setup (unlike PAM50) that the ODX/
external-validation chapters build on.

---

*Artifacts. Primary analysis (bootstrap CIs, figures, slide+case units):
`tools/evaluate_er_ablation.py` → `.scratch/results/er/report/` (metrics CSVs, `paired_tests_*.csv`,
`report.md`, 10 figures). Independent cross-check: `tools/analyze_er_ablation.py` →
`.scratch/results/er/analysis/` (`metrics.json`, `case_predictions_*.csv`, `per_site_auroc.csv`).
Preparation baton: `docs/implementation-research/handoff.md`. Lessons: `.scratch/er_pipeline_notes.md`.*
