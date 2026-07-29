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

---

# Part C — the novel fusion mechanism (FiLM-conditioned attention MIL)

**Headline: the pre-registered primary endpoint is a NULL.** A FiLM-conditioned attention
mechanism does **not** significantly outperform CLAM's stock gated fusion on either modality.
What the chapter does establish is different and, in one respect, more interesting: the
*established* patch-conditioning operator (MCAT-style co-attention) **actively harms** when the
second modality is uninformative, while the FiLM mechanism degrades gracefully to the image-only
model. Design, pre-registration and implementation record:
`docs/implementation-research/novel-fusion-design.md`.

All numbers below are case-level, out-of-fold, over the same 10 tissue-site-holdout folds, same
seed 1, same frozen per-fold WSI checkpoints as Parts A and B. Reproduce with
`python tools/evaluate_er_ablation.py`.

## C1. Why this chapter exists — the diagnosis that preceded it

Three findings, computed from the Part B artifacts without retraining, reframed the Part A/B
result and are contributions in their own right.

1. **A single gene beats the entire multimodal model.** Raw *ESR1* expression used directly as a
   score, with **zero fitted parameters**, reaches AUROC **0.9605** against the gated fusion arm's
   0.9412 on the same 956 matched cases (DeLong p = 0.017). *ESR1* survives the variance top-10000
   selection in 10/10 folds, so the model was handed the gene and still underperformed it. The
   stock gated operator does not merely fail to exploit the transcriptome — it *destroys*
   information a one-line baseline preserves.
2. **The logged gate statistic is misleading.** Both fusion arms report an image-gate near an even
   blend (0.55 RNA / 0.66 clinicopath), which reads as healthy integration. Functional ablation of
   the trained checkpoints says otherwise: deleting the image costs the RNA arm only 0.006 AUROC
   (it still scores 0.9353, *above* WSI-alone), while deleting the table costs the clinicopath arm
   exactly 0.000. Both arms are functionally unimodal, in opposite directions. **A gate mean is
   not evidence of multimodal integration; report a functional ablation instead.**
3. **There is almost no late-fusion headroom.** A fitting-free rank-average of *ESR1* and the
   image peaks at 0.9610, indistinguishable from *ESR1* alone (p = 0.80). For clinicopathology, a
   stacker with full access to both signals is *significantly worse* than the image alone
   (0.8788 vs 0.8957, p = 0.011) — the null is a property of the data, not of the mechanism.

The mechanistic cause: `gated` is an element-wise convex combination, so the fused vector is
confined to the axis-aligned box between the two projections. It can reweight modalities but not
let them interact, and blending a strong signal with a weaker one lands in between — which is
exactly the observed 0.9412.

## C2. The missing baselines, now measured

`known_gaps` item 1 is closed. Same folds, same train-fold-only selection and standardisation.

| Modality | Table-only logistic regression | Table-only MLP | Best fusion arm | WSI-alone |
|---|---:|---:|---:|---:|
| RNA | **0.9511** | 0.9431 | 0.9502 (co-attention) | 0.8969 |
| Clinicopath | 0.6474 | 0.6457 | 0.8984 (FiLM) | 0.8957 |

**No fusion arm beats the RNA table alone.** FiLM vs the RNA probe: −0.0049, p = 0.581. The
+0.044 that Part B reported over WSI-alone is attributable to the RNA table, not to synergy.

## C3. Full arm table (case level, pooled out-of-fold)

| Arm | N | AUROC | 95% CI | AUPRC | F1(ER+) | per-fold mean ± sd | ECE | Brier |
|---|---:|---:|---|---:|---:|---|---:|---:|
| WSI-alone | 1003 | 0.8957 | [0.869, 0.920] | 0.9609 | 0.9048 | 0.906 ± 0.047 | 0.090 | 0.107 |
| WSI+RNA gated (dim 32) | 956 | 0.9412 | [0.919, 0.961] | 0.9750 | 0.9578 | 0.949 ± 0.040 | 0.049 | 0.060 |
| WSI+RNA gated (dim 96, capacity control) | 956 | 0.9447 | [0.925, 0.962] | 0.9799 | 0.9504 | 0.945 ± 0.041 | 0.042 | 0.064 |
| **WSI+RNA FiLM-attention** | 956 | **0.9462** | [0.928, 0.962] | 0.9821 | 0.9533 | 0.950 ± **0.027** | 0.051 | 0.064 |
| WSI+RNA additive-logit (rank 0) | 956 | 0.9343 | [0.912, 0.953] | 0.9775 | 0.9463 | 0.941 ± 0.031 | 0.047 | 0.074 |
| WSI+RNA adapted co-attention | 956 | 0.9502 | [0.930, 0.969] | 0.9813 | 0.9607 | 0.952 ± 0.041 | 0.054 | 0.055 |
| RNA table only | 956 | 0.9511 | [0.929, 0.969] | 0.9769 | 0.9640 | 0.956 ± 0.040 | 0.031 | 0.051 |
| WSI+clinicopath gated (dim 32) | 1003 | 0.8937 | [0.867, 0.918] | 0.9585 | 0.9117 | 0.898 ± 0.052 | 0.067 | 0.103 |
| WSI+clinicopath gated (dim 96) | 1003 | 0.8950 | [0.869, 0.921] | 0.9549 | 0.9219 | 0.898 ± 0.047 | 0.050 | 0.099 |
| **WSI+clinicopath FiLM-attention** | 1003 | **0.8984** | [0.873, 0.922] | 0.9625 | 0.9061 | 0.909 ± 0.041 | 0.091 | 0.108 |
| WSI+clinicopath additive-logit (rank 0) | 1003 | 0.8984 | [0.872, 0.922] | 0.9623 | 0.9092 | 0.906 ± 0.047 | 0.084 | 0.104 |
| WSI+clinicopath adapted co-attention | 1003 | 0.8731 | [0.843, 0.900] | 0.9487 | 0.8841 | 0.879 ± 0.061 | 0.112 | 0.128 |
| Clinicopath table only | 1003 | 0.6474 | [0.607, 0.686] | 0.8571 | 0.6877 | 0.663 ± 0.111 | 0.238 | 0.234 |

## C4. Pre-registered primary endpoint — NULL on both modalities

DeLong, paired on matched cases, Holm-corrected across the two primary comparisons. Fixed before
any test number was read.

| Comparison | N | AUROC | vs | Δ | DeLong p | Holm p | Verdict |
|---|---:|---:|---|---:|---:|---:|---|
| RNA: FiLM vs gated | 956 | 0.9462 | 0.9412 | +0.0050 | 0.462 | **0.838** | not significant |
| Clinicopath: FiLM vs gated | 1003 | 0.8984 | 0.8937 | +0.0047 | 0.419 | **0.838** | not significant |

**The novel mechanism does not improve on the stock gated operator.** The small positive
case-level delta for RNA is not even robust to the aggregation unit: at slide level it is
**−0.0000 (p = 0.999)**, versus +0.0050 at case level. It is noise.

## C5. Secondary comparisons

Fourteen pre-registered secondary comparisons, reported **uncorrected**; a strict Bonferroni
threshold over them would be α = 0.0036, and only the starred row clears it.

| Comparison | N | Δ AUROC | p |
|---|---:|---:|---:|
| RNA FiLM vs WSI-alone | 956 | +0.0493 | 3.5×10⁻⁸ |
| RNA FiLM vs gated dim-96 capacity control | 956 | +0.0015 | 0.739 |
| RNA FiLM vs its own additive-logit ablation | 956 | +0.0119 | 0.0092 |
| RNA FiLM vs adapted co-attention | 956 | −0.0040 | 0.572 |
| RNA FiLM vs RNA table only | 956 | −0.0049 | 0.581 |
| RNA additive-logit vs gated | 956 | −0.0068 | 0.350 |
| RNA co-attention vs gated | 956 | +0.0091 | 0.173 |
| Clinicopath FiLM vs WSI-alone | 1003 | +0.0028 | 0.458 |
| Clinicopath FiLM vs gated dim-96 | 1003 | +0.0034 | 0.524 |
| Clinicopath FiLM vs its own additive-logit ablation | 1003 | +0.0001 | 0.974 |
| **Clinicopath FiLM vs adapted co-attention** ★ | 1003 | **+0.0253** | **0.0030** |
| Clinicopath FiLM vs clinicopath table only | 1003 | +0.2511 | 3.8×10⁻³³ |
| Clinicopath additive-logit vs gated | 1003 | +0.0047 | 0.432 |
| Clinicopath co-attention vs gated | 1003 | −0.0206 | 0.0089 |

## C6. What the chapter does establish

**1. Graceful degradation distinguishes the mechanisms, and this is the positive result.** The
pre-registered success criterion for clinicopathology was *do no harm* — statistical
indistinguishability from WSI-alone, because §C1 shows there is no signal there to extract.

- FiLM: 0.8984 vs 0.8957, Δ +0.0028, p = 0.458 — **criterion met**.
- Adapted co-attention: 0.8731 vs 0.8957, Δ **−0.0226, p = 0.012** — **criterion failed**; it is
  also significantly worse than the stock gated arm (−0.0206, p = 0.0089).

So the established patch-conditioning operator *damages* an image model when handed an
uninformative second modality, while the FiLM mechanism does not. That is the "one unmodified
mechanism serves two modalities with opposite statistics" claim, and it is the one claim the data
supports. It is a robustness property, not an accuracy gain.

**2. The attention conditioning does contribute — just not enough.** On RNA, FiLM beats its own
`film_rank 0` ablation by +0.0119 (p = 0.0092, nominal). The conditioning is doing real work
beyond simply removing the gated bottleneck. Consistent with this, the learned modulation moves
well away from identity (mean |γ − 1| = 1.20 on the selection folds). But it cannot clear the
ceiling in §C1, because no late-fusion arrangement of these two signals can.

**3. Capacity is not the explanation for anything.** The gated arm at `fusion_hidden_dim 96`
(93,026 mechanism parameters, more than FiLM's 83,714) reaches 0.9447 — indistinguishable from
FiLM (p = 0.739) and from gated dim-32. No arm's result is attributable to parameter count.

**4. FiLM is markedly more stable across folds on RNA**: per-fold sd 0.027 versus 0.040 for gated
and 0.041 for co-attention, the lowest of any RNA arm. With ten folds this is descriptive, not
tested, but it is consistent with the zero-initialised design starting every fold from the
image-only solution.

**5. The ceiling predicted from the diagnosis held exactly.** Every RNA arm lands in
0.934–0.951 and the table-only probe (0.9511) sits at the top. The §C1 headroom analysis
predicted ≈0.96 as the late-fusion ceiling before any of these arms were trained.

## C7. Calibration

Brier is decomposed (Murphy) into reliability (calibration, lower better) and resolution
(discrimination, higher better), which separates "ranks better" from "reports better-calibrated
numbers" in a way binned ECE cannot.

| Arm | ECE | Brier | reliability | resolution |
|---|---:|---:|---:|---:|
| RNA table only | 0.031 | 0.051 | 0.0032 | 0.1264 |
| WSI+RNA co-attention | 0.054 | 0.055 | 0.0048 | 0.1232 |
| WSI+RNA gated (dim 32) | 0.049 | 0.060 | 0.0053 | 0.1193 |
| WSI+RNA FiLM | 0.051 | 0.064 | 0.0057 | 0.1149 |
| WSI+clinicopath FiLM | 0.091 | 0.108 | 0.0145 | 0.0816 |
| WSI+clinicopath co-attention | 0.112 | 0.128 | 0.0247 | 0.0711 |
| WSI-alone | 0.090 | 0.107 | 0.0130 | 0.0806 |

The clinicopath co-attention arm is worst on every calibration measure, reinforcing §C6.1: it is
not merely less accurate, it is less reliable and less sharp. No arm is well calibrated in
absolute terms; all would benefit from temperature scaling, which is deferred (see Caveats).

## C8. Conclusion

Under the same leakage-controlled, site-held-out protocol, **a FiLM-conditioned attention-MIL
does not improve ER prediction over CLAM's stock gated fusion (RNA +0.005, Holm p = 0.84;
clinicopathology +0.005, Holm p = 0.84), and the RNA difference disappears entirely at slide
level.** The pre-registered hypothesis that conditioning MIL attention on a second modality would
exceed the late-fusion ceiling is **not supported**.

The chapter's contributions are therefore diagnostic and methodological rather than an accuracy
gain: that a single gene outperforms the published multimodal model; that the commonly-reported
fusion gate statistic is not evidence of multimodal integration and a functional ablation is;
that routine clinicopathology carries no ER signal beyond morphology; and that among
patch-conditioning operators, FiLM degrades gracefully on an uninformative modality where
MCAT-style co-attention significantly harms.

## C9. Caveats specific to Part C

- **Single seed (1)**, inherited from Parts A/B. All Part C differences are small enough that a
  multi-seed estimate would materially change the confidence attached to them.
- **The co-attention arm is not MCAT.** It is MCAT's fusion *operator* re-implemented in this
  harness so that only the operator differs. It also does not reuse the pretrained WSI branch the
  way the other arms do: perturbing CLAM's attention head or bag classifier changes its logits by
  exactly 0.000, because it substitutes its own projection, attention and head. And it carries
  690,692 mechanism parameters against FiLM's 83,714, so its clinicopathology failure may reflect
  overfitting at this cohort size rather than the operator as such. No claim is made about
  published MCAT.
- **Fourteen uncorrected secondary comparisons.** Only clinicopath FiLM vs co-attention
  (p = 0.0030) survives a strict Bonferroni threshold of 0.0036.
- **`film_rank` was selected on validation folds only** (16/32/64; means 0.9546/0.9572/0.9582),
  and rank 64 was indistinguishable from rank 32 (p = 0.449). Test folds were read once, after
  the rank was fixed.
- **The missing-modality evaluation was not run.** It requires a forward pass with the tabular
  modality marked absent, and `evaluate_multimodal.py` cannot construct the new modes; validation
  predictions are also not saved, so leakage-free temperature scaling is unavailable for this
  chapter. Both are documented gaps, not oversights.
- **Prior art.** "RNA-seq improves binary ER prediction on TCGA-BRCA" is already published
  (Zhang et al., MICCAI 2025, arXiv 2508.17213: 0.9331 → 0.9581 on the same cohort and label).
  Parts A/B are a replication with a cleaner single-variable ablation; only the mechanism and the
  diagnosis in Part C are candidate contributions.
