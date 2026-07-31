# ER status: TCGA-BRCA → CPTAC-BRCA external validation

Status as of 2026-07-31. Internal (TCGA) results are in `docs/er-prediction-results.md`;
this document covers the external leg only.

**Headline.** The H&E model's *discrimination* transports to an independent proteogenomic
cohort with no detectable degradation (case-level AUROC 0.925 external vs 0.896 internal).
Its *calibration* does not: at the shipped 0.50 threshold, specificity collapses from 0.771
to 0.108. A threshold shift requiring no external labels recovers internal operating
characteristics. Adding the clinicopath block does not help externally and trends negative.

---

## 1. What was run

| Arm | Trained on | Applied to CPTAC | Retraining needed |
|---|---|---|---|
| `er_wsi_alone` | TCGA, 10 site-holdout folds | yes | none |
| `er_wsi_clinpath_gated` | TCGA, same folds, frozen WSI branch | yes | none |
| `er_wsi_rna_gated` | TCGA | **descoped** | — |

Weights frozen, no fine-tuning, no domain adaptation. Both arms are scored by the same
10-fold ensemble (mean softmax) and aggregated to case level by the mean over each case's
slides — the convention in `tools/analyze_er_ablation.py`, so external numbers are directly
comparable to internal ones.

The RNA arm was descoped deliberately: CPTAC RNA is GDC linear TPM against the model's Xena
log2-RSEM training scale, so it would need FSQN unsupervised domain adaptation plus HGNC
symbol remapping for ~15% of selected genes. That is a weaker claim than the WSI result and
a failure mode unrelated to the science.

## 2. Cohort

`tools/cptac/prepare_cptac_er_manifest.py` → **387 slides / 118 cases (81 ER+ / 37 ER−)**.

This is larger than the PAM50 external cohort (114 cases) because the PAM50 manifest
inherited `download_cptac.py --cohort-only`, which kept only cases carrying RNA *and* a
PAM50 call. ER needs neither, so the manifest is built by indexing the 653-file feature
store directly. The gain is 4 cases and, more importantly, 3 extra ER-negatives — the
minority class that bounds the confidence interval.

ER calls come from cBioPortal `brca_cptac_2020` (`ER_UPDATED_CLINICAL_STATUS`); the CPTAC
pan-cancer clinical table carries no receptor status at all. 118 of 120 ER-labelled cases
have features; 78 further feature-cases have no cBioPortal entry.

### Geometry audit

CPTAC mixes 20x and 40x scans, so this was checked before anything else — a silent
magnification mismatch voided a previous external cohort (see
`hsibc-magnification-mismatch-bug`).

| base mpp | scanner | patch_size | custom_downsample | effective px | FOV |
|---|---|---|---|---|---|
| 0.4942 | 20x | 256 | 1 | 256 | 126.5 µm |
| 0.2501 | 40x | 512 | 2 | 256 | 128.1 µm |

All 653 files reduce to 256 effective px. The rule holds with **zero exceptions** across the
391 slides whose mpp is known, so it extrapolates safely to the 262 with no manifest entry.
TCGA training FOV is 128 µm. `prepare_cptac_er_manifest.py` asserts this and fails loudly.

## 3. WSI-alone: discrimination transports

| | n | prevalence ER+ | AUROC | 95% CI | AUPRC |
|---|---|---|---|---|---|
| internal (TCGA, out-of-fold) | 1003 | 0.774 | 0.8957 | [0.870, 0.920] | 0.961 |
| **external (CPTAC, ensemble)** | 118 | 0.686 | **0.9246** | [0.874, 0.966] | 0.970 |

Per-fold external case AUROC 0.891 ± 0.019. Slide-level 0.888 < case-level 0.925, i.e.
aggregation helps, as internally.

The CIs overlap substantially. **The correct claim is "no detectable degradation", not
"improvement"** — with 37 ER-negatives the external CI is ±0.05 and cannot resolve a
difference of this size.

### 3.1 Not a single-site artefact

CPTAC case ids carry a two-digit site prefix, and the cohort is lopsided: site 11 alone is
46 of 118 cases (39%). Leave-one-site-out (`report_er_external.py`) drops each site in turn
and recomputes on the remainder:

| site dropped | n dropped | AUROC on remainder | Δ vs full |
|---|---|---|---|
| 11 | 46 | 0.9234 | −0.0012 |
| 01 | 18 | 0.9184 | −0.0062 |
| 05 | 13 | 0.9178 | −0.0068 |
| 18 | 10 | 0.9252 | +0.0007 |
| 03 | 7 | 0.9259 | +0.0013 |
| … | ≤6 | 0.9217 – 0.9302 | ±0.006 |

**Full range across all 13 leave-one-out fits: 0.918 – 0.930.** Removing the single largest
site — 39% of the cohort — moves the headline by 0.001. The result is not carried by one
contributing centre.

Per-site AUROCs are reported in `per_site_auroc_case.csv` but should not be over-read: only
sites 11 (n=46), 01 (n=18), 05 (n=13) and 18 (n=10) have both classes and n ≥ 5, and site 20
(n=6, 2 ER+) shows 0.625 on essentially no data.

## 4. WSI-alone: calibration does not transport

At the shipped 0.50 threshold the model calls 98.8% of CPTAC cases ER-positive.

| | probability range | spread | mean p in ER− cases | frac ≥ 0.5 |
|---|---|---|---|---|
| internal | [0.009, 0.996] | 0.988 | 0.289 | 0.734 |
| external | [0.379, 0.943] | 0.564 | **0.611** | **0.958** |

Scores compress and shift upward; nothing on CPTAC is confidently negative. This is **not**
prior shift — CPTAC is 68.6% ER+ against TCGA's 77.4%, which pushes the other way.

| operating point | threshold | sens | spec | balanced acc |
|---|---|---|---|---|
| internal @0.5 | 0.500 | 0.881 | 0.771 | 0.826 |
| internal @internal-Youden | 0.642 | 0.834 | 0.841 | 0.838 |
| external @0.5 | 0.500 | 0.988 | **0.108** | **0.548** |
| external @internal-Youden (transferred) | 0.642 | 0.914 | 0.649 | 0.781 |
| **external @prevalence-matched** | 0.682 | 0.877 | 0.730 | **0.803** |
| external @external-Youden | 0.763 | 0.790 | 1.000 | 0.895 *(ORACLE)* |

The prevalence-matched threshold is the (1 − prevalence) quantile of the external scores. It
needs only a local ER+ prevalence estimate — a published epidemiological number, not
labelled slides — and recovers essentially internal operating characteristics
(0.877/0.730 vs 0.881/0.771). The external-Youden row is chosen on test labels and is an
upper bound, not an achievable result.

**The deployable claim: the model is transportable; its threshold is not.**

## 5. Clinicopath fusion does not help externally

Applied without retraining. Each fold reloads its own fitted standardisation from
`s_{fold}_tabular_transform.json`, and columns are matched to the transform by name.

| | internal | external |
|---|---|---|
| WSI-alone | 0.8957 | 0.9246 |
| WSI + clinicopath (gated) | 0.8937 | 0.9042 |
| **delta** | **−0.0020** (p = 0.74, null) | **−0.0204** |

External paired DeLong: z = 1.702, **p = 0.089**; paired bootstrap 95% CI
[−0.047, +0.003]. So the external drop is **a trend, not a significant difference** — it
should be reported as "no benefit, and directionally harmful", not as a demonstrated harm.

### 5.1 The eligibility-shifted table is *not* the cause

The obvious hypothesis was that three of the six clinicopath fields are eligibility-shifted,
because CPTAC enrolled stage IIA–IIIC only (Krug 2020, PMID 33212010):

| level | CPTAC % | TCGA % |
|---|---|---|
| stage I | 3.4 | 16.9 |
| stage IV | 0.0 | 1.7 |
| pT1 | 5.1 | 25.6 |
| pT2 | 71.2 | 57.7 |
| pM unknown | 40.7 | 15.6 |
| histology unknown | 9.3 | 0.1 |

**That hypothesis is not supported.** Re-running the arm with the standardised tabular
vector replaced by all-zeros — the fold's own training mean, i.e. a table carrying no
case-specific information; the "table absent" condition of `tools/diagnostics/gate_probe.py`
— gives (`infer_cptac_er_fusion.py --ablate_table`):

| condition | external AUROC | vs WSI-alone | paired DeLong |
|---|---|---|---|
| WSI-alone | 0.9246 | — | — |
| clinicopath, table absent | 0.9112 | −0.0133 | z = 1.872, p = 0.061, CI [−0.029, +0.000] |
| clinicopath, intact | 0.9042 | −0.0204 | z = 1.702, p = 0.089, CI [−0.046, +0.003] |

Effect of removing the table: **+0.0070, z = −0.771, p = 0.44, CI [−0.011, +0.027]** —
indistinguishable from zero. Internally the same operation was worth −0.0003.

So the −0.0204 external gap decomposes as roughly **+0.007 table (not significant) and
−0.013 fusion head**. The deficit lives in the head, which was trained on (image, table)
pairs and transports slightly worse than the plain WSI-alone classifier *even when fed a
table with no information in it*. The eligibility shift is real and worth documenting, but
it is not what costs the AUROC here.

Caveat inherited from `gate_probe.py`: the ablation is off-manifold — the branch never saw a
constant input during training — so the absolute ablated AUROC is a lower bound. The
load-bearing result is the relative comparison, and it agrees in both cohorts: the table
contributes nothing (internal −0.0003, external +0.0070 n.s.).

Mechanistically this matches the internal finding: the clinicopath fusion head is
*functionally an image classifier*. Internally, deleting the table cost −0.0003 while
deleting the image cost +0.3378 (0.8937 → 0.5559, near chance), and its case-level
predictions correlate r = 0.973 with the WSI-alone arm. The head learned to route around the
24-dim table entirely, and it does so on CPTAC too.

## 6. Why the clinicopath block is thin in the first place

From `docs/implementation-research/tcga-cptac-clinicopath-harmonisation.md`: only
**age, pN, LN+ count, histology** harmonise across the two cohorts without a significant
shift. Grade is unavailable in *both* (`[Not Available]` for all 1097 TCGA cases; `GX` for
all 114 CPTAC cases), and tumour size is absent from TCGA clinical and 100% empty in CPTAC.
Those two are exactly what drives clinical nomograms.

Measured ceiling of the transportable block alone, under 10-fold **site-holdout** on TCGA
(random CV inflates these by 0.03–0.06 through tissue-source-site leakage):

| target | clinicopath-only AUROC |
|---|---|
| TNBC | 0.704 |
| ER-negative | 0.646 |
| TP53 mutation | ~0.65 |
| ERBB2 amplification | 0.652 |
| nodal pN0 vs pN+ | 0.631 |
| PIK3CA mutation | 0.548 |

Race was dropped: it contributes nothing under site-holdout (ER 0.641 with vs 0.646 without;
TNBC 0.702 vs 0.704) while shifting significantly between cohorts (χ² = 12.25, p = 0.007).

PIK3CA was investigated as a fusion target and **rejected**. The published multimodal result
(Cancer Biology & Medicine 23(3):430, 2026 — clinical-alone 0.694, fusion 0.745) includes
*molecular subtype* in its "clinical" block, and PAM50 is an RNA assay. With a clean,
transportable block PIK3CA is 0.548, essentially chance, so no fusion lift is available.

## 7. Artefacts

Scripts (all under `tools/cptac/`):
- `prepare_cptac_er_manifest.py` → `.datasets/cptac-brca/cptac_brca_er_dataset.csv` + coverage report with named drop reasons
- `infer_cptac_er.py` → `.scratch/cptac_validation/results/er/`
- `summarise_er_external.py` → `.../er/external_calibration.json`
- `prepare_cptac_clinicopath.py` → `.datasets/cptac-brca/cptac_brca_er_clinicopath_clam.csv`
- `infer_cptac_er_fusion.py` → `.scratch/cptac_validation/results/er_clinpath/`

Model reconstruction was taken from the recorded run config
(`experiment_er_wsi_alone.txt`), not guessed: `clam_mb`, `size_arg='big'`, dropout 0.5,
`embed_dim` 1536, `k_sample` 4, `subtyping` False, label orientation
`{'ER-negative': 0, 'ER-positive': 1}` confirmed against `project/CLAM/main.py`. The fusion
loader follows `tools/diagnostics/gate_probe.py`, which validated the same reconstruction
against the saved per-fold pickles to max |Δprob| = 1.8e-07.

## 8. Outstanding

**Blocking nothing — the external leg is reportable as it stands.** In rough priority:

1. ~~Retrain the clinicopath arm on the transportable block~~ — **dropped, 2026-07-31.** The
   table-absent ablation (§5.1) answered the question for free: the table contributes nothing
   externally (+0.0070, p = 0.44), so swapping which variables are in it cannot recover the
   −0.0133 that sits in the fusion head. Retraining would change the table, not the head.
   The conclusion is also cleaner without it: clinicopath adds nothing internally *and*
   nothing externally, and the arm's small external deficit is a property of the fusion head.
2. ~~Report figures for the external leg~~ — **done**, `report_er_external.py` →
   `.scratch/cptac_validation/results/er/report/`: ROC and PR with the internal curve
   overlaid, reliability diagram, score-distribution comparison, leave-one-site-out.
3. ~~Per-site external breakdown~~ — **done**, see §3.1.
4. **Multi-seed.** Only seed 1 has been externally validated; `tools/train_er_multiseed.sh`
   exists but its checkpoints have not been pushed through the external leg. This is the
   largest remaining gap in the external leg's rigour.
5. **TNBC as the next target** — the only one whose transportable clinicopath block reaches
   0.704, and where age is genuinely invisible to the image encoder. Expected lift is small
   (H&E-alone will be ~0.90), so treat as a side quest, not a headline.

Explicitly **not** planned: the RNA arm externally (descoped, §1); PIK3CA (§6); HRD (no CPTAC
label and it cannot be computed from the available CNA data); survival or recurrence (CPTAC
follow-up is ≤601 days with 2 deaths and 0 recurrences).
