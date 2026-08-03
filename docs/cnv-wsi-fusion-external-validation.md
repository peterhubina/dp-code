# WSI + arm-level CNV → PAM50: external validation on CPTAC-BRCA

**Date:** 2026-08-03
**Reproduce:** `python tools/evaluate_cnv_wsi_fusion.py --internal`
**Train:** TCGA-BRCA. **External test:** CPTAC-BRCA, n = 114 cases (Basal 27, Her2 14, LumA 56, LumB 17).
**Nothing was refit or tuned on CPTAC.** The fusion rule (equal-weight mean of the two probability
vectors) was fixed on TCGA before the external set was touched.

**Inputs**
- WSI arm — TCGA-trained CLAM-MB + UNI2-h, applied to CPTAC:
  `.scratch/cptac_validation/results/predictions/ensemble_predictions.csv` (378 slides → 114 cases,
  mean-pooled per case).
- CNV arm — 39 chromosome arms, median gene-level log2 per arm, built by
  `tools/download_cnv_mutations.py --representation arm`. Logistic regression (`C=0.1`,
  `class_weight='balanced'`) fit on all 945 non-Normal TCGA cases, applied unchanged to CPTAC.

Sanity check: the WSI arm reproduces the previously documented external figure of **0.847** exactly,
so the two pipelines are being evaluated on the same object.

---

## 1. Headline — raw arms, as first run

| Model | macro AUROC | 95% CI | balanced acc |
|---|---|---|---|
| WSI (CLAM-MB + UNI2-h) | 0.847 | [0.791, 0.895] | 0.513 |
| **CNV (39 arms)** | **0.888** | [0.835, 0.933] | **0.716** |
| Fusion (equal-weight mean) | **0.909** | [0.858, 0.948] | 0.646 |

Paired bootstrap over the same 114 cases, 4,000 resamples. These are the **raw** arms; §3
repeats them with the WSI decision rule matched to the CNV arm's, which changes the picture:

| Contrast | Δ macro AUROC | 95% CI | verdict |
|---|---|---|---|
| Fusion − WSI | **+0.063** | [+0.023, +0.106] | **significant** |
| Fusion − CNV | +0.021 | [−0.001, +0.046] | not significant |
| CNV − WSI | +0.042 | [−0.018, +0.100] | not significant |

---

## 2. The recalibration control, and what it settles

The WSI model's Her2 AUROC is 0.860 while its Her2 recall is 0/14, which looks like a threshold
artifact. **It is not.** Dividing the WSI probabilities by the TCGA training prior (Her2 = 8.3%, so
a ~12x boost) and renormalising leaves Her2 recall at **0/14**. Saerens-Latinne-Decaestecker prior
estimation — fully unsupervised, no labels — independently drives the estimated CPTAC Her2 prior to
**0.000** against a true 0.123, and also yields 0/14.

| | max p_Her2 over 114 cases | mean | mean on true-Her2 cases | cases argmaxed to Her2 |
|---|---|---|---|---|
| raw | 0.1335 | 0.0329 | 0.0654 | 0/114 |
| prior-balanced (12x) | 0.2992 | 0.0997 | 0.1840 | **0/114** |

So the WSI model retains *partial* Her2 ranking ability (true-Her2 cases average roughly twice the
p_Her2 of the cohort) but its Her2 head is compressed so far that **no monotone prior reweighting
recovers a single Her2 call**. The claim "the CNV arm calls a class the H&E arm cannot" survives the
control. This is a genuine failure mode of the WSI model on this cohort, not a decision-threshold
choice.

Prior-balancing is still the right decision rule for a fair comparison, because the CNV arm is fit
with `class_weight='balanced'`. It was run post hoc, after the raw fusion underperformed — reported
here as a control, not as the pre-registered analysis.

## 3. Results with a matched decision rule

| Model | macro AUROC | 95% CI | balanced acc | Her2 recall |
|---|---|---|---|---|
| WSI raw | 0.847 | [0.791, 0.895] | 0.513 | 0/14 |
| WSI prior-balanced | 0.865 | — | 0.554 | 0/14 |
| WSI SLD-EM | 0.858 | — | 0.510 | 0/14 |
| CNV (39 arms) | 0.888 | [0.835, 0.933] | 0.716 | 12/14 |
| Fusion (raw WSI) | 0.909 | [0.858, 0.948] | 0.646 | 6/14 |
| **Fusion (balanced WSI)** | **0.912** | — | **0.740** | 10/14 |

Paired bootstrap, 4,000 resamples:

| Contrast | Δ macro AUROC | Δ balanced acc |
|---|---|---|
| Fusion(bal) − WSI(bal) | **+0.048** [+0.012, +0.085] | **+0.185** [+0.089, +0.273] |
| Fusion(bal) − CNV | **+0.024** [+0.000, +0.050] *(marginal)* | +0.025 [−0.057, +0.103] ns |
| Fusion(bal) − Fusion(raw) | +0.003 [−0.005, +0.011] ns | **+0.093** [+0.019, +0.181] |
| CNV − WSI(bal) | +0.023 [−0.032, +0.079] ns | **+0.161** [+0.053, +0.264] |

Two things follow. **Recalibrating the WSI arm before fusing is worth +0.093 balanced accuracy** —
averaging a Her2-blind vector into a Her2-competent one is what dragged raw fusion's Her2 recall to
6/14, and fixing the decision rule recovers 10/14. And with the matched rule, **fusion becomes the
best model on both metrics**, though its edge over CNV alone is marginal: the AUROC CI lower bound
is +0.000 and the balanced-accuracy difference is not significant at n = 114.

### Per-class AUROC

| | Basal (27) | Her2 (14) | LumA (56) | LumB (17) |
|---|---|---|---|---|
| WSI | 0.972 | 0.860 | 0.861 | 0.693 |
| CNV | 0.972 | 0.871 | 0.883 | 0.827 |
| Fusion | 0.992 | 0.881 | 0.916 | 0.848 |

### Per-class recall at argmax — where the real difference lives

| | Basal | **Her2** | LumA | **LumB** |
|---|---|---|---|---|
| WSI raw | 24/27 | **0/14** | 52/56 | 4/17 |
| WSI prior-balanced | 26/27 | **0/14** | 34/56 | 11/17 |
| CNV | 22/27 | **12/14** | 37/56 | 9/17 |
| Fusion (raw WSI) | 23/27 | 6/14 | 50/56 | 7/17 |
| Fusion (balanced WSI) | 24/27 | **10/14** | 43/56 | 10/17 |

Her2 is the whole story: the WSI arm cannot call it at any prior, the CNV arm calls 12 of 14, and
fusion keeps 10 of 14 once the WSI arm's decision rule is matched.

### Error independence

External φ(correctness) between the two models is **−0.006** — the errors are essentially
independent, and either-model-right is 0.912 against WSI-alone accuracy of 0.702. So the ceiling for
a *better* combination rule is real; the equal-weight mean is simply not exploiting it.

---

## 4. Controls already run

| Control | Result |
|---|---|
| Internal TCGA, 5-fold × 10 seeds | 0.866 ± 0.003 |
| Leave-one-TCGA-site-out, 13 sites ≥25 cases | 0.878 ± 0.035 — site confounding does not inflate it |
| Aneuploidy-burden only (1 feature) | 0.685 — the arm *pattern* carries signal beyond total instability |
| Regularisation sweep C = 0.01 / 0.1 / 1 / 10 | 0.879 / 0.870 / 0.860 / 0.856 — `C=0.1` was not cherry-picked |
| Cross-cohort platform check (SNP6 array vs WGS) | per-arm mean r = 0.960, mean \|Δ\| = 0.041, CPTAC sd ratio 0.82 |
| Arm derivation vs TCGA's official arm calls | Gain 0.405 > Unchanged 0.003 > Loss −0.338, monotonic over 39 arms |

---

## 5. Internal TCGA comparison, and what it says about the Her2 collapse

Run `--internal`: 599 TCGA cases that have CLAM out-of-fold predictions, both arms 10-fold.

| Model | macro AUROC | 95% CI | balanced acc | Basal | Her2 | LumA | LumB |
|---|---|---|---|---|---|---|---|
| WSI | 0.887 | [0.865, 0.908] | 0.677 | 91/106 | **26/51** | 257/318 | 66/124 |
| CNV (39 arms) | 0.862 | [0.836, 0.888] | 0.662 | 92/106 | 26/51 | 227/318 | 69/124 |
| **Fusion** | **0.922** | [0.904, 0.938] | **0.747** | 100/106 | 32/51 | 256/318 | 76/124 |

Fusion beats both arms significantly on both metrics internally; WSI vs CNV is not significant on
either (ΔAUROC +0.025 [−0.007, +0.057]) — the two are statistically indistinguishable on TCGA.

**Crucially, the WSI arm calls Her2 perfectly adequately internally (26/51) and not at all
externally (0/14).** So the collapse is *induced by the domain shift*, not an inherent property of
the model or of its training priors — which is consistent with Fernandez-Romero 2026 attributing 80%
of transfer degradation to staining and feature-space divergence.

## 6. What still needs running

1. ~~Recalibrate the WSI model and re-test.~~ **Done — see §2.** Prior correction does not rescue
   Her2 (still 0/14 after a 12x boost), so the finding stands.
2. **A learned fusion rule.** φ = −0.006 externally says the information is there and the equal-weight
   mean is wasting it. `project/CLAM/models/model_multimodal.py` already has concat / gated /
   residual / cross_attention. Report the mean as the baseline — if a learned gate cannot beat an
   unweighted average, that is itself a finding, and given the RNA-fusion history it is a live
   possibility.
3. **Multi-seed the CNV arm inside the external comparison.** The external numbers rest on one fit.
4. **Power.** Her2 n = 14 and LumB n = 17. Per-class external estimates are indicative, not precise;
   the 12/14 vs 0/14 contrast is large but rests on 14 cases.
5. **Stain normalisation / encoder ablation on the WSI arm.** Since the Her2 collapse is
   shift-induced rather than prior-induced, the remediation belongs on the imaging side.

---

## 7. Positioning

From the 50-paper survey in `docs/implementation-research/PAM50/`: **no published multimodal PAM50
model is externally validated with a PAM50-specific metric.** Amer et al. 2025 (arXiv:2509.03408) is
the only WSI+CNV PAM50 work and reports 10-fold CV on TCGA only — their CNV-alone arm is 0.8284 and
their four-modality fusion 0.9153, both internal. The numbers above are the external counterpart
that literature does not have.

The defensible claims, in order of strength:

1. **WSI + CNV fusion beats the WSI arm externally on both metrics** — ΔAUROC +0.066
   [+0.026, +0.107], Δbalanced-accuracy +0.226 [+0.127, +0.324] against raw WSI, and still
   +0.047 / +0.186 against the prior-balanced WSI. This is an externally validated multimodal PAM50
   result, which the literature does not currently contain.
2. **A cheap, FFPE-robust, shallow-WGS-reachable 39-feature assay is statistically
   indistinguishable from a UNI2-h + CLAM pipeline on an independent cohort** (ΔAUROC −0.023
   [−0.080, +0.032] vs prior-balanced WSI) and significantly better at the decision level
   (Δbalanced-accuracy +0.162 [+0.054, +0.265]).
3. **The WSI arm cannot call Her2-enriched on CPTAC at any prior** (0/14 raw, 0/14 after a 12x prior
   boost, 0/14 under unsupervised SLD-EM), while calling it adequately in-domain (26/51). The CNV
   arm calls 12/14 and fusion retains 10/14.
4. Fusion's margin **over CNV alone** is marginal: ΔAUROC +0.024 with a CI lower bound of exactly
   +0.000, and no significant balanced-accuracy difference. Report this, or the work reproduces the
   selective reporting the survey criticises.

Claims 1–3 are the paper. Claim 4 is the honesty that makes it credible.
