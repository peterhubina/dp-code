# Do the clinicopathological variables predict PAM50 subtype?

**Date:** 2026-07-31
**Cohorts:** TCGA-BRCA n = 910 cases, CPTAC-BRCA n = 114 cases (4-class PAM50; Normal-like
dropped, 33 in TCGA, 0 in CPTAC).
**Reproduce:** `python tools/analyse_clinicopath_pam50.py --n_repeats 10 --n_perm 500`
→ `.scratch/analysis/clinicopath_pam50/`
**Inputs:** `.scratch/harmonisation/{tcga,cptac}_brca_harmonised_clinicopath.csv` (built in
[tcga-cptac-clinicopath-harmonisation.md](tcga-cptac-clinicopath-harmonisation.md)),
PAM50 labels from `tools/data/tcga_brca_pam50_labels.csv` — the same labels the CLAM model
was trained on, so these numbers sit next to the WSI numbers on the same target.

---

## Answer in one paragraph

**Yes, but weakly, and not usefully.** Excluding receptor status, the harmonised
clinicopathological block carries statistically real but small PAM50 signal: macro one-vs-rest
AUROC **0.660** in TCGA and **0.579** in CPTAC, against a permutation null of ~0.49
(p = 0.002 and p = 0.034). That is far below the H&E model's **0.890**, it degrades on transfer
to 0.616 [0.532, 0.692], and it adds **nothing** on top of the WSI model — the delta is
negative at every regularisation strength tested. The only strongly predictive fields in these
tables are ER/PR/HER2 IHC (AUROC 0.797 alone), and those are the *clinical surrogate definition*
of PAM50, so using them as features is circular rather than informative.

---

## 1. Univariate association

Chi-square (categorical) or Kruskal-Wallis (age) against 4-class PAM50, complete cases per
variable, Benjamini-Hochberg across the ten variables within each cohort. Effect size is
bias-corrected Cramér's V, or epsilon² for age.

| Variable | TCGA n | TCGA q | TCGA V | CPTAC n | CPTAC q | CPTAC V |
|---|---|---|---|---|---|---|
| **ER (IHC)** | 866 | <1e-4 | **0.832** | 112 | <1e-4 | **0.841** |
| **PR (IHC)** | 863 | <1e-4 | **0.726** | 109 | <1e-4 | **0.743** |
| **HER2** | 785 | <1e-4 | **0.422** | 91 | <1e-4 | **0.538** |
| histology | 909 | <1e-4 | 0.188 | 103 | 0.251 | 0.095 |
| race | 828 | <1e-4 | 0.172 | 110 | 0.032 | 0.210 |
| stage | 890 | <1e-4 | 0.103 | 103 | 0.026 | 0.229 |
| LN+ count | 765 | 0.0004 | 0.111 | 88 | 0.041 | 0.221 |
| pN | 893 | 0.0002 | 0.095 | 101 | 0.099 | 0.152 |
| pT | 908 | 0.0027 | 0.078 | 102 | 0.251 | 0.103 |
| age | 910 | 0.0096 | 0.009 (ε²) | 114 | 0.592 | 0.000 (ε²) |

Two things to read off this:

- **Everything is significant in TCGA and almost nothing is large.** At n = 910 the tests have
  power to resolve trivial effects. Excluding the three IHC rows, the largest effect in the
  table is histology at V = 0.188 — conventionally a weak association. Significance here is a
  statement about sample size, not about usefulness.
- **The non-IHC associations do not replicate.** Histology is the strongest non-receptor
  variable in TCGA (q < 1e-4) and is **not significant in CPTAC** (q = 0.25). pT and pN also
  fail to replicate. The three that do hold up in CPTAC (race, stage, LN+) are the ones the
  harmonisation analysis already flagged as either shifted (stage, by CPTAC's IIA-IIIC
  eligibility) or usable as a covariate only (race).

### Where the signal actually lives

| PAM50 | n (TCGA) | age median | ductal % | **lobular %** | pN=N0 % | ER+ % | PR+ % | HER2+ % |
|---|---|---|---|---|---|---|---|---|
| LumA | 475 | 60 | 63.4 | **26.3** | 48.5 | 97.6 | 90.3 | 13.4 |
| LumB | 195 | 58 | 85.1 | 5.6 | 41.4 | 98.4 | 80.3 | 22.4 |
| Basal | 165 | 54 | 86.6 | **1.2** | 64.2 | 12.0 | 7.1 | 8.3 |
| Her2 | 75 | 56 | 92.0 | 4.0 | 36.6 | 36.2 | 18.3 | 73.4 |

The single real non-receptor effect is **lobular histology marking LumA** (26.3% of LumA vs
1.2% of Basal). That matters for the interpretation in §5: lobular growth pattern is a
morphological feature, visible on the H&E slide, so it is not information the image model
lacks — it is information the image model already reads directly, and more finely than a
four-level categorical.

---

## 2. Multivariable prediction (10-fold CV × 10 repeats, pooled out-of-fold)

Missingness is encoded as an explicit `unknown` level, never dropped — complete-case filtering
would cost 185 TCGA and 30 CPTAC cases on the LN+ column alone. Chance is macro AUROC 0.500 /
balanced accuracy 0.250.

**TCGA (n = 910)**

| Block | Model | Macro AUROC | Balanced acc | LumA | LumB | Basal | Her2 |
|---|---|---|---|---|---|---|---|
| **A** harmonised (age, pN, LN+, histology, race) | logreg | **0.660** ± 0.004 | 0.311 | 0.681 | 0.607 | 0.720 | 0.631 |
| A | hgb | 0.637 ± 0.006 | 0.334 | 0.646 | 0.583 | 0.684 | 0.633 |
| **B** = A + stage + pT | logreg | 0.669 ± 0.004 | 0.346 | 0.702 | 0.612 | 0.722 | 0.639 |
| B | hgb | 0.657 ± 0.004 | 0.350 | 0.687 | 0.613 | 0.704 | 0.624 |
| **C** = A + IHC *(circular)* | logreg | 0.844 ± 0.002 | 0.573 | 0.824 | 0.710 | 0.957 | 0.887 |
| **D** IHC only *(circular)* | logreg | 0.797 ± 0.002 | 0.558 | 0.773 | 0.616 | 0.939 | 0.860 |

**CPTAC (n = 114)**

| Block | Model | Macro AUROC | Balanced acc | LumA | LumB | Basal | Her2 |
|---|---|---|---|---|---|---|---|
| **A** harmonised | logreg | **0.579** ± 0.011 | 0.292 | 0.561 | 0.569 | 0.732 | **0.456** |
| A | hgb | 0.543 ± 0.021 | 0.275 | 0.584 | 0.610 | 0.647 | **0.331** |
| **B** = A + stage + pT | logreg | 0.598 ± 0.012 | 0.299 | 0.615 | 0.552 | 0.775 | 0.452 |
| **C** = A + IHC *(circular)* | hgb | 0.802 ± 0.015 | 0.553 | 0.856 | 0.614 | 0.942 | 0.795 |
| **D** IHC only *(circular)* | hgb | 0.774 ± 0.008 | 0.524 | 0.810 | 0.586 | 0.892 | 0.809 |

Observations:

- **Block A is weak but not empty.** 0.660 in TCGA is well clear of chance; balanced accuracy
  0.311 vs 0.250 says it barely improves the decision, though.
- **Basal is the only class block A gets anywhere on** (0.720 TCGA, 0.732 CPTAC) — driven by
  the near-total absence of lobular histology and by node negativity. LumB is near chance
  everywhere (0.607 / 0.569).
- **Her2 is below chance in CPTAC** (0.456 logreg, 0.331 hgb) on n = 14 positives. Whatever
  block A learned about Her2 in TCGA points the wrong way in CPTAC.
- **Adding stage/pT buys +0.009.** Not worth the transfer cost documented in the harmonisation
  analysis (CPTAC is stage-truncated by enrolment).
- **IHC does all the work in blocks C and D.** Block C beats block D by only +0.047, i.e.
  the entire non-receptor clinicopath block contributes under 0.05 AUROC once receptor status
  is present. And blocks C/D are not legitimate models of PAM50 — they are the IHC surrogate,
  which is what PAM50 is clinically approximated *by*.

### Permutation null (500 label permutations, block A, logreg)

| Cohort | Observed | Null mean ± sd | Null 95th pct | p |
|---|---|---|---|---|
| TCGA | 0.6614 | 0.4922 ± 0.0211 | 0.5252 | **0.0020** |
| CPTAC | 0.5854 | 0.4848 ± 0.0537 | 0.5709 | **0.0339** |

The signal is real in both cohorts. Note the CPTAC null has sd 0.054 and a 95th percentile of
0.571 — at n = 114 an apparent AUROC of 0.57 would be indistinguishable from noise, so the
observed 0.585 clears the bar narrowly and should not be quoted as a solid estimate.

---

## 3. Cross-cohort transfer (fit on all TCGA, predict CPTAC, no re-fitting)

| Block | Model | Macro AUROC [95% CI] | Balanced acc | LumA | LumB | Basal | Her2 |
|---|---|---|---|---|---|---|---|
| A harmonised | logreg | **0.616** [0.532, 0.692] | 0.290 | 0.606 | 0.613 | 0.620 | 0.624 |
| A harmonised | hgb | 0.553 [0.474, 0.630] | 0.293 | 0.585 | 0.534 | 0.511 | 0.581 |
| C = A + IHC *(circular)* | hgb | 0.838 [0.786, 0.885] | 0.561 | 0.862 | 0.714 | 0.928 | 0.847 |
| D IHC only *(circular)* | hgb | 0.823 [0.779, 0.866] | 0.534 | 0.849 | 0.644 | 0.917 | 0.883 |

Block A transports at 0.616 with the CI's lower bound just clear of chance under the linear
model, and **fails to transport at all under gradient boosting** (CI 0.474–0.630 includes 0.5).
The linear model transferring better than the boosted one is the expected signature of a weak
signal: there is not enough structure for the flexible model to capture that survives a cohort
change.

---

## 4. Reference points on the same target

| Model | Internal TCGA macro AUROC | External CPTAC |
|---|---|---|
| CLAM-MB + UNI2-h, H&E only | 0.888 (case-level, 10-fold) | 0.847 (case-level) |
| Clinicopath block A | 0.660 | 0.579 internal-CV / 0.616 transferred |
| IHC surrogate (circular) | 0.797 | 0.823 transferred |

The H&E model is 0.23 AUROC ahead of the entire non-receptor clinicopath block internally, and
that gap widens externally.

---

## 5. Does clinicopath add anything on top of the WSI model?

Stacked multinomial logistic regression on the CLAM-MB out-of-fold log-probabilities, with and
without block A, on the **599 TCGA cases** that appear in at least one CLAM test split.
Regularisation swept because block A costs the stacker ~20 columns on 599 cases, and a negative
delta at a single penalty could be nothing but the parameter count.

| C | WSI only | WSI + clinicopath (A) | clinicopath (A) only | Δ macro AUROC [95% CI] | p |
|---|---|---|---|---|---|
| 0.05 | 0.8916 | 0.8827 | 0.6588 | **−0.0081** [−0.0161, −0.0002] | 0.043 |
| 0.20 | 0.8911 | 0.8806 | 0.6530 | **−0.0104** [−0.0187, −0.0023] | 0.013 |
| 1.00 | 0.8903 | 0.8781 | 0.6500 | **−0.0124** [−0.0214, −0.0038] | 0.004 |

**No incremental value.** The delta is negative at every penalty, and it shrinks monotonically
as regularisation tightens (−0.0124 → −0.0081) — the signature of cost-of-parameters, not of
actively misleading variables. The defensible claim is therefore *"clinicopath adds nothing
over H&E"*, not *"clinicopath harms"*: in the limit of infinite regularisation the delta goes
to exactly zero, which is the honest ceiling.

§1 explains why. The one substantive non-receptor association is lobular histology marking
LumA, and lobular growth pattern is exactly what a patch-level morphology model sees directly.
The clinicopath block's signal is close to a coarse, four-level quantisation of information the
image already carries.

### Caveats on this section

1. **n = 599, not 910.** CLAM's `create_splits_seq.py` draws an independent test fraction per
   fold rather than partitioning, so the 10 test sets overlap (260 of 643 slides recur) and
   their union misses cases that were never sampled into a test set. The subset is random with
   respect to the label, so this costs power, not validity.
2. **Stacking optimism.** The WSI probabilities are out-of-fold for the model that produced
   them, but a stacker trained on other folds still sees probabilities from models that had the
   held-out cases in *their* training data. This inflates both arms identically, so the delta
   is the trustworthy quantity, not the absolute 0.89.

---

## 6. Consequences for the thesis

1. **Do not build a WSI + clinicopath fusion arm for PAM50 and expect a gain.** The measurement
   says the ceiling is zero and the realistic outcome is a small loss. This mirrors the ER
   result, where the clinicopath fusion arm was also null — two independent targets, same
   verdict, which is worth reporting as a finding rather than an absence.
2. **Never put ER/PR/HER2 in the tabular block when the target is PAM50.** They give AUROC
   0.797 on their own and would manufacture a large, meaningless "fusion gain". The
   harmonisation analysis already made this rule; this quantifies exactly how large the
   artefact would be.
3. **Grade is the missing variable, and it is missing symmetrically.** Nottingham grade is the
   one clinicopathological variable with a strong published association to intrinsic subtype,
   and it is absent in both cohorts (§4 of the harmonisation analysis). Everything above is
   therefore a lower bound on what clinicopathology could do in principle — but it is the
   achievable bound on *these* cohorts, which is what the thesis is constrained by.
4. **What this strengthens.** A null incremental result is the cleanest available argument that
   the H&E branch is not a proxy for routinely-recorded clinical variables: the morphology model
   reaches 0.89 where the clinical record reaches 0.66, and the clinical record adds nothing on
   top. That is a positive claim about the image model, extracted from a negative result.
