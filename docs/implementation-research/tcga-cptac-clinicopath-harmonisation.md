# TCGA-BRCA ↔ CPTAC-BRCA clinicopathological harmonisation

Scope: which clinicopathological variables exist in **both** cohorts at usable coverage, so the
same tabular block can be trained on TCGA and evaluated on CPTAC without re-fitting.

Analysis date 2026-07-30. Cohorts restricted to cases that actually have UNI2-h features on disk:
**TCGA n = 1055 cases** (2252 slide feature files), **CPTAC n = 114 cases** (378 slides).

Reproducible tables: `.scratch/harmonisation/{tcga,cptac}_brca_harmonised_clinicopath.csv`.

## 1. Sources (they are not the same shape)

| Cohort | Source used | What it gives |
|---|---|---|
| TCGA | Xena `TCGA_BRCA_clinicalMatrix.tsv` (194 cols, GDC biospecimen clinical) | ER/PR/HER2 IHC + FISH, pT/pN/pM, stage, histology, menopause, margin, LN counts |
| TCGA | `TCGA-CDR-SupplementalTableS1.xlsx` | race, ethnicity, OS/DSS/DFI/PFI endpoints |
| TCGA | cBioPortal `brca_tcga` (138 attributes) | same fields via API, incl. `ER_STATUS_BY_IHC` |
| CPTAC | `clinical/cptac_pancancer_clinical_breast.csv` (134 cases, 125 cols — **75 of them entirely empty**) | age, race, ethnicity, pT/pN/pM, histology, LN counts, margin, coarse stage |
| CPTAC | `clinical/cbioportal_labels.csv` from `brca_cptac_2020` (36 attributes only) | ER/PR/HER2 clinical status, PAM50, substaged `TUMOR_STAGE`, CD3 TILs |

**cBioPortal alone is not enough for CPTAC.** `brca_cptac_2020` exposes no pT/pN/pM, no histology,
no grade, no menopause, no margin, no LN counts — only age, sex, ethnicity, receptor status, PAM50,
stage, and molecular/immune scores. The T/N/M and histology fields must come from the CPTAC
pan-cancer clinical table. Also note `brca_cptac_2020` values are case-inconsistent
(`negative` vs `Negative`) and must be normalised.

## 2. The harmonisable set

| Variable | TCGA coverage | CPTAC coverage | Verdict | Distribution shift |
|---|---|---|---|---|
| **age** | 1055/1055 | 114/114 | **use** | KS = 0.074, p = 0.60 — no shift |
| **pN** (N0–N3, NX dropped) | 1036/1055 | 101/114 | **use** | χ² = 5.95, p = 0.11 |
| **LN+ count** (0 / 1–3 / 4+) | 891/1055 | 88/114 | **use** | χ² = 2.02, p = 0.37 |
| **histology** (ductal/lobular/mixed/other) | 1054/1055 | 103/114 | **use, with caution** | χ² = 7.80, p = 0.050; CPTAC 84% ductal vs TCGA 72% |
| **race** (White/Black/Asian/Other) | 964/1055 | 110/114 | use as covariate only | χ² = 12.25, p = 0.0066; CPTAC 15% Asian vs TCGA 6% (international CPTAC sites) |
| **stage** (I–IV) | 1034/1055 | 101/114 | **shifted — see §3** | χ² = 15.19, p = 0.0017 |
| **pT** (T1–T4) | 1053/1055 | 102/114 | **shifted — see §3** | χ² = 22.72, p = 4.6e-05; CPTAC 79% T2 |
| ER (IHC) | 1004/1055 | 112/114 | **use** — label or covariate | χ² = 2.96, p = 0.086 |
| PR (IHC) | 1001/1055 | 109/114 | **use** — label or covariate | χ² = 1.13, p = 0.29 |
| HER2 (ISH-over-IHC resolved) | 916/1055 | 98/114 | **use** — label or covariate | χ² = 2.47, p = 0.12 |
| PAM50 | 826/1055 (`PAM50Call_RNAseq`) | 114/114 | **use** — label | χ² = 11.37, p = 0.023 (TCGA has Normal-like, CPTAC 114-case subset does not) |
| histologic grade | **0** | **0** | **unusable** | see §4 |
| menopause status | 975/1055 | **0** | unusable | not collected by CPTAC |
| margin status | 988/1055 | 18/114 usable (76/114 are "Not Applicable") | unusable | |
| tumour size (cm) | not in TCGA clinical | column present but **100% empty** | unusable | |
| pM | 1055/1055 (but 14% MX) | 103/114 (33% MX) | uninformative | CPTAC has 1 M1 case |
| OS / recurrence | OS 1096 cases, median follow-up years | OS 95 cases, **max 601 days, 2 deaths**; recurrence **all 0** | unusable for CPTAC validation | |
| sex | 1043 F / 12 M | 114 F | constant in CPTAC | drop |
| CD3 TILs | — | 114/114 | CPTAC-only | not harmonisable |
| ERBB2 proteogenomic status | — | 114/114 | CPTAC-only | not harmonisable |

### Complete-case counts for candidate covariate blocks

| Block | TCGA | CPTAC |
|---|---|---|
| `age, stage, histology` | 1033/1055 | 103/114 |
| `+ pT, pN` | 1022/1055 | 100/114 |
| `+ race` | 935/1055 | 96/114 |
| `+ LN+ count` | 870/1055 | 84/114 |

## 3. The stage/pT shift is structural, not noise

CPTAC's prospective breast cohort enrolled **"newly diagnosed, untreated patients undergoing
definitive surgery for breast cancer (stage IIA–IIIC)"** (Krug et al. 2020, *Cell* 183:1436,
PMID 33212010). Stage was an eligibility criterion. Our data matches: CPTAC is 64% stage II,
32% stage III, 4% stage I, **0% stage IV**; TCGA spans 17% I / 58% II / 23% III / 2% IV.
pT follows: 79% T2 in CPTAC vs 58% in TCGA.

Consequence: stage and pT carry far less variance in CPTAC than in TCGA, and a fusion head that
leans on them will look better internally than externally. This is a design-shift, so
missing-value imputation or re-fitting on CPTAC will not fix it. Either drop stage/pT from the
tabular block, or report the external metric with and without them.

## 4. Histologic grade is unavailable in *both* cohorts

- **TCGA**: `histological_grade` in TCGA-CDR is `[Not Available]` for all 1097 BRCA cases; the
  Xena matrix has no grade column at all. Published TCGA-BRCA grade labels come from
  Asaoka et al. 2020 re-reviewing the original pathology reports, yielding grade for only
  **521 of 1046** slides (496 patients) — see Predicting Nottingham grade with a foundation model,
  *Breast Cancer Research* 2025, PMC12008962.
- **CPTAC**: `cptac_path/histologic_grade` is `"GX — grading is not applicable, cannot be assessed
  or not specified"` for **all 114** cases. Krug et al. 2020 does not report grade.

So grade is symmetric-absent. Convenient — it means excluding grade costs nothing in comparability,
but it also means we cannot use the single strongest morphological clinicopath predictor.

## 5. Literature cross-check

- **Fernandez-Romero et al. 2026**, *Med Biol Eng Comput*, "Domain generalisation challenges in
  breast cancer molecular classification using foundation models" (PMID 42113320) — the closest
  published analogue to our setup: 13 foundation models × 3 MIL heads, trained on
  **TCGA-BRCA n = 1079**, externally validated on **CPTAC-BRCA n = 120 patients / 387 flash-frozen
  slides**. Our cohort is 119 cases / 391 slides, so we are on the same slide pool. Their
  harmonised variable set is exactly **PAM50 + ER + PR + HER2 and nothing else** — no clinicopath
  covariates. They report severe external degradation concentrated in HER2-enriched and
  Normal-like PAM50 and HER2-positive IHC, and attribute 80% of relative performance drop to
  staining variability + feature-space divergence, with **prevalence shift non-significant**.
- **Yuan et al. 2025**, *Signal Transduct Target Ther*, PROGPATH (PMID 40897689) — pan-cancer
  WSI + clinical fusion, 7999 WSIs training, 17 external cohorts including CPTAC. Across 15 cancer
  types the only clinical variables they could harmonise were **age, sex, and tumour stage**. They
  explicitly frame this as avoiding "molecular data not routinely available".
- **Krug et al. 2020**, *Cell* (PMID 33212010) — CPTAC prospective breast: 134 enrolled, 122 fully
  analysed; ER/PR/HER2 by IHC/FISH per ASCO-CAP; PAM50 reported (14 HER2-E, 29 Basal, 57 LumA,
  17 LumB, 5 Normal-like); stage IIA–IIIC eligibility; one-year follow-up forms only, no
  recurrence/survival endpoint presented.

The literature agrees with the disk: the harmonisable clinicopath surface between these two
cohorts is thin, and everyone who has done TCGA→CPTAC breast either uses receptor status/PAM50 as
*labels* only, or restricts fusion covariates to age + stage.

## 6. Recommendation

**Tabular block for fusion (train TCGA, evaluate CPTAC without re-fitting):**

```
age            z-scored on TCGA statistics, applied to CPTAC
pN             one-hot N0/N1/N2/N3    (+ unknown indicator)
LN+ count      one-hot 0 / 1-3 / 4+   (+ unknown indicator)
histology      one-hot ductal/lobular/mixed/other
race           one-hot White/Black/Asian/Other  (+ unknown indicator)  [optional, fairness covariate]
```

Five variables, no distribution shift at p < 0.05 except histology (borderline) and race.
Complete-case: TCGA 870/1055, CPTAC 84/114 — use unknown-indicator encoding rather than dropping,
which keeps 1055 / 114.

**Excluded and why:** stage and pT (CPTAC eligibility-truncated), grade (absent both),
menopause and margin (absent CPTAC), tumour size and pM (empty/uninformative), sex (constant),
OS and recurrence (CPTAC follow-up ≤ 601 days, 2 deaths, 0 recurrences).

**Do not put ER/PR/HER2 in the tabular block when the target is PAM50** — receptor status is the
clinical surrogate of the subtype and leaks. Keep them as prediction targets in their own right
(all three are harmonised at ≥ 86% coverage in both cohorts), which is also what makes the
existing ER result directly externally validatable.

**Encoding notes for the CPTAC side:**
1. Normalise case (`negative` → `Negative`) — `brca_cptac_2020` is inconsistent.
2. Fix the `Inflitrating` typo in `baseline/histologic_type`.
3. Treat `NX`/`TX`/`MX`/`GX`/`Not Applicable`/`Not Reported/ Unknown`/`Staging is not applicable
   or unknown` as missing, not as levels.
4. Collapse pT/pN to the top-level digit (T1c → T1, N1a → N1); the substage vocabularies differ
   between cohorts (`N0 (i-)` vs `N0(i-)`).
5. Resolve TCGA HER2 as ISH-over-IHC to match CPTAC's `ERBB2_UPDATED_CLINICAL_STATUS`; raw TCGA
   IHC alone leaves 20% equivocal, which has no CPTAC counterpart.
