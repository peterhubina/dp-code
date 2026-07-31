# Prediction-target selection: evidence review and recommendation

**Date:** 2026-07-30
**Question:** which prediction target is most feasible, publishable, and clinically valid for a
WSI-histology branch + a second modality, trained on TCGA-BRCA and externally validated?

**Provenance marking used throughout:**
- `[V]` = verified directly against data files or live APIs in this session
- `[L]` = from the literature review (four subagent research streams)
- `[U]` = reported but NOT independently verified — treat as a lead, not a fact

**Coverage gap:** two of six research streams (multimodal fusion-baseline novelty; journal/venue
reporting requirements) terminated on an API limit before reporting. Sections that would have
depended on them are marked as open questions at the end.

---

## 1. Recommendation

**Primary target: genomic instability and actionable mutation status from H&E — specifically
a continuous chromosomal-instability / HRD score, with PIK3CA and TP53 as a secondary panel.**

**Second modality: RNA-seq (as a feature), with the label coming from DNA.** This is the only
configuration examined where the label is not derivable from the second modality, so the
multimodal ablation asks a real question instead of a tautological one.

**External validation: CPTAC-BRCA.** Verified below to have complete label coverage.

### Why this one

| Criterion | Status |
|---|---|
| Label in TCGA | `[V]` 1,066 sequenced / 1,070 CNA |
| Label in CPTAC | `[V]` **119 of 119 WSI cases** have mutation + CNA + expression |
| Genomic-instability attributes in CPTAC | `[V]` `CHROMOSOME_INSTABILITY_INDEX_CIN_`, `MUTATION_COUNT`, `TMB_NONSYNONYMOUS` |
| Non-circular with the RNA branch | Yes — DNA label, RNA feature |
| Clinically actionable today | Yes — HRD → PARP inhibitors (OlympiA); PIK3CA → alpelisib (SOLAR-1) |
| Crowding | `[L]` HRD moderate (~40 papers); **PIK3CA ~5 papers — genuinely uncrowded** |
| Open problem to attack | `[L]` Marmé et al., *EJC* 2025: HRD model 0.72 hold-out → **0.57 external** |

That last row is the reason this target is publishable rather than merely feasible. The field has
a documented external-generalisation collapse for exactly this task. An honest TCGA→CPTAC
external validation with unimodal baselines is therefore a contribution, not a benchmark run —
and it is robust to the outcome, because a well-diagnosed partial collapse is as informative as
a success.

Reported prior performance `[L]`: point mutations AUC 0.68–0.85 (Qu et al., *npj Precis Oncol*
2021); HRD 0.86 luminal (Lazard, *Cell Rep Med* 2022), 0.81 DeepSMILE, 0.887 SuRe-Transformer
2025, 0.78 breast-trained (Loeffler, *BMC Biol* 2024).

---

## 2. What was ruled out, and on what evidence

### 2.1 Survival / PFS on TCGA → CPTAC — not viable

Two independent failures.

**CPTAC has no outcomes** `[V]`: 1 recurrence event, 2 deaths, median follow-up 377 days
(max 601), 0 cases with both a slide and usable RFS. Independently confirmed `[L]` at source:
GDC reports `vital_status="Not Reported"` for 134/135 breast cases; cBioPortal `brca_cptac_2020`
has 36 clinical attributes, none of them survival.

**TCGA-BRCA histology barely predicts survival at all** `[L]`, from the benchmark papers' own tables:

| Model | TCGA-BRCA c-index | Note |
|---|---|---|
| PORPOISE, histology-only AMIL | **0.560** (0.489–0.615) | **p = 0.270, not significant** |
| PORPOISE, **age + gender Cox** | **0.645** (0.564–0.704) | p = 5.4e-4 — beats every deep model |
| MCAT | 0.580 ± 0.069 | its worst of 5 cancer types |
| SurvPath, ABMIL WSI-only | 0.493 ± 0.126 | below chance |
| CLAM-SB / CLAM-MB (via MOTCat) | 0.573 / 0.578 | our own architecture |

A target where age and sex outperform the image model is not a defensible primary endpoint.

**Correction to an earlier assumption:** TCGA-CDR PFI does *not* count death from any cause.
Liu et al. verbatim: *"In our definition of PFI, the events included deaths with tumor, but do not
include deaths from other causes"* — deaths without tumour are censored. That is already
cause-specific competing-risk handling, so no Fine-Gray adjustment is needed.

Also worth knowing `[L]`: BRCA **failed** the CDR's own supplemental follow-up check for PFI and
was admitted by judgement (median time-to-event 26 months vs median time-to-censor 25 months).
Liu et al. caution explicitly against BRCA OS and DSS.

### 2.2 Derived recurrence labels on TCGA-BRCA — feasible to build, too weak to use

This was requested specifically, so it gets its own analysis. All figures `[V]`, restricted to
the 1,003 cases holding UNI2-h embeddings.

**Label options:**

| Definition | n | events |
|---|---|---|
| CDR PFI | 1,003 | 132 |
| CDR DFI | 876 | 77 |
| Derived: any `new_tumor_event_type` | 1,003 | 93 |
| Derived: **true recurrence** (distant met + locoregional; excludes 19 second primaries) | 1,003 | **74** |

Breakdown of `new_tumor_event_type`: Distant Metastasis 57, New Primary Tumor 19, Locoregional
Disease 9, Locoregional Recurrence 8. `treatment_outcome_first_course` is `[Not Available]` for
all 1,003 cases and is useless.

**The censoring trap.** Median follow-up among event-free cases is **792 days (2.2 y)**, against
a disease that recurs for 15+ years. Requiring controls to be event-free *and* followed at least
the horizon:

| Landmark | usable cases | events | event rate | discarded as censored-too-early |
|---|---|---|---|---|
| 2 y | 525 | 39 | 7.4% | 423 |
| 3 y | 405 | 48 | 11.9% | 552 |
| 5 y | 271 | 66 | 24.4% | 704 |

**`project/CLAM/dataset_csv/tcga_brca_recurrence.csv` must not be used as it stands.** It labels
892 slides `no_recurrence`, but **552 event-free cases have under 3 years of follow-up**. A model
trained on it partly learns who was censored early — which tracks enrolment site and calendar
year, precisely the confound the site-holdout protocol exists to control. Howard et al.
(*Nat Commun* 2021) `[L]` show TCGA submitting site is learnable from H&E and biases survival
prediction specifically.

**Alternative label sources are not better** `[L]`: cBioPortal PanCancer Atlas reproduces CDR
exactly (DFS 84/942, PFS 145/1083). GDC harmonised follow-up is worse (`new_event_type` populated
for 0 cases; 96 with `progression_or_recurrence="Yes"`, only 83 with a date). Only legacy Firehose
`brca_tcga` reports more (DFS 113/1005), via a looser derivation with no stage-IV exclusion and no
landmark — i.e. more events because it is less careful.

**Verdict:** recurrence is derivable but caps at ~74 clean events, or ~48 within a defensible
3-year landmark. Not a primary endpoint. It *is* defensible as a **pre-specified secondary
confirmatory test** of one frozen score in multivariable Cox — direct precedent: Wulczyn et al.
(*PLOS ONE* 2020) did exactly this on TCGA-BRCA, reporting HR 2.86 [1.42, 5.76], p=0.0034, while
conceding their c-index CI [0.555, 0.873] was too wide to conclude from. Conditions: score frozen
on a different target, one score and one test, PFI from TCGA-CDR, exclude the 17 stage-IV cases
(13 of which are events — ~10% of the event mass is de-novo metastatic), 90-day landmark, report
events-per-variable, c-index CI descriptive only.

### 2.3 Molecular risk score (Oncotype DX / MammaPrint / ROR) — closed this cycle

Three 2026-cycle papers occupy it `[L]`:
- **Shamai et al., *Lancet Oncology* 2026** — GigaPath + clinicopathology on TAILORx, N=10,273
  (test 2,407), AUC **0.898** for RS≥26; six external cohorts N=5,497, AUC 0.858–0.903;
  TCGA with RNA-estimated RS 0.832. Also demonstrates chemo-benefit prediction.
- **Boehm et al., *Nature Communications* 2025 ("Orpheus")** — 6,172 patients, RS>25 AUC 0.89;
  continuous r=0.63 (WSI) / 0.68 (+report) against the **true assay**.
- **Kaczmarzyk et al., "MAKO", *npj Digital Medicine* 2026** — ROR-P + attention-MIL, 12 foundation
  models, external TCGA n=613. Two results that kill the plan: externally **no model beat a
  ResNet50 baseline** after FDR correction (*"limited cross-cohort generalizability of continuous
  ROR-P models"*), and **transcriptomic ROR-P itself failed to stratify recurrence in TCGA
  ER+/HER2−** (C = 0.535 binarised, 0.468 continuous, both n.s.).

Compounding problems:
- **Subset arithmetic** `[V]`: CPTAC ER+/HER2− with slides = **63 cases**. MAKO's external arm had
  613 and still failed.
- **genefu is the wrong tool** `[L]`: `oncotypedx()` drops the assay's five reference genes,
  min–max scales each gene across whatever samples are passed in (making the score cohort-relative),
  and hard-codes pre-TAILORx 18/31 cutoffs rather than 25/26. No published genefu-vs-assay
  concordance exists.
- **No precedent for the design** `[L]`: published proxy-label work (Howard *npj Breast Cancer*
  2023; Shamai; Van Alsten *JCO PO* 2024) always anchored the proxy to a true assay or to real
  recurrence events. A proxy-train → proxy-validate design with neither has no precedent.

### 2.4 Other targets screened out `[L]`

| Target | Blocking reason |
|---|---|
| Bulk expression from H&E | SEQUOIA (*Nat Commun* 2024) already did TCGA→CPTAC, r=0.636 on BRCA. And RNA *is* the label — fusion is tautological. |
| Ki-67 / histologic grade | TCGA `GRADE` returns 0 records; GDC `tumor_grade` missing for 1097/1098; no Ki-67 field. CPTAC grade "not reported" ×134. DeepGrade/Stratipath (CE-marked) own the space. |
| HER2-low | CPTAC has only Positive/Negative/equivocal — no external validation possible. Zaakouk (*The Breast* 2023): 16 expert pathologists reached absolute agreement on 6% of cases. |
| Nodal status | Best TCGA label of all (pN, n=1097) but **CPTAC has no pN at all**. Clinicopath-only ensemble already reaches 0.762. |
| Neoadjuvant pCR | TCGA has **13** neoadjuvant patients; CPTAC is treatment-naive by protocol. |
| TILs / immune | Viable but small: CPTAC CD3-IHC gives ~68 usable cases. Keep as a secondary analysis. |

---

## 3. External cohorts — the full picture `[L]`

**There is no open, at-scale public WSI cohort with mature follow-up for *invasive* breast cancer.**
~45 datasets were checked. Three options not previously considered:

| Cohort | Content | Caveat |
|---|---|---|
| **HTAN Duke** (TBCRC 038 + RAHBT) `[U]` | 248 patients with `.svs` + RNA-seq + outcome, **135 recurrence events, 8.0 y median FU**; open access | **DCIS**, not invasive — a disease-biology shift. Numbers are subagent-computed from the HTAN metadata graph; **my own verification attempt failed (API 404). Verify before relying on this.** |
| **METABRIC images** `[U]` | EGA `EGAD00010000270`, 564 Aperio H&E slides, paired with 1,144 OS deaths / 13.1 y | DAC-controlled application required |
| **Nightingale `brca-psj-path`** `[U]` | 4,200 cases / 72,400 WSIs | Gated; ICD-proxy endpoints; **no RNA** |
| **AURORA-Metastatic** (TCIA) `[U]` | 55 patients, 184 H&E, RNA/WGS/WES, PAM50, 46 OS events | Metastatic; tiny |

**Traps** `[L]`: TUPAC16, BCSS, NuCLS and 151/195 TIGER slides are TCGA-derived and are **not**
valid external sets. Beck 2011's NKI+VGH TMA images are no longer retrievable from any host.

For **molecular/phenotype** endpoints the ranking is: BCNB (1,058 cases, real scanner shift),
then CPTAC-BRCA — which remains the right external cohort for our actual task.

For validating a **score → outcome** link without images, METABRIC (1,980 with expression +
survival, 13.1 y) and SCAN-B (7,868, 6.95 y) are open and headless.

---

## 4. The in-house recurrence cohort `[V]`

`.scratch/hsi_bc_recurrence/` — 47 cases, previously set aside as label-poor. It is not.

- **22 relapse events, 24 deaths**
- DFS units are **months**: relapsers recur at median 39 mo (3.3 y); non-relapsers are followed a
  median of **150 mo (12.5 y)**. Mature follow-up — the one thing TCGA (2.2 y) and CPTAC (1.0 y)
  both lack.
- **ER+ 38/47; grade 2 in 25/47; LumB 29 / LumA 10 / Basal 4 / Her2 4.** ER+/HER2− subset = 32
  cases with 14 relapses.
- **All 47 slides already carry UNI2-h features** (1536-dim; verified `100.h5` = 53,119 × 1536),
  same encoder as TCGA and CPTAC.
- Clinicopathology includes **treatment** (hormonal, chemo, trastuzumab, RT) — which TCGA lacks
  entirely — plus LVI, PNI, KI67, LN counts.
- No RNA-seq; 677 hyperspectral ENVI images instead.

At n=47 with a 47% event rate the selection is enrichment-designed, so it validates
**discrimination, not calibration**, with wide CIs. Its role is a sanity check, not a headline.

---

## 5. Modality availability — the binding constraint `[V]`

| Modality | TCGA | CPTAC | HSI-BC |
|---|---|---|---|
| WSI (UNI2-h) | 1,003 | 119 | 47 |
| RNA-seq | 955 | 133 | ✗ |
| Mutation + CNA | 1,066 / 1,070 | **119 / 119** | ✗ |
| Clinicopathology | 1,046 | partial | 47 (+ treatment) |
| Proteome (TMT) | ✗ | 127 (not downloaded) | ✗ |
| **Outcomes** | 132 PFI, 2.2 y FU | **1 event — unusable** | **22 relapses, 12.5 y FU** |

Clinicopathology is the only second modality present in all three cohorts. RNA covers TCGA and
CPTAC only. **DNA labels are the only endpoint family with complete TCGA + CPTAC coverage.**

---

## 6. Design requirements carried forward from our own results

Both completed experiments show the same failure mode, and it must be designed against:

- **PAM50 / CPTAC**: RNA-only 0.988 > fusion 0.981 > WSI-only 0.847. With RNA held at the training
  mean, the fusion model predicts LumA for all 114 cases (balanced accuracy 0.250 = chance) despite
  a near-even gate (0.462 / 0.538). *A balanced gate weight is not evidence of a balanced decision.*
- **ER / TCGA**: WSI 0.896 → +RNA 0.941 (p = 1.6e-5); +clinicopath 0.894 (null). The unimodal
  baselines were subsequently measured in Part C of `docs/er-prediction-results.md`, and they
  close the question: **RNA table alone = 0.9511, which no fusion arm beats** (best is
  co-attention at 0.9502; FiLM 0.9462 vs RNA-only is −0.0049, p = 0.581). Raw **ESR1 expression
  with zero fitted parameters reaches 0.9605**, beating the gated fusion arm (p = 0.017).
  Functional ablation shows both fusion arms are unimodal in opposite directions: deleting the
  image costs the RNA arm 0.006; deleting the table costs the clinicopath arm 0.000.

Non-negotiable for the next experiment:
1. **Every unimodal baseline reported** — WSI-only, RNA-only, clinicopath-only, on the same folds.
2. **Modality-ablation at inference** — report what the model does with each modality zeroed.
3. **Site-holdout retained** (38 TSS sites; Howard 2021).
4. **The clinical baseline is a competitor, not a bonus** — stage/grade/age must be beaten, not added.
5. **Pre-register** the primary endpoint before reading test folds.

---

## 7. Proposed study design

**Title shape:** externally validated multimodal prediction of genomic instability and actionable
mutation status from breast H&E, with unimodal baselines.

- **Train:** TCGA-BRCA, site-holdout folds, same splits as the ER ablation.
- **Primary endpoint:** CPTAC external AUROC for HRD/CIN-high, pre-registered, against the
  WSI-only baseline.
- **Arms:** WSI-only; RNA-only; clinicopath-only; WSI+RNA fusion; WSI+clinicopath.
- **Secondary panel:** PIK3CA (actionable, uncrowded), TP53 (positive control — the easiest and
  best-benchmarked mutation, so it calibrates the pipeline against published numbers).
- **Deployment-realistic arm:** RNA as training-time supervision only, H&E-only at inference.
  This is the clinically meaningful configuration since RNA is unavailable in routine practice.
- **Secondary confirmatory:** does the H&E-predicted instability score stratify TCGA PFI in
  multivariable Cox adjusted for stage, grade, age, subtype? One score, one test, stage IV
  excluded. With 132 events, SE(log HR) ≈ 0.087 — powered for HR ≈ 1.25 per SD. This is the only
  outcome analysis the data supports.
- **Optional mechanistic section:** CPTAC TMT proteome (PDC000120, 127 cases) as an orthogonal
  readout — does predicted instability track proteomic proliferation modules? No proteomic
  training required.
- **Optional sanity check:** HSI-BC (47 cases, 22 relapses) — the only cohort where an H&E-only
  model can be tested against real recurrence, and one no RNA-requiring model can run on at all.

---

## 8. Open questions (research streams that did not complete)

1. **Is the "missing unimodal-omics baseline" critique already published?** Our two results are
   strong evidence for it, but novelty was not established. Before framing any paper around it,
   check whether PORPOISE / MCAT / SurvPath / MOTCat / CMTA / HEALNet report omics-only baselines,
   and whether a benchmark or reproducibility paper has already made this argument.
2. **What do target venues require in 2026?** TRIPOD+AI / CLAIM adoption, minimum event counts for
   external validation (Riley et al.), and whether negative/benchmark papers are placeable.
   Jennings et al. (*Cancer Informatics* 2026) `[L]` reviewed 48 studies and found *"all studies
   were assessed as having high or unclear risk of bias — most often due to limited external
   validation"* — which suggests the external-validation bar is the discriminator, but the venue
   specifics are unconfirmed.
3. **HTAN Duke numbers require verification** before any plan depends on them.

---

## 9. Immediate next steps

1. Download CPTAC mutation + CNA from cBioPortal `brca_cptac_2020` and join to the 119 WSI cases.
2. Fix the HRD label definition on the TCGA side — decide between the Knijnenburg et al. 2018
   HRD score, `FRACTION_GENOME_ALTERED`, `ANEUPLOIDY_SCORE`, and CPTAC's `CHROMOSOME_INSTABILITY_INDEX_CIN_`,
   and confirm the two cohorts' scores are on a comparable scale before anything else. The RNA
   scale-mismatch lesson applies here too.
3. Rebuild `tcga_brca_recurrence.csv` with a 3-year landmark, or retire it.
4. Retrieve TCGA HER2 status from cBioPortal (absent from `tools/data/tcga_brca_clinicopath.csv`)
   if any ER+/HER2−-restricted analysis is kept.
