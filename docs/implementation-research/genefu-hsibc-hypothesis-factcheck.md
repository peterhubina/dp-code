# Fact-check: train on genefu-derived labels (TCGA-BRCA) → validate on HistologyHSI-BC-Recurrence

**Date:** 2026-07-30
**Hypothesis under test:** derive `genefu` signature labels (Oncotype DX / MammaPrint / ROR-P / GGI)
from TCGA-BRCA bulk RNA-seq, train a UNI2-h + CLAM model on H&E to predict them, then externally
validate on the HistologyHSI-BC-Recurrence cohort.

**Method:** 6 parallel literature/source agents → 25 adversarial fact-check agents (one per
load-bearing claim) → synthesis; plus direct computation on the raw data in this repo.

**Provenance marking** (same convention as `prediction-target-recommendation.md`):
- `[V]` = verified by me directly against the data files / slide headers in this session
- `[L]` = from literature, adversarially fact-checked against the primary source
- `[U]` = reported but not independently verified — a lead, not a fact

---

## 1. Verdict

**Not viable as stated.** The design cannot support the conclusion you would want ("our H&E-derived
genefu surrogate externally validates against real recurrence") — and, worse, it cannot support the
opposite conclusion either. A null result would be uninterpretable.

Three independent reasons, each sufficient on its own:

1. **Pre-empted.** Howard et al. (*npj Breast Cancer* 2023;9:25) already trained deep learning on
   TCGA-BRCA n=1,039 using formula-recomputed Oncotype DX **and** MammaPrint labels from RNA-seq,
   and externally validated on 427 UChicago patients against both the true assay and real
   recurrence-free interval. That is this study, three years old, with a 9× larger external
   cohort. `[L]`
2. **The training label is not prognostic in the training cohort.** Recomputed RS in TCGA gives
   HR 1.7, p=0.160 for DFI (Shamai et al., *Lancet Oncol* 2026); transcriptomic ROR-P in TCGA
   ER+/HER2− (n=613, 47 events) gives C=0.535, p=0.107 (Kaczmarzyk et al. "MAKO",
   *npj Digit Med* 2026;9:149). You would be asking an image model to regress a quantity that
   demonstrably fails to order outcomes in the cohort it is computed from. `[L]`
3. **The external cohort cannot arbitrate.** With 22 relapses / 25 controls, an observed AUROC
   must exceed **0.667** to reach p<0.05 at all, and 80% power needs a *true* AUROC ≥ **0.730**.
   Cox needs **HR > 2.31** for significance, ≈3.7–3.9 for 80% power. Published external effect
   sizes in this exact literature are HR 1.55, 1.91, 2.88, 3.14 — **three of those four
   best-in-class results would come out non-significant here.** `[V]` (my computation) `[L]`
   (the comparison HRs)

There is one unclaimed element — no ML baseline of any kind exists on HistologyHSI-BC — but
novelty of a 47-case test set is not novelty of a method, and it is a case-control set at that.

**Note on how this differs from the July 12 analysis.** `oncotypedx-tcga-proxy-publishability-findings.md`
assessed the same genefu-training idea with **BCR-Net** as the external cohort and found one genuinely
strong angle: *silver-label training → gold-label validation*, because BCR-Net carries the **true**
Oncotype DX assay. Substituting HSI-BC deletes exactly that angle. HSI-BC has no RNA and no assay, so
the proxy-fidelity check that made the earlier framing defensible is impossible by construction.

---

## 2. Blocking problems, ranked

### F1 — Pipeline defect: 2.055× magnification mismatch `[V]`

**Every number currently in `.scratch/hsi_bc_recurrence/` is uninterpretable.** Verified from the
files:

| | source | patch geometry | field of view per token |
|---|---|---|---|
| TCGA features | Trident, `savetodir=.../trident/20x_256px_0px_overlap` | `level0_magnification=40`, `target_magnification=20`, `patch_size=256`, `patch_size_level0=512` | **128.0 µm** |
| HSI-BC features | `tools/hsi_bc/run_pipeline.sh:41-49` → CLAM `create_patches_fp.py` | `patch_level=0`, `patch_size=256`, slide `mpp-x=0.24328` | **62.3 µm** |

That is **2.055× linear, 4.22× in tissue area**. Nothing compensates: `--target_patch_size 224` is a
plain resize, and `--preset tcga.csv` only sets segmentation thresholds. The header comment at
`tools/hsi_bc/run_pipeline.sh:5` claiming the pipeline "match[es] the TCGA feature extraction
protocol" is false.

Corroboration `[V]`: HSI-BC median **48,398** patches/slide (min 6,775, max 126,340) vs TCGA median
**11,731** (min 1,223, max 34,469) on comparably sized resection sections.

**HSI-BC is the only cohort affected.** `[V]` The CPTAC features (1,306 h5) were tiled
`patch_size=512, step_size=512, patch_level=0, custom_downsample=2.0` — 512 level-0 px on 40×
slides → 128 µm, downsampled to 256 px, i.e. the TCGA recipe exactly. So the existing note that
CPTAC geometry matches TCGA holds; only the HSI-BC script deviates.

**Fix:** `--patch_level 1` on these slides gives 256 px @ 0.4866 µm/px = **124.6 µm**, within 3% of
TCGA's 128 µm. (`--patch_size 512 --step_size 512` at level 0 is equivalent.) Re-extract UNI2-h
afterwards.

Related hygiene `[V]`: `.datasets/tcga-brca/h5_files` is a **dangling symlink** → `/workspace/dp-code/.datasets/embeddings`,
which does not exist. The real 1,126 files are at `.datasets/tcga-brca/embeddings/`.

### F2 — The clinicopathological baseline is at ceiling, and it is not a proliferation baseline `[V]`

Computed from the raw xlsx (bootstrap 95% CI, 4,000 resamples):

| predictor | AUROC vs relapse | 95% CI |
|---|---|---|
| **Tumour diameter** | **0.839** | 0.714–0.937 |
| **N stage** | **0.818** | 0.714–0.920 |
| Age | 0.704 | 0.540–0.857 |
| LVI | 0.678 | 0.540–0.807 |
| **Histologic grade** | **0.615** | 0.471–0.762 |
| Ki67 (binary) | 0.609 | 0.486–0.734 |
| ER-negative | 0.491 | 0.381–0.607 |

Grade and Ki67 — the proliferation axis that ODX / GGI / ROR-P / MammaPrint principally encode — are
the **weakest** predictors here, with CIs crossing chance. The axes that do carry the outcome (nodal
burden, tumour size) are not readable from a single tumour section. Your surrogate targets the one
thing this cohort's outcome does not track.

### F3 — Outcome-dependent case selection, with complete separation `[V]`

```
              relapse=0   relapse=1
node-negative     25          8
node-positive      0         14
```

**Every node-positive case relapsed; every non-relapser is node-negative.** Tumour diameter:
mean 15.8 mm (non-relapse) vs 30.3 mm (relapse). The descriptor declares the design — "a
retrospective case-control study", controls unmatched, breast-only recurrences excluded — so
controls are extreme long-term survivors (median DFS 150 months) against cases recurring at
median 39 months. `[L]` Lijmer et al. (*JAMA* 1999;282:1061-1066, n=184 studies) measured the
inflation from exactly this design at a relative diagnostic odds ratio of **3.0 (2.0–4.5)**.

Consequence: an H&E model that has learned nothing but "large, high-grade tumour" will score well
here for reasons unrelated to your surrogate. And calibration, PPV/NPV and absolute risk are
formally non-estimable — only discrimination and relative associations may be reported.

### F4 — Statistical power `[V]`

Hanley-McNeil variance, 22 positives / 25 negatives:

| true AUROC | 95% CI | power vs 0.5 |
|---|---|---|
| 0.60 | 0.436–0.764 | 21% |
| 0.65 | 0.491–0.809 | 42% |
| 0.70 | 0.548–0.852 | 66% |
| 0.80 | 0.670–0.930 | — |

- Observed AUROC must exceed **0.667** for p<0.05; **0.720** for p<0.01.
- Minimum *true* AUROC for 80% power: **0.730**.
- Cox (d=22): SE(log HR)=0.4264 → significance needs **HR>2.31**; 80% power ≈ HR 3.7–3.9.
- EPV budget (Peduzzi 1995): **2.2 covariates**. "Independent prognostic value after adjustment
  for standard clinicopathology" is *not estimable* here and must be conceded, not run.
- Strata: ER+/HER2− n=32/14 events (needs HR 2.85); ER+/HER2−/N0 n=23/**5 events** (needs HR 5.77,
  AUROC 0.792) — the last is not analysable.
- Multiplicity: 5 signatures × 3 endpoints × 3 strata ⇒ FWER ≈ 0.90 under the global null.

`[L]` Accepted minimum for external validation of a prognostic model is **100 events, ideally 200+**
(Collins, Ogundimu & Altman, *Stat Med* 2016;35:214-226). You have 22.

### F5 — 15/47 patients get a formally meaningless prediction `[L]`

ASCO 2022 (Andre et al., *JCO* 2022;40(16):1816-1837) states these assays should **not** be used in
HER2-positive or triple-negative disease. The cohort has 9 ER− and 10 HER2+ (15/47 outside
ER+/HER2−) `[V]`. And the scores do not degrade gracefully outside the stratum — they saturate,
because the ER module enters the Paik formula with a negative coefficient.

### F6 — A null result is non-identifiable `[V]`

Three failure modes are stacked in series: genefu score ≠ true assay; H&E model ≠ genefu score;
genefu construct ≠ relapse outside its validated stratum. HistologyHSI-BC has **no gene expression
of any kind** (verified column-by-column: 47 rows × 37 columns, IHC only), so a null cannot be
attributed to any one of them. This is the deepest weakness — it makes the experiment unfalsifiable
in the diagnostic sense.

### F7 — Treatment confounding with no available remedy `[V]` `[L]`

41/47 radiotherapy, 38/47 endocrine, 22/47 chemotherapy, 5/47 trastuzumab. Near-universal RT and
endocrine therapy means **non-positivity**, so inverse-probability weighting is formally
unavailable. Howard 2023 found its proxy-trained model prognostic *only* in the 322 chemo-untreated
patients, reporting no association with RFI among chemotherapy recipients. Your analogous
chemo-naive stratum is ~25 patients.

### F8 — Site-signature inflation in the internal numbers `[L]`

Howard et al. (*Nat Commun* 2021;12:4423) showed deep models identify TCGA submitting site at OVR
AUROC 0.964–0.998, surviving colour normalisation, and that preserved-site CV drops measured AUROC
by **0.069** on average. If the TCGA CV is not site-stratified, the internal surrogate AUROCs are
inflated by roughly that before the external cohort is even reached.

---

## 3. Two things I measured that decide it

### 3a. The genefu label is largely redundant with the PAM50 model you already have `[V]`

Using the existing `pam50_final_s1` **held-out CV** predictions on TCGA (n=643 slides) and an
ODX-like score recomputed from the Xena matrix (Paik 2004 formula, reference-gene normalisation,
per-cohort 2.5–97.5 percentile rescaling — the same structural steps genefu documents):

| | Spearman r | AUROC for RS ≥ 26 |
|---|---|---|
| all TCGA | **0.648** | **0.827** |
| ER+/HER2− only (n=325) | 0.587 | 0.783 |

A model that was **never trained on any recurrence score** — it was trained on PAM50 — already sits
at published-SOTA level for H&E→ODX. `[L]` Orpheus WSI-only reports r=0.60 and external AUROC
0.80–0.85 from 5,145 training cases with the *true* assay; Howard's pathology-only external AUROC
is 0.798. Training a fresh model on a genefu label would buy essentially nothing over the checkpoint
already on disk.

The mechanism is visible in the score itself `[V]`: the ODX-like RS correlates with the 5-gene
proliferation module at **Spearman 0.885**, and PAM50 class alone explains **η²=0.377** of its
variance. It is mostly "ER-negativity + proliferation", which is what PAM50 already encodes.

*Caveat:* the TCGA CV numbers are likely site-inflated (F8), and the ODX-like score is my
reimplementation, not genefu itself (see §3b).

### 3b. The per-cohort rescaling produces a miscalibrated label `[V]`

Same recomputation, applied to all of TCGA-BRCA:

- **67.6%** of the cohort scores RS ≥ 26 ("high risk"), including **48.5%** of PAM50-LumA cases and
  **61.8%** of the ER+/HER2− stratum.
- Median RS saturates at 100 for Basal, LumB and Her2.
- `[L]` In TAILORx only ~17% of the screened ER+/HER2−/N0 population had RS ≥ 26.

So the derived score is a **cohort-relative rank**, not a clinically calibrated quantity. Binarising
it at the clinical cutpoints is not meaningful; binarising by quantile makes the label definitionally
cohort-dependent — which is precisely MAKO's documented cross-cohort failure mode.

*This is my reimplementation, not a genefu run* — R is not installed in this environment
(`Rscript: command not found` `[V]`), so genefu's exact behaviour was checked against its
documentation and the literature rather than executed.

**This over-calling is a documented, citable phenomenon — not an artefact of my code.** `[L]`

- **Bartlett JMS et al., OPTIMA TMG. *PLoS One* 2020;15(9):e0238593** (PMID 32881987) took 274
  OPTIMA tumours with **true** Oncotype DX, Prosigna and MammaPrint results, re-profiled on
  NanoString, and deliberately tested the genefu-style approach — published algorithms after a
  single global normalisation, chosen because "it recapitulates the approaches taken by previous
  authors". The MammaPrint arm used **`genefu::gene70` v1.14.0 by name**. Result:
  **RS_like = 1.26 + 1.95 × RS_true** — a ~2× slope. R=0.837 but ternary concordance only **42.3%**,
  binary (TAILORx cut-points) **54.7%**, and **56.9% of cases were pushed into a higher risk group
  while only 2 were called lower**. Training against paired true results lifted ternary concordance
  to 75.2% — still 19.0% upgraded. Notably the MammaPrint-like/genefu arm was *well-behaved*
  (83.2% concordance, balanced); the pathology is specific to the RS algorithm's fixed offsets.
- **Li H, …, Perou CM, Giger ML. *Radiology* 2016;281(2):382-391** (PMID 27144536) computed
  research-version MammaPrint / Oncotype / PAM50 on **TCGA mRNA-seq, n=1,030** and hit exactly this:
  "the categorization of these values … yielded conflicting outputs from one of the three multigene
  tests (ie, Oncotype DX), most likely due to the application of the published assay thresholds …
  on the messenger RNA sequencing expression data. Therefore, for this single assay, we simply put
  the patients into rank expression order and created tertiles." They also had to re-median-centre
  TCGA against a balanced 157 ER+ / 157 other subsample before PAM50, because TCGA is ~80% ER+.
  **The Perou lab hit this failure on this exact cohort and abandoned the clinical thresholds.**
- Convergent evidence that **cohort-relative rescaling, not the reference genes, is the dominant
  fault**: Fan et al. (*NEJM* 2006) *did* use the five reference genes and still scaled per cohort,
  reporting 65.1% high-RS in 295 patients / 53.3% in the ER+ 225. Against TAILORx's 14% and
  TransATAC's 10.3% with the true RT-PCR assay.

Practical consequence: if a genefu score is used at all, use it as a **cohort-relative proliferation
rank evaluated by rank metrics** (Li 2016's tertiles), never at the clinical 11/25 cut-points.

**Related: the signatures do not even agree with each other.** `[L]` Buus R et al.
(*J Clin Oncol* 2021;39(2):126-135, PMID 33108242), n=785 TransATAC with all four **commercial**
assays: Spearman ρ 0.63–0.74 among them **except RS vs ROR ρ=0.32 and RS vs BCI ρ=0.35**; the RS
proliferation module alone explained **72.5%** of ROR variance while its oestrogen module explained
0.6%. OPTIMA pairwise kappas: Oncotype vs MammaPrint κ=0.40, vs Prosigna κ=0.44 (range 0.33–0.60).
Buus's own caveat, verbatim: "if working with expression data obtained on different platforms
(e.g. RT-PCR, microarray), the normalisation and adjustment factors described here are **not
applicable**." Note `[U]`: Sestak 2018 (*JAMA Oncol*) contains no inter-signature agreement
statistics — a full-text grep returns zero hits; cite Buus 2021 instead.

### 3c. What the current (magnification-broken) external transfer already shows `[V]`

For completeness, the existing TCGA→HSI-BC PAM50 transfer in
`.scratch/hsi_bc_recurrence/results/predictions/ensemble_predictions.csv`:

- PAM50 4-class accuracy against the cohort's IHC-surrogate subtype: **40.4%** (LumB is
  systematically called LumA or Her2).
- `1 − p(LumA)` vs real relapse: AUROC **0.729** (95% CI 0.576–0.863) — above the 0.667 significance
  floor, univariable Cox HR 35.4 (3.1–411), p=0.004.
- **But** it does not add over the clinical variables: logistic LR-test for `risk` on top of
  {diameter, N} gives **p=0.268**; AUROC 0.965 → 0.969.

Read this as a null with a wide CI, not as encouragement — and remember it was produced at the
wrong magnification (F1), so it will change when the pipeline is fixed.

---

## 4. What is genuinely sound — do not throw this away

1. **The design pattern is legitimate and publishable.** `[L]` imCMS is the template:
   Sirinukunwattana et al. (*Gut* 2021;70:544-554) trained on transcriptomic CMS in FOCUS (n=278),
   tested on TCGA (n=430, AUC 0.84) and GRAMPIAN (n=144, AUC 0.85), with imCMS1 associated with
   worse OS in TCGA (HR 1.88, p=0.027) where transcriptomic CMS1 was not. The sequel
   (*npj Precis Oncol* 2024;8:89) validated image calls against pathological complete response in
   169 patients.
2. **The cohort provenance is impeccable and citable.** `[L]` TCIA collection
   HistologyHSI-BC-Recurrence, DOI 10.7937/6KPY-YT49, CC BY 4.0, 47 subjects, ~1.2 TB;
   peer-reviewed descriptor Quintana-Quintana et al., *Sci Data* 2025;12:1886
   (10.1038/s41597-025-06157-4).
3. **No ML baseline exists on it.** `[L]` The descriptor's Technical Validation is clinical
   statistics, pathologist annotation review and instrument characterisation — no AUC, accuracy or
   c-index anywhere. You would be first on the WSI side (but see §6 item 1).
4. **Two hard domain-shift axes are favourable.** `[V]` All 1,126 TCGA slides in
   `.datasets/tcga-brca/embeddings/` are **DX** (diagnostic FFPE) — zero frozen BS/TS slides; exactly
   one is a `-06Z` metastatic sample (`TCGA-E2-A15E-06Z-00-DX1`). The external cohort is FFPE H&E
   resection material too. `[L]` Neoadjuvant cases were excluded, so the H&E reflects untreated
   primary biology, matching TCGA. Residual shift is scanner (3DHISTECH Pannoramic vs Aperio),
   one stain lab, and block age — not fixation, not tissue type.
5. **The endpoint quality is excellent.** `[V]` Mature 12.5-year follow-up, time-to-event on every
   case, one WSI per patient. Median DFS 150 vs 39 months; median OS 150 vs 66.5 months. This is the
   one thing TCGA (median PFI follow-up 2.1 y, only 21.3% of censored cases reaching 5 y `[V]`) and
   CPTAC both lack.
6. **Assay-guided chemotherapy almost certainly did not operate** (Spain, 2006–2015, pre-reimbursement),
   removing the sharpest form of confounding by indication. State this explicitly; it is a point in
   the design's favour.

---

## 5. Recommendation

**Do not run the hypothesis as stated.** In order of preference:

### R1 (recommended) — HRD / genomic-instability from H&E, external on CPTAC

Already identified in `prediction-target-recommendation.md` as the only target with 119/119 CPTAC
label coverage. Labels exist natively in *both* cohorts, so there is no surrogate chain, no
construct-validity problem and no non-identifiable null; the target is directly actionable (PARPi);
and your TCGA→CPTAC pipeline already works at case-level macro AUROC 0.847. Keep HSI-BC as a small,
honestly-labelled secondary probe.

### R2 — Turn the magnification defect into the contribution

F1 is an accidental clean natural experiment: identical encoder, identical MIL head, identical task,
one cohort at 128 µm FOV and one at 62.3 µm. Quantify how much apparent "external generalisation
failure" in foundation-model pathology is silent MPP/FOV mismatch, by re-running TCGA→CPTAC and
TCGA→HSI-BC at matched and mismatched scale. Costs almost nothing (features and pipelines exist),
needs no outcome labels, so none of §2's power constraints apply.

### R3 — WSI + hyperspectral fusion on HistologyHSI-BC

677 HSI cubes (826 bands, 400–1000 nm, ENVI, 10×) `[V]` sit unused, nested inside pathologist-annotated
ROIs on the *same physical slides* as the WSIs, with GeoJSON polygons giving approximate spatial
registration `[L]`. Genuinely unclaimed modality combination. Same 22-event ceiling, so the claim
must be methodological ("first WSI+HSI fusion baseline"), not clinical. **Risk:** see §6 item 1.

### R4 — Extend the finished ER-prediction line

You already hold a positive, robust result (RNA fusion beats H&E-alone, +0.044 AUROC, p=1.6×10⁻⁵).
The natural extension — does the fusion advantage survive to CPTAC, and does the gate degrade
gracefully when RNA is withheld at test time — directly addresses the failure mode the PAM50 CPTAC
multimodal work exposed.

### R5 — If you keep the genefu thread anyway

Reframe from "external validation" to a **decomposition study**, and design backwards from the
caveats:

- Fix F1 first; discard every existing HSI-BC number.
- Restrict TCGA training to **ER+/HER2−**: `[V]` **502 patients** have ER+/HER2− status *and* WSI
  features *and* RNA (PAM50: 347 LumA / 134 LumB / 16 Basal / 5 Her2). This resolves open item #2
  of the July 12 doc. Note only 63 PFI / 33 DFI events exist in that stratum.
- Pre-register **one** primary signature (`rorS` gives you MAKO as a direct published benchmark);
  others Bonferroni-corrected.
- **Site-preserved CV** on TCGA, and report the naive-vs-preserved delta — that delta is itself a
  small contribution.
- Primary endpoint: continuous predicted score vs DRFS in the **ER+/HER2− stratum** of HSI-BC
  (n=32, 14 events) as a Cox HR per 1 SD **with a CI, not a significance test**. `[L]` ER+/HER2− is
  the right stratum rather than ER+/HER2−/N0, because SWOG-8814 (Albain et al., *Lancet Oncol*
  2010;11:55-65) validated the 21-gene score in node-*positive* ER+ disease; the binding constraints
  are ER and HER2, not nodal status. This roughly triples the events (14 vs 5).
- Declare **tumour diameter alone (AUROC 0.925 in that stratum)** as the bar up front. If the model
  does not beat a ruler, say so.
- Do **not** report PPV, NPV, calibration, absolute risk, or any multivariable Cox with >2 covariates.
- Add a construct-fidelity check on **CPTAC-BRCA** (has WSI *and* RNA) — the only place you can
  measure the H&E→genefu link separately from the genefu→outcome link. Harmonise the scale first
  (Xena log2 RSEM vs GDC linear TPM).

Realistic outcome: a well-executed negative or ambiguous result plus a genuine methodological
contribution. A solid thesis chapter and a defensible workshop/methods paper — not a
Lancet-Oncology-adjacent contribution, and it should not be pitched as one.

---

## 6. Open questions you must resolve personally

**Blocking:**

1. **DESP-Net — `[U]`.** Lan P, Qiu S, "DESP-Net: Learning Disease Evolution States from
   Hyperspectral Pathology for Survival Prediction," ICIC 2026, DOI 10.1007/978-981-92-3485-1_32.
   Surfaces in an exact-phrase search for "HistologyHSI-BC Recurrence", but whether it uses this
   cohort, and all its metrics, are **unverified** — Springer redirects to auth, ResearchGate 403s,
   OpenAlex has a null abstract, and it was absent from the dblp ICIC 2026 TOC. If it reports a
   c-index on these 47 patients it becomes a mandatory comparator, and if it used the HSI cubes it
   substantially weakens R3. **Resolve before committing to R3.**
2. **Quintana-Quintana PhD thesis (ULPGC accedaCRIS, 2026) — `[U]`.** Not retrievable this session.
   May contain a baseline from the data authors themselves.
3. **Fix F1 and confirm nothing downstream of it has been written up.**

**High priority:**

4. **genefu fidelity — now largely answered, see §3b.** Bartlett 2020 quantifies the RS-like vs
   RS-true relationship at a ~2× slope with 56.9% upgraded, and Li 2016 documents the same failure
   on TCGA RNA-seq specifically. R is still not installed here, so §3a/3b used my reimplementation;
   if you proceed, install R + genefu and confirm the numbers reproduce, but the *direction and
   magnitude* are now externally sourced rather than resting on my code.

   *Resolved sub-question:* the genefu agent flagged that
   `oncotypedx-tcga-proxy-publishability-findings.md:92` cites Shamai (*Lancet Oncol* 2026) as
   externally validating on TCGA-BRCA without specifying the ground truth, and noted Li 2016's
   statement that no commercial ODX results exist for TCGA. Those are consistent, not contradictory:
   the workflow's adversarial checker retrieved the verbatim text — "As the TCGA-BRCA cohort does
   not have the Oncotype DX scores of the patients, we estimated the scores using transcripts per
   million (TPM) expression data. Formulas from the published development of Oncotype DX were then
   applied." So Shamai's TCGA arm used a **computed** RS, and that is precisely the arm where both
   the model (HR 1.8, p=0.111) and the computed RS (HR 1.7, p=0.160) failed to separate outcomes.
   The existing citation is accurate but should say "computed RS", not "assay".
5. **Is the TCGA CV site-preserved?** If not, expect internal AUROCs to fall by ~0.069.
6. **Howard et al. 2023 contradicts itself on the headline HR** `[L]`: Results text says
   HR 2.04 (1.18–3.53), Table 1 says HR 1.55 (1.13–2.12) for the same model, same n=322 subgroup,
   same C-index 0.743. Cite the Table 1 value and note the discrepancy; do not quote 2.04 bare.
7. **Contact the data authors** (corresponding author at IISPV/Tortosa) about whether any
   assay-guided chemotherapy assignment occurred and how the 47 were drawn from the biobank pool.
   The descriptor gives no matching criteria and no source-pool size.
8. **Pre-register the analysis population in writing before looking at any external score.**

**Hygiene:**

9. Dangling symlink `.datasets/tcga-brca/h5_files` (F1).
10. The sweeps concluding "no genefu+WSI paper exists" are **medium confidence** — read as
    "none found", not "none exists".
11. Use **HistologyHSI-BC-Recurrence** (TCIA form) in writing; the xlsx's internal
    "HistologyHSI-BRCA-Recurrence" is an inconsistency. Do not describe the cohort as
    "Hispanic or Latino" in a way implying ancestry-based generalisation — those fields are caDSR
    CDE artefacts forcing US OMB categories onto a Catalan single-hospital cohort.

---

## 7. Prior-art table `[L]`

✓ = adversarially verified this session; ○ = reported, unchecked.

| Paper | Year | Train N | Label | External | Result |
|---|---|---|---|---|---|
| ✓ **Howard et al.**, *npj Breast Cancer* 9:25 | 2023 | TCGA-BRCA 1,039 | **RNA-recomputed ODX + MammaPrint** — this exact trick | UCMC 427 (ODX), 88 (MP); real assay + real RFI | path-only AUROC 0.798; combined 0.828 vs nomogram 0.764, p=0.0005; RFI C 0.743 vs true ODX 0.776 in 322 chemo-naive; null in chemo-treated |
| ✓ **Shamai et al.**, *Lancet Oncol* 27(4):512-526 | 2026 | TAILORx 8,284, **true assay in an RCT** | ODX RS | 6 cohorts, 5,497 pts **incl. TCGA-BRCA n=594 with RNA-computed RS** | test AUC 0.898 (0.879–0.913); external 0.858–0.903; **TCGA: AUC 0.832, model HR 1.8 p=0.111, computed-RS HR 1.7 p=0.160** |
| ✓ **Boehm et al. (Orpheus)**, *Nat Commun* 16:2106 | 2025 | MSK-BRCA 5,145 | true ODX RS | IEO 452, MDX 575 | WSI-only r 0.60/0.60/0.58, AUROC 0.85/0.81/0.80; multimodal (WSI+report text) r 0.70, AUROC 0.88–0.89 |
| ✓ **Kaczmarzyk et al. (MAKO)**, *npj Digit Med* 9:149 | 2026 | CBCS 1,339 | PAM50-derived ROR-P, ABMIL over 12 encoders incl. UNI | TCGA-BRCA 1,050 | no model beat ResNet50 in TCGA after FDR; **transcriptomic ROR-P itself C=0.535, p=0.107**; "limited cross-cohort generalizability" |
| ✓ **CPMP (Yan et al.)**, *Adv Sci* 13(4):e10307 | 2026 | TJMUCH-MP 477 | MammaPrint risk score | TCGA-BRCA | MP AUROC 0.824 internal / 0.772 external; DMFS HR 3.14 (1.83–5.37) |
| ✓ **Sirinukunwattana et al. (imCMS)**, *Gut* 70:544-554 | 2021 | FOCUS 278 | transcriptomic CMS | TCGA 430, GRAMPIAN 144 | AUC 0.84 / 0.85; TCGA imCMS1 OS HR 1.88, p=0.027 |
| ○ **Goyal et al.**, *npj Breast Cancer* 10:93 | 2024 | 950 ER+/HER2−, true ODX | ODX | UChicago 405 | internal 0.91, external 0.84 |
| ○ **Wang/Acs (DeepGrade)**, *Ann Oncol* 33:89-98 | 2022 | NHG1-vs-3 | Nottingham grade | n=1,262 | recurrence HR 1.91 (1.11–3.29) |

The field has also moved past surrogate labels to **direct outcome supervision**, and beats the assay
when it does: ○ Ataraxis (Witowski et al., *Nat Commun* 2026), 8,161 patients / 15 cohorts, DFI
C-index 0.71, head-to-head vs real ODX in n=858 — AI 0.67 vs ODX 0.61.

**One thing not done:** no paper uses the **genefu package** specifically (all hand-implement the
published formulas), and no ML has been run on HistologyHSI-BC. But "we used genefu instead of
hand-coded formulas" is a weakness, not a novelty — see §3b.
