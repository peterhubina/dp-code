# Is AI prediction of ER status a real clinical need? — Literature-grounded analysis

Companion to `tcga-brca-reliable-fusion-task-report.md`. This document grounds two things in primary
literature retrieved via `paper-search` (PubMed/Crossref): (a) the technical claims of the prior report,
and (b) the clinical-feasibility question — *is predicting ER status from H&E a real clinical need?*

**Verdict up front:** Predicting **ER status** from H&E is technically feasible and validated, but as a
*standalone clinical product its need is weak*, because ER is already measured by a cheap, mandated,
irreplaceable IHC assay. Its defensible clinical role is **QC / discordance flagging**. The tasks with
genuine unmet need — and where the same pipeline delivers real value — are **Oncotype-DX-style recurrence
scoring** and **HER2-low discrimination**, where the reference assay is expensive, slow, or poorly
reproducible. This is now grounded in named, dated primary sources below.

---

## Part A — Grounding the prior (technical) report

| Prior-report claim | Grounding source | Status |
|---|---|---|
| TCGA-BRCA foundational dataset (modality counts, PAM50 distribution) | **TCGA Network 2012**, *Nature* 490:61–70, `10.1038/nature11412` (10,955 citations) | Confirmed as the primary source; modality counts were verified 3-0 against PMC3465532 in the prior workflow |
| PFI/DFI are the recommended BRCA survival endpoints; OS/DSS underpowered | **Liu et al. 2018, TCGA-CDR** — grounded via the authors' AACR abstract `10.1158/1538-7445.am2018-3287` ("PFI derived with high confidence… recommended for 27 of 33 pan-cancer types… OS for 23") and the SABCS abstract `10.1158/1538-7445.sabcs17-p3-16-01` (BRCA-specific: 1097 cases, median follow-up **27.7 months**; ER+ vs ER− PFI **p=0.005**, DFI **p=0.001**, OS **not significant p=0.09**; "PFI and DFI are valid… OS and DSS with some caution") | **Confirmed** — the BRCA-specific numbers match the prior report exactly |
| Multimodal WSI+genomic fusion is feasible but a niche, mostly for prognosis | **Unger & Kather 2024**, *BMC Med Genomics* `10.1186/s12920-024-01796-9` — systematic survey of 534 articles: "Multimodal DL… remains a niche topic… primarily focusing on prognosis predictions" | **Confirmed** |
| Fusion survival baselines exist on TCGA | **CATfusion** (Hu et al. 2025, *Brief Bioinform* `10.1093/bib/bbaf121`); **UMPSNet** (Zhang et al. 2025, *Am J Pathol* `10.1016/j.ajpath.2025.06.006`, mean c-index **0.725** across 5 TCGA cohorts, zero-shot transfer c-index 0.652) | **Confirmed** — corroborates PORPOISE-class feasibility |

Note: the Liu 2018 and TCGA 2012 *full texts* are paywalled; grounding uses same-author conference
abstracts and the Crossref record, which reproduce the identical quantitative claims. The prior
deep-research workflow independently verified the TCGA-2012 modality counts 3-0 against the PMC full text.

---

## Part B — Is ER-from-H&E a real clinical need? (the core question)

### B1. ER cannot be *replaced* — the assay is cheap, mandated, and irreplaceable

The **ASCO/CAP 2020 ER/PgR testing guideline** (Allison et al., *J Clin Oncol* `10.1200/JCO.19.02309`;
mirrored in *Arch Pathol Lab Med* `10.5858/arpa.2019-0904-SA`) is decisive:

> "The Expert Panel continues to recommend ER testing of invasive breast cancers by validated
> immunohistochemistry as the standard for predicting which patients may benefit from endocrine therapy,
> **and no other assays are recommended for this purpose.**"

This is the single strongest argument against ER-from-H&E as a product: an H&E prediction *cannot* be used
to decide endocrine therapy — guideline-mandated IHC is required, and it is cheap (tens of dollars),
fast (~1 day), reimbursed, and one of the more reproducible IHC markers. **A predictor of an assay that is
already trivial to run has little standalone clinical need.** (Confidence: high.)

### B2. Where a genuine — but narrow — need exists: QC / discordance flagging

The guideline itself opens the door. It created a new **"ER Low Positive" (1–10%)** reporting category and
concedes "**limited data on endocrine therapy benefit** for cancers with 1% to 10% of cells staining ER
positive," recommending laboratory SOPs to "confirm/adjudicate" low/no-ER results. That adjudication step
is exactly where an orthogonal H&E signal helps.

This QC use-case is grounded, not hypothetical. **Shamai et al. 2019** (*JAMA Network Open*
`10.1001/jamanetworkopen.2019.7700`; 5,356 patients, 20,600 H&E TMA sections) predicted ER from H&E with
**PPV 97–98%, NPV 68–76%, accuracy 91–92%, "noninferior to traditional IHC,"** and — critically — reported:

> "Morphological analysis of patients with ER-negative/PR-positive status by IHC revealed resemblance to
> patients with ER-positive status… This **suggests a false-negative IHC finding** and warrants
> antihormonal therapy for these patients."

That is a documented instance of an H&E model catching a likely IHC error — the concrete clinical value of
ER-from-H&E is as a **cheap automated second-reader / QC layer**, not a replacement. (Confidence: high for
feasibility; medium for whether QC deployment measurably reduces error rates in prospective clinical use —
that trial evidence does not yet exist.)

### B3. The tasks with *real* unmet need — where the same pipeline pays off

The literature is unambiguous that the clinically-needed targets are the ones whose reference assay is a
*burden*, which is precisely what ER is not:

**Oncotype-DX recurrence scoring from H&E** (expensive genomic assay → real economic/access need):
- **Boehm, … Kather 2025 — "Orpheus"** (*Nat Commun* `10.1038/s41467-025-57283-x`, 6,172 cases): infers the
  Recurrence Score from H&E, identifying high-risk (RS>25) at **AUC 0.89 vs 0.73 for the leading
  clinicopathologic nomogram**; motivation stated as ODX "cost and lag time have limited global adoption."
- **Shamai et al. 2026** (*Lancet Oncol* `10.1016/S1470-2045(25)00727-2`): trained/validated on the
  **TAILORx** randomised trial (8,284 pts), RS≥26 **AUC 0.898**, externally validated on **six** independent
  cohorts (**AUC 0.858–0.903**), explicitly framing the need as ODX being "inaccessible to many patients
  because of high cost and logistical barriers… particularly in resource-limited settings where genomic
  testing remains unavailable or unaffordable." It even reclassified 31% of clinically-high-risk
  postmenopausal women to low-AI-risk with no chemo benefit — a direct de-escalation decision.
- Supporting: **BCR-Net** (Su et al. 2023, *PLoS One* `10.1371/journal.pone.0283562`, H&E AUC 0.775),
  **BPMambaMIL** (Guo et al. 2025, *Comput Methods Programs Biomed* `10.1016/j.cmpb.2025.109039`, AUC 0.839),
  **Magee-DL** (Li et al. 2022, *Front Med* `10.3389/fmed.2022.886763`) — all frame ODX as "expensive,
  time-consuming, and tissue destructive."

**HER2-low discrimination** (new actionable category for trastuzumab deruxtecan; IHC poorly reproducible at
the low end → real diagnostic need):
- **Turashvili et al. 2024** (*J Clin Pathol* `10.1136/jcp-2023-209055`): among *subspecialised* breast
  pathologists, binary agreement at the 1% cutoff was only **moderate (κ≈0.56)** — "An **urgent need remains
  for a new assay/algorithm** to reliably evaluate HER2-low breast cancer."
- **Zaakouk et al. 2023** (*The Breast* `10.1016/j.breast.2023.06.005`, 16 UK/Ireland expert pathologists):
  HER2-low agreement only "fair to moderate"; ~10% of cases remain irreducibly challenging.
- **Farshid et al. 2024** (*Mod Pathol* `10.1016/j.modpat.2024.100535`): **41%** of locally-scored HER2-0
  cases were reclassified as HER2-low on expert review — a large, clinically-consequential error rate that
  an AI adjunct could address.

---

## Synthesis: reliability ≠ clinical need

The prior report ranked ER **first on reliability** (best AUROC, best label quality, best external-validation
path). This report shows ER ranks only **moderate on clinical need**, because its ground truth is cheap and
mandated. The two are different axes, and conflating them is the central risk in framing a thesis around ER.

**Recommended framing** (defensible to a methodological reviewer):
1. Use **ER-from-H&E as the validation / proof-of-concept task** — you can demonstrate the UNI2+CLAM pipeline
   works and externally validate it (per the prior report).
2. Position the **clinical value proposition** as either (a) a **QC / discordance-flagging** adjunct for ER/PgR
   IHC — grounded in ASCO/CAP's own "adjudicate low-ER results" recommendation and the Shamai 2019
   false-negative finding — or, more compellingly, (b) **extend the identical pipeline to a genuinely-needed
   target**: Oncotype-DX recurrence score or HER2-low, where the reference assay is expensive or unreliable
   and the recent literature (Orpheus, Shamai 2026 Lancet Oncol) already shows H&E prediction clears a
   clinically meaningful bar with external validation.

This lets your *most reliable result* (ER) and your *most clinically relevant claim* (recurrence / HER2-low)
be different tasks — avoiding the reviewer critique: "you built an AI to predict something we already
measure trivially."

---

## Confidence and open questions

- **High confidence:** ER IHC is standard-of-care, cheap, mandated, and clinically irreplaceable by an H&E
  prediction (ASCO/CAP 2020); ER-from-H&E is technically feasible and noninferior to IHC in accuracy
  (Shamai 2019); Oncotype and HER2-low carry genuine, literature-documented unmet need.
- **Medium confidence:** that an AI QC layer *measurably* reduces ER/PgR IHC error rates in prospective
  deployment — the mechanism is grounded (Shamai 2019 caught a likely false-negative) but no prospective
  QC-deployment trial exists yet.
- **What would change the recommendation:** a regulator or guideline body accepting H&E-predicted receptor
  status as a reflex-triage or QC tool (would broaden ER's clinical need), or evidence that fusion (RNA-seq)
  adds nothing over H&E-alone for ER (would further weaken ER as a fusion task specifically — an open
  question flagged in the prior report).

## Key references (retrieved and read this session)

- **Allison et al. 2020**, ASCO/CAP ER/PgR guideline, *JCO* `10.1200/JCO.19.02309` — ER IHC is the mandated
  standard; defines the ER-Low-Positive (1–10%) category with acknowledged evidence gaps.
- **Shamai et al. 2019**, *JAMA Netw Open* `10.1001/jamanetworkopen.2019.7700` — ER from H&E, noninferior to
  IHC; documents an H&E-caught likely false-negative IHC (the QC use-case).
- **Boehm & Kather et al. 2025 (Orpheus)**, *Nat Commun* `10.1038/s41467-025-57283-x` — Oncotype RS from H&E,
  AUC 0.89 vs 0.73 nomogram.
- **Shamai et al. 2026**, *Lancet Oncol* `10.1016/S1470-2045(25)00727-2` — TAILORx-validated ODX-from-H&E,
  external AUC 0.858–0.903; states the cost/access clinical need explicitly.
- **Turashvili et al. 2024**, *J Clin Pathol* `10.1136/jcp-2023-209055` — HER2-low IHC poorly reproducible;
  "urgent need for a new assay/algorithm."
- **Zaakouk et al. 2023**, *The Breast* `10.1016/j.breast.2023.06.005`; **Farshid et al. 2024**, *Mod Pathol*
  `10.1016/j.modpat.2024.100535` — corroborate HER2-low reproducibility crisis (41% HER2-0 reclassified).
- **Liu et al. 2018 (TCGA-CDR)**, AACR `10.1158/1538-7445.am2018-3287` + SABCS
  `10.1158/1538-7445.sabcs17-p3-16-01` — PFI/DFI recommended for BRCA, OS/DSS with caution.
- **TCGA Network 2012**, *Nature* `10.1038/nature11412` — foundational BRCA multi-omic dataset.
- **Unger & Kather 2024**, *BMC Med Genomics* `10.1186/s12920-024-01796-9` — multimodal DL is a niche, mostly
  prognosis; **UMPSNet** `10.1016/j.ajpath.2025.06.006`, **CATfusion** `10.1093/bib/bbaf121` — fusion baselines.
