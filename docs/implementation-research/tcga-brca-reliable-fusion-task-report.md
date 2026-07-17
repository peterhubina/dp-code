# The single most reliable WSI + fusion task on TCGA-BRCA

**Bottom line:** The most reliable target is **ER (estrogen-receptor) status prediction**, paired with **RNA-seq expression** as the fusion modality, externally validated on **CPTAC-BRCA**. It beats the more fashionable PAM50 four-class subtyping and the survival-fusion tasks on every axis that matters for reliability: label quality, sample size, cross-study reproducibility, and the existence of a like-labeled external cohort. This is a research-grade synthesis from 24 fetched primary sources and 22 adversarially-confirmed claims (3 were killed during verification) — but it is web-literature-derived, not a full-text paper-search run.

## 1. TCGA-BRCA dataset profile

TCGA-BRCA is large on imaging but sharply constrained the moment you require a *matched* second modality. The foundational TCGA Network paper (Nature 2012, PMC3465532) reports the platform counts that still define fusion feasibility: **mRNA n=547, methylation n=802, copy-number n=773, miRNA-seq n=697, WES n=507, and RPPA protein n=403**, with **only 348 primary tumors complete across all platforms** (confirmed 3-0; corroborated by the GDC brca_2012 publication page). RPPA (n=403) is the binding constraint for any WSI+protein fusion; mRNA (n=547) bounds WSI+expression fusion. **Important caveat:** these are the 2012 freeze figures and represent a *historical floor* — current GDC releases have grown to ~1000+ RNA-seq cases, so your usable fusion N today is almost certainly larger. Quantifying that current N is the single biggest open question (see §6).

Labels split into "clinically-measured" versus "derived." ER/PR/HER2 come from routine **clinical IHC assays** — high reliability, minimal derivation ambiguity. PAM50 subtypes are a **derived gene-expression signature** (curated file `BRCA.547.PAM50.SigClust.Subtypes.txt` on GDC) with real class imbalance: **LumA n=225, LumB n=126, HER2E n=57, Basal n=93** (3-0). Outcome labels should come from the **TCGA-CDR** curation (Liu et al. 2018, Cell; PMID 29625055), which standardizes OS/PFI/DFI/DSS across 11,160 tumors — and its per-cancer guidance is decisive for BRCA: **PFI and DFI are recommended; OS and DSS are underpowered** because of insufficient follow-up in this indolent cohort (ER+ vs ER− separates on PFI p=0.005 and DFI p=0.001 but *not* OS p=0.097; confirmed 3-0).

## 2. Candidate-task comparison

| Task | Label reliability | Usable N (WSI + fusion) | Published WSI perf (variance) | Leakage risk | External cohort | Overall |
|---|---|---|---|---|---|---|
| **ER status** | **High** (clinical IHC) | ~547 (mRNA) / 403 (RPPA); larger in current GDC | **AUROC 0.82** (Kather 2020), 0.806 (Arslan 2024), up to **0.951** 5-fold CV (CAT 2024) | Site-signature risk (Howard 2021) | **CPTAC-BRCA, Carmel, ABCTB, HEROHE-adjacent** | **★ Best** |
| PR status | High (IHC) | same | AUROC ~0.74–0.79 | same | same | Good |
| HER2/ERBB2 | Moderate (IHC+ISH) | same | AUROC ~0.82 | same | **HEROHE (509 WSI), Warwick** | Good |
| PAM50 4-class | Moderate (derived signature) | ~547 | macro-F1 **0.727**; avg AUROC **0.752±0.080**; HER2E class collapses (F1 **0.545**) | Imbalance + site | CPTAC-BRCA (PAM50) | Fragile |
| TP53 mutation | High (WES) but weak signal | ~507 | AUROC up to **0.785** (peak); SNVs avg 0.636±0.117 | same | Limited | Weak |
| PIK3CA mutation | High (WES), weak signal | ~507 | Generally <0.70 | same | Limited | Weak |
| OS survival | **Low** (underpowered, per TCGA-CDR) | ~443 (PORPOISE risk-split) | c-index feasible but noisy | Follow-up + site | Scarce | Weak |
| PFI/DFI | Moderate (recommended endpoint) | ~443 | Less-reported | same | Scarce | Modest |
| HRD / genomic instability | Derived score | <500 | Sparse WSI literature | same | Very limited | Weak |
| Treatment response | **Poor** (not systematically annotated) | Too small | — | — | None | Not viable |

The pattern across three *independent* primary studies is the same ordering — **ER > PR > HER2** — and it reflects biology: ER status correlates most strongly with H&E morphology. Kather et al. 2020 (Nature Cancer, TCGA-BRCA N=995) report ER AUROC 0.82, PR 0.74; Arslan et al. 2024 (Comm Med) report ER 0.806, PR 0.744; the CAT-Train study (Comm Med 2024) reports ER 0.951, PR 0.792, ERBB2 0.822 in 5-fold CV (all confirmed 3-0). By contrast, PAM50's much-cited "0.78–0.87" figures are the *Basal* subtype in isolation, not the four-class average, and the minority HER2E class (n=57) collapses to F1 0.545 — the multi-class task is fragile precisely where the clinic cares most. Driver-mutation tasks sit ~0.1 AUROC below biomarkers (SNV mean 0.636 vs biomarker mean 0.742). Survival fusion is architecturally feasible — PORPOISE (Chen et al. 2022, Cancer Cell) fuses attention-MIL WSI with a molecular network across 14 types including BRCA (443 patients) — but it trains on the very OS/DSS endpoints TCGA-CDR flags as underpowered for BRCA.

## 3. Recommendation

**Predict ER status from H&E WSIs, fused with RNA-seq expression.** ER wins on all four reliability axes: (1) its label is a routine clinical assay, not a derived signature; (2) it has the highest and most *reproducible* H&E signal (0.80–0.82 consistently across three independent groups); (3) matched RNA-seq is the largest fusion modality (n≥547, more today); and (4) it has a genuine external-validation pathway. RNA-seq is the right fusion partner because ER is fundamentally a transcriptional-program readout (*ESR1* and its downstream signature), so expression adds a biologically-orthogonal signal to morphology and is the modality most cohorts also carry.

**The strongest counter-argument** — and it is real — comes from Kather et al. 2020 itself: in that influential paper, external validation was performed *only* for colorectal BRAF/CIMP (DACHS, N=408); **every breast prediction, including the ER 0.82, was TCGA cross-validation only** (confirmed 3-0). Internal CV on TCGA is optimistic, and Howard et al. 2021 (Nat Commun) showed TCGA models readily learn tissue-submitting-site signatures (a leakage/confounder). Two things blunt this counter-argument: ER's external validation *does* exist in the later literature (Arslan 2024 validated on CPTAC; CAT-Train used held-out external cohorts), and ER still beats PAM50 and survival even under the pessimistic reading. A secondary caveat: the brief calls for *fusion*, and none of the verified evidence isolates whether RNA-seq actually improves ER prediction over H&E-alone — ER may already be near-saturated from morphology. That is a genuine open question, not a refutation.

## 4. External-validation plan

**Primary: CPTAC-BRCA.** It shares PAM50 gene-expression labels with TCGA-BRCA, and Arslan 2024 externally validated H&E-biomarker prediction on CPTAC (3,481 images, 1,329 patients across 7 cancer types) with "comparable overall performance across almost all tested cancer types." PMC11667687 confirms CPTAC-BRCA (382 WSIs, PAM50 labels) matches TCGA-BRCA (980 WSIs). **Honest caveat:** Arslan's CPTAC *absolute* AUCs were modest (0.567–0.672) for the pan-cancer set — it validated *comparability and generalization*, not high absolute performance. Expect a drop under scanner/staining domain shift.

**For receptor status specifically:** the CAT-Train pathway (Carmel + ABCTB + TCGA) demonstrates ER/PR/HER2 generalizes across cohorts. **For HER2** as a fallback biomarker, HEROHE offers a purpose-built external cohort (509 H&E WSIs, HER2 labels by IHC+ISH: 359 train / 150 test). METABRIC has expression and outcome but **reportedly no public WSIs**, so it cannot serve as an imaging external set. Concrete risks to plan for: (a) access — CPTAC/TCGA open, Carmel/ABCTB gated; (b) domain shift — different scanners and H&E protocols; (c) label-definition drift — IHC cutoffs (1% vs 10% ER positivity) have shifted with ASCO/CAP guideline revisions, so harmonize label definitions before comparing.

## 5. Annotated key references

- **TCGA Network 2012, Nature (PMC3465532)** — the dataset's platform counts and the PAM50 subtype distribution; the source of the fusion-N ceiling.
- **Liu et al. 2018, Cell — TCGA-CDR (PMID 29625055)** — the standardized outcome-label resource; establishes PFI/DFI over OS/DSS for BRCA.
- **Kather et al. 2020, Nature Cancer (s43018-020-0087-6)** — pan-cancer H&E biomarker benchmark; ER 0.82; also the paper whose breast tasks were *never* externally validated.
- **Arslan et al. 2024, Comm Med (s43856-024-00471-5)** — systematic multi-omic-from-H&E benchmark; ER 0.806, PAM50 0.752, TP53 0.785; CPTAC external validation.
- **CAT-Train 2024, Comm Med (s43856-024-00695-5)** — Carmel+ABCTB+TCGA receptor prediction with external test cohorts; ER up to 0.951.
- **Chen et al. 2022, Cancer Cell — PORPOISE (S1535-6108(22)00317-8)** — the canonical WSI+molecular *survival* fusion baseline; proves fusion feasibility, wrong endpoint for BRCA.
- **Howard et al. 2021, Nat Commun (s41467-021-24698-1)** — TCGA site-signature leakage; the reason to hold out by submitting site.
- **PMC11667687** — CPTAC-BRCA/TCGA-BRCA/HER2-Warwick label matching for external validation.

## 6. Confidence and open questions

**High confidence:** ER is the most reliable H&E-predictable breast biomarker; PAM50 is more fragile than its popularity implies; OS is underpowered for BRCA.

**Lower confidence / open:**

1. **Current usable fusion N** — the 547/403/348 figures are the 2012 floor; the real present-day matched WSI+RNA-seq / +RPPA N needs to be pulled from the current GDC release before you size experiments.
2. **Marginal value of the second modality for ER** — no verified evidence isolates whether RNA-seq beats H&E-alone for ER. If ER is morphology-saturated, "fusion" may be cosmetic; test H&E-alone vs fusion head-to-head.
3. **Externally-validated (not CV) ER AUROC** under a fixed model on CPTAC/Carmel/ABCTB with real domain shift.
4. **Under-surfaced cohorts** — BRACS, BACH/ICIAR, TUPAC16, Post-NAT-BRCA, AIDPATH weren't confirmed to carry matched WSI + the required fusion label; verify their access terms and label drift before counting on them.

**Three claims were killed in verification** and are worth flagging so they are not reused: a "825 patients / 466 complete across 5 platforms" figure (0-3, wrong), a specific "171 antibodies × 403 samples" RPPA detail (1-2), and the assertion that *no* PAM50 external cohort exists (0-3 — refuted; CPTAC-BRCA provides matched WSI+PAM50).

---

## Methodological note

This report was produced by the `/deep-research` harness running WebSearch agents over web-hosted primary sources (Nature, Cell, PMC, GDC), not a full-text `paper-search` (arXiv/PubMed/Semantic Scholar) run. Claims are adversarially verified (22/25 confirmed 3-0). Two paywalled full-texts (Liu 2018, PORPOISE) returned HTTP 403 and were verified via PubMed abstracts + concordant paraphrase rather than raw full text. Sample counts are 2012-freeze historical floors; current GDC releases are larger.

### Verification stats

- 5 search angles · 24 sources fetched · 113 claims extracted · 25 verified · **22 confirmed / 3 refuted / 0 unverified**
- 106 agent calls · ~2.53M subagent tokens
