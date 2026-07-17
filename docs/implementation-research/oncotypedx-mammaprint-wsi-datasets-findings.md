# Public WSI datasets with Oncotype DX / MammaPrint labels — Deep-research findings

**Purpose:** source data for a multimodal breast-cancer pipeline with **WSI as the primary modality** and a **second modality fused** with it, ideally plus an **independent external cohort** for evaluation.

**Method:** produced by the `deep-research` harness (5 search angles → 18 sources fetched → 82 claims extracted → 25 adversarially verified, 3-vote, 2/3 to kill → 19 confirmed). Run date 2026-07-12. A `paper-search` verification pass follows this report (see final section).

**Status of each claim below:** confidence is carried from the verification vote. Refuted/uncertain items are quarantined in their own section so they are not chased as dead ends.

---

## Bottom line up front (BLUF)

A genuinely public WSI + **Oncotype DX** dataset *does* exist, but it is small and single-institution — the **BCR-Net** dataset on Zenodo (CC BY 4.0): pre-extracted 40× H&E patch bags for 151 patients (99 with complete labels), plus paired Ki67 IHC for 50 of them. **No truly public dataset releases raw MammaPrint results paired with WSI.**

For scale, the realistic path is twofold:

1. **Recomputable-target route (open data, proxy labels).** Pair public WSI cohorts — **TCGA-BRCA** (WSI via GDC + RNA-seq) and **CPTAC-BRCA** (134 SVS subjects on TCIA, CC BY 4.0, + proteomics/genomics) — with their released expression data, and recompute **proxy** Oncotype DX (21-gene) and MammaPrint-like (70-gene) labels with the **`genefu`** R/Bioconductor package. These are research reimplementations, *not* the FDA/proprietary assay scores — proxy fidelity is the central caveat.
2. **Controlled-access direct-label route (DUA/ethics-gated).** Request institutional/trial cohorts that hold *true* Recurrence Scores: **Dartmouth/BMIRDS** (~990 WSI, DUA), **TAILORx** (8,284 pts / 9,383 WSI via ECOG-ACRIN + NCTN/NCORP), **MSKCC "Orpheus"** (6,172 cases), and a **five-cohort medRxiv set** (Israel/Australia/US). Availability is not the constraint here — **access latency (DUA/ethics)** is.

**Fusion modality:** for every direct-label WSI+ODX study, the co-released fusible second modality is **clinicopathologic variables** (age, tumor size, grade, ER/PR/HER2). For the recomputable route, the fusible partner is **RNA-seq / proteomics** (which also generates the label — beware label leakage if the same expression both trains the target and feeds the fusion head).

**External validation:** plentiful *in principle* (US / Israel / Australia cohorts exist), but every large ODX cohort is access-gated. The only frictionless external pairing is **BCR-Net vs. CPTAC/TCGA** — at the cost of a label-definition mismatch (direct RS vs. recomputed proxy).

---

## Comparison table

| Dataset | WSI (format / #cases) | Oncotype DX label | MammaPrint label | Recomputable from expression? | 2nd modality for fusion | Access path & license | External cohort? | Confidence |
|---|---|---|---|---|---|---|---|---|
| **BCR-Net (OSU + Wake Forest)** | Pre-extracted 40× H&E patch bags (HDF5, 224×224), Leica Aperio ScanScope CS2; 151 pts (99 labeled: 64 low ODX<25, 35 high ODX≥25) | ✅ **Direct** RS per WSI (binarized) | ❌ | n/a (direct label) | **Ki67 IHC** (50 pts, adjacent slides) — second *imaging* modality | **Public**, Zenodo [7514392](https://doi.org/10.5281/zenodo.7514392) / [7514394](https://doi.org/10.5281/zenodo.7514394), **CC BY 4.0** | ✅ vs. TCGA/CPTAC (label-def mismatch) | **High** |
| **CPTAC-BRCA (TCIA)** | 642 H&E WSI, **SVS**, 20×, 134 subjects, 113 GB | ❌ (recompute) | ❌ (recompute) | ✅ via `genefu` on CPTAC transcriptomics | **Proteomics + genomics + RNA + clinical** | **Public**, [TCIA](https://www.cancerimagingarchive.net/collection/cptac-brca/), CC BY 4.0, IBM Aspera | ✅ (mutually external w/ TCGA) | **High** |
| **TCGA-BRCA** | H&E diagnostic + frozen WSI via GDC (~1,000+ cases) | ❌ (recompute) | ❌ (recompute) | ✅ via `genefu` on TCGA RNA-seq | **RNA-seq + clinical** | **Public** GDC (open tier); recompute labels | ✅ | **High** (implied; confirm counts) |
| **Dartmouth / BMIRDS** | ~990 FFPE H&E WSI, **.tif**, Aperio AT2, 20×/40×, ER+/HER2− | ✅ **Direct** RS (binarized low/high) | ❌ | n/a | **Clinicopathologic** (age, size, grade, histtype, ER/PR/HER2) | **Controlled-access DUA**, [BMIRDS](https://bmirds.github.io/) / [npj Breast Cancer 2024](https://www.nature.com/articles/s41523-024-00700-z) | ✅ (multi-institution) | **High** (per-site split counts unverified) |
| **TAILORx** | 9,383 WSI, 8,284 pts | ✅ **Direct** 21-gene RS | ❌ | n/a | **Clinicopathologic** | **Controlled-access**: request from **ECOG-ACRIN** + **NCTN/NCORP Data Archive** (code only on [GitHub](https://github.com/shachar5020/TransformerWSI4OncoDXPrediction)/Zenodo) — [Lancet Oncol 2025](https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(25)00727-2/fulltext) | ✅ (large, multi-site) | **High** |
| **MSKCC "Orpheus"** | WSI, 6,172 cases, 3 institutions | ✅ **Direct** RS | ❌ | n/a | **Clinicopathologic** | **Institutional / not public** — [Nat Commun 2025](https://www.nature.com/articles/s41467-025-57283-x) | ✅ | **High** |
| **Five-cohort set** (Carmel, Haemek, Sheba, ABCTB, UChicago) | ~5,546 slides / 4,227 pts | ✅ **Direct** RS (subset of cohorts) | ❌ | n/a | **Clinicopathologic** | **Institution-held / biobank ethics** — [medRxiv 2025](https://www.medrxiv.org/content/10.1101/2025.07.21.25331907.full.pdf) | ✅ (Israel/Australia/US domain shift) | **High** (which cohorts carry RS is uncertain) |

---

## Recomputing the labels: the `genefu` route

`genefu` (Gendoo et al., *Bioinformatics* 2016) implements 12 breast-cancer prognostication algorithms, including:

- **`oncotypedx()`** — 21-gene Recurrence Score after Paik et al. 2004; ships `sig.oncotypedx` (21 genes, EntrezGene.ID + Affy probe mapping); returns a **continuous score + binary risk**.
- **`gene70()`** — 70-gene signature after van 't Veer et al. 2002 (the MammaPrint-like signature); returns **continuous score + binary high/low risk**; `sig.gene70` = 70 probes.

The manual states microarray-defined signatures "can be used to reliably evaluate RNA sequencing data," so the functions apply to **TCGA-BRCA / CPTAC RNA-seq**. An independent study (ESMO Open 2025, PMC12088756) recomputed all three scores (ODX, 70-gene, PAM50-ROR) from 1,527 RNA-seq/microarray samples.

Sources: [`genefu` paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6410906/) · [Bioconductor manual](https://www.bioconductor.org/packages/devel/bioc/manuals/genefu/man/genefu.pdf) · [ESMO Open 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088756/)

> **Central caveat (proxy fidelity):** `oncotypedx()` / `gene70()` are research reimplementations of the *published* signatures, **not** the proprietary RT-PCR–calibrated commercial Recurrence Score or the FDA-cleared MammaPrint. Concordance with true assay scores is signature- and platform-dependent and imperfect. The research goal explicitly permits "close approximation," but any thesis claim must foreground this.

---

## Recommended shortlist

**Train on:** the **recomputable-target route** — **TCGA-BRCA** (largest public WSI + RNA-seq) as the primary training corpus, with **CPTAC-BRCA** as a second public WSI+omics cohort. Recompute proxy ODX (21-gene) and 70-gene labels with `genefu`.

**Fuse:** WSI (primary, via your existing UNI2-h → CLAM MIL flow) + **clinicopathologic tabular features** as the second modality. Prefer clinical over RNA-seq for the fusion head when the RNA is *also* the label source, to avoid the label-leakage of feeding the target's own expression into the classifier. (If using RNA as fusion input, hold out the ODX/70-gene genes from the fusion features.)

**External validation:** **BCR-Net** (Ohio State / Wake Forest, public, *direct* ODX) as a held-out external set — different institution, scanner, and a real assay label rather than a proxy. This is the cleanest zero-friction external test. For a stronger, direct-label external benchmark, pursue a **Dartmouth/BMIRDS DUA** (or TAILORx via ECOG-ACRIN) in parallel, accepting the access-latency risk.

**Key risks / caveats:**
- **Proxy vs. true label mismatch** — training on recomputed proxies but validating on BCR-Net's true RS measures proxy-to-assay transfer, not pure model skill. Quantify the gap.
- **Small external n** — BCR-Net has 99 labeled patients; wide CIs.
- **BCR-Net ships patches, not raw WSI** — fine for a patch-MIL pipeline, but no re-tiling flexibility.
- **Domain shift** — scanner/stain/population differences between TCGA and BCR-Net are real; stain normalization advisable.
- **Access latency** — all direct-label large cohorts (TAILORx, MSKCC, MINDACT, ABCTB, Israeli centers) are DUA/ethics-gated with unquantified timelines and possible cost.

---

## Mention-only / could NOT confirm obtainable (do not chase)

- **MINDACT (EORTC 10041 / BIG 3-04)** — 6,693 pts, the largest prospective **MammaPrint** cohort, but **EORTC-owned; no public WSI or patient-level genomic release.** ([Lancet Oncol 2021](https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(21)00007-3/abstract))
- **I-SPY2 (TCIA)** — **DCE-MRI radiology only, no WSI.** ([TCIA](https://www.cancerimagingarchive.net/collection/ispy2/))
- **BRAIN study (MDPI Cancers 2024)** — 2,565 Korean ER+/HER2− pts with **2,039 ODX + 526 MammaPrint** labels, but predicts risk from **clinicopathological variables only — no WSI, no RNA released.** ([MDPI 16(4):774](https://www.mdpi.com/2072-6694/16/4/774))
- **RASTER** — real per-patient MammaPrint results exist (observational sibling of MINDACT), but no confirmed WSI pairing / public download surfaced.
- **17-dataset public breast H&E WSI scoping review** (10,385 WSIs) — a useful WSI catalog, but the review itself notes **none explicitly include Oncotype DX or MammaPrint labels.** ([arXiv 2306.01546](https://arxiv.org/pdf/2306.01546))

## Refuted during deep-research, then re-checked with paper-search

- Exact Dartmouth per-institution split (198 Dartmouth internal / 418 UChicago external) — **refuted 0-3** in the deep-research pass; treat cohort-split specifics as unverified (the paper is real; the split numbers are not confirmed).
- Orpheus AUC 0.89 — deep-research quarantined this (0-3), but **paper-search reinstates it**: the *Nature Communications* 2025 abstract states verbatim "identifies TAILORx high-risk cases (RS > 25) with an area under the curve (AUC) of 0.89, compared to a leading clinicopathologic nomogram with 0.73." So the deep-research refutation was a false negative — the figure is genuine.

## Open questions to resolve next

1. Empirical concordance (AUC / correlation) between `genefu`-recomputed proxy labels and true commercial assay scores — needed to quantify proxy-label training risk.
2. Whether TCGA-BRCA / CPTAC RNA-seq covers the 21 ODX + 70 MammaPrint genes at adequate quality, and how many **WSI-linked** cases survive.
3. Concrete timeline / cost / eligibility for TAILORx (ECOG-ACRIN + NCTN/NCORP) and Dartmouth/BMIRDS DUAs as an outside academic.
4. Whether any MammaPrint-labeled cohort (RASTER / MINDACT-adjacent) can be paired with WSI via EGA/dbGaP, or whether recomputed 70-gene proxy is the *only* viable MammaPrint route.

---

## Sources

Primary: [BCR-Net / PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0283562) · [Zenodo 7514392](https://doi.org/10.5281/zenodo.7514392) · [Zenodo 7514394](https://doi.org/10.5281/zenodo.7514394) · [CPTAC-BRCA / TCIA](https://www.cancerimagingarchive.net/collection/cptac-brca/) · [`genefu` paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6410906/) · [`genefu` manual](https://www.bioconductor.org/packages/devel/bioc/manuals/genefu/man/genefu.pdf) · [ESMO Open 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088756/) · [Dartmouth npj Breast Cancer 2024](https://www.nature.com/articles/s41523-024-00700-z) · [BMIRDS](https://bmirds.github.io/) · [TAILORx / Lancet Oncol 2025](https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(25)00727-2/fulltext) · [Orpheus / Nat Commun 2025](https://www.nature.com/articles/s41467-025-57283-x) · [Five-cohort / medRxiv 2025](https://www.medrxiv.org/content/10.1101/2025.07.21.25331907.full.pdf) · [MINDACT / Lancet Oncol 2021](https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(21)00007-3/abstract) · [I-SPY2 / TCIA](https://www.cancerimagingarchive.net/collection/ispy2/) · [BRAIN / MDPI Cancers 2024](https://www.mdpi.com/2072-6694/16/4/774) · [WSI scoping review / arXiv 2306.01546](https://arxiv.org/pdf/2306.01546) · [GDC data-access policy](https://gdc.cancer.gov/access-data/data-access-policies)

*Deep-research stats: 5 angles · 18 sources fetched · 82 claims · 25 verified (19 confirmed / 6 refuted) · 7 findings after synthesis.*

---

## paper-search confirmation pass

Each load-bearing claim was independently re-checked against the literature (Semantic Scholar / CrossRef / PubMed). All five resolve to real, citable publications:

| # | Claim | Verified citation | Verdict |
|---|---|---|---|
| 1 | **BCR-Net** public WSI + Oncotype DX + Ki67 dataset | Su, Niazi, Tavolara, Niu, Tozbikian, Wesolowski, Gurcan — *"BCR-Net: A deep learning framework to predict breast cancer recurrence from histopathology images"*, **PLOS ONE 2023** (pone.0283562). Abstract confirms 99 anonymized patients, H&E + Ki67 WSI, ODX recurrence, MIL; AUC 0.775 (H&E) / 0.811 (Ki67). | ✅ **Confirmed** |
| 2 | **genefu** recomputes ODX (21-gene) + gene70 (70-gene) | Gendoo et al. — *"Genefu: an R/Bioconductor package for computation of gene expression-based signatures in breast cancer"*, **Bioinformatics 2016**. | ✅ **Confirmed** |
| 3 | Deep learning predicting **Oncotype DX RS from H&E on TAILORx** | Shamai, Cohen, Binenbaum, … Kimmel, Aran — *"Deep learning on histopathological images to predict breast cancer recurrence risk and chemotherapy benefit: a multicentre… study"*, **Lancet Oncology 2026** (10.1016/S1470-2045(25)00727-2); preprint medRxiv 2025.05.15.25327686. Trained on TAILORx n=8,284; **externally validated on six cohorts n=5,497** (Carmel, Haemek, Sheba, UChicago, ABCTB, **TCGA-BRCA**); RS≥26 AUC 0.898. | ✅ **Confirmed** |
| 4 | **Dartmouth** WSI + Oncotype DX recurrence-risk dataset | Matched to *"A multi-model approach integrating whole-slide imaging and clinicopathologic features to predict breast cancer recurrence risk"*, **npj Breast Cancer 2024** (s41523-024-00700-z). | ✅ **Confirmed** |
| 5 | **MSKCC "Orpheus"** multimodal WSI→RS across institutions | Boehm, El Nahhas, Marra, … Shah, Kather — *"Multimodal histopathologic models stratify hormone receptor-positive early breast cancer"*, **Nature Communications 2025** (10.1038/s41467-025-57283-x). Abstract confirms 6,172 cases / 3 institutions / Orpheus / RS>25 AUC **0.89** vs. nomogram 0.73. | ✅ **Confirmed** |

**Note surfaced by the confirmation pass:** the "five-cohort medRxiv set" (item in the comparison table) and the TAILORx Lancet Oncology paper are the **same Shamai/Aran group** — the medRxiv 2025.07.21 preprint and the Lancet paper describe overlapping cohorts, and the external-validation set explicitly *includes public **TCGA-BRCA***. This strengthens the recommendation: TCGA-BRCA is already an established external-validation cohort for WSI→RS models in the published literature.

**Additional relevant works found (not in the deep-research set, worth noting):**
- Romo-Bucheli, Janowczyk, Gilmore, Romero, Madabhushi — tubule-nuclei quantification correlated with Oncotype DX risk categories in ER+ WSI, *Sci Rep 2016* (srep32706) — early WSI↔ODX evidence, no released dataset.
- Q-Plasia OncoReader Breast (QPORB) SABCS 2022 abstract — H&E WSI predicting recurrence in low-ODX cases (198 slides, St James's UK) — institutional, not public.
