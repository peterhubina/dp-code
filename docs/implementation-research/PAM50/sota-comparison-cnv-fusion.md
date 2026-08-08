# State-of-the-art comparison — H&E + arm-level CNV → PAM50, TCGA → CPTAC

Companion to `README.md` (the 50-paper survey, cutoff 2026-07-31). This document places the numbers in
`docs/cnv-wsi-fusion-external-validation.md` against the published literature as of **2026-08-06**.
It does not restate our results; they are the given. It says what is comparable, what is not, and why.

---

## 1. What this is and how far to trust it

**Seven search axes**, run 2026-08-05/06, ~317 queries total:

1. WSI + copy-number fusion for molecular subtype, and specifically whether **arm-level** CNV has ever
   been used as a fusion modality (53 queries).
2. Copy number → breast subtype **without imaging**: broad-vs-focal representations, aneuploidy burden
   as a scalar, shallow/low-pass WGS classifiers, and the assay-replacement framing (36).
3. Untrained averaging / late fusion **beating** trained fusion operators, plus the mechanism
   literature — ensemble diversity, error correlation, modality collapse, shared initialisation (47).
4. Fusion operators in computational pathology (concat / gated / cross-attention / FiLM / co-attention),
   and any head-to-head against a trivial baseline (43).
5. CPTAC-BRCA as an external cohort for PAM50, and any PAM50-specific metric on a never-trained-on
   cohort (36).
6. HER2-enriched failure under cohort transfer, and the proposed remediations — stain normalisation and
   encoder choice (36).
7. Recency sweep, hard window 2026-07-31 → 2026-08-06, widened to 2026-06-01 (66).

**Registries that worked.** Europe PMC REST (search + `fullTextXML`), PubMed eutils, OpenAlex API,
Crossref API, the arXiv API queried directly with `sortBy=submittedDate`, the Semantic Scholar Graph
API (which returned abstracts for paywalled papers that Crossref and OpenAlex both carried as null),
the bioRxiv and medRxiv `details` APIs, and Unpaywall. Direct API access was materially better than
the `paper-search` CLI on every axis.

**Registries that failed, with the failure mode.** Google Scholar — the prior survey's single
highest-yield source for 2026 conference and preprint material — returned rows for roughly the first
ten queries of a session and then `{'google_scholar': 0}` with an **empty errors dict** on every
subsequent query, on three separate axes, and hard-failed with HTTP 429 on a fourth. This is a silent
rate limit indistinguishable from a genuine zero result. `arxiv` and `semantic` *search* through the
CLI showed the same silent mid-session zeroing (direct API and `read` kept working). `biorxiv` and
`pmc` through the CLI reproduced the prior survey's recency-feed failure verbatim — a query for
`PAM50 breast cancer subtype histopathology` returned X-chromosome inactivation in chondrogenesis,
synovial fibroblasts, CAR-T in prostate and curcumin in *C. elegans* — and were bypassed by driving
the details APIs directly. `base` and `ssrn` returned zero rows; `dblp` returned two, both already in
the survey; `zenodo` raised a CLI exception. One trap worth recording: **Europe PMC's
`FIRST_INDEX_DATE` field returns `hitCount = 0` for every query with no error**, which silently
emptied an entire 17-query sweep before it was caught. `FIRST_PDATE` and `CREATION_DATE` both work.

**Publisher walls observed** (statuses actually seen, not assumed): ScienceDirect/Elsevier HTTP 403;
IEEE Xplore HTTP 202 with an empty body; Springer HTTP 303 to `idp.springer.com`, and a 3 KB
JavaScript "Client Challenge" served at HTTP 200 on the CC-BY PDF endpoint; MDPI HTTP 403; Nature HTTP
303 to `idp.nature.com`; AACR HTTP 403; Wiley HTTP 402; BMC a Cloudflare challenge page at HTTP 200.

**Counts.** 253 raw candidates → 231 unique → 13 already in the survey → **20 extracted and
independently verified**. Of the 20: **18 full-text, 2 abstract-only, 0 title-only**; **14 refereed,
6 unrefereed preprints**. Verification (a second pass that re-resolved every URL and DOI and re-read
every quoted number at source) returned **11 `citable`, 9 `citable-with-caveat`, 0 `do-not-cite`**.
Two quoted sentences were marked CONTRADICTED and are corrected at the point of use in §7. One
row — UGES — is full-text only in the sense that the authors' accepted manuscript on GitHub was read;
the IEEE version of record was never reachable.

**A third pass.** After the document was drafted, a reader who had performed neither the extraction
nor the verification re-checked the four load-bearing claims that change what the thesis says: the
Amer Table 1 / Table 2 / Table 3 cells (§2 and §4d, re-extracted from the PDF and confirmed), the
`compare_fusion_ladder` bootstrap constants (§6.13, confirmed at source lines), the two φ values in
`dpcode/cli/analysis.py:335-336` (§6.10, confirmed), and the `weighted_sample: true` sampler chain
behind §6.11 (confirmed through `core_utils.py:418` → `utils.py:65-66,156`). It also re-resolved all
20 URLs and DOIs independently: 20/20 URLs return HTTP 200; the DOI non-200s are IEEE 202, MDPI 403
and `spj.science.org` 403, all of them the publisher behaviours already recorded above.

**What this still is not.** The recency null (§5) is well supported for preprint servers and for
journals indexed by Europe PMC/Crossref/OpenAlex, and poorly supported for IEEE-indexed regional
conference proceedings — the venue class that carries the survey's own P01 (ICMI 2026) and P04
(ICKECS 2026). Google Scholar is the only route that indexes that class well and it was blocked.

---

## 2. Reused from the existing survey

All 50 existing rows were re-triaged against this thread. **One is a fair comparator. Nine are near
comparators that need a caveat carried with them. Forty were set aside.**

### The one fair comparator

**P11 — Amer et al. 2025, arXiv:2509.03408 (UNREFEREED).** WSI as patches *and* as a cell graph, plus
CNV, plus clinical/EHR; TCGA-BRCA only, 977 patients after exclusions; 4-class PAM50 (LumA 53.5%,
LumB 20.6%, Basal-like 18.1%, Her2-enriched 7.8%), labels from Netanely et al. 2016; patient-level;
10-fold CV; macro-AUROC described as "average AUC across classes". This is our label space, our
prediction unit, our internal protocol and our metric, with both a CNV-alone arm and WSI+CNV fusion
arms. It was re-opened at full text and re-verified for this document; §4 uses it in three places.

### Near comparators, with the caveat that must travel with each

| Row | What it gives us | The caveat that cannot be dropped |
|---|---|---|
| **P10** Multimodal CustOmics (PLOS Comp Biol 2025) | The only other TCGA-BRCA paper with H&E and CNV in one model *and* WSI-only / omics-only / fused arms under one protocol | PAM50 class count **never stated**; AUC averaging scheme NR; CNV is never ablated alone on the BRCA task; RNA-seq is in the bag, so 98.7 ± 1.1 is target-leaking under our own rule |
| **P20** Borji 2026 (arXiv, UNREFEREED) | Same cohort pair, same 4 classes, external macro AUC 0.9523 | Patch-indexed at exactly 10,000 patches per class; 201/627 TCGA labels are IHC surrogate; CPTAC label basis NR; version drift |
| **P22** Fernandez-Romero 2026 (Med Biol Eng Comput) | The closest methodological sibling — UNI-2 + CLAM, TCGA → CPTAC — and the strongest independent corroboration of the HER2 collapse | 5-class, slide-level, **macro-F1 only, no AUROC anywhere**; TCGA restricted to flash-frozen |
| **P29** Zhang 2025 (Int J Surg) | The only other internal→external H&E PAM50 AUC *pair*: 0.803 → 0.6515 | Class count NR, averaging scheme NR, slide-level, external figure pools four cohorts |
| **P30** Liu 2024 (Comput Biol Med) | Numerically nearest internal WSI-only figure, 0.875 average ROC-AUC | Slide-level; split type **not stated**; number read from the bioRxiv preprint (ScienceDirect 403) |
| **P32** OmicsFootPrint (Nucleic Acids Res 2024) | The only rows-26–50 paper with copy number as a model input; independent instance of fusion barely beating the strong unimodal arm | **No imaging of any kind**; CNV-only AUC is **NR**; "Overall" is a median of four one-vs-rest AUCs, not a documented macro |
| **P39** Ektefaie 2021 (npj Breast Cancer) | The canonical H&E→PAM50 reference point; site-stratified splits, stronger than ours | Top-1 **accuracy** 0.654 (0.636–0.672), not an AUROC; **no external PAM50 number exists** |
| **P40** Jaber 2020 (Breast Cancer Res) | Patient-level 4-class, RNA-derived labels, Normal-like dropped | Accuracy 67.27%, no macro AUROC; binary Basal AUC 0.8607 is one-vs-rest |
| **P48** hist2RNA 2023 (Cancers) | The only per-class 4-class PAM50 AUROC set at patient level with RNA-derived labels | Two-stage cascade; subtype-stage split **not stated**; external TMA has no RNA, so no external classification metric exists |

### The forty set aside, by category

- **Omics-only, no imaging, no CNV-alone arm** (7): P05 GCOA-Net, P09 HyperCLSA, P17 moBRCA-net,
  P18 MOGONET, P24 GAIN-BRCA, P25 Omran, P43 Qiu. Useful only as the RNA→PAM50 circularity ceiling.
- **Copy number present but bundled with RNA and never ablated** (4): P06 UMMT, P13 Li & Nabavi,
  P15 CustOmics 2023, P19 Ahmad (unsupervised clustering, ARI not AUROC).
- **WSI + RNA fusion ladders — our leaky-modality foils** (6): P01, P02, P03, P04, P08, P12. All
  accuracy-only, none external, none reports a trivial-average baseline.
- **Binary or IHC-surrogate endpoints** (9): P14 Lee, P27 Kurian, P28 Kiraz, P31 Jimenez-Martin,
  P34 Huang, P35 Liu 2022, P41 Couture, P45 Ben Rabah, P49 Zhang MDL-IIA.
- **Wrong modality, wrong organ, or no subtype classifier** (14): P16 Furtney (MRI+IHC), P21 Li (TMA),
  P23 Nateghi (prostate), P26 Tafavvoghi (CPTAC pooled *into* training), P33 MBFusion (no usable
  numbers, ScienceDirect 403), P36/P37 Ming (DCE-MRI), P38 Phan (patch-level split — leakage),
  P42 Kaczmarzyk (ROR-P), P44 TEMI (CRC/GBM), P46 Kunhoth (review), P47 SEQUOIA (no subtype metric),
  P50 Wang 2021 (transcriptome regression), P07 M3FusionNet (re-extracted here; see §3 and §7).

### One correction to the existing survey's record of Amer 2025

The survey and the project's own claim inventory attribute macro-AUC **0.8604 / 0.8616** and accuracy
**69.51** to Amer's **CNV + WSI-image** arm. On a layout-preserving re-extraction of the arXiv PDF
(pdfplumber, page 7), that pair belongs to the **Image + Graph** row — two WSI representations, no
CNV. The CNV + Image row reads **WL 0.8835 / WLB 0.8836** (macro-AUC) and **WL 75.41 / WLB 75.12**
(accuracy). The row-to-modality mapping was confirmed from checkmark column positions rather than
inferred. This makes Amer's two-modality arm materially stronger than the survey records, which is
the honest direction. Treat 0.8835/0.8836 as the like-for-like Amer comparator.

**This cell has been independently re-confirmed.** `arxiv.org/pdf/2509.03408v1` was re-downloaded
and re-extracted with `pdfplumber(layout=True)` by a second reader who did not perform the original
extraction. Table 3's column order is `CNV | WSI-Image | WSI-Graph | Clinical`, and its five rows in
order are Image+Graph (WL **0.8616†**, WLB 0.8604), CNV+Image (WL **0.8835**, WLB **0.8836†**),
CNV+Image+Graph (WL **0.9000†**, WLB 0.8995), CNV+Image+Clinical (WL 0.8976, WLB 0.9008) and
all-four (WL 0.9153, WLB **0.9153†**). Table 1's single-modality figures — CNV 70.25 / **0.8284**,
WSI-Image 66.96 / 0.8080, WSI-Graph 70.23 / 0.8350, Clinical 70.43 / **0.8522** — and the cohort
description (977 patients after exclusions; LumA 53.5%, LumB 20.6%, Basal-like 18.1%,
Her2-enriched 7.8%; labels from Netanely et al. 2016; 10-fold CV; "macro-AUROC (average AUC across
classes)") were confirmed in the same pass.

---

## 3. New papers

**Reading the cells.** *Cls* = class count (4 / 5 / 2 / 10 / NR). *Unit* = prediction unit
(case = case or patient, slide, patch, sample). *Label* = RNA-PAM50 (RNA-derived PAM50), IHC (IHC
surrogate), mixed, other (a non-PAM50 endpoint), NR. *Regime* = INT (internal CV or an internal
hold-out), EXT (a never-trained-on cohort), POOL (cohorts pooled into training). *Ev* = FT
(full-text), AB (abstract-only). *Ref* = R (refereed), **P (UNREFEREED preprint)**. *Ver* = the
independent verification verdict: ✓ citable, ⚠ citable-with-caveat.

| # | Paper | Modalities → endpoint | Cls | Unit | Label | Regime | Headline number, verbatim with metric and split | Ev | Ref | Ver |
|---|---|---|---|---|---|---|---|---|---|---|
| N1 | **Amer 2025**, arXiv:2509.03408 | WSI (patch + graph) + CNV + clinical → PAM50 | 4 | case | RNA-PAM50 (Netanely 2016) | INT | CNV-alone macro-AUC **0.8284**; CNV+Image **0.8835/0.8836**; four-modality **0.9153**; all 10-fold CV, TCGA-BRCA n=977 | FT | P | ⚠ |
| N2 | **Multimodal CustOmics**, PLOS Comp Biol 21(6):e1013012 | WSI + RNA + CNV + methylation → PAM50 | **NR** | case | RNA-PAM50 (caller NR) | INT | WSI-only **73.2 ± 3.1** AUC %; WSI+multi-omics **98.7 ± 1.1**; stratified 5-fold, TCGA-BRCA | FT | R | ✓ |
| N3 | **PathLUPI**, arXiv:2506.19681 | WSI at inference; transcriptomics as privileged info at training → subtype | 4 | case | INT RNA-PAM50 (TCGAbiolinks); **EXT NR** | INT + EXT | BRCA internal CLAM AUC **0.867** (0.866–0.868) → external **0.706** (0.705–0.706); PathLUPI **0.876** → **0.727**; 5-fold, external = private Center-2 n=2,045 | FT | P | ⚠ |
| N4 | **UGES**, IEEE TCBBIO 22(6):3000-3016 | mutation + CNA + methylation (no imaging) → PAM50 | 4 | case | RNA-PAM50 (caller NR) | POOL (+EXT) | Overall AUC **0.963** on a pooled TCGA+METABRIC test split; **CNA-only 0.877**; genuine cross-cohort **0.735** (METABRIC→TCGA) and **0.846** (TCGA→METABRIC) | FT¹ | R | ⚠ |
| N5 | **Belay Ascent**, Cancers 18(8):1277 | LP-WGS of CSF → arm-level aneuploidy + gene CNV calls | n/a | specimen | other | EXT-like | Tissue equivalence: **25/25** arm-level events, 100% PPA; CSF validation **7/9**, 78% PPA; 243 production cases | FT | R | ✓ |
| N6 | **Wissel 2023**, Cell Rep Methods 3:100461 | 7 molecular/clinical modalities, **no imaging** → overall survival | n/a | case | other | INT | Late (Mean) Antolini's C **0.631 (0.0056)** → **0.627 (0.0054)**; Late (MoE) **0.599** → **0.564**; 17 TCGA datasets, 5×5-fold = 25 splits | FT | R | ⚠ |
| N7 | **Papagoras 2025**, bioRxiv 10.64898/2025.12.19.695372 | WSI (UNI-2) → PAM50 | 4 | **slide** | **mixed** (TCGA/CPTAC PAM50 + 80 IHC Warwick slides) | **POOL** | Attention-MIL balanced accuracy **0.84**, macro F1 **0.83**, n=287 WSI hold-out from the pooled set. **"AUROC" occurs zero times** | FT | P | ⚠ |
| N8 | **CLOVER**, bioRxiv 10.1101/2025.01.12.632280 | WSI aligned to RNA+methylation+CNV by contrastive pretraining; slide-only at inference → PAM50 | 4 | slide | RNA-PAM50 (caller NR) | INT | k=10 macro-AUROC **87.2 ± 2.2** %, TANGLE 86.8 ± 2.6, UNI+ABMIL 76.6 ± 4.3; 5-fold × 50 trials, TCGA-BRCA 610 patients | FT | P | ✓ |
| N9 | **Borji 2026**, arXiv:2604.01798v4 | WSI (ResNet-18, frozen) → PAM50 | 4 | **patch (contested)** | **mixed** (426/627 RNA, 201/627 IHC; CPTAC NR) | INT + EXT | External CPTAC Macro Avg AUC **0.9523**, F1 **0.7995**; HER2 recall **0.7670** — every CPTAC row indexed at **10,000 patches per class** | FT | P | ⚠ |
| N10 | **MRSVM** (Nakach 2024), Multimed Tools Appl 84:32671-32703 | WSI + CNV + gene expression + clinical → subtype | **5** (incl. Normal-like) | case | **NR** | INT | Best **88.07% accuracy** (early fusion, gene expression + CNV — **no WSI in the winning arm**); WSI alone **60.32%**; stratified 5-fold. **No AUC anywhere** | FT² | R | ⚠ |
| N11 | **CopyClust**, Sci Rep 14:11861 | DNA copy number only → **IntClust** (not PAM50) | 10 | case | other (iC10) | EXT | METABRIC→TCGA SNP recall **0.811**, balanced accuracy **0.893**; TCGA WES **0.786** / **0.879**; micro-averaged. **No AUROC anywhere** | FT | R | ✓ |
| N12 | **Killcoyne 2020**, Nat Med 26:1726-1732 | 0.4× sWGS copy number → progression to HGD/IMC (Barrett's) | 2 | sample | other | EXT | Encoding = "**589 5Mb windows and 44 chromosome arms**"; only AUC in text **0.89**, and it pools discovery + validation (n=164) | FT | R | ✓ |
| N13 | **Sucre 2025**, CSBJ 27:4505-4516 | WSI + gene-level CNV + 4 more → overall survival | n/a | case | other | INT | Late integration OM+CL C-index **0.740 ± 0.002**; CNV alone **0.49 ± 0.031**; MCAT **0.658 ± 0.051** (Table 3) vs **0.685 ± 0.051** (text) | FT | R | ✓ |
| N14 | **Wagner 2026**, npj Precis Oncol 10:198 | WSI (UNI2/Virchow2 + CLAM) → endometrial molecular class | 4 | slide | other | POOL discovery + real EXT | Internal Virchow2+CLAM macro-AUC **0.860** → external UNI2+CLAM **0.780**; **dMMR external recall 0.14 at AUC 0.759 (0.718–0.800)**, n=160 patients | FT | R | ✓ |
| N15 | **Xia/Perou 2019**, Nat Commun 10:5666 | 536 segment-level CNA scores, no imaging → intrinsic subtype | 4, as **four separate binaries** | case | RNA-PAM50 (nearest centroid) | EXT | **One-vs-rest binary** AUC: Basal ">0.9", HER2-E ">0.82", LumA **0.82**, LumB **0.76** (METABRIC). Prosigna ROR **0.81**, OncotypeDX **0.79**, MammaPrint **0.87** | FT | R | ✓ |
| N16 | **Pan 2019**, Mol Genet Genomics 294(1):95-110 | CNV only (20,649 probes) → subtype | 4 | sample | **NR** | EXT | METABRIC→TCGA independent test **MCC 0.492**, accuracy **0.647**, weighted-F1 **0.653**. **No AUROC anywhere** | FT³ | R | ✓ |
| N17 | **CNApp**, eLife 9:e50267 | Segmented copy number → colorectal CMS and MSI | 4 (CMS) / 2 (MSI) | case | other | EXT (MSI threshold only) | Burden scalar alone: **MSI-vs-MSS AUC 0.917**, accuracy 82.2%, validated at **81%** on n=147. **CMS: no burden-alone metric exists** — only a t-test | FT | R | ✓ |
| N18 | **UniCat**, arXiv:2310.18812 (NeurIPS 2023 UniReps workshop) | RGB + NIR + TIR → re-identification | n/a | image | other | INT | RGBNT100 ViT-B mAP: Fusion-avg **76.1 ± 0.3**, Fusion-concat **75.9 ± 0.7**, UniCat **81.3 ± 0.9**. Appendix C same-modality, shared init: UniCat wins all four | FT | P | ✓ |
| N19 | **M3FusionNet**, Comput Biol Chem 124:109228 | WSI + RNA + miRNA + proteomics + clinical → 5 biomarkers incl. PAM50 | **NR** | **NR** | **NR** | INT + EXT (E1) + POOL (E3) | TCGA "in-distribution" **PAM50 macro-AUC 0.96**, split type NR; **E1 Overall AUC 0.82 pooled across five endpoints**; E3 (TCGA+CPTAC combined) 0.90 | **AB** | R | ⚠ |
| N20 | **Liu 2022**, IRBM 43(1):62-74 | "gene modality" + "image modality" → molecular subtype | **NR** | **NR** | **NR** | INT | **88.07% accuracy**, 10 times 10-fold CV; "average AUC value obtained was **0.9427**" (mean of per-subtype AUCs). Abstract never says CNV, PAM50, or whole-slide | **AB** | R | ⚠ |

¹ IEEE Xplore returned HTTP 202 with an empty body; the text read is the authors' accepted manuscript
on GitHub, whose title, nine-author list and abstract match the published record exactly. Not the
version of record. ² Verified through WebFetch's extraction of the Springer HTML rather than raw
parsed text — a weaker evidence level than the other full-text rows. ³ `link.springer.com` serves a
3 KB JavaScript bot challenge; the body came from the `rd.springer.com` mirror.

---

## 4. The comparison, number by number

### (a) WSI-only, external CPTAC macro AUROC 0.847 [0.791, 0.895]

**Closest published comparator by protocol: PathLUPI (arXiv:2506.19681, UNREFEREED).** WSI-only at
inference (transcriptomics is privileged information used during training and absent at test time),
4-class PAM50 with the classes enumerated as Luminal A / Luminal B / HER2-enriched / Basal-like,
case-level, CONCH encoder at 512×512 px / 20× / 512-dim, five-fold CV on TCGA. Its **CLAM baseline
scores AUC 0.867 (0.866–0.868) internally on 505 TCGA cases and 0.706 (0.705–0.706) on the external
Center-2 cohort of 2,045 patients** — a drop of 0.161, against our 0.887 → 0.847, a drop of 0.040.
The comparison is legitimate on class count, prediction unit and metric family, and illegitimate on
three axes that must be stated together: the encoder differs (CONCH 512-dim at 512 px against UNI2-h
1536-dim at 256 px), the external cohort differs (a private institutional cohort against CPTAC), and
**the external cohort's label basis is NR** — the paper says only "Molecular Subtype / Luminal A,
Luminal B, HER2-enriched, Basal-like / Center-2", with no assay named. A 2,045-patient RNA-derived
PAM50 assay in one hospital is implausible; an IHC surrogate mapped onto the same four names is more
likely, and would mean their external number is not a PAM50 number at all. Its AUC averaging scheme
is also never defined — "macro" and "one-vs-rest" occur zero times in the paper.

**The number a reviewer will raise: Borji 2026, external macro AUC 0.9523.** Same cohort pair, same
four classes, Normal-like dropped. The comparison is **not legitimate** and the reasons are
checkable rather than rhetorical. Its Table 1 has exactly one sample-size column, headed *Number of
patches*, reading **10000 for every CPTAC class row** against a cohort composition of LumA 51 /
LumB 28 / HER2 12 / Basal 31 slides — so the external metrics are computed on a class-balanced patch
set, not at natural prevalence. Each CPTAC per-class *Accuracy* equals that row's *Recall*
digit-for-digit, which is what class-balanced one-vs-rest evaluation looks like. No slide-level or
case-level denominator accompanies any metric anywhere; the string "case-level" occurs zero times in
the paper. 201 of its 627 TCGA training labels are IHC surrogates and the authors state the results
"should be regarded as predicting IHC-aligned intrinsic subtypes"; the CPTAC label basis is never
stated at all. It is unrefereed, and the arXiv v4 listing abstract (F1 0.8812 / AUC 0.9841 internal,
0.7952 / 0.9512 external, "a custom CNN head") disagrees with the v4 PDF abstract (0.8964 / 0.9865,
0.7995 / 0.9523, "a fully connected head") — and the PDF's own Table 3 says "a custom CNN classifier",
so the PDF contradicts itself. Its external **HER2-enriched recall of 0.7670** will be raised against
our 0/14; it is a recall over 10,000 balanced patches, not over 14 cases.

**Directional comparators.** Fernandez-Romero 2026 (survey P22) runs UNI-2 + CLAM on TCGA → CPTAC and
reports 5-class **macro-F1 0.575 ± 0.061 internal → 0.325 external**; not an AUROC, not four classes,
slide-level, and TCGA restricted to flash-frozen material. Zhang 2025 (P29) reports **0.803 → 0.6515**
with class count and averaging scheme NR across four pooled external cohorts. Both degrade far more
than we do; neither number can be differenced against ours.

**Papagoras et al. 2025 is the closest thing in existence to our setup and is not an external
validation.** TCGA-BRCA + CPTAC-BRCA + Warwick, four classes with Normal-like dropped, frozen UNI-2
at 1536-dim, 20×, non-overlapping tiles — and TCGA class counts (LumA 502, LumB 221, HER2 76, BL 172)
within a slide or two of our own manifest. It then pools all three cohorts into one 80/20 split
**stratified by subtype *and* by data source**, which puts CPTAC on both sides. Its 80 Warwick slides
are IHC HER2-positive biopsies pooled into the same HER2 class as expression-derived HER2-enriched
cases, so its HER2 F1 of 0.88 is computed on a label-mixed class. And it reports **no AUROC at all** —
"AUROC" occurs zero times; "AUC" occurs three times, all as figure-caption labels with no value
attached. Its function here is to establish our negative rather than to contest it.

**The plain statement: no published case-level, 4-class, RNA-derived-PAM50, TCGA → CPTAC macro AUROC
exists to place against 0.847.** Four papers evaluate a PAM50 endpoint on a never-trained-on cohort
(Borji, Fernandez-Romero, Zhang, PathLUPI) and not one of them is simultaneously case-level, AUROC,
4-class and RNA-labelled on both sides.

### (b) CNV-only — internal 0.862–0.872 (published headline 0.866 ± 0.003) and external 0.888 [0.835, 0.933]

**Amer et al. 2025, against our CNV-only arm.** CNV-alone macro-AUC **0.8284**, accuracy 70.25, on
TCGA-BRCA n=977, 4-class PAM50 with Netanely 2016 labels, patient-level, 10-fold CV. The comparison
is legitimate on class count, label basis and prediction unit, and illegitimate on regime: their
0.8284 is internal 10-fold and our 0.888 is a held-out cohort, so the numbers sit on opposite sides
of a transfer boundary and cannot be differenced. The defensible statement is that their CNV arm is
0.8284 internal against our 0.862–0.872 internal, with our external 0.888 having no counterpart in
their work. Their CNV representation is discrete gene-level GISTIC-style calls in {−2,−1,0,+1,+2} fed
to an 8192–2048 Self-Normalizing Network — roughly 500× our feature count and at the opposite end of
the assay-cost spectrum from 39 arm medians. Their **clinical modality is the strongest single arm**
(accuracy 70.43%, macro-AUC 0.8522), and the EHR vector's contents are unspecified; if it encodes
ER/PR/HER2 the comparison is circular, and that must be stated whenever their fusion number is quoted.
Amer's Table 3 also reports no standard deviations across the 10 folds, so none of their deltas is
testable.

**UGES (IEEE TCBBIO 2025) is the number that will be thrown at 0.888, and it defuses cleanly.** Its
headline is overall AUC **0.963** on 4-class PAM50 from DNA alone — but METABRIC is *pooled into
training* (800 balanced training samples drawn from the combined TCGA+METABRIC pool, 1,265 held out),
so 0.963 is a within-pool test split, not an external cohort. Its genuine cross-cohort arms, reported
in a supplementary figure and absent from the abstract, are **0.735 (METABRIC→TCGA) and 0.846
(TCGA→METABRIC)** — both below our 0.888, which is the comparison that matters. Its feature set is
50,831 features (16,770 mutations + 25,594 gene-level CNAs + 8,467 methylation probes), so it is not
an sWGS-reachable assay; the paper's ctDNA framing is aspirational and its Limitations concede that
chromosomal-scale structural change was not modelled. **The one genuinely apples-to-apples cell is its
CNA-only ablation: overall AUC 0.877 from 25,594 gene-level CNAs**, against our 0.866 ± 0.003 internal
from 39 arms. That is the strongest single piece of published evidence that coarsening copy number to
arm scale costs very little, and it — not 0.963 — is the number to quote. Caveat it three ways:
different cohorts, a pooled rather than external split, and cross-cohort PAM50 label harmonisation
between TCGA and METABRIC that the paper never describes. Provenance caveat: IEEE Xplore returned
HTTP 202, so everything beyond the PubMed abstract comes from the authors' accepted manuscript.

**Xia/Perou 2019 (Nat Commun) is the real prior art and its metric forbids the obvious comparison.**
CNA-only, TCGA-trained, validated on METABRIC across *different* copy-number and expression platforms
(TCGA GISTIC2 gene-level → METABRIC CBS segments mapped by the GISTIC2 extreme method; mRNA-seq →
microarray) — structurally our TCGA-SNP6 → CPTAC-WGS design, and direct support for our per-arm
r = 0.960 platform check. But its subtype numbers are **four independent one-vs-rest binary Elastic
Net models**: Basal-like ">0.9", HER2-Enriched ">0.82" (both printed as inequalities, exact values in
a supplement not opened), Luminal A **0.82**, Luminal B **0.76** on METABRIC. There is no multiclass
model, no argmax decision, no macro AUROC and no confusion matrix anywhere — "macro" occurs zero
times in the paper. **A per-subtype one-vs-rest AUC and a 4-class macro AUROC are not the same
quantity and must never share a column.** What Xia/Perou does supply, uniquely, is the only
quantitative precedent for our assay-replacement framing: Prosigna ROR predicted from CNA at METABRIC
test AUC **0.81**, OncotypeDX **0.79**, MammaPrint **0.87**, all as top-third-versus-bottom-two-thirds
binary targets on research-based implementations. Cite that in the motivation, not in the results.
Also worth stating: they use 536 predefined segment-level scores in which whole chromosome arms are
*eligible* features, and **no arms-only model is evaluated anywhere in the paper**.

**Pan et al. 2019 is the closest design and reports the wrong currency.** METABRIC (n=1608) → TCGA
(n=499) with a genuinely held-out independent test, 4-class, and *exactly* our class names. Test-set
**MCC 0.492, overall accuracy 0.647, weighted-average F1 0.653**; ten-fold CV on METABRIC gave MCC
0.515 / accuracy 0.675. It reports **no AUROC of any kind** — case-sensitive counts of "AUC", "ROC"
and "AUROC" over the full text return zero each — so nothing in it converts to 0.888, and its
accuracy 0.647 on a set that is 45% LumA is not our balanced accuracy 0.716. Its label basis is
**NR**: the Methods never say how a sample was assigned to a subtype, and the only subtype definition
in the paper is a *five*-category IHC/Ki-67 scheme in the Introduction that does not match the four
classes modelled. Its per-class CV accuracies — LumA 0.864, Basal 0.684, LumB 0.507, **Her2 0.429** —
are the mirror image of our CNV arm, which recovers 12/14 Her2 externally. Use it as the counterweight
to inflated single-modality CNV claims: 8,715 of 20,649 features buys MCC 0.492 across cohorts.

**CopyClust (Sci Rep 2024) transports and predicts the wrong taxonomy.** METABRIC-trained,
TCGA-validated on **two assay platforms** — SNP array recall 0.811 / balanced accuracy 0.893, WES
0.786 / 0.879, micro-averaged over 10 IntClust classes. That is the strongest published evidence that
a copy-number-only breast classifier survives both a cohort change and a platform change. It reports
no AUROC, its endpoint is IntClust rather than PAM50, and "PAM50" occurs exactly once in the paper, as
Introduction background. The often-quoted sentence "the algorithm does not predict PAM50 intrinsic
subtypes" **is not in the article body** — the point must be argued from the design, not quoted. Its
478 regions are also never characterised in base pairs or relative to chromosome arms ("arm" occurs
zero times), and there is no sWGS, coverage or cost claim anywhere.

**Nothing published gives a 4-class macro AUROC for a CNV-only breast subtype model on an external
cohort.** That is the finding. The two CNV-only papers with a genuine external cohort report MCC
(Pan) and one-vs-rest binary AUC (Xia/Perou); the one that reports a macro-like AUROC (UGES) pools
its second cohort into training.

**Arm-level, specifically: no work anywhere uses chromosome-arm medians as the feature set for a
subtype endpoint.** The negative rests on named, reproducible searches rather than on absence of
effort: OpenAlex title+abstract `(arm-level copy number) OR (chromosome arm) AND (deep learning) AND
(histology)` → 8 records, none relevant; `(chromosome arm) AND (whole slide image)` → 12, none
relevant; `(arm-level copy number) AND (classification)` → 127, not one with imaging;
`(shallow whole genome sequencing) AND (histopathology) AND (deep learning)` → **count 0**; the arXiv
full-record query `all:"copy number" AND all:"whole slide"` returns **five papers in the entire
archive**; `all:"copy number" AND all:"histopathology"` returns **two**; PubMed tiab
`(CNV OR CNA OR SCNA) AND ("whole slide image(s)")` returns **ten records in all of PubMed**, none a
subtype-endpoint fusion. Every WSI+CNV work found uses gene-level or segment-level features. The
intersection is genuinely almost empty.

**Aneuploidy burden alone at 0.685 has no published counterpart.** Not one paper reports how well a
scalar genome-instability measure alone discriminates PAM50 classes. The nearest analogue in any
disease is CNApp (eLife 2020), which supplies the only peer-reviewed formal definition of an
arm-scale burden scalar — BCS = the sum of amplitude weights over *broad* events, where "broad" is
defined as ≥50% of a chromosome arm or ≥90% of a chromosome — and shows it carries real signal on a
**binary** endpoint (MSI vs MSS, AUC **0.917**, accuracy 82.2%, validated at 81% on an independent
n=147 cohort). For its **multiclass** endpoint it reports **no burden-alone metric at all**: BCS
versus CMS is only a significance test (p ≤ 0.0001, Student's t-test), and the only CMS classifier is
built on four *arm regions* (13q, 17p, 18, 20q) at 55% four-class accuracy. So the burden-versus-
pattern distinction is a published concern and our 0.685 is the first number attached to it for
PAM50 — which also means it cannot be sanity-checked against anyone.

### (c) The equal-weight probability mean — external 0.909 [0.858, 0.948], internal 0.922–0.926

**There is no external comparator. None.** No published multimodal PAM50 model reports a
PAM50-specific fusion number on a never-trained-on cohort (see §5). The only candidate, M3FusionNet,
reports its cross-cohort protocol E1 as a single "Overall AUC 0.82" **pooled across five heterogeneous
endpoints** (ER, PR, HER2, PAM50 classification and MKI67 regression), with no PAM50-specific external
value anywhere in the abstract — and its stronger E3 number of 0.90 is explicitly "the combined
TCGA + CPTAC model", so CPTAC is inside its training data. Whether a PAM50-only E1 figure exists in a
table could not be determined (§7).

**Internal, the fair comparator is again Amer 2025.** Their CNV + WSI-image fusion reaches macro-AUC
**0.8835 (WL) / 0.8836 (WLB)** and CNV + Image + Graph reaches **0.9000**; the four-modality headline
is **0.9153**. Against our internal probability-mean 0.9259 those are legitimate on class count, unit,
label family and regime — this is the one place in the literature where a like-for-like internal
comparison is available. Two caveats. Their four-modality 0.9153 includes a clinical/EHR branch whose
top attributions the survey records as "HER2 IHC score, ER/PR Status, and Fraction Genome Altered",
i.e. the IHC surrogate for the label being predicted; quoting 0.9153 as *the* comparator sets our
2-modality fusion against a 4-stream model with a near-leaky branch. And their four-checkmark row is
a four-*model* ensemble (image and graph are two separate WSI models), not a three-modality one.

**Multimodal CustOmics is the second WSI+CNV PAM50 paper and its numbers are target-leaking under our
own rule.** TCGA-BRCA, stratified 5-fold CV, 60-20-20, patient-level: WSI-only CustOmics **73.2 ± 3.1**
AUC %, multi-omics **98.3 ± 1.0**, WSI + multi-omics **98.7 ± 1.1**. Every BRCA arm above the WSI-only
73.2 has RNA-seq in the bag, and PAM50 labels are computed from that expression matrix, so the 98.7 is
the signature of an RNA branch reading its own label — which is precisely why our thread chose copy
number. Three further blockers on any numeric comparison: **the number of PAM50 classes is never
stated** (grepping the full text for Basal / Luminal / LumA / Normal-like returns zero hits, and
"PAM50" appears exactly once in the body), the AUC averaging scheme is NR, and its **CNV-alone
figure of 75.1 ± 2.7 belongs to the pan-cancer task, not to BRCA** — the S3 Table caption says so
explicitly. Its useful contribution is the fusion-gain *shape*: +0.4 points over multi-omics alone,
inside one fold standard deviation. Same null as ours, cited as a null.

**Two more WSI+CNV subtype papers exist and neither is comparable.** MRSVM (Nakach et al. 2024,
Multimedia Tools Appl) fuses CNV + clinical + gene expression + WSI on 1,031 TCGA-BRCA patients — but
it is **5-class including Normal-like**, its metrics are accuracy / recall / precision / F1 with **no
AUC anywhere**, its label basis is NR, its CNV granularity and feature count are NR, and its best
result (88.07% accuracy, early fusion) comes from **gene expression + CNV with no WSI in the winning
arm** while WSI alone scores 60.32%. Its ensemble combines members by **majority vote**, not by
averaging probabilities — a vote-based untrained combiner, not a probability-mean analogue. CLOVER
(bioRxiv 2025) uses CNV as one of three contrastive pretraining targets with slide-only inference, so
it is not fusion at test time; its k=10 macro-AUROC of 87.2 ± 2.2 % is internal 5-fold, few-shot
prototype-based and slide-level. Its per-omics silhouette ranking against PAM50 — RNA-seq 0.058 >
**CNV 0.039** > methylation 0.033 — is independent corroboration that copy number carries real but
sub-RNA PAM50 structure.

**Liu et al. IRBM 2022 is the oldest structural antecedent of our probability mean and cannot be
read.** Its abstract states "we fuse the output of the two feature networks based on the idea of
weighted linear aggregation. Finally, the fused features are used to predict breast cancer subtypes",
reporting 88.07% accuracy over 10 times 10-fold CV and "average AUC value obtained was 0.9427". The
abstract names its modalities only as "gene modality" and "image modality" and contains **no
occurrence of copy number, CNV, gene expression, whole slide or PAM50**; the WSI+CNV+PAM50 attribution
comes entirely from one sentence in Amer's related-work section. Its own wording ("the fused
*features*") points at feature-level rather than probability-level combination, so it may not be an
antecedent of our mean at all. ScienceDirect returns HTTP 403 and Unpaywall reports the paper closed
with no repository copy. Class count, prediction unit and label basis are permanently unverified.

### (d) The operator ladder and its untrained-mean baseline

**There is no published head-to-head of a learned fusion operator against an equal-weight probability
mean in WSI + omics computational pathology.** Roughly 40 queries across nine registries did not close
it. Every "trivial baseline" in that literature is either a *unimodal* model or a *feature-level*
concatenation or Kronecker product — never an average of two independently trained models' output
probabilities.

**Amer's own Table 3 contains an unremarked instance of our result.** In the CNV + Image + Clinical
row the untrained **"Simple Ensemble" scores 0.9074 and carries the best-performance dagger**, above
both of the authors' trained Weighted-Logits operators (WL 0.8976, WLB 0.9008); in the all-modality
row it **ties them exactly at 0.9153**. The text concedes it: "Regarding macro-AUC, our method
surpasses SOTA in nearly all cases, except for one combination—CNV, image, and clinical data—where it
is the second best performance." One hard limit on how far this can be pushed: **"Simple Ensemble" is
never defined in the paper.** It appears only in the two table captions as "SE: Simple Ensemble (Tang
et al., 2024)" and in one Related Work sentence. Whether it averages logits or probabilities, and over
what, is traceable only to FusionBench (Tang et al. 2024), which was not opened. Until it is, write
"an untrained ensemble baseline", not "a probability mean". Note also that on *accuracy* Amer's
operators win every row including that one, so the SE advantage is macro-AUC-specific.

**Independently re-confirmed** from a fresh `pdfplumber(layout=True)` extraction of
`arxiv.org/pdf/2509.03408v1` by a reader who did not do the original extraction: Table 3's
CNV+Image+Clinical row reads `WE 0.8965 | SE 0.9074† | MP 0.8873 | ML 0.8912 | IF 0.8429 |
T 0.8592 | WL 0.8976 | WLB 0.9008`, with the dagger — defined in the caption as "Best performance is
bolded and marked with '†'" — on SE. The all-four row reads `WE 0.8978 | SE 0.9153 | MP 0.8889 |
ML 0.9006 | IF 0.8369 | T 0.8541 | WL 0.9153 | WLB 0.9153†`. The Table 2 accuracy counterpart of the
CNV+Image+Clinical row is `WE 74.10 | SE 76.23 | MP 74.73 | ML 75.10 | IF 72.08 | T 73.49 |
WL 76.88† | WLB 76.78`, which is where the macro-AUC-specific caveat comes from. The caption's only
gloss on SE is "SE: Simple Ensemble (Tang et al., 2024)"; the paper contains no other definition.

**The strongest published precedent for the ordering is Wissel et al. 2023 (Cell Reports Methods), and
it has no imaging.** Twelve integration methods on 17 TCGA datasets, seven modalities, 5×5-fold =
25 test splits per dataset, endpoint overall survival. **Late (Mean)** — an untrained mean pooling of
per-modality partial-hazard predictions — is the *only* method of twelve that does not degrade
significantly when modalities are added (Antolini's C **0.631 (0.0056) → 0.627 (0.0054)**), while
**Late (MoE)** — a *learned gate* over exactly the same per-modality predictions — is the **worst of
all twelve in both settings (0.599 (0.0056) → 0.564 (0.0054))** and degrades most. That is
probability-mean-beats-gated, independently, on multi-omics survival. Three caveats: no WSI modality
anywhere; the endpoint is a survival C-index and an Integrated Brier Score, so none of its numbers is
comparable to any AUROC; and their untrained-mean advantage is a *non-degradation* result, not an
outright win (BlockForest 0.637 edges Late (Mean) 0.631 in the Clinical+GEX setting), whereas our
probability mean 0.9259 beats every trained operator outright.

**Sucre et al. 2025 (CSBJ) is the closest design on the modality side and supports only half the
claim.** TCGA-BRCA, H&E WSI plus gene-level CNV among six modalities, one protocol, three re-run
published benchmarks. Its headline direction matches ours — "late fusion models consistently
outperformed early fusion approaches and late and intermediate benchmark methods", and its re-runs of
MCAT / PORPOISE / MGCT collapse from training C-indices above 0.936 to test C-indices of 0.658 / 0.546
/ 0.552. **But its "late integration" is not a probability mean**: it concatenates the unimodal models'
output *logits* and passes them to a secondary fusion network — a learned stacker, closer to our
`stack_wsi_cnv` arm than to our mean, and our stacker did not beat the mean either. So it corroborates
"train the modalities separately" and does *not* corroborate "the untrained average beats trained
operators"; §8 must say which of the two it supports. Two further usable facts: its CNV-alone arm is
the worst omics modality at C-index **0.49 ± 0.031** (at chance), which is a striking contrast with
our CNV arm being the *stronger* unimodal arm externally and is best explained by endpoint — copy
number predicts subtype far better than it predicts survival. And it reports MCAT at 0.658 ± 0.051 in
Table 3 against 0.685 ± 0.051 in its own Results text; quote whichever, say which, do not average them.

**UniCat (NeurIPS 2023 UniReps workshop, UNREFEREED) is the single most valuable row for our open
warm-start confound.** Independently trained unimodal backbones fused only at inference beat jointly
trained fusion-by-averaging *and* fusion-by-concatenation on RGBNT100 (ViT-B mAP: Fusion-avg
76.1 ± 0.3, Fusion-concat 75.9 ± 0.7, UniCat 81.3 ± 0.9). More importantly, **Appendix C runs the
control our ladder lacks**: a *same-modality* two-backbone ensemble in which both backbones start
from the **same ImageNet checkpoint** ("we use ImageNet pre-trained weights. The classifiers and
bottlenecks are randomly instantiated"), varying only whether the loss is global or local. Independent
training still wins on all four tasks — RGB 60.4 vs 58.4/57.7, NIR 50.7 vs 48.9/48.6, TIR 49.7 vs
48.6/47.4, Market1501 89.1 vs 87.9/87.6 — and the authors state it: "in all unimodal tasks, training
the ensemble using a global loss (Fusion-avg/concat) resulted in a lower performance than would
otherwise have been obtained if each member were trained independently". That holds shared
initialisation fixed and still produces the diversity collapse, which is direct published support for
reading our φ = 0.656 versus 0.193 as a *joint-training* effect rather than a *shared-warm-start*
effect. It does not eliminate the confound — it shows the effect survives shared init, not that
shared init contributes nothing — so the `--no_warm_start` arm remains worth running, and UniCat's
Appendix C is the template. Three limits: the domain is person and vehicle re-identification with
mAP/Rank-1 metrics, so no number transfers; their Fusion-avg averages **embeddings**, not output
probabilities, so the operator that loses in their paper is closer to our concat/gated arms than to
our mean; and their result **reverses on RGBNT201**, where jointly trained Fusion-concat beats UniCat
on both backbones (63.0 vs 38.1 mAP on ResNet-50). Any citation omitting that reversal is selective.

**One caution on the general-ML literature.** Do not cite Wang, Tran & Feiszli (CVPR 2020) as
evidence that averaging beats learned fusion: full text confirms their "late fusion" is late fusion
*by concatenation*, a trained operator, and they never report an untrained probability average
anywhere. The same caution applies to Wu 2022, Peng 2022 and the PLoS One 2026 MGMT study — those
support "best unimodal beats jointly trained multimodal", which is a different and weaker claim.

**Learned operators losing to a unimodal arm is published five times over, and always in a table
rather than an abstract**: SurvPath (a unimodal transcriptomics MLP at 0.599 beats all eight
multimodal baselines), MOTCat ("most multimodal methods are inferior to the unimodal model of genomics
in UCEC dataset", section 4.2), HEALNet (on UCEC the unimodal WSI arm 0.630 beats all six multimodal
models including their own 0.626), MMP (MCAT 0.610 below unimodal Pathways 0.614 in the same table,
unremarked) and MOAD-FNet (omics-only SNN 0.726 F1-macro above MCAT 0.402 and SurvPath 0.424). All are
survival c-indices except MOAD-FNet's brain-tumour macro-F1, so none may enter a PAM50 comparison
table — they belong in an operator-provenance discussion marked as survival work.

---

## 5. Claims that survive

### The central novelty claim survives only in its qualified form

**Surviving wording: "no published *multimodal* PAM50 model reports an external, never-trained-on,
PAM50-specific evaluation."** The word *multimodal* is load-bearing and must stay.

*What was done to break it.* Axis 5 (36 queries: Europe PMC, PubMed, OpenAlex, Crossref, arXiv,
WebSearch, plus full-text reads of every candidate) plus axis 7's hard-window recency sweep. The sweep
enumerated **17,529 bioRxiv and medRxiv preprints by date** (2,883 in the six-day window since the
prior survey's cutoff, 14,646 in the widened window back to 2026-06-01), keyword-screened every title
and abstract, and found zero on-topic hits; Europe PMC cursorMark enumeration of five broad queries
across two date fields returned 893 unique records in the hard window of which two cleared a relevance
floor and neither is a breast-subtype paper; an arXiv `abs:"breast cancer"` listing sorted by
submission date returns nine papers from 2026-07-25 onward, of which only two are pathology and
neither has a molecular-subtype endpoint. Crossref and OpenAlex hard-window filters independently
caught the two journal items Europe PMC had not yet indexed.

*It held, and it also broke in its unqualified form.* Four papers do report a PAM50-or-PAM50-derived
metric on a cohort they did not train on — **Borji 2026** (TCGA→CPTAC, 4-class, external macro AUC
0.9523), **Fernandez-Romero 2026** (TCGA→CPTAC, 5-class, macro-F1 0.325–0.379 external), **Zhang 2025**
(TCGA→four cohorts, PAM50 AUC 0.6515 external) and **PathLUPI** (TCGA→private Center-2 n=2,045,
AUC 0.727). **Every one of them is unimodal at inference.** The unqualified claim "no published PAM50
model has an external evaluation" is false and must never be written; cite Zhang 2025 and
Fernandez-Romero 2026 explicitly as the unimodal external precedents so a reviewer does not find them
first.

*The paper that would have pre-empted it.* Its shape: a model taking H&E plus at least one molecular
modality at inference, trained on one cohort, evaluated on a second it never saw, reporting an AUROC
(or any metric) **computed on the PAM50 endpoint alone** rather than pooled with other endpoints, at
a stated prediction unit and a stated class count. **M3FusionNet is one full-text retrieval away from
being that paper** and could not be retrieved (§7). Its abstract gives an internal TCGA PAM50
macro-AUC of 0.96 with split type NR, a cross-cohort E1 "Overall AUC 0.82" pooled across five
endpoints, and an E3 figure of 0.90 that explicitly pools CPTAC into training. On the evidence
obtainable the claim is not broken; it is also not verified against this paper, and the thesis should
say so rather than assert the negative.

*One correction our positioning must absorb.* **"Amer et al. 2025 is the only WSI+CNV PAM50 work" is
false.** Four further published works pair copy number with whole-slide images against a breast
molecular-subtype endpoint: Multimodal CustOmics, MRSVM (Nakach 2024), CLOVER, and — on a secondary
attribution only — Liu et al. IRBM 2022. What survives is narrower and still ours: **Amer 2025 is the
only published work in which copy number is the *sole* molecular modality paired with H&E for a PAM50
endpoint**, and even there it is internal-only, 10-fold CV on TCGA n=977, with zero occurrences of
"CPTAC" in the full text.

### The ladder result: known in general ML, unreported in this field

Novel *here*, not novel anywhere. The phenomenon is established with numbers in general multimodal ML
(UniCat; Jeffares et al. NeurIPS 2023; Huang et al. Sci Rep 2020, averaging late fusion 0.947 AUROC
against joint fusion 0.796; Du et al. ICML 2023 proposing "Uni-Modal Ensemble" as a first-class
method) and in multi-omics survival with no imaging (Wissel 2023; SurvBoard). So "nobody has ever seen
this" is not the claim to make.

What is absent is the instantiation: **no instance in computational pathology on a molecular-subtype
endpoint; none with an external never-trained-on cohort; and no paper anywhere that runs a
multi-operator ladder against an untrained probability mean *and* reports a pairwise error-correlation
statistic as the mechanism.** The closest ladder-shaped ablation found outside our domain is PARSE
(federated HAR / audio-visual, unrefereed), where the mean is competitive but cross-attention wins by
0.6 points — useful calibration showing our margin (0.9259 against 0.8818–0.8992) is far larger than
what that literature reports. Whether anyone has published a φ / Q-statistic / disagreement comparison
between jointly trained fusion operators and independently trained unimodal arms could not be
established either way across ~45 queries; if genuinely unpublished it is a second, smaller novelty
claim, but absence cannot be certified.

*The paper that would have pre-empted it.* A WSI + omics study on a classification endpoint that
tabulates ≥3 trained fusion operators alongside the equal-weight probability mean of the same two
unimodal models on the same splits, and reports pooled out-of-fold metrics. It does not exist. The
nearest miss is Amer's Table 3, which contains the effect and does not name it.

### The sWGS-reachability claim: the assay class is real, the depth claim is untested

The *reachability* half is now supported by clinical precedent rather than assertion. **Belay Ascent**
(Cancers 2026) is a deployed, clinically validated low-pass WGS assay whose literal reported output is
"chromosome arm-level aneuploidy and gene-level copy number variants", with 243 production cases,
100% PPA (25/25 events) against CMA/NGS for arm-level calls in tissue, and an arm-level call threshold
of |log₂r| ≥ 0.09 at 91% sensitivity / 99% specificity from ≥20 ng of input DNA. **Killcoyne et al.
2020 (Nat Med)** encode a 0.4× sWGS classifier as "**589 5Mb windows and 44 chromosome arms**" plus one
complexity scalar — a direct precedent that chromosome-arm-level copy number is a legitimate feature
scale at low-pass depth. Two limits: neither paper states a sequencing coverage figure for arm-level
calling (Belay never reports read depth) and neither states a cost (Belay's "cost-effective" is
qualitative; Killcoyne's "low cost" carries no figure), and Killcoyne's model is not arm-only — the
589 windows dominate. And **no paper anywhere predicts PAM50, or intrinsic subtype under any name,
from shallow or low-pass WGS**; the searches that support that negative include an OpenAlex
`(shallow whole genome sequencing) AND (histopathology) AND (deep learning)` returning count 0 and a
targeted WebSearch that came back empty. Every deployed sWGS classifier found — Barrett's, ovarian
HRD, urinary cfDNA, breast CUTseq, CSF LP-WGS — hits a detection, prognosis, HRD or receptor-status
endpoint. Not one hits an intrinsic subtype.

The *indistinguishability* half — "statistically indistinguishable from a UNI2-h + CLAM pipeline on an
independent cohort" — is a **paired within-cohort statement** and a cross-paper comparison is not
legitimate evidence for or against it. It stands on our own Δ, not on the literature.

### The HER2-enriched collapse and the refuted calibration explanation

Both halves survive, and the corroboration is now stronger than the survey recorded.

*The collapse is reproduced.* Fernandez-Romero 2026 reports HER2-enriched "complete performance
collapse (RPD = 1.000 across all models)" on TCGA→CPTAC across 13 foundation models and 3 MIL
architectures, with a domain-shift regression attributing 80.0% of RPD variance to staining
variability plus feature-space divergence and **prevalence shift not significant**. Independently,
**Wagner et al. 2026 (npj Precision Oncology)** is the closest structural analogue in any organ: UNI2
and Virchow2 with CLAM, a four-class molecular target, a genuine third-institution external cohort,
and **dMMR external recall of 0.14 while its AUC holds at 0.759 (0.718–0.800)** — the same
discrimination-holds-while-decisions-collapse dissociation as our 0/14 Her2 at AUROC 0.860. Crucially
dMMR is *not* a rare class externally (160 patients / 294 slides), which strengthens the analogy
rather than weakening it, and the misclassification pattern is consistent across all their foundation
encoders, so it is not encoder-specific. Their internal-best encoder (Virchow2+CLAM, 0.860) is *not*
their external-best (UNI2+CLAM, 0.780) — a direct caution against selecting an encoder on internal
performance. Mechanistically, Wang et al. 2021 (Cancer Research) state explicitly that ERBB2 could not
be predicted from H&E in a transcriptome-wide regression over 17,695 genes with both internal and
external validation.

*The refutation appears to be unprecedented.* Across seven differently-worded queries on Europe PMC,
Crossref, OpenAlex, PubMed and arXiv, **zero computational-pathology papers apply BBSE or the
Saerens–Latinne–Decaestecker EM to a whole-slide model, and zero report prior correction failing to
recover a collapsed class.** Wagner et al. attempt no prior correction, threshold adjustment or
recalibration at all (case-insensitive "recalibrat" and "calibrat" both return zero hits). The
methodology exists only in the general ML literature. Our refutation therefore appears to be the first
such negative report in computational pathology — worth claiming, but hedged as "we found no prior
report", never as "none exists": prior-shift work publishes under "quantification" and "prevalence
estimation" in venues these registries index poorly. The sharpest available framing is that **BBSE
assumes p(x|y) is unchanged and only p(y) moves; Her2 persisting at 0/14 after a 12× prior boost is
direct evidence the shift is not label shift.**

*One caution on the proposed remediation.* Wagner et al. **deliberately omit stain normalisation**, on
the grounds that "color normalization may impair generalization of foundation models by altering
biologically meaningful stain-morphology correlations learned during large-scale pretraining". That is
a cited counter-argument to one of our two named remediations, not support for it.

---

## 6. Claims that must be softened or dropped

**1. "Amer et al. 2025 is the only WSI+CNV PAM50 work."** False. Replace with: *"Amer et al. 2025 is
the only published work in which copy number is the sole molecular modality paired with H&E for a
PAM50 endpoint. Three other works pair copy number with whole-slide images against a breast
molecular-subtype endpoint — Multimodal CustOmics, MRSVM (Nakach et al. 2024) and CLOVER — and in
each of them copy number is bundled with gene expression, or enters only as a pretraining signal."*

**2. "No published multimodal PAM50 model has an external, never-trained-on, PAM50-specific
evaluation."** Keep, but never drop *multimodal*, and name the near misses in the same paragraph.
Replace the bare sentence with: *"No published multimodal PAM50 model reports an external,
never-trained-on, PAM50-specific evaluation. Four unimodal H&E models do — Borji 2026,
Fernandez-Romero 2026, Zhang 2025 and PathLUPI — and one multimodal paper, M3FusionNet, has a genuine
TCGA→CPTAC protocol whose only reported cross-cohort figure is an overall AUC pooled across five
endpoints; its full text is behind an Elsevier paywall and we could not determine whether a
PAM50-specific external value exists in its tables."*

**3. "None of them reports the trivial average."** Overstated. Replace with: *"None reports the
untrained equal-weight probability mean of two independently trained unimodal models as the baseline
an operator must clear. Amer et al. 2025 comes closest, benchmarking a 'Simple Ensemble' that beats
both of their trained operators in the CNV+image+clinical combination (0.9074 against 0.8976 and
0.9008) and ties them in the four-modality combination — but the paper never defines what Simple
Ensemble computes."*

**4. Amer's like-for-like arm is 0.8835/0.8836, not 0.8604/0.8616.** The pair currently recorded in
the survey and in the claim inventory as "CNV + WSI image" belongs, on a layout-verified re-extraction,
to the **Image + Graph** row. Correct the record before either number is quoted, and quote 0.9000
(CNV + Image + Graph) rather than 0.9153 whenever a genomic-plus-imaging comparator is wanted, since
0.9153 includes a clinical branch whose top attributions are receptor status.

**5. §9 claim 1's headline delta is a mixed contrast that appears in no table.** The quoted
ΔAUROC +0.066 [+0.026, +0.107] and Δbalanced-accuracy +0.226 [+0.127, +0.324] are arithmetically
*prior-balanced fusion minus raw WSI* — a post-hoc-corrected model against a pre-registered one. §2
labels the prior balancing post hoc; §9 does not. Either restate claim 1 from the contrasts that exist
in §1/§3 (Fusion(raw) − WSI(raw) = +0.063 [+0.023, +0.106]; Fusion(bal) − WSI(bal) = +0.048 / +0.185)
or add the mixed-contrast row to §3 carrying the post-hoc label. Reporting rule 3 requires one or the
other.

**6. "The equal-weight mean is the baseline, not the WSI-only model" is violated by §9's own lead
claim**, which is fusion versus WSI. Either §9 leads with the CNV-alone contrast, or reporting rule 2
is amended. As it stands §7 calls "CNV alone is statistically indistinguishable from the fusion" the
central result while §9 ranks it fourth and calls claims 1–3 "the paper"; a reviewer reading both in
sequence will ask which the authors believe.

**7. "Shallow-WGS-reachable, cheap, FFPE-robust."** Design rationale presented as a validated
property — there is no sWGS-depth simulation, no FFPE-degradation ablation and no cost model in this
repo. Replace with: *"The 39 arms are chosen at a scale that deployed low-pass WGS assays already
report — a clinically validated CSF assay reports chromosome-arm-level aneuploidy directly, and a
0.4× sWGS Barrett's classifier encodes 44 chromosome arms among its features — but we have not
simulated sWGS depth, tested FFPE degradation, or costed the assay."*

**8. "H&E stays primary in fact and not just in framing."** Architectural, not evidential: every
headline number has the CNV arm outperforming the WSI arm externally (0.888 against 0.847/0.865;
12/14 against 0/14 on Her2). Replace with a statement about the training design (warm-started and not
frozen) and drop the evidential reading.

**9. §8's balanced-accuracy half is decision-rule-unmatched.** The mean's CNV component uses
`class_weight='balanced'`; the five operators have no output-side prior correction, and §3 of the same
document establishes that matching the decision rule is worth +0.093 balanced accuracy externally.
Either add a prior-balanced row to §8 or scope the balanced-accuracy half explicitly. The AUROC half
is unaffected.

**10. The mechanism claim should say which CNV cross-validation design produced its φ.** Two internal
WSI-vs-CNV values exist — 0.193 (per-CLAM-fold refit, §8) and 0.269 (`StratifiedKFold(10, seed 0)`,
recorded in `dpcode/cli/analysis.py:336` as the published value and appearing nowhere in the results
document). Under the other protocol the contrast reads 0.656 against 0.269 — same direction, ~40%
smaller gap.

**11. Correct the premise of the prior-balancing control.** `evaluate_cnv_wsi_fusion.py`'s docstring
says the WSI arm "was trained under TCGA's natural class frequencies (Her2 = 8.3%)", but
`experiment/pam50_wsi_final.yaml` sets `weighted_sample: true`, which becomes an
inverse-class-frequency `WeightedRandomSampler`. The direction of effect strengthens the Her2 finding
— an over-correction still yields 0/14 — but the stated premise is checkable and wrong.

**12. Two further reporting fixes the inventory flagged.** Print the unrounded lower bound of
Fusion(bal) − CNV, since "+0.024 [+0.000, +0.050]" leaves the significance of the central contrast
undeterminable from the published table. Add protocol labels to the §4 controls table, which
juxtaposes single-seed numbers (aneuploidy burden 0.685, the C sweep, the site holdout) with a
10-seed figure (0.866 ± 0.003) and no protocol column, and quote 0.6893 ± 0.0033 beside 0.685 wherever
0.866 ± 0.003 is the comparator.

**13. Record `compare_fusion_ladder`'s bootstrap constants.** Every CI and every "sig" verdict in §8
rests on `n_boot = 2000, seed = 13`, which the frozen-constants list does not name. Confirmed
against source: `tools/compare_fusion_ladder.py:59` (`--n-boot`, default 2000) and `:97`
(`bootstrap_indices(y.values, args.n_boot, seed=13)`).

**13b. Nominate one internal CNV figure.** Three exist under three protocols and the results document
never says which the thesis quotes: **0.862** [0.836, 0.888] (§5, `StratifiedKFold(10, seed 0)` on
the 599 CLAM-covered cases), **0.872** (§7, CNV refit per CLAM fold so both arms are out-of-fold on
the same fold) and **0.866 ± 0.003** (the published headline, 5-fold × 10 reseeds on the full
945-case set). All three are defensible; leaving the reader to pick is not, and the 0.010 spread is
larger than several of the contrasts the chapter calls significant. Every comparator in §4(b) is
placed against one of the three, so say which.

**14. Say that `concat` and `film_attention` do not clear the WSI-only arm.** §8 says the operators
"barely clear H&E alone"; `concat` (0.8827) and `film_attention` (0.8818) are *below* WSI-only
(0.8872). The document acknowledges this for `film_attention` and not for `concat`.

**15. State label provenance per cohort.** TCGA labels come from cBioPortal
`brca_tcga_pan_can_atlas_2018` patient `SUBTYPE`; CPTAC from `brca_cptac_2020` `PAM50`. Two consortia,
two pipelines. Cross-cohort label-definition shift is currently absorbed into the domain-shift story
without being named, and the survey's own finding 1 is that the label space is not the same object
across papers — the same critique applies within this project across its two cohorts.

**16. Do not repeat a citation error the literature is now propagating.** Borji 2026 states that
Ektefaie et al. 2021 "studied an independent cohort and reported a modest PAM50 classification
accuracy of 65.4%". It did not: 0.654 (0.636–0.672) is top-1 on a site-stratified *held-out TCGA* test
set, and Ektefaie states verbatim that fine-tuning and external evaluation were performed only for the
tumour-vs-normal and histological-subtype models. This finding strengthens our novelty claim.

---

## 7. Still unverified

**No row received a `do-not-cite` verdict.** Nine of twenty are `citable-with-caveat`; the caveats are
stated inline in §3 and §4.

### Paywalls that blocked verification

| Paper | Wall, with the status observed | What remains unknown |
|---|---|---|
| **M3FusionNet**, Comput Biol Chem 124:109228 | ScienceDirect **HTTP 403**; Unpaywall `is_oa false / oa_status "closed" / oa_locations [] / has_repository_copy false`; Europe PMC `pmcid null, availabilityCode "S"`; Semantic Scholar `CLOSED`. A search across Crossref, OpenAlex, Semantic Scholar, Europe PMC, BASE and two WebSearch passes found **no preprint or repository copy anywhere** | The definitions of protocols E1/E2/E3; **whether any PAM50-specific external number exists in its tables**; class count; prediction unit; label basis. This is the single highest-value retrieval remaining — if a PAM50-only E1 AUC exists, the central novelty claim must be rewritten |
| **Liu et al. 2022**, IRBM 43(1):62-74 | ScienceDirect **HTTP 403** (the DOI itself resolves 200 via `linkinghub.elsevier.com`); Unpaywall closed, no repository copy | Class count; prediction unit; label basis; **whether copy number is an input at all**; whether the aggregated quantity is probabilities or features; how the weights were optimised; whether an equal-weight baseline was reported |
| **UGES**, IEEE TCBBIO 22(6):3000-3016 | IEEE Xplore **HTTP 202** with an empty body | Everything beyond the PubMed abstract rests on the authors' accepted manuscript on GitHub. The per-subtype AUCs, the CNA-only 0.877, the 800/1,265 pooled split and the 0.735/0.846 cross-cohort arms are **author-manuscript-level**, not version-of-record |
| **MRSVM** (Nakach 2024), Multimed Tools Appl | Reachable only through a `?error=cookies_not_supported` redirect chain, and read via WebFetch's HTML summarisation rather than parsed raw text | Weaker evidence level than the other 17 full-text rows. Label basis, CNV granularity, CNV feature count and per-subtype class distribution are all genuinely absent from the source |
| **Pan et al. 2019**, Mol Genet Genomics | `link.springer.com` serves a 3 KB JavaScript bot challenge at HTTP 200; the `rd.springer.com` mirror worked. Unpaywall closed | Label basis (PAM50 vs IHC) is unstated in the paper itself and is the fact that decides whether it is the same task as ours |
| **Akbari et al. 2025**, Complex & Intelligent Systems | Springer **HTTP 303** to `idp.springer.com`; the CC-BY PDF endpoint returns a 3 KB JavaScript "Client Challenge" at HTTP 200 | By design the nearest relative of our operator ladder (breast, WSI + genomics, early/intermediate/late/hybrid), and its abstract states **no numeric result at all**. Which strategy won, and whether CNV is among its genomic features, are both unknown |
| **J Transl Med** 10.1186/s12967-026-08743-5 (published 2026-07-31) | BMC serves a Cloudflare "Client Challenge" at HTTP 200; **absent from PubMed and Europe PMC** as of 2026-08-06 | Title-only. The only hard-window title pairing image analysis with PAM50. Retry in a week |
| **Inter-MIL**, Med Image Anal 2025;101 | Elsevier serves a 2,762-byte JavaScript stub at HTTP 200 | Claims four morpho-molecular subtyping tasks including breast; **whether its breast label is PAM50 is unknown**, and if it is, it is a missing baseline |
| **AACR/SABCS PO3-07-04** | `aacrjournals.org` **HTTP 403** | Resolved as PathAI's SABCS 2023 abstract, 961 TCGA-BRCA cases with **no external cohort**, so it cannot threaten the claim. Its AUROC table is in a Table 1 that Crossref does not carry; all values remain NR |
| **Fernandez-Romero 2026** supplementary Tables S4/S5 | Not fetched (an open-access route via PMC13269319 now exists) | Per-class RPD and per-class external F1 for HER2-enriched. Would turn our strongest corroboration from qualitative into quantitative. Cheap and worth doing |
| **Tang et al. 2024 (FusionBench)** | Not opened | The definition of Amer's "Simple Ensemble". Decides whether the only prior WSI+CNV PAM50 paper contains an instance of *our* result or merely of a weaker one |

### Two quoted sentences corrected

- **Wissel et al. 2023.** The sentence "Intermediate (Embrace), Late (MoE), and RSF performed
  significantly worse in terms of Antolini's C compared to using only clinical data" is **not what the
  paper says**. Verbatim: *"When integrating one noise modality, Intermediate (Embrace), Late (MoE),
  and RSF performed significantly worse in terms of Antolini's C compared to using only clinical data
  **and gene expression**."* Two material omissions: the leading noise-injection condition (this is a
  different experiment from Table 1) and the Clinical+GEX comparator. Use the full sentence or none.
- **Papagoras et al. 2025.** "All tissue tiles retained with no tumour detector" is wrong. Tiles *are*
  filtered on texture (grayscale standard deviation < 15 or Laplacian variance < 100 discarded, a
  median of 44%). What is absent is tumour-region preselection.

Neither correction touches a headline number used in §4.

### Axes that returned nothing, and what that means

Five negatives are well supported and are findings rather than failures. **(i)** No work uses
chromosome-arm-level copy number as a fusion modality (queries and counts in §4b). **(ii)** No work
predicts PAM50 from shallow or low-pass WGS. **(iii)** No work reports a scalar aneuploidy-burden
baseline for PAM50. **(iv)** No FiLM conditioning of a pathology MIL attention network was found via
five independent routes (arXiv keyword, Crossref, OpenAlex, Europe PMC, WebSearch), which suggests our
`film_attention` formulation is unprecedented in pathology MIL — with the hedge that Google Scholar
was dead for most of that session and Scholar is exactly where a 2026 workshop paper doing this would
surface. **(v)** No head-to-head of a learned fusion operator against an equal-weight probability mean
exists in WSI+omics computational pathology.

One negative is *not* well supported: **the six-day recency window is under-searched for IEEE-indexed
regional conference proceedings**, because Google Scholar was hard-blocked (HTTP 429 across two
attempts forty minutes apart) and nothing else indexes that venue class well. If a competing WSI+CNV
PAM50 paper appeared at an IEEE conference in that window, this sweep would not have seen it.

### A gap in our own data, surfaced by this search

Song et al. 2022 (Diagnostics 12(11):2623) attributes a TCGA→CPTAC transfer failure on the *same
cohort programme* (UCEC, external AUROC 0.826 [0.727, 0.925]) to CPTAC containing frozen sections as
well as FFPE. `.datasets/cptac-brca/wsi_manifest.csv` (391 rows) carries **no tissue-preservation
field** — its columns are `collection, case_id, slide_id, filename, url, width, height, mpp_x,
mpp_y` — while the TCGA side is documented as diagnostic FFPE only. Whether our 378 CPTAC-BRCA slides
are FFPE, frozen or mixed is not recorded anywhere in this repo. The one file with the right *column
names* is no help: `.datasets/cptac-brca/clinical/cptac_pancancer_clinical_breast.csv` (134 rows)
carries `procurement/tumor_tissue_collection_tumor_type`, `..._frozen_with_oct`, `..._clamps_used`
and two segment-count fields, and **all five are null in all 134 rows**. The schema promises the
answer and the data never delivered it, which is worse than the field being absent — a later reader
will find the column name and assume the question was settled. The cohort-size spread across
published CPTAC-BRCA subsets shows other groups are drawing different slide sets from the same
programme — Fernandez-Romero used **387 flash-frozen** slides from 120 patients, Borji **122 FFPE**
WSIs, EXPAND 168 WSIs from 89 patients, FLEX 323, Tafavvoghi 382 subtype-labelled of 653 — and those
two papers report macro-F1 0.358 and 0.7995 for nominally the same task. Resolving our own slides'
preservation type from PathDB/TCIA metadata is a larger, more specific and more checkable confound
than staining, and it should be settled before the domain-shift attribution in §6 of the results
document is finalised.
