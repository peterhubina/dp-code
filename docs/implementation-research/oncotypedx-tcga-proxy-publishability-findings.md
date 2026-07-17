# genefu Oncotype DX proxy on TCGA-BRCA → external validation on BCR-Net: applicability & publishability

**Scope:** answers two questions for a WSI-primary multimodal recurrence pipeline —
(1) is `genefu::oncotypedx()` applicable to TCGA-BRCA and has it been done, and
(2) is "train on TCGA proxy labels → externally validate on BCR-Net" publishable / conference-acceptable.

**Basis:** paper-search over PubMed / Semantic Scholar / CrossRef (2026-07-12) plus the prior deep-research report
(`oncotypedx-mammaprint-wsi-datasets-findings.md`). Verified citations are named inline.

---

## 1. Is `oncotypedx()` applicable to TCGA-BRCA? Has anyone done it?

**Technically yes — with four caveats that determine whether the label is meaningful.**

`genefu::oncotypedx()` takes an expression matrix + gene annotations (EntrezGene.ID / probe mapping) and returns a
continuous score plus a risk group. TCGA-BRCA RNA-seq contains all 21 genes (16 cancer + 5 reference), and the genefu
manual states the microarray-defined signatures "can be used to reliably evaluate RNA sequencing data." So the
mechanics work. But:

1. **Cohort filtering is mandatory.** Oncotype DX is only meaningful in **ER+/HER2−** (typically node-negative) early
   breast cancer. TCGA-BRCA is a *mixed* cohort (includes TNBC, HER2+, node+). An RS proxy on a triple-negative tumor is
   noise. Subsetting to ER+/HER2− drops TCGA-BRCA from ~1,000+ cases to roughly **500–600**, and fewer after
   intersecting with usable diagnostic WSIs.
2. **The genefu score is NOT calibrated to the clinical 0–100 RS.** `oncotypedx()` implements the *published*
   Paik 2004 algorithm on a different measurement platform; its output is a scaled/rank score, not the RT-PCR–calibrated
   commercial Recurrence Score. Rank-order concordance is decent, but the absolute clinical cutpoints (11 / 25) do **not**
   transfer directly. In practice: binarize by quantile or recalibrate — a defensible but arguable modeling choice
   reviewers will scrutinize.
3. **Tissue / spatial mismatch.** TCGA RNA is extracted from a *different* tissue piece than the diagnostic H&E slide,
   so image and label are from adjacent-but-not-identical tissue — a known weak-label source.
4. **FFPE vs. frozen.** Use TCGA **FFPE diagnostic** slides, not the artifact-heavy frozen slides.

**Prior work:** recomputing 21-gene RS *proxies* from public expression is established — e.g. ESMO Open 2025,
*"Refining prognostic tools for luminal breast cancer"* (esmoop.2025.105080; PMC12088756) recomputed Oncotype DX +
MammaPrint + PAM50-ROR from RNA-seq — and genefu is purpose-built for it (Gendoo et al., *Bioinformatics* 2016).
**However, no located paper does the specific chain "genefu RS on TCGA-BRCA → predict from H&E."** Every H&E→RS paper
found (BCR-Net / PLOS ONE 2023; Orpheus / *Nat Commun* 2025; Shamai–Aran / *Lancet Oncol* 2026) used the **true clinical
assay** as ground truth, not a genefu proxy. That gap is the crux of question 2.

---

## 2. Publishable? Will conferences accept it?

**Honest verdict: publishable at a workshop, thesis, or mid-tier journal if framed carefully — NOT competitive at a top
venue (MICCAI main track, CVPR/Nature-family) as a standalone "we predict Oncotype DX" contribution.**

### Why the naive framing struggles

- **Circularity / weak-label attack (a reviewer's first move).** The training target is a deterministic function of gene
  expression, so the model learns to approximate *a computational transform of RNA*, not the clinical assay. Expect:
  "why is predicting a proxy of expression better than predicting expression directly?" You need an answer ready.
- **Crowded, high SOTA bar.** Orpheus (6,172 cases, true RS, AUC 0.89) and Shamai–Aran (TAILORx n=8,284, true RS,
  externally validated on six cohorts **including TCGA-BRCA**, AUC ~0.90) already own this problem with real labels at
  scale (2025–2026). A TCGA-proxy → BCR-Net paper is weaker on both label quality and scale, and reviewers will compare.
- **Thin external validation.** BCR-Net's 99 labeled patients give wide CIs; one small external cohort is soft. Reviewers
  increasingly expect ≥2 external cohorts.

### The one genuinely strong angle you DO have

BCR-Net carries the **true** Oncotype DX assay label. So the defensible story is **"silver-label training (TCGA proxy) →
gold-label external validation (BCR-Net true assay)"** — testing whether a proxy-trained model transfers to real
Oncotype DX. That is legitimate and novel-ish, but only if the BCR-Net numbers actually hold up.

### What makes it publishable / accepted

- **Lead with the multimodal FUSION contribution**, not RS prediction per se — "WSI + [clinical] fusion beats WSI-alone
  for recurrence-risk stratification" survives the proxy-label critique far better than "we predict Oncotype DX."
- **Include a proxy-fidelity analysis** — quantify genefu-RS vs. true-RS concordance where both exist (e.g. on BCR-Net's
  true labels). Turns the biggest weakness into a documented, honest result.
- **Avoid label leakage in fusion:** if you fuse RNA-seq *and* derive the label from RNA-seq, you leak the answer. Fuse
  clinicopathologic features instead, or hold the 21 RS genes out of any RNA fusion input.
- **Add a second external cohort** if feasible (CPTAC with recomputed proxy; or pursue a Dartmouth/BMIRDS DUA) — even one
  more materially improves acceptance odds.
- **Target realistically:** MICCAI workshops (COMPAY, MOVI), *Journal of Pathology Informatics*, *Scientific Reports*,
  *Cancers*, or a thesis chapter — all realistic. Top-tier main tracks / high-impact clinical journals — unlikely without
  real assay labels at scale.

### Recommendation

Frame it as a **multimodal recurrence-risk stratification** study with proxy-fidelity honesty and silver→gold external
validation — **not** as an "Oncotype DX predictor." The predictor framing walks straight into a fight with Orpheus and
Shamai–Aran that you lose on labels and scale. The fusion + honest-proxy framing is a solid, publishable contribution.

---

## Key references

- **BCR-Net** — Su et al., *PLOS ONE* 2023 (pone.0283562) — H&E+Ki67, 99 patients, true Oncotype DX, MIL.
- **genefu** — Gendoo et al., *Bioinformatics* 2016 — `oncotypedx()` (21-gene, Paik 2004), `gene70()` (70-gene, van 't Veer 2002).
- **Proxy recomputation precedent** — ESMO Open 2025, esmoop.2025.105080 (PMC12088756) — ODX/MammaPrint/PAM50-ROR from RNA-seq.
- **SOTA with TRUE labels (the competition):** Orpheus — Boehm et al., *Nat Commun* 2025 (10.1038/s41467-025-57283-x), 6,172 cases, RS>25 AUC 0.89; Shamai, Aran et al., *Lancet Oncol* 2026 (10.1016/S1470-2045(25)00727-2), TAILORx n=8,284, six-cohort external incl. TCGA-BRCA, AUC ~0.90.

## Open items to resolve before committing

1. Empirical genefu-RS ↔ true-RS concordance (run on BCR-Net; needed to justify proxy labels).
2. Exact count of ER+/HER2− TCGA-BRCA cases with usable FFPE diagnostic WSI after filtering.
3. Whether a second external cohort (CPTAC proxy, or Dartmouth DUA) is obtainable in the project timeline.
