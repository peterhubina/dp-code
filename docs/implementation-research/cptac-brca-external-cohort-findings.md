# CPTAC-BRCA as an external cohort — download routes and comparability findings

**Purpose:** stand up CPTAC-BRCA as an independent external-evaluation cohort for the
TCGA-BRCA PAM50 / ER pipelines, with WSI as the primary modality and RNA-seq fused alongside.

**Method:** every endpoint below was probed live from inside the project container
(2026-07-27/28) rather than taken from documentation. Counts and sizes are measured, not
estimated. Implemented in `tools/download_cptac.py`.

---

## Bottom line up front (BLUF)

1. **Aspera is not required.** TCIA support recommends the Aspera Faspex browser plugin, which
   needs a GUI and is a dead end in this container. TCIA's PathDB exposes every slide as a
   direct HTTPS file URL that works headless, with Range support (resumable, parallelisable).
2. **The usable cohort is 119 cases** with WSI + RNA + PAM50 (391 slides, 68 GB).
   For the 4-class PAM50 task that becomes **114** after dropping Normal-like.
3. **Two traps, both load-bearing:**
   - CPTAC-BRCA is **unusable for survival or recurrence** (1 recurrence event, 2 deaths).
   - The **RNA is on a different scale than our TCGA RNA** — see the section below, which is
     the single most important thing in this document.

---

## Modality map

Each modality lives in a different portal and none of them cross-reference the others. All are
open access — no token, no dbGaP application, nothing to sign.

| Modality | Source | Scale | Notes |
|---|---|---|---|
| WSI | TCIA PathDB, collection id **521** | 654 `.svs`, 199 cases, **114.3 GB** | 40x, mpp 0.25 — same optics as TCGA-BRCA |
| RNA-seq | GDC, project **CPTAC-2** | 133 STAR-Counts files, **559 MB** | `access: open`; merged to a TPM matrix |
| PAM50 + receptors | cBioPortal **`brca_cptac_2020`** | 122 cases | Krug et al. 2020 |
| Recurrence + OS | Zenodo record **8394329** | 134 breast cases | the `cptac` package's own data mirror |
| Proteome / phospho / acetyl | PDC000120 / 121 / 239 | 127 cases, TMT10 | not wired into the script |
| Radiology | — | **none** | CPTAC-BRCA is absent from NBIA entirely |

### Non-obvious things that cost time

- **The prospective breast cohort sits under GDC project `CPTAC-2`, not `CPTAC-3`.** CPTAC-3
  contains exactly 1 breast case. Filtering on `CPTAC-3` looks like "CPTAC has no breast RNA".
- In the GDC **files** index the field path is `cases.project.project_id`, not
  `project.project_id`. The wrong path returns 0 hits silently rather than erroring.
- **Neither GDC nor PDC clinical carries PAM50 or receptor status.** Both expose stage,
  morphology, age and grade only. cBioPortal is the only source for the labels we need.
- cBioPortal case ids are **X-prefixed** (`X01BR001`) and must be stripped to join to PathDB
  and GDC. Receptor values also mix `Negative` / `negative` casing.
- The **`cptac` pip package will not install here** — its `pyranges` → `sorted_nearest`
  dependency needs a C compiler the image lacks. Its data is plain HTTP on Zenodo, so the
  script bypasses the package and fetches the files directly.

---

## The cohort

```
199  cases with WSI
133  cases with RNA
122  cases with PAM50 / receptor labels
---
119  cases with WSI + RNA + PAM50   (391 slides, 67.9 GB)
```

| Label | Distribution |
|---|---|
| PAM50 | LumA 56, Basal 27, LumB 17, Her2 14, Normal-like 5 → **114 for 4-class** |
| ER | 80 positive, 37 negative, 2 missing |

**Join verified biologically, not just by row count:** ESR1 median **150 TPM** in ER-positive
cases vs **1.6 TPM** in ER-negative — a ~100x separation, which is what a correct
WSI ↔ RNA ↔ label join should produce.

---

## ⚠ Trap 1 — RNA scale mismatch (settle this before downloading 114 GB)

**The CPTAC RNA and the TCGA RNA in this repo are not on the same scale and are not directly
comparable.** A fusion model trained on one will not transfer to the other as-is.

| | TCGA side | CPTAC side |
|---|---|---|
| Fetched by | `tools/rna/download-rna.py` | `tools/download_cptac.py --modality rna` |
| Source | UCSC Xena (`HiSeqV2`) | GDC STAR-Counts |
| Quantification | RSEM | STAR + GDC TPM |
| Units | **log2(x+1)** normalised | **linear TPM** (columns sum to 1e6) |
| Gene model | HiSeq-era annotation | GENCODE v36 |
| Gene ids | HGNC symbols | `gene_name` (symbols) + Ensembl ids |

Feeding a linear-TPM matrix to a head trained on log2 RSEM is a silent failure: it produces
numbers, not an error, and external performance collapses for reasons that look like domain
shift but are actually a units bug.

**Three ways out, in decreasing order of rigour:**

1. **Re-pull TCGA from GDC STAR-Counts** so both cohorts share quantification pipeline, gene
   model and units. Most defensible for a thesis claim, and the only option that removes the
   confound rather than damping it. Costs a TCGA RNA re-download and re-running the RNA arm.
2. **Rank / quantile transform per sample** on both sides. Discards absolute expression but is
   invariant to the RSEM-vs-TPM difference, and is standard practice for cross-cohort transfer.
3. **Per-cohort z-score per gene.** Cheapest, but it only removes location/scale, not the
   quantification difference, and it leaks cohort-level statistics into evaluation. Acceptable
   as a sanity check, weak as a headline result.

Whichever is chosen, the gene sets must also be intersected — the two annotations do not cover
identical symbol sets.

---

## ⚠ Trap 2 — do not use CPTAC-BRCA for survival or recurrence

The clinical table *populates* `Recurrence status`, `Overall survival, days` and
`Survival status`, so it looks usable. Across the 119-case cohort it contains:

- **1** recurrence event
- **2** deaths (97 alive, 20 missing)
- follow-up on the order of a year

That is far too few events to fit or evaluate the AMIL survival model or a recurrence head.
CPTAC-BRCA is a **PAM50 / ER cohort only**. The HistologyHSI-BC and TCGA threads remain the
places to do survival work.

---

## Usage

```bash
python tools/download_cptac.py --modality clinical         # ~150 KB, start here
python tools/download_cptac.py --modality rna              # 559 MB + merged TPM matrix
python tools/download_cptac.py --modality wsi --workers 8  # 114 GB
python tools/download_cptac.py --modality all --cohort-only --dry-run
```

`--cohort-only` restricts the WSI pull to slides whose case has both RNA and a PAM50 label —
391 slides / 68 GB instead of 654 / 114 GB. `--dry-run` writes manifests and reports sizes
without transferring. Re-running is idempotent: complete files are skipped, partial files
resume via Range.

Output layout under `.datasets/cptac-brca/`:

```
wsi/                                    *.svs
rna/                                    per-case STAR-Counts + tpm_matrix.tsv.gz
clinical/cbioportal_labels.csv          PAM50, ER, PR, HER2, TNBC, stage
clinical/cptac_pancancer_clinical_breast.csv
cohort.csv                              one row per case: modalities present + labels
wsi_manifest.csv, rna_manifest.csv
```

`--collection` fetches WSIs for any other CPTAC collection (CPTAC-LUAD, CPTAC-CCRCC, …); the
RNA and clinical wiring is BRCA-specific.

---

## Related

- `docs/implementation-research/oncotypedx-mammaprint-wsi-datasets-findings.md` — CPTAC-BRCA as
  a genefu-proxy cohort for Oncotype DX / MammaPrint.
- `docs/er-prediction-results.md` — the TCGA ER task this cohort would externally validate.
