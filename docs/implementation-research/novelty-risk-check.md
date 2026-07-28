# Novelty risk check — FiLM-on-MIL-attention

Scope: verify one novelty claim (FiLM-style affine conditioning of attention-MIL attention scores
in WSI analysis) against three specific risks. Open sources only; no paywall bypass.
Checked 2026-07-27.

---

## Q1 — What does iMIL (Lai, Vong, Yan, Liang, ESWA 2026) actually do?

**Bibliographic record — VERIFIED.**
"Interactive multiple instance learning network for whole slide image analysis", Qi Lai (corresponding),
Chi-Man Vong, Tao Yan, Xiaokun Liang. *Expert Systems with Applications* **297**:129338.
DOI `10.1016/j.eswa.2025.129338`. Online 2025-08-21, issue date 2026-02. Confirmed independently via
Crossref, OpenAlex, Semantic Scholar (`CorpusId 280812737`, DBLP `journals/eswa/LaiVYL26`).
Unpaywall: `oa_status: closed`, `has_repository_copy: false` — **no legitimate full text exists anywhere**.
No preprint, no repository copy, no code release found.

**Abstract — VERIFIED verbatim** (retrieved from the colab.ws mirror of the Elsevier record; ScienceDirect,
ResearchGate and X-MOL all returned 403 to direct fetch):

> Whole Slide Image (WSI) classification presents unique challenges in digital pathology due to
> gigapixel-scale images and complex tissue structures. While Multiple Instance Learning (MIL) has emerged
> as a promising approach for WSI analysis, existing methods often overlook crucial textual information and
> contextual correlations between instances. We propose an interactive multiple-instance learning (iMIL)
> framework that addresses these limitations through two novel perspectives: Multiple-instance Aggregation
> and Prompt-Guided Attention. The Multiple-instance aggregation module effectively combines instance-level
> features (local details), slide-level features (global context), and clinical textual information,
> providing a more comprehensive representation of each WSI. The Prompt-Guided Attention module employs
> learnable prompts to modulate the network's attention toward specific features within instances, enabling
> the model to focus on lesion-relevant areas even with weak annotations. Additionally, an Interactive
> Refinement Module enables continuous model improvement through multiple-level features and contextual
> information feedback. Our framework uniquely integrates visual patterns with associated clinical textual
> information to establish accurate correspondences from slide-level to instance-level features under weak
> supervision. Experimental results verify that iMIL consistently outperforms SOTA MIL models across
> multiple public WSI datasets, significantly improving both WSI classification and positioning performance
> through interactive pseudo-label reasoning.

**Finding — the pre-emption risk is low, for two independent reasons:**

1. **The conditioning signal in Prompt-Guided Attention is *learnable prompts*, not the patient's second
   modality.** The abstract is explicit: PGA "employs **learnable prompts** to modulate the network's
   attention". Learnable prompts are free parameters shared across the dataset — a task/class-level prior.
   They are not a per-patient encoding. The thesis mechanism conditions attention on a *patient-specific*
   20530-dim RNA-seq or 24-dim clinicopathology vector. Different conditioning source entirely.
2. **The clinical text enters a different module.** Clinical textual information is fused in the
   **Multiple-instance Aggregation** module (concatenated/combined with instance- and slide-level features),
   *not* used to modulate attention. So iMIL does not occupy "the second modality reshapes MIL attention"
   either — its clinical modality is a pooling-path input, which is exactly the conventional late/intermediate
   fusion the thesis is departing from.

**Operator inside PGA — COULD NOT VERIFY.** The abstract says only "modulate the network's attention toward
specific features within instances". It does not state whether this is affine (γ, β), a multiplicative gate,
or prompt-token cross-attention. With no full text, no preprint and no code, the operator cannot be
established from open sources. Contextually, "prompt" in the WSI-MIL literature (TOP, ViLa-MIL, PTCMIL,
PA-MIL) almost always denotes prompt tokens consumed by cross-attention or cosine similarity rather than a
FiLM generator, but that is inference, not evidence, and should not be asserted in the thesis.

**Confidence:** bibliographic record and abstract **VERIFIED**; mechanism **PARTIAL** (conditioning source
and module placement verified from the abstract; exact operator COULD NOT VERIFY).

**Recommended handling in the chapter:** cite iMIL, state what the abstract establishes (learnable prompts
modulate attention; clinical text is fused at the aggregation stage), and distinguish on the *conditioning
source* — patient-specific tabular modality vs. dataset-level learnable prompts. Do **not** characterise
iMIL's operator; the paper is paywalled and the operator is not recoverable.

URLs used:
- https://www.sciencedirect.com/science/article/abs/pii/S0957417425029537 (403 to fetch; indexed by search)
- https://colab.ws/articles/10.1016%2Fj.eswa.2025.129338 (abstract retrieved here)
- https://api.crossref.org/works/10.1016/j.eswa.2025.129338
- https://api.openalex.org/works/doi:10.1016/j.eswa.2025.129338
- https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.eswa.2025.129338
- https://api.unpaywall.org/v2/10.1016/j.eswa.2025.129338

---

## Q2 — Does any published work apply FiLM-style (γ, β) modulation to MIL attention scores in WSI?

**Finding: no such work found. The claim survives.**

Searches run: arXiv API (phrase-scoped), Semantic Scholar, OpenAlex, Crossref, and web search, crossing
{FiLM, feature-wise linear modulation, conditional affine, DAFT, conditional batch norm, conditional
layer norm} × {multiple instance learning, attention-MIL, ABMIL, CLAM, whole slide image, histopathology,
computational pathology}.

Key negative results (arXiv phrase search, which does hit the relevant venues):
- `"feature-wise linear modulation" AND "multiple instance"` → **0 results**
- `"FiLM" AND "whole slide image"` → **0 results**
- `"feature-wise linear modulation" AND "pathology"` → 2 results, both irrelevant
  (FiLM-conditioned SpeechLLM; a nuclei detection/classification model)
- `abs:"multiple instance learning" AND abs:"FiLM"` → 1 result, a false positive (blood *films*)

**Candidates checked and cleared:**

| Work | Why it does not kill the claim | Confidence |
|---|---|---|
| **TDA-MIL** (Reisenbüchler et al., MICCAI 2025) "Top-Down Attention-based MIL" | Read the full open-access PDF. The "top-down" signal is derived from the WSI's *own* instances: self-attention pass → feature-selection module → selected instances re-injected into attention for a second pass. Unimodal; no second modality; no affine modulation. The strings `FiLM`, `feature-wise` and `affine` do not occur in the paper. | VERIFIED |
| **PA-MIL** (Yang et al., arXiv 2602.02558) | Language prompts + a Genotype-to-Phenotype network provide "multi-level guidance"; phenotype knowledge base drives feature aggregation. Not affine modulation of attention; genotype is a training-time knowledge source, not a per-patient inference input. Operator not stated in the abstract. | PARTIAL |
| **CAMIL** (channel attention-based MIL, Bioinformatics 2025) | Channel attention is unimodal self-gating over feature channels; no external conditioning signal. | VERIFIED (from abstract) |

**Adjacent work worth citing defensively** (FiLM-style conditioning of *attention* by non-imaging data exists
in medical imaging, but never in MIL/WSI). Reporting these strengthens the chapter rather than weakening it:

- **INSIDE** — Jacenków et al., "INSIDE: Steering Spatial Attention with Non-Imaging Information in CNNs",
  MICCAI 2020, arXiv 2008.10418. Explicitly builds on FiLM: non-imaging variables (lesion location/size,
  cardiac phase, slice index) drive a parametrised spatial attention function *prior to* feature-wise
  modulation. **Segmentation CNN** (CLEVR-Seg, ACDC cardiac MR) — not MIL, not WSI, not pathology.
  VERIFIED from the arXiv abstract.
- **TabAttention** — Grzeszczyk et al., "TabAttention: Learning Attention Conditionally on Tabular Data",
  MICCAI 2023, arXiv 2310.18129, code at github.com/SanoScience/Tab-Attention. Tabular clinical data
  conditions CBAM-style channel/spatial attention inside a 3D CNN, for fetal birth-weight prediction.
  **Not MIL, not WSI, not pathology.** This is the closest conceptual neighbour to the thesis mechanism
  and should be cited. VERIFIED from the arXiv record and MICCAI page.
- **DAFT** — Pölsterl et al., MICCAI 2021, arXiv 2107.05990. Already in the chapter's prior-art list; confirmed.

**Confidence: PARTIAL-to-VERIFIED.** Absence of evidence across five independent indexes with phrase-scoped
queries is strong, but a negative existence claim over all of the literature can never be fully VERIFIED —
notably, paywalled Elsevier/Springer venues (exactly where iMIL sits) are poorly covered by phrase search.
Recommend the chapter uses "to our knowledge" hedging rather than a bare universal negative.

URLs used:
- https://export.arxiv.org/api/query (phrase-scoped queries above)
- https://papers.miccai.org/miccai-2025/paper/2460_paper.pdf (TDA-MIL, full PDF read)
- https://papers.miccai.org/miccai-2025/0933-Paper2460.html
- https://arxiv.org/abs/2008.10418 (INSIDE)
- https://arxiv.org/abs/2310.18129 , https://conferences.miccai.org/2023/papers/639-Paper2389.html (TabAttention)
- https://arxiv.org/abs/2107.05990 (DAFT)
- https://arxiv.org/abs/2602.02558 (PA-MIL)
- https://academic.oup.com/bioinformatics/article/41/2/btaf024/7958575 (CAMIL)

---

## Q3 — MKD-CLOD: ER on TCGA-BRCA, 0.9331 → 0.9581, Kronecker fusion

**Finding: every element checks out. VERIFIED from the full arXiv PDF (read end to end).**

- **Paper.** Qibin Zhang, Xinyu Hao, Qiao Chen, Rui Xu, Fengyu Cong, Cheng Lu, Hongming Xu,
  *"Multi-modal Knowledge Decomposition based Online Distillation for Biomarker Prediction in Breast Cancer
  Histopathology"*, arXiv 2508.17213. arXiv comment field: **"Accepted at MICCAI 2025"** — venue confirmed.
  Note the title is *not* "MKD-CLOD"; MKD and CLOD are the two named components (Multi-modal Knowledge
  Decomposition; Collaborative Learning for Online Distillation). Cite by the real title.
- **Task.** IHC biomarker prediction — **ER, PR and HER2**, each binary. ER is one of three, not the sole task.
- **Cohort.** TCGA-BRCA internal (ER+ 1822 / ER− 562, n = 2384), plus an in-house **QHSU** external test set
  (ER+ 551 / ER− 165). Cases with missing/low-quality genomics or slides excluded; one diagnostic slide per patient.
- **Second modality.** RNA-seq: log-transformed, Z-score-normalised expression; top-K genes selected by a Cox
  proportional-hazards model against overall survival. So "WSI + RNA" is correct.
- **Numbers (Table 1, TCGA-BRCA, ER column, percentages).**
  - Pathology-only, their model: **AUC 93.31**, ACC 88.47, F1 92.58
  - Multimodal, their model: **AUC 95.81**, ACC 90.24, F1 93.67
  → **0.9331 → 0.9581 confirmed exactly.** For reference, pathology-only baselines in the same table:
  ABMIL 88.91, CLAM 89.20, DTFD 89.89; multimodal baselines: MCAT 94.64, Porpoise 92.64, CMTA 93.91.
  Note the comparison is *their* patho-only vs *their* multimodal — the +2.5 AUC point is not a
  clean modality ablation, since MKD/SKD/CLOD differ between the two rows.
- **Fusion mechanism.** Confirmed as described. Pathology features → FC compression → **ABMIL** gated
  attention pooling (student `S_P`). Genomic features → **SNN** (Klambauer self-normalising network) →
  **ABMIL** (teacher `T_G`). A third teacher `T_M` "fuse[s] the global representations of two modalities
  learned by ABMIL using the **Kronecker product**". Training adds CORAL domain alignment, similarity-
  preserving KD, and online collaborative learning; **inference supports pathology alone, genomics alone,
  or both** — the pathology-only student is the deployment path.

**Implication for the thesis is confirmed:** "RNA-seq improves binary ER prediction on TCGA-BRCA over
H&E-alone" is already published at MICCAI 2025 with numbers above this project's. The *result* is not
available as a contribution; only the *mechanism* and the framing can be.

**Confidence: VERIFIED** (full text retrieved and read).

URLs used:
- https://arxiv.org/abs/2508.17213
- https://arxiv.org/pdf/2508.17213
- https://export.arxiv.org/api/query?id_list=2508.17213 (venue field)

---

## Verdict

**The novelty claim survives**, but the current wording is a bare universal negative and should be hedged
and made precise about the conditioning *source*. Proposed replacement wording:

> To our knowledge, FiLM-style affine conditioning (Perez et al., 2018) has not previously been applied to
> the attention-scoring pathway of attention-MIL (Ilse et al., 2018) in whole-slide-image analysis. Related
> mechanisms condition attention on non-imaging data in other medical-imaging settings — INSIDE
> (Jacenków et al., 2020) steers spatial attention in a segmentation CNN, TabAttention
> (Grzeszczyk et al., 2023) conditions CBAM attention on tabular data in a 3D CNN, and DAFT
> (Pölsterl et al., 2021) affinely modulates 3D-CNN feature maps — but none operates on MIL attention over
> patch bags. Within WSI-MIL, iMIL (Lai et al., 2026) modulates attention using *learnable* prompts, i.e. a
> dataset-level prior rather than a patient-specific second modality, and fuses clinical text in its
> aggregation module rather than in the attention pathway; PersAM (Takagi et al., 2023) reshapes MIL
> attention using clinical records, but by multiplicative similarity filtering rather than affine (γ, β)
> modulation. The contribution here is therefore the specific combination: a per-patient tabular modality
> generating (γ, β) that modulate attention logits only, leaving the pooled representation unmodulated.

Two caveats to carry into the chapter:
1. iMIL's exact operator is **unverifiable** from open sources — do not assert it is or is not FiLM.
2. Do not claim novelty for the *result* "RNA improves ER prediction on TCGA-BRCA"; Zhang et al.
   (MICCAI 2025, arXiv 2508.17213) published 0.9331 → 0.9581 on that exact task and cohort.
