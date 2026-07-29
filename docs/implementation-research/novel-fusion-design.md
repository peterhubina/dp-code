# A fusion mechanism for asymmetric second modalities — design document

**Status:** Phase 2 implemented and independently verified (2026-07-28). Sections 0–6 are the
design record as approved; **§7 records what was actually built and where it deviates, and §8
records the independent audit and the two findings that constrain how the co-attention baseline
may be described.** The mechanism is implemented and unit-tested (48 checks), a single-fold
smoke test has run, and the multi-fold ablation is handed off to the author — no multi-fold
training has been launched from here. Every number below was computed
from a tool result in this session or is cited from `docs/er-prediction-results.md`, which was
itself independently verified; numbers that could not be verified are marked as such.

**Purpose.** The preceding chapter established a leakage-controlled ER-status ablation on
TCGA-BRCA and produced a real result: RNA fusion helps (+0.044 AUROC, DeLong p = 1.6×10⁻⁵),
clinicopathology fusion does not (−0.002, p = 0.74). That chapter used CLAM's stock `gated`
fusion. This chapter must contribute a *fusion mechanism*. This document decides which one.

---

## 0. Summary

**Diagnosis (§3).** Three findings, all computed this session, none requiring retraining.

1. **A single gene beats the entire multimodal model.** Raw ESR1 expression, zero fitted
   parameters, scores AUROC **0.9605** against the gated fusion arm's **0.9412** on the same 956
   matched cases — DeLong p = 0.017. RNA-only logistic regression scores 0.9511. The stock gated
   operator *destroys* information rather than merely failing to exploit it.
2. **The gate statistic is misleading.** Both arms' gates sit near an even blend (0.55 / 0.66 on
   the image), yet functional ablation of the trained checkpoints shows the RNA arm is effectively
   an RNA classifier (deleting the image costs 0.006) and the clinicopath arm is effectively an
   image classifier (deleting the table costs 0.000). Reporting a gate mean as evidence of
   multimodal integration is a mistake — this is itself a result worth stating.
3. **There is almost no late-fusion headroom.** No weighting of ESR1 and the image beats ESR1
   alone (best blend 0.9610, p = 0.80), and for clinicopath a stacker with full access to both
   signals is *significantly worse* than the image alone (−0.017, p = 0.011). Clinicopathology
   carries no ER signal beyond morphology — the null is a property of the data, not of the
   mechanism.

**Mechanistic cause.** `gated` is an element-wise convex combination, so the fused vector is
confined to the axis-aligned box between the two projections. It can reweight the modalities but
cannot let them interact — and blending a strong signal with a weaker one lands in between, which
is exactly the observed 0.9412.

**Literature (§2).** Of six candidate directions, five are already claimed — tabular→patch
cross-attention is MCAT verbatim, low-rank bilinear ships in PORPOISE's own code, per-case
reliability weighting is *weaker* than the repo's existing gate, gene grouping is standard
preprocessing, and modality dropout is ModDrop (2014). One slot is open: **FiLM-style affine
(γ, β) modulation of the MIL attention pathway**, with no hits found in WSI work. A dedicated
follow-up check cleared the three plausible threats, including the paywalled iMIL, whose
Prompt-Guided Attention turns out to use dataset-level *learnable prompts* rather than a
patient-specific modality. Separately, and verified from the full paper: "RNA improves ER
prediction on TCGA-BRCA" is **already published** (Zhang et al., MICCAI 2025, 0.9331 → 0.9581), so
the previous chapter is a replication and only the mechanism can be this chapter's contribution.

**Recommendation.**

- **Primary — Design C, FiLM-conditioned attention MIL.** The tabular vector predicts a scale and
  shift applied to the input of CLAM's frozen attention-scoring network, re-ranking patches, while
  pooling still uses the original unmodulated patch embeddings. Plus undiluted direct logit paths
  for both modalities, plus modality dropout. Verified numerically: at identity it reproduces the
  WSI-alone **logits** to 0.00e+00 (so training provably starts at the image-only model), and a
  non-identity γ re-ranks 129 of 137 patches (so it genuinely reorders attention rather than
  merely sharpening it). As built it is **42,754** parameters, not the ≈51.5k estimated here —
  see §7.2b for measured counts.
- **Fallback — Design A, additive logit fusion.** Delivers the corrective result without the
  novelty claim. As built it is **770** parameters and is reachable as `--film_rank 0`, so it is
  an ablation arm rather than a separate implementation.

**Honest expectation.** Most likely C matches RNA-alone (≈0.951–0.960) and beats the gated arm,
while clinicopath stays at WSI-alone. The chapter is structured so its value does not depend on
the headline number moving — the §3 diagnosis stands on its own, and a null on the interaction
hypothesis will be reported as a null.

**Decisions taken (2026-07-28).** Implement the FiLM design, with `film_rank 0` giving the
additive-logit design as an ablation rather than as a separate implementation. Add an adapted
MCAT-style co-attention operator as a baseline inside this harness (§6.3b), while declining to
reproduce MCAT or SurvPath end to end — that stays a stated limitation.

---

## 1. Grounding — what the four existing fusion modes actually compute

Source: `project/CLAM/models/model_multimodal.py` (class `CLAMRNAFusion`),
`project/CLAM/models/model_clam.py` (classes `CLAM_MB`, `Attn_Net_Gated`), read in full this
session.

### 1.1 The shared WSI pathway

`CLAM_MB` with `size_arg='big'` and `embed_dim=1536` computes, for a bag of *N* patches:

```
h  = Dropout(ReLU(Linear(1536 → 512)))(patches)          # N × 512   projected patch embeddings
A  = Attn_Net_Gated(L=512, D=384, n_classes=2)(h)        # N × 2     raw attention logits
A  = softmax(Aᵀ, dim = patches)                          # 2 × N     one attention row per class
M  = A · h                                               # 2 × 512   attention-pooled, per class
```

`CLAMRNAFusion` then calls `_pool_wsi_features(M)`, which **averages the two class rows** into a
single 512-dim vector. Every one of the four fusion modes consumes only that one pooled vector.

This is the single most important structural fact for this chapter: **all four existing modes are
late fusion over a pooled slide embedding. In none of them can the second modality influence
`A` — which patches the model looks at.** The attention distribution is computed from the image
alone, before the tabular vector is ever consulted.

### 1.2 The four operators, precisely

Let `w` = pooled WSI vector (512-d), `t` = raw tabular vector, `d` = `fusion_hidden_dim`.
`TabularMLPEncoder` is `[Linear → LayerNorm → ReLU → Dropout] × num_layers`, producing
`e = enc(t)` of dimension `tabular_hidden_dim`. "Project" below means
`Linear → LayerNorm → ReLU → Dropout` into `d` dimensions.

| Mode | Operator | Where it acts |
|---|---|---|
| `concat` | `logits = MLP([w ; e])` — plain concatenation into a 2-layer head. | pooled |
| `gated` | `p_w = proj(w)`, `p_t = proj(e)`, `g = sigmoid(W[p_w ; p_t]) ∈ ℝᵈ`, `fused = g ⊙ p_w + (1−g) ⊙ p_t`, `logits = Linear(fused)`. An element-wise **convex combination** in a d-dim space; `g` is the weight on the image. | pooled |
| `residual` | A full `RNA_MLP` produces `rna_logits`; a head over `[proj(w) ; proj(rna_feat)]` produces `Δ`; `logits = rna_logits + 0.2·Δ`, with the Δ head **zero-initialised**. The image only *corrects* an RNA predictor. | logit |
| `cross_attention` | `p_w`, `p_t` are treated as a length-2 token sequence through a 1-head `nn.MultiheadAttention` with residual + LayerNorm; flattened; `logits = Linear([·;·])`. | pooled |

Note the naming trap: `cross_attention` here is **self-attention over exactly two pooled tokens**.
It is not attention between the tabular vector and the thousands of individual patch tokens. A
reviewer familiar with MCAT will expect the latter; the repo implements the former. This matters
for the novelty claim in §2 and is stated plainly rather than glossed.

`residual` is the only mode that preserves an undiluted direct path for the tabular modality — and
it is hardcoded to an `RNA_MLP` requiring a pretrained RNA checkpoint (`main.py` errors with
`--fusion_mode residual requires --pretrained_rna_ckpt`). **It was never run in the ER ablation**;
only `gated` was. Verified: `.scratch/results/er/` contains exactly three experiment directories
(`er_wsi_alone_s1`, `er_wsi_rna_gated_s1`, `er_wsi_clinpath_gated_s1`).

### 1.3 The deployed configuration, and its bottleneck

Read from `.scratch/results/er/*/experiment_*.txt` this session. Both fusion arms ran with:

```
fusion_mode = gated          fusion_hidden_dim  = 32
tabular_hidden_dim = 256     tabular_num_layers = 2
tabular_top_n_features = 10000 (RNA) / 0 = all 24 (clinicopath)
freeze_wsi_branch = True     pretrained_wsi_ckpt = er_wsi_alone_s1/s_{fold}_checkpoint.pt
model_size = big             drop_out = 0.5    lr = 1e-4    seed = 1
```

So the entire multimodal decision is funnelled through **32 dimensions** and then a
`Linear(32 → 2)`. The RNA pathway is `20530 → (variance top-10000) → 256 → 256 → 32`. Hold that
number; §3 shows it is the proximate cause of the mechanism's failure.

### 1.4 The implementation surface a new mode must touch

Verified by reading the files:

- `project/CLAM/main.py:155` — `--fusion_mode` `choices=['concat','gated','residual','cross_attention']`, and the settings dict at `main.py:246`.
- `project/CLAM/models/model_multimodal.py` — the `fusion_mode` validation set at line 74 and the branch construction / `forward` dispatch.
- `project/CLAM/utils/core_utils.py:333` — model construction, then `load_wsi_checkpoint` / `freeze_wsi_branch` at 348–358.
- `project/CLAM/utils/core_utils.py:24` — `FUSION_RESULT_KEYS`, a **whitelist**; a new diagnostic metric is only logged if its key is added here.
- `project/CLAM/utils/core_utils.py:569` — the fusion arms use the generic `train_loop`, which calls `model(data)` **without the label**. Consequence: any training-time-only behaviour (e.g. modality dropout) must live inside `forward`, keyed on `self.training`. This is convenient — it means no change to the training loop is required.
- `project/CLAM/dataset_modules/multimodal_dataset.py:228` — `__getitem__` returns `((wsi_features, tabular_features), label)`, one case-level tabular vector per slide.

A new mode is therefore an **additive** change in three places: the argparse choices, the
validation set in `CLAMRNAFusion.__init__`, and a new branch in `forward`. Nothing about the
leakage controls (train-fold-only transform fitting, `case_id` join, label-consistency check)
needs to move.

---

## 2. Literature positioning

Full survey with the method-by-method table, verification status and sources:
`docs/implementation-research/fusion-literature-survey.md` (29 methods; 27 sources verified
directly, 8 of those by reading the released implementation; 15 verified by delegated search;
9 explicitly marked UNVERIFIED). The summary below is what bears on the design decision.

### 2.1 What the literature has already claimed

| Move | Owned by | Consequence for this chapter |
|---|---|---|
| Second modality attends over individual patch tokens (omics tokens as queries) | **MCAT** (ICCV 2021), MOTCat (ICCV 2023), CMTA (ICCV 2023), SurvPath (CVPR 2024), Luo et al. (Sci. Rep. 2025) | Cannot be claimed. MCAT's vendored code is literally `coattn(h_omic_bag, h_path_bag, h_path_bag)`. This is a **baseline to implement**, not a contribution. |
| Clinical/tabular variables reshaping MIL attention | **PersAM** (J. Pathol. Inform. 2023) — multiplicative filtering of class-wise MIL attention by patch↔clinical-factor similarity | The single most dangerous paper for the novelty claim. **Must be cited.** Its operator is similarity filtering, not affine modulation. |
| Gated convex combination on pooled embeddings | Pathomic Fusion (TMI 2022), PORPOISE (Cancer Cell 2022) | This is what the repo's `gated` mode already is. |
| Low-rank bilinear / tensor pooling | **LMF** (ACL 2018), shipped in PORPOISE's own code as `LRBilinearFusion(rank=16)`; HFBSurv (Bioinformatics 2022) | Kills candidate Design D as a contribution. |
| Per-case modality reliability weighting | MRePath (IJCAI 2025), TMC (ICLR 2021 / TPAMI 2023), EsurvFusion (preprint) | Worse than that: the repo's existing `gated` already computes a per-case, per-*dimension* gate, so a scalar reliability weight is **weaker than the baseline**. |
| Gene grouping / pathway priors over the transcriptome | MCAT (6 signature groups), SurvPath (331 pathway tokens), P-NET (Nature 2021) | Standard preprocessing. Frame as engineering, never as novelty. |
| Modality dropout for graceful degradation | ModDrop (2014), superseded by DRIM (MICCAI 2024), DisPro (CVPR 2025), G-HANet (TMI 2025) | A sensible regulariser and ablation. Not a contribution. |
| Tabular→image affine conditioning in medical imaging | **DAFT** (MICCAI 2021) — FiLM with a tabular conditioner on 3D-CNN feature maps | The primitive is claimed; only its application to MIL attention is not. |

### 2.2 Two findings that change the chapter's framing

**"RNA improves ER prediction on TCGA-BRCA" is already published.** **VERIFIED** by reading the
full arXiv PDF: Zhang, Hao, Chen, Xu, Cong, Lu, Xu, *"Multi-modal Knowledge Decomposition based
Online Distillation for Biomarker Prediction in Breast Cancer Histopathology"*, MICCAI 2025
(arXiv 2508.17213). Cite by that real title — "MKD-CLOD" is the name of two components, not the
paper. Table 1, TCGA-BRCA, ER column: pathology-only AUC **93.31**, multimodal **95.81** — the
+0.025 gain is confirmed exactly, on the same task and cohort, with numbers above this project's.
The previous chapter's headline is therefore a *replication*, not a discovery, and must be written
as one. **Only the mechanism can be this chapter's contribution.**

Two details that soften this slightly and are worth using. Their cohort is n = 2384 (ER+ 1822 /
ER− 562), larger than this project's 1003, and ER is one of three biomarkers rather than the sole
task. More usefully: their 93.31 → 95.81 comparison is *their model against their own model*, with
the knowledge-decomposition and distillation components differing between the two rows — so it is
**not a clean modality ablation**. This project's site-held-out, single-variable ablation is
methodologically cleaner than the published one, and that is a legitimate thing to say.

**The stack's ER numbers are in range, not exceptional.** Høibø et al. (Front. Med. 2025) report
ER 0.95 internal / 0.91 external using **UNI + CLAM — this thesis's exact stack**; ReceptorNet
(Nat. Commun. 2020) reports TCGA 0.861; Shamai et al. (Commun. Med. 2024) TCGA 0.930. The
WSI-alone 0.896 sits comfortably inside that band, which is reassuring for the previous chapter's
"exceeds the literature" caveat.

**A defensive point worth making explicitly.** MKD-CLOD's HER2 result jumps 0.746 → 0.958 with
RNA, and HERO (MICCAI 2026) reports ER 0.994 from methylation + miRNA. Those are leakage
signatures — *ERBB2* / *ESR1* expression encodes the IHC readout almost directly. This thesis's
+0.044 is modest enough to be credible, and saying so pre-empts the obvious reviewer suspicion.

### 2.3 The one open slot, and the honest claim

Sweeps across arXiv, Semantic Scholar, Crossref, PubMed, DBLP and OpenAlex returned **no work
applying FiLM-style affine (γ, β) modulation to MIL attention in WSI analysis**. Phrase-scoped
arXiv queries return zero for `"feature-wise linear modulation" AND "multiple instance"` and zero
for `"FiLM" AND "whole slide image"`. A dedicated follow-up check (recorded in
`docs/implementation-research/novelty-risk-check.md`) cleared the three plausible threats:

- **iMIL** (Expert Systems with Applications 297:129338, 2026), the paywalled paper that looked
  most dangerous, does **not** pre-empt the claim. Its abstract, retrieved verbatim, establishes
  two things: its Prompt-Guided Attention modulates attention using **learnable prompts** — a
  dataset-level prior, not a patient-specific second modality — and its clinical text enters the
  *Multiple-instance Aggregation* module, not the attention pathway. Its exact operator **could
  not be verified** (no full text, preprint, or code exists), so the thesis must cite it and
  distinguish on conditioning *source* while **not characterising its operator either way**.
- **TDA-MIL** (MICCAI 2025) — full PDF read; its "top-down" signal comes from the WSI's own
  instances, so it is unimodal, and the words FiLM/affine do not appear.
- **PA-MIL** — language prompts and training-time genotype guidance, not affine attention
  modulation.

Two genuinely adjacent works must be cited defensively, and reporting them strengthens the chapter
rather than weakening it: **INSIDE** (Jacenków et al., MICCAI 2020) steers spatial attention with
non-imaging variables in a segmentation CNN and is explicitly FiLM-derived; **TabAttention**
(Grzeszczyk et al., MICCAI 2023) conditions CBAM attention on tabular data in a 3D CNN. Neither is
MIL or WSI. TabAttention is the closest conceptual neighbour to this mechanism.

Stated honestly, and this is the wording the thesis should use:

> To our knowledge, FiLM-style affine conditioning (Perez et al., 2018) has not previously been
> applied to the attention-scoring pathway of attention-MIL (Ilse et al., 2018) in whole-slide-image
> analysis. Related mechanisms condition attention on non-imaging data in other medical-imaging
> settings — INSIDE (Jacenków et al., 2020) steers spatial attention in a segmentation CNN,
> TabAttention (Grzeszczyk et al., 2023) conditions CBAM attention on tabular data in a 3D CNN, and
> DAFT (Pölsterl et al., 2021) affinely modulates 3D-CNN feature maps — but none operates on MIL
> attention over patch bags. Within WSI-MIL, iMIL (Lai et al., 2026) modulates attention using
> *learnable* prompts, i.e. a dataset-level prior rather than a patient-specific second modality,
> and fuses clinical text in its aggregation module rather than in the attention pathway; PersAM
> (Takagi et al., 2023) reshapes MIL attention using clinical records, but by multiplicative
> similarity filtering rather than affine modulation. The contribution is therefore the specific
> combination: a per-patient tabular modality generating (γ, β) that modulate **the input to the
> attention-scoring network only**, leaving the pooled representation unmodulated — evaluated on a
> binary IHC-derived label rather than survival, with one unmodified mechanism serving both a
> 20530-dim transcriptome and a 24-dim clinicopathology vector, and with a second modality known
> *a priori* not to help used as a graceful-degradation control.

Note the phrase "modulate the input to the attention-scoring network" is deliberate and differs
from the looser "modulate attention logits". Modulating the *logits* would be a monotone transform
that cannot re-rank patches (§4, Design B); modulating the *input* can, and does — 129 of 137
patches in the verification run.

**Hedging is required, not optional.** A negative existence claim over the whole literature cannot
be fully verified, and paywalled Elsevier/Springer venues — exactly where iMIL sits — are the
blind spot of phrase search. Use "to our knowledge" rather than a bare universal negative.

What must **not** be claimed: "we are the first to let non-image data influence which patches
matter" (MCAT and PersAM own that); any claim to a new fusion *paradigm* (it would not survive
contact with MCAT); or novelty for the *result* that RNA improves ER prediction on TCGA-BRCA
(§2.2).

---

## 3. Diagnosis — what is actually wrong with the stock gated fusion

The diagnosis is presented after the literature because it is what determines which of the open
moves is worth taking.

All numbers in this section were computed in this session from files on disk. No model was
retrained. The three findings below are ordered by how much they change the design.

### 3.1 Reproduction check

Recomputing the published headline directly from the per-fold pickles
(`.scratch/results/er/<exp>_s1/split_{i}_results.pkl`, case-level mean of slide probabilities)
reproduces `docs/er-prediction-results.md` exactly: WSI-alone 0.8957 on 1003 cases; WSI+RNA
0.9412 versus 0.8969 on the 956 matched cases (Δ +0.0442); WSI+clinicopath 0.8937 versus 0.8957
on 1003 (Δ −0.0020). The substrate is sound and everything below is built on it.

### 3.2 Finding 1 — the gate value is a **misleading** diagnostic; functional ablation is the real one

`known_gaps` item 2 supposed that with a frozen WSI branch and a 20530-dim RNA vector, the gate
may have collapsed onto RNA. **Numerically it did not — but functionally it did, and the gate
statistic hides that completely.** This is worth stating as a result in its own right, because
`fusion_wsi_gate_mean` is exactly the kind of number a paper reports as evidence of "balanced
multimodal integration".

The per-fold gate means were already logged during training in `fold_{i}_history.csv`
(`train/`, `val/fusion_wsi_gate_mean`), so this needed no forward passes at all. The gate is the
weight on the **image**. A separate probe re-ran the trained checkpoints over the held-out test
slides; its forward pass reproduces the saved predictions to a maximum absolute discrepancy of
**1.8×10⁻⁷**, so the reconstruction is exact.

| Arm | Image-gate, val (10 folds) | Image-gate, test slides | Spread across the 32 gate dims |
|---|---|---|---|
| WSI + RNA | 0.547 ± 0.040 | 0.542 | 0.160 |
| WSI + clinicopath | 0.659 ± 0.012 | 0.612 | 0.182 |

By this measure both arms look like healthy, near-even blends, stable across all ten folds. Now
the functional test — delete one modality at inference from the *trained, unmodified* model
("table absent" = the fold's standardised training mean, i.e. all-zeros; "image absent" = the
pooled 512-d WSI vector replaced by its mean over the fold's own training slides):

| Arm | Intact | Table removed | Image removed | WSI-alone, same cases |
|---|---|---|---|---|
| WSI + RNA | 0.9412 | 0.8643 (**−0.077**) | 0.9353 (**−0.006**) | 0.8969 |
| WSI + clinicopath | 0.8937 | 0.8940 (**−0.000**) | 0.5559 (**−0.338**) | 0.8957 |

The two arms are mirror images of each other, and neither is what its gate suggests:

- **The RNA arm is effectively an RNA classifier.** Delete the image entirely and it still scores
  0.9353 — *above* the WSI-alone arm's 0.8969 on the same cases. The frozen image branch is worth
  about **0.006 AUROC** of the reported multimodal gain. Yet its gate says 0.54.
- **The clinicopath arm is genuinely unimodal on the image.** Its table can be deleted with *zero*
  measurable effect (−0.0003), while deleting the image collapses it to near chance (0.556). The
  fusion head routed around the 24-dim table completely. Yet its gate says 0.61.

Consistent with this, case-level prediction correlation against the WSI-alone arm is Pearson
0.884 (Spearman 0.879) for the RNA arm and Pearson **0.973** (Spearman 0.888) for the clinicopath
arm — the latter is close to a copy of the image model.

Caveat recorded by the probe: both ablations are off-manifold, since neither branch ever saw a
constant input during training, so the absolute ablated AUROCs are lower bounds. The load-bearing
comparison is between the two ablations *within* each arm, which is unambiguous in both cases.

**Methodological upshot for the thesis:** a near-0.5 gate mean is not evidence of multimodal
integration. Reporting it without a functional ablation is a mistake this chapter can name and
correct.

### 3.3 Finding 2 — a single gene beats the entire multimodal model

This is the finding that reframes the chapter. Taking the raw ESR1 expression value as the score,
with **zero fitted parameters** and therefore no train/test leakage of any kind:

| Predictor | AUROC on the same 956 matched cases |
|---|---|
| WSI-alone (CLAM-MB, UNI2-h) | 0.8969 |
| **WSI + RNA, stock `gated` fusion** | **0.9412** |
| GATA3 raw expression alone | 0.9494 |
| **ESR1 raw expression alone** | **0.9605** |

DeLong, paired on those 956 cases: ESR1-alone versus the gated fusion arm is **+0.0194, z = 2.388,
p = 0.0169** — the single gene is significantly *better* than the multimodal model. (ESR1-alone
versus WSI-alone is +0.0636, z = 5.398, p = 6.8×10⁻⁸.) Across the full 20530-gene table, 42 genes
individually exceed AUROC 0.90 and the single best is ESR1 itself at 0.9606, so this is not an
artefact of privileged gene selection — a purely data-driven choice lands on the same gene.

**And the model had the gene.** Reading the saved per-fold transforms
(`er_wsi_rna_gated_s1/s_{i}_tabular_transform.json`), ESR1 survives the variance-based top-10000
selection in **10 of 10 folds**. So this is not a case of the informative feature being filtered
out before the model saw it: the gated fusion arm was handed ESR1 in every fold and still returned
0.9412 against that gene's own 0.9605.

So the stock gated mechanism does not merely fail to exploit the transcriptome; it **destroys**
information that a one-line baseline preserves.

### 3.3b The missing tabular-only baselines, now measured

`known_gaps` item 1 is closed. Both probes were fitted with the same leakage control as CLAM
(variance top-N selection **and** standardisation fitted on the training fold only), on the same
10 site-holdout folds, pooled out-of-fold at case level.

| Modality | Tabular-only logistic regression | Tabular-only MLP | Stock gated fusion | WSI-alone | N |
|---|---|---|---|---|---|
| RNA | **0.9511** (per-fold 0.956 ± 0.038) | 0.9431 | 0.9412 | 0.8969 | 956 |
| Clinicopath | **0.6474** (per-fold 0.663 ± 0.105) | 0.6457 | 0.8937 | 0.8957 | 1003 |

Paired DeLong of fusion against the tabular-only probe, on identical cases:

| Modality | Δ (fusion − table alone) | z | p | Verdict |
|---|---|---|---|---|
| RNA | **−0.0099** | −1.27 | 0.205 | **fusion does not beat the RNA table alone** |
| Clinicopath | +0.2463 | 11.97 | 5.4×10⁻³³ | fusion beats the clinicopath table decisively |

An independent quick probe I ran separately (fixed C, validation cases folded into training)
gave RNA 0.9503 and clinicopath 0.6500, agreeing with the above.

The RNA row is the one that matters. The reviewer question "does fusion beat simply using the RNA
table?" now has an answer, and the answer is **no** — the gated fusion arm is numerically *below*
a plain logistic regression on the same table, and significantly below ESR1 alone. The +0.044
gain over WSI-alone is attributable to the RNA table, not to any synergy between the modalities.
The clinicopath row confirms the mirror image: the 24 features carry little marginal ER signal
(0.647), and the fusion arm is statistically indistinguishable from the frozen WSI branch.

### 3.4 Finding 3 — the mechanism can only interpolate, and there is almost no late-fusion headroom

**Why it interpolates.** `gated` computes `fused = g ⊙ p_w + (1−g) ⊙ p_t` with `g` element-wise
over `fusion_hidden_dim` dimensions. Each coordinate of the fused representation is therefore
confined to the interval between the corresponding coordinates of the two projections: the fused
vector lies in the axis-aligned box spanned by `p_w` and `p_t`. The operator has no way to form a
*conjunction* — "the transcriptome says ER-positive **and** the morphology says high-grade" — as
that lies outside the box. It can reweight the two modalities; it cannot let them interact. Since
the deployed `fusion_hidden_dim` is **32**, the whole multimodal decision is additionally
funnelled through 32 dimensions and a `Linear(32 → 2)`, with RNA travelling
`20530 → top-10000 → 256 → 256 → 32`.

Blending a strong signal (ESR1, 0.9605) with a weaker one (WSI, 0.8969) lands in between. The
observed 0.9412 is almost exactly what an interpolation should give. For clinicopath, the gate
mixes in a near-signal-free table at weight 0.34 and mildly dilutes the image, giving the −0.002
null. Both arms' results follow from one mechanism-level property.

**How much headroom is there for *any* late fusion?** Almost none. A fitting-free rank-average of
ESR1 and the WSI-alone probability, swept over mixing weights on the 956 matched cases:

| Weight on ESR1 | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 | 1.0 (ESR1 alone) |
|---|---|---|---|---|---|---|
| AUROC | 0.9353 | 0.9500 | 0.9585 | 0.9601 | 0.9610 | 0.9605 |

The best blend, 0.9610, is statistically indistinguishable from ESR1 alone (DeLong Δ −0.0004
against the 0.8/0.2 blend, p = 0.80). The two signals are only moderately correlated
(Spearman 0.596 between ESR1 and the WSI-alone probability), yet blending them buys nothing: at
the operating point, the image rescues 17 cases that ESR1 gets wrong, while ESR1 rescues 97 that
the image gets wrong, and 39 cases are missed by both.

**Conditional headroom, by stacking.** The rank-average is deliberately crude, so I also fitted a
per-fold logistic stacker on the image's out-of-fold logit **plus** the full tabular feature block
(train-fold-only selection and standardisation), which is a strictly more expressive late fusion
than anything the four modes implement:

| Stacked predictor | AUROC | Image alone | Δ | z | p |
|---|---|---|---|---|---|
| RNA features + image logit | 0.9501 | 0.8969 | +0.0532 | 4.62 | 3.9×10⁻⁶ |
| Clinicopath features + image logit | 0.8788 | 0.8957 | **−0.0169** | −2.55 | 0.011 |

(Mild caveat: the stacker's *training* features for a held-out fold come from models that saw
that fold's cases, so this is an optimistic estimate of stacking quality. It is used here only as
an upper bound, which is exactly the role it needs to play.)

Two conclusions, and they point in opposite directions:

- **RNA:** even an optimistic linear stacker reaches only 0.9501 — no better than RNA-only
  logistic regression (0.9511) and below ESR1 alone (0.9605). The image adds essentially nothing
  to the transcriptome through *any* late-fusion route.
- **Clinicopath:** the stacker with full access to both signals is **significantly worse** than
  the image alone (−0.017, p = 0.011). There is no ER signal in routine clinicopathology beyond
  what morphology already encodes. This closes `known_gaps` item 3: the null is **a property of
  the data, not a failure of the mechanism**. The stock gated arm's −0.002 was, if anything, a
  mild success — it diluted less than a linear stacker would have.

**This is the single most important constraint on the design.** No amount of cleverness in
*weighting* two pooled predictions will beat 0.96 on RNA, and nothing at all will beat 0.8957 on
clinicopath. A mechanism can only exceed the strong modality if it makes the modalities genuinely
*interact* — most plausibly, by letting the tabular vector change **how the image is read** (which
patches are attended) rather than merely how the image's summary is weighted. That is exactly the
operation none of the four existing modes can perform (§1.1).

### 3.5 What this implies for the bar, and for the experiment plan

Three consequences, all of which are carried into §6:

1. **A capacity-matched `gated` control is mandatory.** The deployed `fusion_hidden_dim` is 32.
   If a new mechanism with more capacity beats it, a reviewer will attribute the gain to capacity,
   not to the mechanism. `gated` must be re-run at the new mechanism's width.
2. **The honest bar is RNA-alone (≈0.951–0.960), not the gated arm (0.9412).** Beating the gated
   arm is nearly free — any mechanism that stops destroying RNA information will clear it. Both
   comparisons must be reported, and the RNA-alone comparison is the one that decides whether the
   contribution is real.
3. **The two modalities need different success criteria, and this must be pre-registered.**
   Given §3.4 there is no headroom to chase on clinicopath: the correct target there is
   *statistical indistinguishability from WSI-alone* (do no harm), not an improvement. Promising
   an improvement on clinicopath would be promising to beat the data. For RNA the target is to
   recover to at least RNA-alone while remaining genuinely bimodal under functional ablation.
4. **"Genuinely multimodal" must be measured by ablation, not by a gate value** (§3.2). The
   functional-ablation table is part of the deliverable for every new arm.

### 3.6 A practical motivation that the diagnosis also surfaces

Of the 1003 cases with an ER label and WSI embeddings, only 956 have RNA; **47 cases (4.7%) have
no transcriptome**, while clinicopathology covers all 1003. H&E is always available, RNA-seq
often is not. A mechanism trained with modality dropout can serve all 1003 cases with a single
model, degrading gracefully to image-only where RNA is absent — a testable property that the
current arms simply cannot offer, since they drop those 47 cases entirely.

---

## 4. Candidate designs

Notation throughout: `t` = raw tabular vector (20530→top-10000 for RNA, 24 for clinicopath);
`c = enc(t) ∈ ℝ²⁵⁶` from the existing `TabularMLPEncoder`; `A ∈ ℝ^{C×N}` = the frozen CLAM
branch's **pre-softmax** attention logits; `h ∈ ℝ^{N×512}` = its projected patch embeddings;
`w = mean_c (softmax(A)·h)_c ∈ ℝ⁵¹²` = the pooled vector the existing modes consume.

§1.4 verified that a new mode is an additive change in three places. A feasibility probe run this
session verified (to a max absolute difference of 0.0) that `self.wsi.attention_net(h)` yields `A` and `h`, and that
recomputing `M = softmax(Aᵀ)·h` reproduces the stock pooled feature exactly — so any design below
can be made to start *bit-identical* to the stock path.

Measured parameter budget for context: the stock `gated` head is **26,914** trainable parameters;
the tabular encoder is 2,627,072 (RNA) or 73,216 (clinicopath); the frozen WSI branch is 1,184,776.

### Design A — Additive logit fusion with a learned per-case reliability ("delta fusion")

**Operator.**
```
z_tab = W_tab · c                       # 256 → 2, the tabular modality's UNBOTTLENECKED logits
z_img = W_img · w                       # 512 → 2, the image's own logits
λ     = σ( u · [c ; w] ) ∈ (0,1)        # one scalar per case: how much to trust the image
logits = z_tab + λ · z_img
```
**Attaches** as a new branch in `CLAMRNAFusion.forward`, replacing the `gated` block. `W_img` is
initialised from the frozen CLAM classifier where shapes allow, and `W_tab` is zero-initialised so
training starts exactly at the WSI-alone predictor.

**Added parameters:** 514 + 1026 + ~770 ≈ **2.3k** — an order of magnitude *fewer* than `gated`.

**Handles the asymmetry** by never forcing a shared bottleneck. Each modality's path is sized to
its own information content: RNA's 2.6M-parameter encoder feeds two logits directly rather than
being squeezed to 32 dims; clinicopath's near-zero-information path simply contributes
near-zero logits and λ moves toward the image.

**Failure mode.** It is still late fusion. Per §3.4 it is therefore capped at roughly 0.96 on RNA.
It fixes the *destruction* but by construction cannot exceed the strong modality.

**Falsifiable hypothesis.** *If the gated arm's deficit is caused by the convex-combination
bottleneck, then delta fusion on RNA will reach at least RNA-only (0.9511) and beat the gated arm
(0.9412) by ≥ +0.01, and its functional ablation will show a large table-removal cost with a small
image-removal cost.* If it fails to recover, the bottleneck explanation in §3.4 is wrong and the
whole diagnosis needs revisiting.

### Design B — FiLM-conditioned attention MIL ("FiLM-attention")

The one operation none of the four existing modes can perform, taken in the one form §2.3 found
unoccupied: the tabular vector predicts an **affine transform of the attention network's input**,
re-ranking patches — rather than acting as a query in cross-attention (MCAT) or multiplicatively
filtering attention scores (PersAM).

**Operator.** Let `fc = attention_net[:3]` (the frozen `Linear(1536→512) → ReLU → Dropout`) and
`head = attention_net[3]` (the frozen `Attn_Net_Gated`), so `h = fc(patches)`.

```
c    = enc(t) ∈ ℝ²⁵⁶
γ    = 1 + Uγ(Vγ · c)      ∈ ℝ⁵¹²        # rank-r factorisation, r = 32; Uγ ZERO-initialised
β    =     Uβ(Vβ · c)      ∈ ℝ⁵¹²        #                              Uβ ZERO-initialised
h̃_n  = γ ⊙ h_n + β                       # FiLM on the attention pathway ONLY
A'   = softmax( head(h̃)ᵀ , dim = patches )
M'   = A' · h                            # pool the ORIGINAL, unmodulated patch embeddings
w'   = mean_c M'_c
logits = W_img · w' + W_tab · c          # Design A's direct paths, both undiluted
```

The essential asymmetry of this operator: the tabular modality changes **which patches are
attended**, but the thing being pooled remains the frozen model's own representation. It cannot
distort the image features, only re-weight them.

**Attaches** in `forward` by splitting `self.wsi.attention_net` into its `fc` and `head` parts and
recomputing the pooling. All three properties this depends on were verified numerically this
session: calling the two parts separately reproduces `attention_net` exactly (max difference
0.00e+00); FiLM at identity (γ = 1, β = 0) reproduces the stock pooled feature exactly (0.00e+00),
so with zero-initialised `Uγ`, `Uβ` the model **provably starts bit-identical to WSI-alone**; and a
non-identity γ re-ranked **129 of 137** patches, confirming the operator genuinely reorders
attention rather than merely sharpening it. That last check mattered — a scalar temperature on the
attention logits would have been a monotone transform, unable to re-rank anything, and would have
made the whole design vacuous.

**Added parameters:** `Vγ,Uγ` = 256×32 + 32×512 = 24,576; same for β = 24,576; heads 1,026 + 514.
Total **≈ 50.7k**.

**Handles the asymmetry** through the rank-32 conditioning bottleneck: whatever its input
dimension, the tabular modality reaches the attention pathway only through 32 latent directions,
so RNA cannot dominate the image branch by brute force. Clinicopath, which §3.4 shows carries no
conditional signal, should drive `Uγ, Uβ → 0`, whereupon the mechanism degenerates *exactly* to
Design A and then to WSI-alone. The do-no-harm property is architectural, not hoped for.

**Failure mode.** With the WSI branch frozen, `h` is fixed. If the ER-relevant morphology is not
recoverable by re-ranking the *existing* patch embeddings, the FiLM parameters stay near zero and
B collapses to A. There is also a genuine overfitting risk — the conditioning is fitted on only
~900 training cases per fold.

**Falsifiable hypothesis.** *If molecular context genuinely changes which morphology is
informative, then (i) the learned ‖γ − 1‖ and ‖β‖ will be non-negligible on the RNA folds,
(ii) permuting the RNA vectors across cases will measurably change the attention distribution for
a fixed slide, and (iii) the RNA arm will exceed the late-fusion ceiling of ≈0.9610 established in
§3.4.* If the FiLM parameters collapse to identity, or AUROC ≤ 0.96, the interaction hypothesis is
falsified — a clean, reportable null, because §3.4 predicts exactly that ceiling for everything
else. Note (ii) is measurable without any retraining once a model exists, and (i) and (ii) can
disagree with (iii); reporting all three is what makes the null interpretable rather than merely
disappointing.

### Design C — Design B plus modality dropout (**the full proposal**)

**Operator.** Design B, plus: during training only (keyed on `self.training`, which §1.4 confirmed
is the only available hook, since `train_loop` calls `model(data)` without the label), with
probability `p_drop` replace `c` by a learned "modality absent" embedding and force γ = 1, β = 0;
independently, with probability `p_drop`, replace `w'` by a learned "image absent" embedding.

**Added parameters:** Design B + two absent embeddings ≈ **51.5k**.

**Why it earns its place** — three distinct jobs, each tied to a §3 finding, not three features
bolted together. (i) §3.2 showed the RNA arm became *functionally* unimodal while its gate said
otherwise; dropout forces both paths to stay independently predictive, and makes that measurable.
(ii) §3.6: 47 of 1003 cases (4.7%) have no transcriptome — dropout lets one model serve all 1003,
degrading to image-only instead of dropping them. (iii) It regularises the attention conditioning,
which is B's main overfitting risk.

Per §2.1 modality dropout is **not itself a contribution** — ModDrop (2014) owns the idea and
DRIM, DisPro and G-HANet have better-engineered versions. It is used here as a regulariser and as
the enabler of an ablation, and the thesis must say so.

**Failure mode.** Dropout may cost peak AUROC; `p_drop` must be chosen on validation folds only.

**Falsifiable hypothesis.** *If modality dropout keeps both pathways predictive, then the trained
model evaluated with RNA absent will score at least WSI-alone (0.8957) on the same cases, whereas
the identical architecture trained without modality dropout will score clearly below it — and
full-modality AUROC will not degrade by more than about 0.005.*

### Design D — Low-rank bilinear pooling (**demoted to a mechanistic control**)

**Operator.** `logits = W·[ (U·w) ⊙ (V·c) ] + z_tab + z_img`, `U ∈ ℝ^{r×512}`, `V ∈ ℝ^{r×256}`,
r = 64 — multiplicative interaction on the *pooled* vectors. **≈ 49.3k** added parameters,
deliberately close to Design B's 50.7k.

§2.1 rules this out as a contribution: it is LMF (ACL 2018), and PORPOISE's released code already
ships `LRBilinearFusion(rank=16)` for exactly WSI+omics. It is retained here for one reason — it
is the control that isolates B's mechanism. §3.4 ruled out *additive* late fusion (rank-average,
linear stacker) but not *multiplicative* pooled-level interaction. If D reaches ≈0.95 while B
exceeds 0.96, the gain is attributable specifically to conditioning the attention rather than to
interaction in general. Run it only if B/C clears the ceiling.

---

## 5. Selection

| Criterion | A (delta fusion) | **C (FiLM-attention + dropout)** | D (low-rank bilinear) |
|---|---|---|---|
| Expected effect vs gated arm (RNA) | +0.01–0.02, recovers to ≈0.95 | +0.01–0.02 floor, and the only credible path above 0.96 | +0.01–0.02, small chance above 0.95 |
| Expected effect (clinicopath) | do no harm | do no harm, architecturally guaranteed as the FiLM parameters → 0 | do no harm |
| **Novelty defensibility (per §2)** | **low** — close to the existing `residual` mode plus a reliability scalar, and §2.1 shows per-case reliability weighting is both published *and weaker than the repo's own baseline* | **the only open slot found** — no work applies FiLM (γ, β) to MIL attention scores in WSI; nearest prior DAFT (no MIL) and PersAM (similarity filtering, not affine) | **none** — this is LMF (ACL 2018), already shipped in PORPOISE's own code |
| Implementation risk in CLAM | lowest | moderate, and the risky dependencies are already verified numerically (§4) | low–moderate |
| Compute cost | negligible | negligible (arms early-stop in 7–24 epochs) | negligible |
| Serves both modalities | yes | yes, and the only one with a *structural* argument for why | yes |

**Recommended primary: Design C.** Three reasons, in order of weight. First, it is the only
candidate that occupies an unclaimed position in the literature (§2.3) — A and D are both
explicitly published, and D is in a competitor's released source. Second, it contains Design A as
its γ = 1, β = 0 special case, so it inherits A's floor: it cannot do worse than the corrective
result, and it provably begins training bit-identical to WSI-alone. Third, it is the only
mechanism with a route above the late-fusion ceiling that §3.4 establishes, because it is the only
one that changes what the image branch *reads* rather than how its summary is weighted. Its
handling of the RNA-versus-clinicopath asymmetry is architectural — a rank-32 conditioning
bottleneck plus zero-initialised outer factors — rather than a hope about optimisation.

**Recommended fallback: Design A.** If conditional attention proves unstable or α collapses, A
still delivers the chapter's core corrective result — that the stock gated operator destroys a
strong modality, and that an additive-logit formulation recovers it — with ~2.3k parameters and
almost no implementation risk.

**Honest expectation.** Given §3.4, the most likely outcome is that C matches RNA-alone
(≈0.951–0.960) and beats the gated arm, while clinicopath stays at WSI-alone. That is still a real
contribution — a mechanism with a demonstrated do-no-harm property across two modalities with
opposite statistics, plus graceful degradation on the 4.7% of cases with no transcriptome — even
if the interaction hypothesis is falsified. **A null on the interaction hypothesis is a publishable
result here and will be reported as one.** The chapter has been deliberately structured so that
its value does not depend on the headline number moving: the diagnosis in §3 (a single gene beats
the fusion model; the gate statistic is misleading; clinicopath has no conditional signal) stands
on its own regardless of how Design C performs.

---

## 6. Experiment plan

### 6.1 Fixed across every arm (non-negotiable, so all comparisons are paired)

Split directory `splits/tcga_brca_er_100` (10 site-holdout folds), `--seed 1`, `--model_type
clam_mb --model_size big --embed_dim 1536`, the same frozen per-fold checkpoints
`er_wsi_alone_s1/s_{fold}_checkpoint.pt` with `--freeze_wsi_branch`, `--tabular_top_n_features
10000` (RNA) / `0` (clinicopath), and the existing train-fold-only transform fitting. Case-level
out-of-fold aggregation by per-case mean of slide probabilities.

### 6.2 Selection phase — validation folds only

Variants of Design C are chosen on **mean validation AUROC across the 10 folds**, never on test:
`p_drop ∈ {0, 0.25, 0.5}`, `r ∈ {32, 64}`, and whether the image-absent path is included. That is
at most 12 configurations per modality, all cheap. The test predictions are not read during this
phase. The chosen configuration is frozen before §6.3 begins and recorded in this document.

### 6.3 Test phase — read once

| # | Arm | Purpose | Status |
|---|---|---|---|
| 1 | WSI-alone | reference | **already trained** |
| 2 | `gated` + RNA, `fusion_hidden_dim 32` | published baseline | **already trained** |
| 3 | `gated` + clinicopath, `fusion_hidden_dim 32` | published baseline | **already trained** |
| 4 | RNA-only, clinicopath-only probes | tabular-only baselines | **already computed** (§3.3b) |
| 5 | **novel + RNA** | primary | to run |
| 6 | **novel + clinicopath** | primary | to run |
| 7 | `gated` + RNA, `fusion_hidden_dim 64` | **capacity control** | to run |
| 8 | `gated` + clinicopath, `fusion_hidden_dim 64` | **capacity control** | to run |

Arms 7–8 are mandatory, not optional. Measured: Design C's fusion machinery is ≈50.7k parameters
against the stock head's 26,914; `gated` at `fusion_hidden_dim 64` is 57,922, i.e. the control is
given slightly *more* capacity than the novel mechanism — deliberately conservative, so a win
cannot be attributed to parameter count.

### 6.3b One scope decision that is the author's to make

§2.3 of the survey notes that reviewers will now expect **MCAT and SurvPath adapted to binary
classification** as baselines, because HERO (MICCAI 2026) already ran two of them, and
**MKD-CLOD** as the direct same-task same-cohort competitor. MCAT is vendored in this repo at
`project/MCAT`, so adapting it is feasible but is a genuine piece of work — it is a survival model
with 6 gene-signature groups and its own data pipeline, not a flag on the existing runner.

This is a real trade-off and it is not mine to settle: including them makes the chapter far more
defensible against the "you only compared against your own weak baseline" objection; excluding
them keeps the chapter to the leakage-controlled ablation it already owns. My recommendation is to
**defer them to a clearly-labelled limitation** for a master's-thesis chapter and state plainly
that the comparison is against the four operators implemented in this repository plus the
tabular-only and capacity-matched controls — but flag it now rather than discover it at review.

### 6.4 Pre-registered endpoints — fixed before any test number is read

- **Primary endpoint:** paired DeLong of the novel arm against the **`gated` arm**, per modality,
  case-level, out-of-fold, on matched cases. Two primary comparisons (RNA, clinicopath), so
  **Holm correction** across those two.
- **Secondary, pre-registered:** DeLong against WSI-alone; DeLong against the tabular-only probe
  (the decisive one for RNA); DeLong against the capacity-matched control (arms 7–8); the
  functional-ablation table of §3.2 recomputed for each new arm; bootstrap CIs on every Δ.
- **Success criteria, stated per modality because §3.4 says they must differ:**
  - *RNA:* beat the gated arm significantly **and** be no worse than RNA-only (0.9511). Exceeding
    0.9610 significantly would additionally confirm the interaction hypothesis.
  - *Clinicopath:* be statistically **indistinguishable from WSI-alone** (0.8957). This is a
    do-no-harm criterion. Claiming an improvement here would mean claiming to beat the data, which
    §3.4 shows is not available.
- **Missing-modality evaluation** (Design C only): evaluate arm 5 over all 1003 WSI cases, with
  the tabular input marked absent for the 47 that have no RNA, and report AUROC on those 47
  against WSI-alone.
- **Anti-fishing rule:** if more than one variant is ever reported on test, that will be stated
  explicitly and the correction widened accordingly. A null result will be reported as a null.

### 6.5 What is *not* in scope (as of Phase 1)

No re-extraction of features, no rebuilt splits or tables, no CPTAC or external cohort, no third
analysis script (`tools/evaluate_er_ablation.py` is extended), no change to the four existing
fusion modes, and no multi-seed sweep in this chapter (single seed 1 remains a stated limitation
inherited from the previous chapter).

---

## 7. Implementation as built (Phase 2)

Approved 2026-07-28: primary design = FiLM-conditioned attention MIL; plus an adapted
co-attention baseline. Recorded here are the points where the built mechanism differs from
the Phase-1 sketch above, so the design record stays honest.

### 7.1 What was added

Two new `--fusion_mode` values, as a purely additive change to five files:
`models/model_multimodal.py` (new `FUSION_MODES` tuple, two constructor branches, and a
separate `_attention_level_fusion` forward path so the four original modes' code is not
touched), `main.py` (argparse choices plus `--film_rank`, `--modality_dropout`,
`--tabular_group_spec`, and the settings dict), `utils/core_utils.py` (constructor kwargs,
token-group construction, four new `FUSION_RESULT_KEYS`), the new
`utils/tabular_groups.py`, and the new `tests/test_fusion_modes.py`. Outside CLAM, the runner
`tools/train_er_novel_fusion.sh` holds the exact command set; `RUNNER=echo bash
tools/train_er_novel_fusion.sh test` prints the resolved commands without executing them.

`main.py` also gained one guard: `--log_heatmaps` is rejected for `coattn`, whose attention
tensor is token-to-patch (`1 × n_tokens × n_patches`) rather than the per-class patch attention
the heatmap logger expects. It defaults to off, so this only converts a latent crash into a
clear error.

### 7.2 Three deviations from the Phase-1 design

1. **The image logits come from the frozen CLAM classifier, not a new head.** The sketch had
   `logits = W_img·w' + W_tab·c` with a fresh `W_img`. As built, the FiLM-conditioned
   per-class pooled features are passed through the WSI branch's *own* `classifiers`, so at
   γ = 1, β = 0 with the zero-initialised tabular head the model reproduces the WSI-alone
   logits **exactly**. This is stronger than the sketch promised: the do-no-harm property is
   now an identity, verified at max absolute difference **0.000e+00** for both the 20530-dim
   and 24-dim inputs. It also means the image pathway adds no new parameters at all.
2. **No per-case reliability scalar.** Design A was sketched with a learned λ weighting the
   image logits. It was dropped: §2.1 established that per-case reliability weighting is both
   published and *weaker* than the repo's existing per-dimension gate, so it would have added
   a parameter without adding a defensible claim. The `film_rank 0` ablation is therefore
   plain additive-logit fusion, `logits = wsi_logits + W_tab·c`.
3. **Modality dropout applies to the tabular modality only.** The sketch dropped either
   modality. In this setting the whole-slide image is always available and only the
   transcriptome can be missing (47 of 1003 cases), so image dropout would have regularised
   against a scenario that never occurs. Evaluation-time missing-modality inference is exposed
   through the `force_tabular_absent` attribute.

### 7.2b Measured parameter counts (these supersede the §4 estimates)

Trainable parameters with the WSI branch frozen, counting the tabular encoder separately
because it is identical across arms and dwarfs every fusion mechanism.

| Arm | Total trainable | Tabular encoder | **Fusion mechanism** |
|---|---:|---:|---:|
| FiLM + RNA (rank 32) | 2,669,826 | 2,627,072 | **42,754** |
| FiLM + clinicopath (rank 32) | 115,970 | 73,216 | **42,754** |
| Additive-logit ablation (rank 0) + RNA | 2,627,842 | 2,627,072 | **770** |
| `gated` + RNA, dim 32 (published arm) | 2,653,986 | 2,627,072 | 26,914 |
| `gated` + RNA, dim 64 (capacity control) | 2,684,994 | 2,627,072 | **57,922** |
| Adapted co-attention + RNA, dim 64 | 3,317,764 | 2,627,072 | **690,692** |
| Adapted co-attention + clinicopath, dim 64 | 125,380 | 73,216 | 52,164 |

Two things follow. The FiLM mechanism is **42,754** parameters, not the ≈50.7k estimated in §4,
because the image logits reuse the frozen classifier rather than a new head; the capacity
control at `fusion_hidden_dim 64` (57,922) therefore still exceeds it by 35%, so the control
remains conservative. And the additive-logit ablation is **770** parameters — a linear head on
the tabular code plus the absent embedding — which makes it a remarkably cheap way to test
whether the entire published gated arm was simply the wrong operator.

### 7.3 A property worth knowing before reading training curves

Because `film_gamma` and `film_beta` are zero-initialised to obtain the identity property,
`film_bottleneck` receives **exactly zero gradient on the first optimiser step** — the chain
rule runs through a zero matrix. It becomes active from step 1 once the output layers move off
zero. This is the ControlNet zero-convolution pattern and is not a bug; it is pinned down in
both directions by `test_film_bottleneck_activates_after_first_step`. Measured on a fixed
batch: loss 0.873 → 0.003 over four steps, bottleneck gradient 0.0 at step 0 and 1.5e-1 at
step 1.

### 7.4 Verification performed

`python tests/test_fusion_modes.py` from `project/CLAM`: **46 checks, all passing.** They
cover output shapes for both a 20530-dim and a 24-dim tabular input on both new modes; the
FiLM identity property; that `film_rank 0` creates no FiLM parameters and reproduces
WSI-alone; that a non-identity γ re-ranks patches (137 of 137 rank positions moved); that
pooling uses the **unmodulated** patch embeddings; gradient flow into every new parameter;
that `freeze_wsi_branch` leaves all WSI parameters with `requires_grad=False`, gradient-free
and numerically unchanged after an optimiser step, with an explicit non-vacuity check;
missing-modality inference; that modality dropout is inactive in eval mode; and a **regression
test confirming all four original modes — `concat`, `gated`, `cross_attention` and `residual` —
produce bit-identical logits (max difference 0.000e+00) to the original implementation.** That
regression test is pinned to commit `60a96391`, the last commit touching
`model_multimodal.py` before this change, deliberately *not* to `HEAD`: once this work is
committed, comparing against `HEAD` would compare the file with itself and pass vacuously.

The tests were confirmed to be discriminating by deliberately breaking the mechanism four
ways and checking each was caught: pooling the modulated embeddings instead of the original
(2 checks failed), removing the zero-initialisation (3 failed), making `freeze_wsi_branch` a
no-op (4 failed), and altering the existing `gated` operator (the regression check failed).
The file was restored and re-verified identical afterwards.

**Smoke tests on real data** (fold 0, 2 epochs, `--freeze_wsi_branch`, not converged and not
to be read as results): FiLM + clinicopath reached val AUROC 0.892 / test 0.912; FiLM + RNA
val 0.947 / test 0.949 with the FiLM diagnostics moving off identity (`fusion_film_gamma_dev`
0.062 → 0.139); adapted co-attention + RNA train loss 0.318 → 0.263, val 0.926 → 0.927.

### 7.5 One measurement that affects how the co-attention baseline is read

With `--tabular_top_n_features 10000`, the variance-based selection retains only **1787** of
MCAT's curated signature genes, so the token sizes are Tumor Suppressor 23, Oncogenes 159,
Protein Kinases 269, Cell Differentiation 356, Transcription Factors 677, Cytokines 303, and
an **unassigned token of 8419** features. ESR1 falls in Transcription Factors, so the baseline
does have the informative gene. But the unassigned token dominates the parameter count: the
co-attention arm's mechanism is **690,692** parameters against the FiLM arm's 42,754, a factor
of 16. That is the conservative direction for a baseline — it is given far more capacity, not
less — but it must be reported prominently rather than hidden, and it carries a real
overfitting risk on ~900 training cases per fold. If the co-attention arm underperforms, the
honest reading is "co-attention with this tokenisation overfits at this cohort size", not
"co-attention is a worse operator".

---

## 8. Independent verification and its consequences

A fresh-context verifier audited the change with no prior knowledge of it, trusting nothing in
the description. Full report: `docs/implementation-research/phase2-verification.md`. Its
evidence is stronger than the author's own tests, so it supersedes them where they overlap.

### 8.1 The three safety claims: all PASS

- **The four original modes are bit-identical.** 32 model pairs (2 backbones x 4 modes x
  2 dropout settings x frozen/unfrozen), each evaluated in train and eval mode with and
  without `return_features`, plus `instance_eval=True` and the `attention_only` path —
  **128 forward comparisons, max absolute difference 0.0**, requiring exact equality rather
  than `allclose`. State dicts identical in all 32 pairs. Only six lines are deleted across
  the whole vendored CLAM, all argparse metadata or constructor validation.
- **The freeze holds.** 16 configurations, including a stricter variant handing *every*
  parameter (frozen ones included) to Adam with weight decay: 0 WSI parameters with
  `requires_grad`, 0 receiving gradients, **0 changing value**, with 6-24 non-WSI parameters
  moving each time so the check is not vacuous.
- **Splits and the transform are untouched.** Every dataset module, `utils/utils.py`,
  `create_splits_seq.py` and `evaluate_multimodal.py` are byte-identical to HEAD by md5, and
  the leakage controls were re-verified *functionally*: the fitted selection and mean/std
  match a train-only fit exactly (0.0) and differ from an all-rows fit (2.79 in the mean), so
  the check is not blind; a missing `case_id` raises; and the label-disagreement check raises
  in the disagreeing case and not in the agreeing one.

### 8.2 Two findings that change how the co-attention arm may be described

These are the important ones. Both were measured, not inferred.

1. **`coattn` does not actually reuse the pretrained WSI branch the way the other arms do.**
   Perturbing each frozen sub-module and reading the change in logits:

   | perturbed sub-module | `film_attention` change in logits | `coattn` change in logits |
   |---|---|---|
   | `attention_net[0]` (patch-encoder trunk) | 2.888 | 1.008 |
   | `attention_net[3]` (CLAM attention head) | 0.363 | **0.000** |
   | `classifiers` (CLAM bag classifier) | 4.820 | **0.000** |

   `coattn` uses only the frozen patch-encoder trunk and substitutes its own projection,
   attention and head for the rest. That is a faithful rendering of the co-attention operator
   — MCAT replaces the pooling entirely — but it means "all arms share the same frozen WSI
   branch" is materially weaker for `coattn` than for `film_attention`. **The write-up must
   say this explicitly rather than implying an equal footing.**

2. **`coattn`'s missing-modality fallback is not the image-only model.** With the tabular
   modality marked absent, its zeroed query tokens make the attention degenerate to
   near-uniform mean pooling, landing 2.2218 away from the WSI-alone logits at
   initialisation, where `film_attention` is exactly 0.0. **The graceful-degradation claim of
   6.4 therefore applies to the FiLM arm only**; reporting a missing-modality number for
   both arms side by side would be comparing two different fallbacks.

### 8.3 Other findings, and what was done about them

- **Fixed.** `utils/tabular_groups.py` had an unreachable error branch, so a signature CSV
  that matched nothing degraded silently into a single giant `unassigned` token — which would
  train happily and look like a valid baseline while testing nothing. It now raises with a
  diagnostic. A companion guard rejects `prefix` grouping when it would yield more than 64
  tokens, which is what happens if it is applied to bare gene symbols. Both are covered by
  new tests (the suite is now **48 checks**).
- **Documented, not changed.** The zero-gradient first step (7.3) affects the whole tabular
  encoder in both new modes, not just the FiLM bottleneck, because `tabular_head` is
  zero-initialised. It self-resolves at step 1 (0.0 -> 2.5e-3 for FiLM, 8.8e-3 for coattn).
  Removing it would mean giving up the exact identity property, which is the mechanism's
  central claim, in exchange for one step out of several hundred. Not worth it.
- **Open gap.** `evaluate_multimodal.py` cannot construct either new mode — it passes no
  `film_rank`, `modality_dropout` or `tabular_group_indices`, and its `infer_fusion_mode`
  does not recognise the new state-dict keys. This does not affect the main ablation, which
  is computed offline from the per-fold `split_{i}_results.pkl` files, but the
  missing-modality evaluation needs a forward pass with `force_tabular_absent=True` and
  therefore needs a path that does not exist yet. To be handled alongside the analysis
  extension after training.
- **Pre-existing, unrelated.** `--fusion_mode residual` is only trainable with
  `--freeze_rna_branch`, because `RNA_MLP` uses `BatchNorm1d` which cannot run in training
  mode at MIL's batch size of 1. Present at HEAD; worth knowing, but not introduced here.

### 8.4 A caveat about the audit itself

Other sessions were writing to this repository during the audit, and the working tree moved
underneath it: the `.detach()` calls on the new metrics and the `--log_heatmaps` guard landed
mid-run. The verifier re-derived every result against final file hashes and re-checked those
hashes afterwards, so the report certifies specific bytes. The `tabular_groups.py` fix in 8.3
was made *after* the audit and is covered by the test suite rather than by the audit.

---

## 9. Selection phase outcome (2026-07-28)

The rank sweep of 6.2 has been run. **Only validation AUROC was read**; the analysis selected
the `val_auc` column explicitly, so no test number from these runs entered the decision. The
selection runs live in `.scratch/results/er_selection/`, a separate tree from the reportable
arms, and their test predictions must stay unread.

### 9.1 Result

Mean validation AUROC over the same 10 site-holdout folds:

| film_rank | mean val AUROC | sd | median | mechanism params |
|---:|---:|---:|---:|---:|
| 16 | 0.9546 | 0.0122 | 0.9542 | 22,274 |
| 32 | 0.9572 | 0.0122 | 0.9586 | 42,754 |
| **64** | **0.9582** | 0.0109 | 0.9619 | 83,714 |

**Selected: `film_rank = 64`**, by the pre-registered rule (highest mean validation AUROC).

Two honest qualifications. Rank 64 beats rank 16 by +0.0036, which is consistent
(rank 16 better in only 2 of 10 folds; paired t p = 0.032). But rank 64 versus rank 32 is
+0.0010 with a paired t of p = 0.449 and Wilcoxon p = 0.492 — **statistically
indistinguishable**, with rank 32 better in 4 of 10 folds. The choice between 32 and 64 is
therefore essentially arbitrary, and nothing in the chapter should be presented as depending on
it. Following the pre-registered rule is the right move precisely because it removes author
discretion at the point where the data cannot decide.

### 9.2 Consequence: the capacity control had to move

This is the part that would have been easy to miss. §6.3 specified the capacity control as
"`gated` at a width whose fusion head exceeds the novel mechanism's", and instantiated it at
`fusion_hidden_dim 64` (57,922 params) on the assumption of `film_rank 32` (42,754). At the
selected `film_rank 64` the FiLM mechanism is **83,714** parameters, which *exceeds* the dim-64
control — inverting the conservatism and handing a reviewer the objection that the novel arm
won on parameter count.

The control has therefore been re-derived from its own original rule, not re-chosen freely:
**`gated` at `fusion_hidden_dim 96` = 93,026 params > 83,714**. The runner reflects this
(`GATED_CAP_DIM=96`, exp codes `er_wsi_rna_gatedcap` / `er_wsi_clinpath_gatedcap`). The
published dim-32 arm remains the primary comparison; the dim-96 arm exists solely to rule out
capacity.

### 9.3 The mechanism is learning, which satisfies hypothesis measurement (i)

From the validation-side FiLM diagnostics at the last epoch of each fold:

| film_rank | mean \|γ − 1\| | mean \|β\| | epochs to early stop |
|---:|---:|---:|---|
| 16 | 0.494 ± 0.166 | 0.258 ± 0.161 | 7–14 |
| 32 | 0.808 ± 0.534 | 0.267 ± 0.130 | 7–19 |
| 64 | 1.196 ± 0.821 | 0.330 ± 0.167 | 8–16 |

The conditioning moves substantially and monotonically away from identity (rank 64, fold 0:
0.250 → 0.902 across epochs). So measurement (i) of the falsifiable hypothesis in §4 — that the
learned modulation is non-negligible — **is satisfied**. That is necessary but not sufficient:
it shows the mechanism is being used, not that using it helps. Measurements (ii) attention
divergence under permuted RNA and (iii) exceeding the ≈0.9610 late-fusion ceiling remain open
and are test-set questions. Note also the large fold-to-fold spread at rank 64 (sd 0.82), which
is the overfitting risk §4 flagged; validation AUROC does not show it biting, but it is worth
re-checking on the test folds.

---

## 10. Metrics and calibration: what is tracked, and one accepted limitation

Decided 2026-07-29, before the test phase was launched.

### 10.1 Why AUROC stays the primary endpoint

AUROC is rank-based and therefore invariant to any monotone recalibration, temperature scaling
included. That is the right property for comparing fusion *mechanisms*: if one arm ranks cases
better but is overconfident, a single temperature parameter repairs the calibration, whereas
nothing repairs a worse ranking. Promoting calibration to a primary criterion would let an arm
win or lose on something trivially fixable. Calibration is therefore reported as a descriptor of
clinical usability, not used as a decision rule.

No calibration term was added to the loss. Binned ECE is not differentiable, and a calibration
penalty would confound the mechanism comparison — a win could no longer be attributed to fusion
rather than to regularisation.

### 10.2 Training is already calibration-aware

`EarlyStopping.__call__` sets `score = -val_loss`, so per-fold checkpoint selection is driven by
validation **cross-entropy**, not AUROC. Cross-entropy is a proper scoring rule, minimised only
by correctly calibrated probabilities, so the selected checkpoint already balances calibration
against discrimination. This is a better default than early-stopping on AUROC and belongs in the
methods text.

### 10.3 Nothing extra needs logging for the results

Every reported metric — AUPRC, F1, balanced accuracy, ECE, calibration curves, per-site
breakdowns and the DeLong tests — is computed offline from `split_{i}_results.pkl`, which stores
per-slide `{slide_id, prob, label}` for each test fold. Adding metrics to W&B would change only
what is visible during training, not what can be concluded afterwards. **Brier score with its
reliability/resolution/uncertainty decomposition will be added to the offline analysis**, because
it is a proper scoring rule and separates "ranks better" from "is better calibrated", which
binned ECE cannot. No re-run is required for that.

### 10.4 Accepted limitation: no validation predictions

`core_utils.train` computes per-slide validation predictions at line 460 and discards them; only
test predictions are saved. The author decided on 2026-07-29 not to change this before launching,
to avoid touching independently verified code immediately before a long run. The consequence,
stated plainly so it is not mistaken for an oversight:

- **Post-hoc temperature scaling cannot be fitted leakage-free for this chapter.** The only
  correct way is to fit the temperature on validation and apply it to test; without saved
  validation predictions that is impossible, and fitting on test would invalidate the results.
- Validation-side calibration cannot be computed retrospectively.

Neither affects the pre-registered analysis: the primary endpoint is AUROC, which is invariant to
temperature scaling, and test-side calibration (ECE, and now Brier) remains fully recoverable.
The thesis should carry this forward as "calibrated probability output is deferred", consistent
with the previous chapter, which already listed temperature scaling as future work. Recovering
the capability later costs one re-run of the affected arms.
