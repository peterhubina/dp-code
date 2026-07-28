# Fusion literature positioning: WSI + omics / WSI + clinical

Scope: what the multimodal computational-pathology literature has actually claimed as a *fusion
mechanism*, and where a defensible novelty claim is still available for this thesis.

Reference point for the thesis. The four operators already in this repo
(`project/CLAM/models/model_multimodal.py`) — `concat`, `gated`, `residual`, `cross_attention` —
all run **after** `_pool_wsi_features(...)` mean-pools the CLAM per-class features (line ~279).
The WSI branch is executed to completion first; the tabular branch never touches the attention
scores `A` over patches. All four are therefore **late (pooled-embedding) fusion**, and
`cross_attention` is self-attention over a 2-token sequence, not attention against patch tokens.

Where operator descriptions below say "code-read", the claim comes from reading the released
implementation, not from paraphrasing an abstract.

---

## 1. Method table

| Method | Venue + Year | Second modality | Where fusion happens | Fusion operator (one line) | Second modality modulates MIL attention over patches? |
|---|---|---|---|---|---|
| **ABMIL** (Ilse et al.) | ICML 2018 | none | n/a | Gated attention pooling `A = softmax(w'(tanh(Vh) ⊙ sigm(Uh)))`; the unimodal baseline every method below inherits | n/a |
| **CLAM** (Lu et al.) | Nature Biomed. Eng. 2021 | none | n/a | ABMIL + instance-level clustering loss; per-class attention branches (CLAM-MB) | n/a |
| **Pathomic Fusion** (Chen et al.) | IEEE TMI 2022 (arXiv 2019) | genomics (+ cell graph) | pooled | Gated Kronecker product: each pooled vector is sigmoid-gated by a linear map of *both* vectors, a 1 is appended, then outer product → MLP | **No** |
| **MCAT** (Chen et al.) | ICCV 2021 | RNA-seq split into 6 gene-signature groups (SNN per group) | **patch-level** | code-read (`project/MCAT/models/model_coattn.py:79`): `coattn(h_omic_bag, h_path_bag, h_path_bag)` — the 6 **genomic tokens are the queries**, the N patch tokens are keys/values, 1 head. Output = 6 genomics-conditioned summaries of the patch bag; a Transformer + `Attn_Net_Gated` then pools those 6 tokens. Final concat or gated-Kronecker `BilinearFusion` on the two pooled vectors | **Yes** — genomics sets the weight of every patch |
| **PORPOISE** (Chen et al.) | Cancer Cell 2022 | RNA-seq + CNV + mutation (SNN) | pooled | code-read (`mahmoodlab/PORPOISE`, `models/model_porpoise.py`, `PorpoiseMMF.forward`): gated-attention MIL pools patches **first** (`h_path = A @ h_path`), then fusion over two 256-d vectors, selectable `concat` / `bilinear` (gated Kronecker) / **`lrb` = `LRBilinearFusion`, an explicit rank-16 low-rank bilinear pooling with per-modality factor matrices** | **No** |
| **MOTCat** (Xu & Chen) | ICCV 2023 | same 6 gene-signature groups | **patch-level** | code-read (`Innse/MOTCat`): MCAT with the dot-product co-attention replaced by `OT_Attn_assem` — an unbalanced mini-batch **optimal-transport plan** π between patch embeddings and gene-signature embeddings, used as the co-attention matrix; rest of the pipeline is MCAT | **Yes** |
| **CMTA** (Zhou & Chen) | ICCV 2023 | 6 gene-signature groups | **patch-level (bidirectional)** | code-read (`FT-ZHOU-ZZZ/CMTA`, `models/cmta/network.py:189-198`): two encoder–decoder towers plus **two** cross-attentions — `P_in_G` (patch tokens query genomic tokens) and `G_in_P` (genomic tokens query patch tokens); per-modality CLS tokens of encoder+decoder are averaged, then concat or Kronecker bilinear | **Yes** |
| **SurvPath** (Jaume et al.) | CVPR 2024 | transcriptomics grouped into **331 pathway tokens** (281 Reactome + 50 Hallmark, over ~4,999 genes) | **patch-level** | code-read (`mahmoodlab/SurvPath`, `models/layers/cross_attention.py:79-101`): one asymmetric MM-attention over `[pathway tokens ; patch tokens]`. Pathway tokens attend to pathways **and** patches; **patch tokens attend only to pathway tokens**, and `out_histology = softmax(q_hist·k_path) @ v[:num_pathways]` — patch–patch self-attention is deleted for memory, so each patch token is *replaced* by a pathway-weighted mixture. Then per-modality **mean** pooling → concat → MLP | **Yes** for the patch *representation*; note there is no learned attention-pooling at all (uniform mean), so the "attention" is the cross-attention itself |
| **TANGLE** (Jaume et al.) | CVPR 2024 | bulk transcriptomics, **pretraining target only** | none at inference | code-read (`mahmoodlab/TANGLE`, `core/models/mmssl.py`, `core/loss/tangle_loss.py`): symmetric InfoNCE between an ABMIL slide embedding and an RNA-MLP embedding (+ optional RNA-reconstruction head from the slide embedding). `get_features()` takes WSI only — downstream use is unimodal | **No** — expression is supervision, not an input |
| **PathOmics** (Ding et al.) | MICCAI 2023 | genomics | not inspected | Pathology-and-genomics multimodal Transformer; pretrain-then-finetune with genomics optionally absent at test time. Operator not code-read — treat description as indicative | Unclear |
| **BioFusionNet** (Mondol et al.) | IEEE JBHI 2024 | genetic **and** clinical, ER+ breast cancer survival | pooled (patient-level) | From abstract only (code not read): DINO/MoCoV3 patch features → VAE fusion → self-attention → patient-level vector; a "co-dual-cross-attention" combines it with genetics; **clinical data enters through a separate feed-forward network**. Closest published work to this thesis's cohort, but three separate mechanisms, not one | Unclear (aggregation happens before the cross-attention as described) |
| **EsurvFusion** (Huang et al.) | arXiv preprint 2024 — **no peer-reviewed venue confirmed** | clinical / imaging / genomics | **logit / decision level** | Each modality yields a prediction plus aleatoric+epistemic uncertainty (Gaussian random fuzzy numbers); a **reliability-discounting layer estimates a per-modality reliability coefficient**, then evidential decision fusion | **No** |
| **PersAM** (Takagi et al.) | J. Pathol. Informatics 14:100185, 2023 | **clinical records** (tabular), malignant-lymphoma subtyping — a *classification* task | **patch-level / attention-score level** | Clinical-factor tokens are concatenated into the same Transformer token sequence as patch tokens; a per-patch relevance `ψ_ℓ = (1/M) Σ_m σ(q_mᵀ k_ℓ)` between each patch and the M clinical factors then **multiplicatively filters the class-wise MIL attention scores**. 842 patients | **Yes** — the single closest prior art to "clinical variables reshape MIL attention" |
| **G-HANet** (Wang et al.) | IEEE TMI 44(5):2170–2181, 2025 | genomics, **training only** | patch aggregation weights | Cross-modal associating branch distills genotype↔phenotype associations during training; a hyper-attention survival branch combines the distilled associations with morphology-based weights to aggregate patches. **Genomics not required at inference** | **Yes at training**, no at inference |
| **HFBSurv** (Li et al.) | Bioinformatics 38(9):2587–2594, 2022 | genomics + clinical | pooled | Hierarchical **factorised (low-rank) bilinear** pooling: modality-specific attentional factorised bilinear, then cross-modality attentional factorised bilinear. **Caveat: the pathology modality is ~2,343 CellProfiler hand-crafted nuclear features reduced to 80 — no WSI patches, no MIL** | **No** (no MIL at all) |
| Keshvarikhojasteh et al. | arXiv 2605.13897, 2026 — **no peer-reviewed venue confirmed** | RNA-seq **and** clinical | pooled | **Low-rank bilinear cross-modal fusion** over three independent pairwise modules, on top of an ABMIL slide embedding. Closest published attempt at low-rank bilinear for WSI-MIL + omics; still strictly post-pooling | **No** |
| **DRIM** (Robinet et al.) | MICCAI 2024 (Spotlight) | methylation + RNA + MRI + WSI | pooled | Shared/unique disentangled encoders (MI minimisation) + masked-Transformer attention over a **variable-length modality sequence**, so absent modalities are simply omitted rather than zero-imputed | **No** |
| **MMP** (Song et al.) | ICML 2024 | pathway-grouped transcriptomics | prototype level (post-condensation) | Both modalities condensed to a small fixed set of prototype tokens (morphological / pathway), then fused by Transformer or optimal-transport alignment | **No** — patches are condensed before fusion |
| **DisPro** (Xu et al.) | CVPR 2025 | genomics, possibly missing | not established from abstract | Two-stage prompt distillation; available modalities prompt an LLM to infer the missing one. Operator level UNVERIFIED | Unclear |
| **MKD-CLOD** (Zhang et al.) | MICCAI 2025 | RNA-seq (Cox-selected top-K genes) — **task is binary ER / PR / HER2 on TCGA-BRCA** | pooled | **Kronecker product of two independently ABMIL-pooled vectors** (genomics via SNN then ABMIL). Multimodal branch is a *teacher*; a pathology-only *student* runs at inference (online distillation). Reported TCGA-BRCA AUC: pathology-only ER 93.31 → multimodal ER 95.81 | **No** |
| **HERO** (Li & Su) | MICCAI 2026 (early accept) | DNA methylation (14,115 CpGs) + miRNA — **binary ER / PR / HER2 + subtype + risk on TCGA-BRCA** | patch *retrieval* level | Omics → sparse pathway-to-morphology prior → K=16 "intent vector" → TF-IDF retrieval over structured captions **selects** endpoint-relevant regions; cosine gate triggers repair. Omics ranks which patches enter the bag; it does not modulate learned attention weights | **No** (selection, not weighting) |
| **MRePath** (Qu et al.) | IJCAI 2025, pp. 1802–1810 | genomics | pooled | **Per-case dynamic modality weighting**: learnable MLPs produce mono-confidences `w_p, w_g`, combined via log-ratio holo-confidence and softmax, scaling the pooled pathology and genomic representations before hypergraph alignment fusion | **No** |
| Luo et al. | Scientific Reports 2025 | genomics | patch-level **and** logit-level | Two stages: genomic-embedding-as-query co-attention over patch tokens (MCAT-like), then subjective-logic evidence with Dempster–Shafer combination of per-modality risk scores, each carrying a confidence mass | **Yes** (co-attention stage) |

Foundational operators borrowed by the above (not pathology papers):

| Method | Venue + Year | What it is |
|---|---|---|
| **FiLM** (Perez et al.) | AAAI 2018 | Feature-wise linear modulation: a conditioning vector predicts per-channel `(γ, β)` applied as `γ ⊙ h + β` |
| **LMF** (Liu et al.) | ACL 2018 | Low-rank multimodal fusion — the exact factorised-tensor scheme PORPOISE re-implements as `LRBilinearFusion` |
| **OGM-GE** (Peng et al.) | CVPR 2022 | On-the-fly gradient modulation to stop one modality dominating optimisation |
| **P-NET** (Elmarakeby et al.) | Nature 2021 | Sparse, pathway-structured (Reactome) network over genes — the canonical gene-grouping prior |
| **DAFT** (Pölsterl et al.) | MICCAI 2021 | Dynamic Affine Feature Map Transform: tabular clinical variables predict a scale+offset applied affinely to 3D-CNN feature maps — a FiLM variant for tabular→image conditioning in medical imaging (no MIL, no patches) |
| **ModDrop** (Neverova et al.) | arXiv 2014/2015 (widely cited as IEEE TPAMI 2016 — **venue UNVERIFIED here**) | Randomly dropping whole modalities during training so the network degrades gracefully when one is absent |

---

## 2. What is already occupied

These moves are clearly claimed. The thesis must cite them, not claim them.

1. **Letting the second modality determine patch weighting via cross-attention.** MCAT (ICCV 2021)
   owns the basic form: omics tokens as queries over patch tokens as keys/values. MOTCat swaps the
   similarity for an optimal-transport plan; CMTA makes it bidirectional; SurvPath scales it to
   hundreds of pathway tokens and deletes patch–patch attention for memory. Four papers, three
   venues, one idea. **"We let the second modality see individual patches" is not novel.**

2. **Gated convex combination on pooled embeddings.** Pathomic Fusion's gated Kronecker unit and
   PORPOISE's `PorpoiseMMF` are this. The repo's existing `gated` operator is a simplification of it.

3. **Low-rank bilinear / tensor pooling for WSI+omics.** Not merely "recombinable" — PORPOISE's own
   released code contains `LRBilinearFusion(rank=16)` with modality-specific factor matrices, i.e.
   LMF (ACL 2018) already ported into the pathology fusion setting by the PORPOISE authors.

4. **Grouping the transcriptome by biological prior before fusing.** MCAT (6 functional signature
   groups), SurvPath (Hallmarks+Reactome pathway tokens), P-NET (sparse Reactome-structured layers).
   Reducing 20530 genes via gene sets is standard practice, not a contribution.

5. **Kronecker / outer-product fusion of two pooled vectors.** Pathomic Fusion, and as the
   `bilinear` option in MCAT, MOTCat, CMTA and PORPOISE.

6. **Using omics as a training-time signal that is absent at inference.** TANGLE (contrastive
   slide↔expression alignment) and its reconstruction head. If the thesis wants "RNA improves the
   image model but is not needed at test time", TANGLE has the strongest prior claim.

7. **Per-modality reliability weighting under an evidential framework.** EsurvFusion (preprint)
   does exactly this for multimodal survival, at decision level.

8. **Clinical/tabular variables reshaping MIL attention over patches.** PersAM (J. Pathol. Inform.
   2023) already does this: clinical-factor tokens are put in the same Transformer sequence as
   patch tokens and their patch-similarity scores multiplicatively filter the MIL attention. This
   is the single most dangerous paper for this thesis's novelty claim and **must** be cited.
   The mechanism is attention-based score filtering, not affine (γ, β) modulation.

9. **Tabular→image affine conditioning in medical imaging.** DAFT (MICCAI 2021) is FiLM with a
   tabular conditioner, on 3D-CNN feature maps. The *idea* of "clinical variables produce a scale
   and shift" in medical imaging is claimed; only its use on MIL attention logits is not.

10. **Omics as a training-time-only signal so the deployed model is image-only.** G-HANet (IEEE TMI
    2025) distils genotype↔phenotype associations into the patch-aggregation weights and needs no
    genomics at test time; TANGLE does the contrastive-pretraining version.

11. **Factorised / low-rank bilinear pooling for cancer multimodal prognosis.** HFBSurv
    (Bioinformatics 2022) — though with hand-crafted CellProfiler features rather than MIL.

---

## 2b. Where the thesis's own ER numbers sit

Necessary calibration before claiming anything. Image-only ER-from-H&E is a solved-ish problem:

| Work | Venue+Year | Cohort | ER AUROC |
|---|---|---|---|
| ReceptorNet (Naik et al.) | Nat. Commun. 11:5727, 2020 | ABCTB+TCGA, 3,399 pts | 0.92 test; **TCGA 0.861** |
| Shamai et al. | Communications Medicine 4:276, 2024 | 7,950 pts, 4 cohorts | 0.951 CV; **TCGA test 0.930** |
| Høibø et al. | Front. Med. 2025 | 4 Norwegian TMA cohorts, 2,220 pts | 0.95 internal / 0.91 external — **UNI + CLAM, i.e. this thesis's exact stack** |
| MKD-CLOD (Zhang et al.) | MICCAI 2025 | TCGA-BRCA 5-fold | pathology-only **0.9331**, +RNA **0.9581** |

So the thesis's 0.896 → 0.941 is in-range but not record-setting, and **the +RNA gain has already been
reported at almost exactly the same magnitude (+0.025) by MKD-CLOD on the same cohort and label**.
The contribution therefore cannot be "RNA helps ER prediction" — that is now published prior art.
It has to be the *mechanism*.

---

## 3. What appears unoccupied or under-tested

1. **FiLM-style affine modulation of MIL attention logits.** Across arXiv/Semantic Scholar/Crossref/
   PubMed/DBLP/OpenAlex sweeps, no paper was found applying FiLM `(γ, β)` conditioning to MIL
   attention scores in WSI analysis. The nearest neighbours are DAFT (MICCAI 2021 — FiLM on 3D-CNN
   feature maps, no MIL) and PersAM (2023 — conditions attention, but by multiplicative
   Transformer-similarity filtering, not affine modulation). This is the cleanest open slot found.

2. **One mechanism required to serve two second-modalities with opposite statistics.** Every method
   in §1 is designed around exactly one second modality — grouped transcriptomics. None is evaluated
   on both a 20530-dim dense vector and a 24-dim sparse one-hot vector with the same operator.
   BioFusionNet is the only paper using genomics *and* clinical data on breast WSIs and it uses
   three separate mechanisms (VAE + co-dual-cross-attention + a clinical FFN).

3. **A modality known to be uninformative as a designed negative control.** The prior chapter's null
   clinicopath result (0.894 vs 0.896) is unusually clean. The fusion literature almost never
   reports a second modality that fails to help, so there is no established evidence about which
   operators degrade gracefully. "Does the operator provably reduce to the unimodal model when the
   second modality is noise?" is an unanswered, testable question, and the two-modality design in
   point 2 makes it answerable within one thesis.

4. **Covariate-conditioned patch attention in a *classification* setting.** Genomics-as-query over
   patch tokens exists only for survival (MCAT, MOTCat, CMTA, SurvPath, Luo et al.). The two
   published WSI+omics papers on binary IHC labels fuse post-pooling (MKD-CLOD) or use omics to
   *retrieve* patches (HERO). HERO does adapt MCAT and SurvPath as classification baselines, so the
   comparison is now expected — but the operators themselves have not been proposed for this task.

5. **Per-case reliability weighting coupled to attention rather than to pooled features.**
   MRePath (IJCAI 2025) and Luo et al. (2025) do per-case modality confidence, but always on pooled
   embeddings or logits, and always for survival.

6. **Modality dominance measured in the pathology fusion setting.** OGM-GE (CVPR 2022) shows one
   modality can suppress another's learning. RNA is plausibly near-sufficient for ER (ESR1), which
   is exactly the dominance regime — yet no pathology fusion paper appears to measure whether the
   WSI branch under-trains when RNA is present. This is a cheap, honest, publishable ablation.

**A warning that belongs in the thesis.** MKD-CLOD's HER2 result jumps 74.56 → 95.76 with RNA, and
HERO reports ER 0.994 using methylation+miRNA. Both are leakage signatures: *ERBB2*/*ESR1*
expression and ER-linked methylation encode the IHC label almost directly. Any RNA-fusion gain on an
IHC label must be argued against this, not just reported. The thesis's +0.044 is modest enough to be
credible, which is itself worth saying explicitly.

---

## 4. Honest novelty assessment

**(a) FiLM-style conditioning of MIL patch-attention scores on the tabular vector**
→ **RECOMBINATION, and the strongest of the six.** The affine-conditioning primitive is FiLM
(AAAI 2018), already brought into medical imaging as DAFT (MICCAI 2021). Conditioning MIL attention
on clinical variables is occupied *in effect* by **PersAM (J. Pathol. Inform. 2023)** — but by
multiplicative attention-similarity filtering, not by predicting `(γ, β)` for the attention network.
No FiLM-on-MIL-attention paper was found. Defensible claim: *"a FiLM-conditioned attention-MIL, a
combination of Perez et al. 2018 and Ilse et al. 2018 not previously applied to WSI attention, and
the first to be required to serve both a 20530-dim and a 24-dim conditioner."* Not defensible:
"we are the first to let non-image data influence which patches matter" — PersAM and MCAT own that.

**(b) Cross-attention from tabular tokens to individual patch tokens before pooling**
→ **ALREADY DONE.** This is literally MCAT (ICCV 2021): `coattn(h_omic_bag, h_path_bag, h_path_bag)`
— omics tokens as queries, patch tokens as keys/values. Also MOTCat (ICCV 2023, OT plan instead of
dot product), CMTA (ICCV 2023, bidirectional), SurvPath (CVPR 2024, 331 pathway tokens with
patch–patch attention pruned), and Luo et al. (Sci. Rep. 2025). Do not claim this. It is a
*baseline the thesis must implement*, not a contribution.

**(c) Modality dropout during training for graceful degradation at inference**
→ **RECOMBINATION, but weak on its own.** ModDrop (Neverova et al., 2014/15) is the generic idea.
In pathology the missing-modality problem is an active 2024–2026 area with better-engineered
answers than dropout: DRIM (MICCAI 2024, variable-length modality sequence), DisPro (CVPR 2025),
Qu et al. (MICCAI 2025, memory bank), and structurally G-HANet (TMI 2025) and MKD-CLOD (MICCAI 2025),
both of which train with omics and infer without it. Plain modality dropout is a sensible
*regulariser and ablation* for this thesis; it is not a contribution.

**(d) Low-rank bilinear / tensor pooling on pooled embeddings**
→ **ALREADY DONE.** LMF (ACL 2018) is the operator; **PORPOISE's released code contains
`LRBilinearFusion(rank=16)` with modality-specific factor matrices**, i.e. the PORPOISE authors
already ported it to WSI+omics. HFBSurv (Bioinformatics 2022) is hierarchical factorised bilinear
for cancer prognosis (though with CellProfiler features, not MIL), and Keshvarikhojasteh et al.
(arXiv 2026) do low-rank bilinear on ABMIL + RNA + clinical. Additionally it is a *pooled-level*
operator, so it cannot address the late-fusion gap that motivates the chapter.

**(e) A learned per-case reliability/confidence weight over modalities**
→ **ALREADY DONE in substance, and partly already in this repo.** The lineage is TMC
(ICLR 2021 → IEEE TPAMI 2023); in pathology, MRePath (IJCAI 2025) computes exactly a per-case
learnable confidence per modality, Luo et al. (2025) does evidential per-modality confidence, and
EsurvFusion (preprint) does reliability discounting. More damning: **the repo's existing `gated`
operator already computes a per-case, per-dimension gate from both modalities** — a scalar
reliability weight is a strictly weaker version of the baseline. Only worth pursuing if the weight
gates *attention conditioning* rather than pooled features, in which case it is a rider on (a), not
a contribution in itself.

**(f) Sparsity or gene-grouping priors over the 20530 genes**
→ **ALREADY DONE.** MCAT (6 functional signature groups), SurvPath (281 Reactome + 50 Hallmark
tokens), MMP (pathway prototypes), P-NET (Nature 2021, sparse Reactome-structured layers), HERO
(sparse pathway-to-morphology prior). Gene grouping is the standard preprocessing step in this
literature, not a mechanism. It is, however, *necessary engineering* if the conditioner is to be
dimension-agnostic — frame it as implementation, never as novelty.

### The defensible claim

Only (a) survives, and only as a conjunction. Stated honestly:

> A FiLM-style conditioning layer that predicts a scale and shift for the gated-attention scoring
> network of CLAM from an arbitrary second-modality vector — evaluated for the first time (i) on a
> binary IHC-derived label rather than survival, (ii) with a single unmodified mechanism serving
> both a 20530-dim transcriptome and a 24-dim clinicopathology vector, and (iii) against a second
> modality that is known *a priori* not to help, as a test of graceful degradation.

Every ingredient is old (FiLM 2018; attention-MIL 2018; clinical-conditioned attention 2023). The
conjunction, and the evaluation regime, are not. That is a master's-thesis-scale contribution and
should be written as such. Claiming a new fusion *paradigm* would not survive contact with MCAT.

**Required baselines** (all now expected by reviewers, given HERO already ran two of them):
CLAM-only; RNA-only; the repo's four late-fusion operators; MCAT and SurvPath adapted to binary
classification; and MKD-CLOD as the direct same-task, same-cohort competitor.

---

## 5. Sources

**VERIFIED — retrieved directly in this session** (venue/year/authors confirmed from a tool result,
and where marked "code-read" the released implementation was read):

- Ilse, Tomczak, Welling. *Attention-based Deep Multiple Instance Learning.* ICML 2018. https://arxiv.org/abs/1802.04712 (venue from arXiv comments)
- Lu, Williamson, Chen, Chen, Barbieri, Mahmood. *Data-efficient and weakly supervised computational pathology on whole-slide images.* Nature Biomed. Eng. 2021. doi:10.1038/s41551-020-00682-w
- Perez, Strub, de Vries, Dumoulin, Courville. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018. https://arxiv.org/abs/1709.07871
- Liu, Shen, Lakshminarasimhan, Liang, Zadeh, Morency. *Efficient Low-rank Multimodal Fusion with Modality-Specific Factors.* ACL 2018. https://arxiv.org/abs/1806.00064
- Peng, Wei, Deng, Wang, Hu. *Balanced Multimodal Learning via On-the-fly Gradient Modulation.* CVPR 2022. https://arxiv.org/abs/2203.15332
- Elmarakeby et al. *Biologically informed deep neural network for prostate cancer discovery.* Nature 2021. doi:10.1038/s41586-021-03922-4
- Chen, Lu, Wang, Williamson, Rodig, Lindeman, Mahmood. *Pathomic Fusion.* IEEE TMI 2022. doi:10.1109/TMI.2020.3021387 · https://arxiv.org/abs/1912.08937
- Chen, Lu, Weng, Chen, Williamson, Manz, Shady, Mahmood. *Multimodal Co-Attention Transformer (MCAT).* ICCV 2021, pp. 4015–4025. https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Multimodal_Co-Attention_Transformer_for_Survival_Prediction_in_Gigapixel_Whole_Slide_ICCV_2021_paper.html — **code-read**, vendored at `project/MCAT/models/model_coattn.py`
- Chen, Lu, Williamson, Chen, Lipkova, Noor, Shaban, Shady, Williams, Joo, Mahmood. *Pan-cancer integrative histology-genomic analysis via multimodal deep learning (PORPOISE).* Cancer Cell 2022. doi:10.1016/j.ccell.2022.07.004 · https://arxiv.org/abs/2108.02278 — **code-read** (`mahmoodlab/PORPOISE`, `models/model_porpoise.py`)
- Xu, Chen. *MOTCat.* ICCV 2023. https://arxiv.org/abs/2306.08330 (venue from arXiv comments: "accepted by ICCV 2023") — **code-read** (`Innse/MOTCat`)
- Zhou, Chen. *Cross-Modal Translation and Alignment for Survival Analysis (CMTA).* ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Cross-Modal_Translation_and_Alignment_for_Survival_Analysis_ICCV_2023_paper.pdf — **code-read** (`FT-ZHOU-ZZZ/CMTA`)
- Jaume, Vaidya, Chen, Williamson, Liang, Mahmood. *Modeling Dense Multimodal Interactions Between Biological Pathways and Histology (SurvPath).* CVPR 2024, pp. 11579–11590. https://openaccess.thecvf.com/content/CVPR2024/html/Jaume_Modeling_Dense_Multimodal_Interactions_Between_Biological_Pathways_and_Histology_for_CVPR_2024_paper.html — **code-read** (`mahmoodlab/SurvPath`)
- Jaume, Oldenburg, Vaidya, Chen, Williamson, Peeters, Song, Mahmood. *Transcriptomics-guided Slide Representation Learning (TANGLE).* CVPR 2024 (Oral). https://arxiv.org/abs/2405.11618 — **code-read** (`mahmoodlab/TANGLE`)
- Song, Chen, Jaume, Vaidya, Baras, Mahmood. *Multimodal Prototyping for cancer survival prediction (MMP).* ICML 2024. https://arxiv.org/abs/2407.00224 (venue from arXiv comments)
- Xiong et al. *MoME: Mixture of Multimodal Experts.* MICCAI 2024 (early accept). https://arxiv.org/abs/2406.09696 — note: **complete-modality** fusion, not a missing-modality method
- Xu, Zhou, Zhao, Wang, Yang, Chen. *Distilled Prompt Learning for Incomplete Multimodal Survival Prediction (DisPro).* CVPR 2025. https://arxiv.org/abs/2503.01653
- Takagi, Hashimoto, Masuda, Miyoshi, Ohshima, Hontani, Takeuchi. *Transformer-based Personalized Attention Mechanism for Medical Images with Clinical Records (PersAM).* Journal of Pathology Informatics 14:100185, 2023. https://arxiv.org/abs/2206.03003
- Ding, Zhou, Metaxas, Zhang. *Pathology-and-genomics Multimodal Transformer (PathOmics).* MICCAI 2023. https://arxiv.org/abs/2307.11952 (venue from arXiv comments) — operator **not** code-read
- Zhang, Hao, Chen, Xu, Cong, Lu, Xu. *Multi-modal Knowledge Decomposition based Online Distillation for Biomarker Prediction in Breast Cancer Histopathology (MKD-CLOD).* MICCAI 2025. https://arxiv.org/abs/2508.17213 (venue from arXiv comments)
- Li, Su. *HERO: Hypothesis-Driven Evidence Retrieval from Omics.* MICCAI 2026 (early accept). https://arxiv.org/abs/2606.21174 (venue from arXiv comments)
- Mondol, Millar, Sowmya, Meijering. *BioFusionNet.* IEEE JBHI 2024. doi:10.1109/JBHI.2024.3418341 · https://arxiv.org/abs/2402.10717 — operator described **from abstract only**
- Huang, Xing, Lin, Ruan, Feng. *EsurvFusion.* https://arxiv.org/abs/2412.01215 — **arXiv preprint, no peer-reviewed venue found**
- Neverova, Wolf, Taylor, Nebout. *ModDrop: adaptive multi-modal gesture recognition.* https://arxiv.org/abs/1501.00102 — arXiv only; the commonly cited IEEE TPAMI 2016 venue is **UNVERIFIED** (no journal-ref on arXiv)
- Keshvarikhojasteh, Pluim, Veta. *Attention-Based Multimodal Survival Prediction with Cross-Modal Bilinear Fusion.* https://arxiv.org/abs/2605.13897 — **arXiv preprint, no venue**
- Lipkova, Chen, Chen, Lu, Barbieri, Shao, Vaidya, Mahmood et al. *Artificial intelligence for multimodal data integration in oncology.* Cancer Cell 2022. doi:10.1016/j.ccell.2022.09.012 — general review
- Zhang, Xu, Chen, Xie, Chen. *Prototypical Information Bottlenecking and Disentangling (PIBD).* ICLR 2024. https://arxiv.org/abs/2401.01646

**VERIFIED BY DELEGATED SEARCH — retrieved by a search agent from primary sources, not re-checked
directly by me.** Treat venue/author strings as reliable but confirm before they enter a bibliography:

- Pölsterl, Wolf, Wachinger. *Combining 3D Image and Tabular Data via the Dynamic Affine Feature Map Transform (DAFT).* MICCAI 2021, doi:10.1007/978-3-030-87240-3_66. https://arxiv.org/abs/2107.05990
- Robinet, Berjaoui, Kheil, Cohen-Jonathan Moyal. *DRIM: Learning Disentangled Representations from Incomplete Multimodal Healthcare Data.* MICCAI 2024 (Spotlight), doi:10.1007/978-3-031-72384-1_16
- Wang, Zhang, Xu, Imoto, Chen, Song. *Histo-Genomic Knowledge Association (G-HANet).* IEEE TMI 44(5):2170–2181, 2025, doi:10.1109/TMI.2025.3526816. https://arxiv.org/abs/2403.10040
- Qu, Yang, Di, Gao, Su, Song, Fan. *Memory-Augmented Incomplete Multimodal Survival Prediction.* MICCAI 2025, doi:10.1007/978-3-032-05127-1_31. https://arxiv.org/abs/2506.19324
- Li, Wu, Li, Wang. *HFBSurv: hierarchical multimodal fusion with factorized bilinear models.* Bioinformatics 38(9):2587–2594, 2022, doi:10.1093/bioinformatics/btac113
- Han, Zhang, Fu, Zhou. *Trusted Multi-View Classification.* ICLR 2021 (https://arxiv.org/abs/2102.02051); extension *Trusted Multi-View Classification With Dynamic Evidential Fusion*, IEEE TPAMI 45(2):2551–2566, 2023, doi:10.1109/TPAMI.2022.3171983
- Qu et al. *MRePath: Multimodal Cancer Survival Analysis via Hypergraph Learning with Cross-Modality Rebalance.* IJCAI 2025, pp. 1802–1810. https://www.ijcai.org/proceedings/2025/0201.pdf
- Luo et al. *Multimodal multi-instance evidence fusion neural networks for cancer survival prediction.* Scientific Reports 2025, doi:10.1038/s41598-025-93770-3
- Naik et al. *Deep learning-enabled breast cancer hormonal receptor status determination from base-level H&E stains (ReceptorNet).* Nature Communications 11:5727, 2020
- Shamai et al. *Clinical utility of receptor status prediction in breast cancer…* Communications Medicine 4:276, 2024, doi:10.1038/s43856-024-00695-5
- Høibø et al. *Predicting estrogen receptor status from HE-stained breast cancer slides using AI.* Frontiers in Medicine 2025, doi:10.3389/fmed.2025.1593143
- Zhang, J. et al. *Prompt-MIL.* MICCAI 2023 (Oral). https://arxiv.org/abs/2303.12214
- Qu, L. et al. *Pathology-knowledge Enhanced Multi-instance Prompt Learning (PEMP).* ECCV 2024, doi:10.1007/978-3-031-73247-8_12. https://arxiv.org/abs/2407.10814
- Shi et al. *ViLa-MIL.* CVPR 2024. https://arxiv.org/abs/2502.08391
- Zhao et al. *PTCMIL.* MICCAI 2025, doi:10.1007/978-3-032-05182-0_49. https://arxiv.org/abs/2507.18848

**UNVERIFIED — mechanism or venue could not be confirmed; do not cite without checking:**

- Lai, Vong, Yan, Liang. *Interactive MIL network for WSI analysis (iMIL).* Expert Systems with Applications 297, 2026, doi:10.1016/j.eswa.2025.129338 — paywalled; the "Prompt-Guided Attention over clinical text" description is from the abstract only, and it is the one paper that could partially pre-empt direction (a). **Check this before finalising the novelty claim.**
- Xing et al. *MIST: Bridging the Modality Bottleneck… via Virtual Molecular Staining.* https://arxiv.org/abs/2605.16392 — preprint, no venue
- Xing et al. *DPsurv.* https://arxiv.org/abs/2510.00053 — preprint, no venue
- Xing et al. *Evidential Fusion Network for Multimodal Survival Prediction under Missing Modalities.* https://arxiv.org/abs/2606.20757 — preprint, no venue
- Raahemi et al. *Adaptive Confidence-weighted Expansion (ACE).* https://arxiv.org/abs/2607.20742 — ICPR 2026 claimed in arXiv comments, acceptance not independently confirmed
- Raza, Azam, Qaiser, Rajpoot. *PS3.* https://arxiv.org/abs/2509.20022 — ICCV 2025 reported by delegated search, not confirmed here
- Wang, M. et al. *DAMLN.* MICCAI 2024, doi:10.1007/978-3-031-72378-0_9 — paywalled; the ER/PR/HER2 numbers quoted in §2b come from a third party's re-implementation, not the original
- Ruffini et al. *Handling Missing Modalities in Multimodal Survival Prediction for NSCLC.* https://arxiv.org/abs/2601.10386 — preprint, mechanism not retrieved
- Jennings et al. *Machine learning-based multimodal prognostic models… a systematic review.* https://arxiv.org/abs/2507.16876 — preprint, no venue
