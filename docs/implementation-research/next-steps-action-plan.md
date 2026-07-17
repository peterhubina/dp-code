    # Next steps: publishable, externally-validated multimodal WSI pipeline

## Current state (verified against code)

- **Stack:** TCGA-BRCA, UNI2-h features (1536-d) → CLAM attention-MIL; matched RNA-seq; AMIL survival thread.
- **Scale:** ~1,055 WSI patients, 1,045 RNA 4-class samples, **1,004 matched**. PAM50 imbalanced: LumA 562 / LumB 209 / Basal 192 / Her2 82.
- **Trained results:** WSI-only PAM50 test AUC **0.893** (acc 0.72); RNA-only **0.64** (acc 0.41); gated WSI+RNA fusion **0.974** (acc 0.89); AMIL survival c-index **0.654 ± 0.060**.
- **The 0.974 fusion number is leakage-inflated.** `CLAMRNAFusion` feeds the full ~20,530-gene transcriptome (top-10k by train-fold variance) to predict PAM50 labels that are computed *from that same expression matrix* (`project/data/pam50.R`). Target leakage by construction — the 0.974 vs 0.893 gap is an artifact.
- **No usable external validation.** Only held-out TCGA folds + WSI-only transfer to tiny HSI-BC (n=47) / CTC-nou (n=203). No CPTAC / BCR-Net / genefu cohort exists on disk; the fusion model has never been externally validated.
- **Bug:** `train_pam50_final.sh` `--data_root_dir .datasets/embeddings` is empty; real data is `.datasets/tcga-brca/embeddings`.

**Reconciliation of the four memos:** reliability ≠ clinical need. ER = most reliable (IHC label, AUROC 0.80–0.82, CPTAC external) but weak standalone need (ER IHC is cheap/mandated). Oncotype-DX/HER2-low = real need, but proxy labels + crowded true-label SOTA (Orpheus 0.89, Shamai–Aran ~0.90). PAM50 4-class fragile (HER2E n=57 → F1 0.545). OS underpowered for BRCA. **Publishability verdict: lead with the fusion contribution, not the task.**

## Ranked next steps

### #1 — ER status (binary), WSI + RNA-seq gated fusion, external = CPTAC-BRCA
Pick this: it converts the leaky fusion into a clean one at near-zero code cost, and CPTAC is the only public cohort with **both** modalities, so it can externally validate the *full fusion*, not just the WSI branch.
- **Why:** ER label is IHC-derived, not RNA-derived → RNA fusion is **not leakage** (unlike PAM50/ODX). Reuse UNI2-h→CLAM + gated head verbatim; only swap the label and extract UNI2 features on CPTAC.
- **Data:** train TCGA-BRCA (public); external CPTAC-BRCA (public, TCIA CC BY 4.0, WSI+RNA+labels).
- **Experiments:** relabel binary ER (harmonize 1%/10% cutoff); **WSI-alone vs WSI+RNA ablation** (mandatory); split held-out by tissue-submitting site (Howard 2021); extract CPTAC UNI2 features, run frozen model, report WSI-alone + fusion externally; stain-normalize.
- **Effort:** Low (label swap + CPTAC feature extraction, no new architecture).
- **Publishability:** contribution = leakage-controlled test of whether transcriptomic fusion beats H&E-alone, externally validated. Venue: COMPAY/MOVI, *J Pathol Informatics*, *Sci Reports*, *Cancers*. **Risk:** ER morphology-saturated → RNA may not add; "predicts something already cheap." Preempt: frame as fusion methodology + honest ablation; position ER as QC/discordance flagging.

### #2 — Oncotype-DX recurrence proxy, WSI + clinicopathologic fusion, external = BCR-Net (true labels)
Clinical-need payload and natural chapter 2 if #1's ablation shows no ER headroom.
- **Why lower:** proxy labels invite circularity attack; true-label SOTA is crowded; external cohort thin (n=99). But highest clinical need + real fusion headroom (WSI-alone ~0.78).
- **Data:** train TCGA-BRCA **ER+/HER2− subset** (~500–600), genefu 21-gene proxy labels; external BCR-Net (public, Zenodo CC BY 4.0, ~99 pts, **true** ODX, ships patches). Optional 2nd external: CPTAC proxy.
- **Experiments:** genefu → quantile-binarize; **fuse clinicopath, NOT RNA** (label is RNA-derived → RNA would leak; reuse the generic tabular encoder with a new feature table); WSI-alone vs fusion ablation; **proxy-fidelity study** (genefu-RS vs true RS on BCR-Net). External validation is clean for the WSI branch; fusion external validation limited (BCR-Net clinicopath unconfirmed).
- **Effort:** Medium.
- **Publishability:** frame as multimodal recurrence stratification with proxy-fidelity honesty + silver→gold external validation; **never** "we predict Oncotype-DX." **Risk:** circularity + loses to Orpheus/Shamai on labels+scale.

### #3 — HER2/ERBB2 (binary), WSI + RNA fusion, external = HEROHE
Clean fusion showcase (ERBB2 amplification only partly visible in morphology → real RNA headroom; non-leaky IHC/ISH label). Ranked third because **HEROHE (509 public WSI) has no RNA** → validates the WSI branch, not the fusion. Keep as strong secondary/robustness cohort. HER2-low (the true unmet need) has no labeled public WSI cohort today.

### #4 — Keep PAM50
Only as a de-leaked **baseline**: re-run holding PAM50 signature genes out of the RNA input (or fuse clinicopath); expect 0.974 → ~0.89. External-validatable on CPTAC (has PAM50). Lowest novelty; fragile minority class; fusion story evaporates once de-leaked — which is why #1 moves to ER.

## Open decisions for the author

- **Demote PAM50 to a de-leaked baseline, make ER primary?** → **Yes.** Keeps all infra, gains a clean clinically-framable story.
- **Accept genefu proxy ODX labels?** → **Yes, but** only under fusion-first framing + proxy-fidelity analysis, as chapter 2 (not the sole contribution).
- **One task or ER→ODX arc?** → **The arc.** ER proves pipeline + external fusion validation; ODX supplies the clinical claim. Reconciles memo 1 (reliability) and memo 2 (need).
- **Pursue gated cohort (Dartmouth/BMIRDS, TAILORx)?** → File in parallel as a stretch; don't block the thesis. All primary work stays public (TCGA, CPTAC, BCR-Net, HEROHE).

## Risks & mitigations

- **RNA-label leakage (in code now):** use IHC-labeled tasks (ER/HER2) for RNA fusion; for ODX/PAM50 fuse clinicopath or hold signature genes out. Always report WSI-alone beside fusion.
- **Fusion may not beat WSI-alone (ER saturation):** make the ablation the primary result; a null motivates the ODX chapter. Don't hide it.
- **Site-signature confounding (Howard 2021):** split/hold out by submitting site; report per-site.
- **Domain shift:** stain-normalize; expect absolute-AUC drop (Arslan CPTAC 0.567–0.672 validated comparability, not high absolute); harmonize label cutoffs.
- **Small external n / weak labels:** report CIs; add a 2nd external cohort; run the proxy-fidelity concordance study.
- **Verify before sizing:** CPTAC WSI count differs across memos (382 vs 642); BCR-Net clinicopath availability unconfirmed — check TCIA/Zenodo live.
- **Infra:** repoint `train_pam50_final.sh` to `.datasets/tcga-brca/embeddings`.
