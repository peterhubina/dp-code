# Her2 collapse: remediation plan

What to do about `docs/cnv-wsi-fusion-external-validation.md` §5, where the WSI arm calls Her2 0/14 on
the 114 external CPTAC cases and 26/51 on the 599 TCGA cases that have CLAM out-of-fold predictions.
Ordering rests on diagnostics run 2026-08-09; nothing was retrained, re-extracted or downloaded. Both
external anchors reproduced exactly through `tools/pam50_arms.py` — macro AUROC / balanced accuracy /
Her2 recall: WSI raw 0.847 / 0.513 / 0-of-14, CNV 39 arms 0.888 / 0.716 / 12-of-14, equal-weight mean
0.909 / 0.646 / 6-of-14.

## What the diagnostics settled

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 4 | Slide aggregation hides the calls | **Refuted** | **0 of 378 slides argmax to Her2**; cohort-wide max single-slide `p_Her2` is 0.2376, on a true-LumA slide. Max, median and geometric-mean pooling all give 0/14. |
| 7 | The CNV arm's 12/14 is one amplicon (17q) | **Refuted** | 17q carries 1.7% of the Her2 coefficient mass. Without 17q: still 12/14, macro AUROC 0.890. 17q alone: 5/14, 0.561. A distributed arm pattern, and a fair yardstick. |
| 5 | CPTAC Her2 is a different class | **Refuted as provenance** | Our labels agree 114/114 with cbioportal's own `PAM50` column. That 7 of 14 PAM50-Her2 cases are `ERBB2_PROTEOGENOMIC_STATUS` positive is normal for the HER2-enriched centroid; the WSI arm detects PAM50-Her2 (0.860) far better than ERBB2 status (0.651). |
| 2 | Tissue segmentation dilutes the bag | **Refuted as dilution** | CPTAC bags are 3.4× smaller (median 2,588 vs 8,772 patches), yet TCGA slides with CPTAC-sized bags (≤2,814, n=63) still average `p_Her2` 0.0921 with 2 Her2 argmaxes against CPTAC's 0.0378 and 0. |
| 3 | Resampling | **Supported, inseparable** | Within-case paired over 47 cases, biology fixed: `p_Her2` is 2.47× higher on the 20×-native slide than the 40×-native one, higher in 45/47, Wilcoxon p < 0.0001; the whole vector rotates (`p_LumB` +0.176, `p_LumA` −0.162). But regime is perfectly collinear with scanner (SS1553 / SS1289, 152 vs 226, zero overlap), magnification, scan year and codec (JPEG/RGB vs JPEG-2000/YUV16), so nothing here isolates resampling. |
| 1 | Stain and colour shift | **Untested across cohorts** | No TCGA slide is on disk, so no cross-cohort pixel comparison exists. Within CPTAC, real patch pixels show the two scanners differ (luminance 182.1 vs 169.0, saturation 30.7 vs 37.2, optical density 0.189 vs 0.217; paired Wilcoxon p = 0.0005). Item 3 would decide it. |
| 6 | Feature-space geometry | **Supported — the mechanism** | Cohort of origin is perfectly readable from a *single* patch feature: AUROC 1.0000, accuracy 0.9990, slide-grouped 5-fold. Scanner within CPTAC: also 1.0000. The shift is **5.4× the Her2 class signal** (median standardised mean difference 0.806 vs 0.148; 621 of 1536 dimensions move over one pooled SD; the *largest* Her2 dimension, 0.649, is below the *median* cohort shift). |
| 8 | Preservation type (added) | **Untested** | All 1009 TCGA training slides are `DX`, FFPE diagnostic. CPTAC's type is recorded nowhere here — not `wsi_manifest.csv`, not `cohort.csv`, and `procurement/*` in `cptac_pancancer_clinical_breast.csv` is null in all 134 rows. Fernandez-Romero 2026 drew 387 *flash-frozen* CPTAC-BRCA slides; Borji drew 122 FFPE. |

Two results fix the ordering. **CLAM's attention head is innocent**: a plain logistic probe on
slide-*mean* UNI2-h features, fitted on TCGA alone, reproduces the MIL head — Her2 AUROC 0.881
internal, 0.833 external, 0 of 378 slides argmax to Her2. And the 56 cases scanned wholly on SS1289
give WSI macro AUROC 0.894 with Her2 one-vs-rest **0.927** — and still 0/7 calls, so a homogeneous
acquisition regime does not fix the argmax. The failure therefore sits in the encoder's feature space,
below the classifier. Fernandez-Romero 2026 corroborates: HER2-enriched "complete performance collapse
(RPD = 1.000)" across 13 encoders and 3 MIL architectures on TCGA→CPTAC, with 80.0% of *RPD variance*
attributed to staining plus feature-space divergence
(`docs/implementation-research/PAM50/sota-comparison-cnv-fusion.md:588` — variance explained, slightly
narrower than the report's "80% of transfer degradation").

## Work items

Measured here: UNI2-h re-extraction runs at **43 patches/s** on one RTX 3090 (14,606 patches in
337.9 s, fp32, batch 256, 8 workers; only one GPU is visible to torch, not the two of record).
CLAM-MB inference runs at **12 ms/slide/fold**. The GDC API reports the TCGA-BRCA diagnostic set as
**1,133 SVS files, 1.159 TB**, 1,062 cases, against 1.4 TB free.

### 1. Label-free feature adaptation, then re-inference — ≈1 min GPU, no new disk

**Objective.** Re-score the frozen `pam50_final_s1` checkpoints on CPTAC features whose first and
second moments are matched to TCGA's. **Tests hypothesis 6**, and is the only free remediation.

**Commands.** Add `--feature_transform {none,moment_match,moment_match_per_scanner}` to
`tools/cptac/infer_cptac_pam50.py`, applied inside `infer_slide` so no adapted `.h5` is written:

```bash
python tools/cptac/infer_cptac_pam50.py --feature_transform moment_match_per_scanner \
    --feature_dir .datasets/cptac-brca/embeddings \
    --dataset_csv .datasets/cptac-brca/cptac_brca_pam50_dataset.csv \
    --ckpt_dir .scratch/results/pam50_final_s1 \
    --output_dir .scratch/cptac_validation/results/predictions_mmps
python tools/cptac/summarise_predictions.py .scratch/cptac_validation/results/predictions_mmps
```

`dp-analysis cnv_wsi_fusion` cannot be pointed at that directory — `assert_paths_reachable` in
`dpcode/cli/analysis.py` refuses `paths.*` overrides deliberately — so recomputing the fusion table
needs a variant of `evaluate_cnv_wsi_fusion.external()` taking the predictions path as an argument.

**Inputs, all on disk.** The 653 CPTAC `.h5` files (34 GB), the 10 checkpoints, the dataset CSV. TCGA
reference moments need one CPU pass over the 1126 TCGA `.h5` files (66 GB, I/O-bound; done twice in
these diagnostics). **Cost.** Under a minute of GPU for 10 folds over 378 slides, plus that pass. No
new disk.

**Pre-specified variants: exactly three** — none, whole-cohort moment matching, per-scanner moment
matching. Per-scanner is admissible because the split comes from `aperio.ScanScope ID`, never a label.
All three are transductive and are reported as label-free transductive adaptation, never as a clean
inductive external evaluation.

**Success.** Her2 recall materially above 0/14 with macro AUROC not falling. The proxy probe sizes the
expectation at **2–4 of 14 cases**: on slide-mean features the three variants move 0/378 → 18, 22 and
17 of 378 slides, macro AUROC 0.793 → 0.802–0.804, Her2 AUROC 0.833 → 0.809. Real, partial, well short
of the CNV arm's 12/14. **Falsified if** all three leave Her2 at 0/14 through the real MIL head: the
shift is then not a moment translation of the feature cloud and no cheap transductive fix exists.

**Risk.** CPTAC's own feature statistics enter the prediction, so this arm is not strictly "nothing
refit on CPTAC". It sits as a labelled extra row, never replacing the published WSI row, and every
fusion number beside it carries the CNV-alone arm (0.888 / 0.716 / 12-of-14) and is read against the
equal-weight mean (0.909), not against WSI-only.

**Gate.** ≥4 of 14 with macro AUROC ≥0.847 → report, go to item 3. 1–3 of 14 → report as partial, go
to item 2. 0/14 → item 2.

### 2. Settle the CPTAC preservation type — 0 GPU, ~1 hour

**Objective.** Determine whether the 378 CPTAC slides are FFPE, frozen or mixed. **Tests hypothesis
8**: every TCGA training slide is FFPE, so if CPTAC is frozen that is a larger shift than staining.

**Commands.** Two routes, the first unverified. (a) Resolve the PathDB node IDs already in
`wsi_manifest.csv` (`slide_id` 211767 …) against the CPTAC pathology data dictionary — two guessed
JSON endpoints on `pathdb.cancerimagingarchive.net` returned HTTP 404 here, so the route needs
finding, not assuming. (b) Dump the `stitch` thumbnails inside the `.h5` files plus
`openslide.associated.thumbnail` and have ~20 read for freezing artefact.

**Inputs.** All on disk. **Cost.** No GPU, no download, no disk. **Success.** A recorded, citable
preservation type. **Falsified if** neither route resolves it; the limitation then names "preservation
type unrecorded" as an open confound rather than assuming FFPE.

**Risk.** If the answer is frozen, every external WSI number acquires a second explanation and the
survey's note that this is "a larger, more specific and more checkable confound than staining"
(`sota-comparison-cnv-fusion.md:803`) becomes a correction to §5.

**Gate.** Frozen or mixed → this becomes the headline explanation and item 4 drops in priority, since
stain normalisation cannot fix a preservation difference. FFPE → item 3.

### 3. Sample 10 TCGA diagnostic slides and compare acquisition — 0 GPU, ~10 GB, ~1 hour

**Objective.** Measure TCGA's stain colour, codec and scanner against CPTAC's. **Tests hypothesis 1**,
closing the gap that made it untestable.

**Commands.** Query the GDC `files` endpoint for `TCGA-BRCA` / `SVS` / `Diagnostic Slide` (done here:
1,133 files, 1.159 TB), take 10 UUIDs spanning several tissue-source sites, fetch via
`https://api.gdc.cancer.gov/data/<uuid>` — `gdc-client` is not installed and is not needed for ten
files — then run this session's per-patch colour read.

**Inputs.** Network; ~10 GB against 1.4 TB free. **Cost.** No GPU. **Success.** A quantified
TCGA-vs-CPTAC gap in luminance, saturation, optical density and codec, comparable to the within-CPTAC
scanner gap already measured (Δ luminance 13.1, Δ saturation 6.5). **Falsified if** TCGA's statistics
sit inside the CPTAC range; staining is then not the axis and item 4 should not run.

**Risk.** Ten slides from 1,062 cases is a sample, not a census, and must be reported as one.

**Gate.** TCGA clearly outside the CPTAC range → item 4. Inside → stop the imaging thread.

### 4. Stain-normalised re-extraction of CPTAC only — 10.7 GPU-h, ~20 GB

**Objective.** Re-extract UNI2-h features for the 378 scored slides with Macenko normalisation to a
TCGA reference, reusing the recorded coordinates so bag composition is fixed and only the pixels vary.
**Tests hypothesis 1** at the level that would fix it.

**Commands.** `pip install torchstain` (absent, as are `staintools`, `histolab`, `spams`). Write the
coordinates as CLAM-shaped `patches/*.h5` — stored `coords` are `(1, N, 2)` while
`Whole_Slide_Bag_FP` indexes `(N, 2)`, so `coords_patching` is the dataset to copy — then drive
`project/CLAM/extract_features_fp.py --model_name uni2-h` with `UNI2H_CKPT_PATH` at
`.scratch/checkpoints/uni2-h/pytorch_model.bin` and a new `--feat_dir`, then item 1's two commands
against that directory.

**Inputs.** The 391 `.svs` files (64 GB) and the coordinates, all on disk. **Cost.** 1,662,971 patches
at 43 patches/s = **10.7 GPU-h** for the scored slides (16.9 for all 653); ~20 GB of new features.
AMP was not used in the pilot and would cut this materially.

**Pre-specified variants: exactly one** — Macenko to a single fixed TCGA reference patch, chosen from
item 3's sample before any CPTAC slide is scored. A second normaliser means amending this count here
first.

**Success.** Her2 recall above item 1's, macro AUROC not falling. **Falsified if** normalised features
give the same 0/14, or no more than item 1's free fix.

**Risk.** Wagner et al. 2026 deliberately omit stain normalisation because it "may impair
generalization of foundation models by altering biologically meaningful stain-morphology correlations
learned during large-scale pretraining" (`sota-comparison-cnv-fusion.md:614`); it can lose ground.
This also changes the test side only — TCGA features stay as Trident produced them, so the two cohorts
still do not pass through one pipeline.

**Gate.** Improvement over item 1 → report and stop. None → the negative result below.

### 5. Re-download TCGA and rebuild both sides — ≈74 GPU-h, 1.16 TB — last resort

Run only if items 1, 3 and 4 all fail and the timeline permits. 1.159 TB against 1.4 TB free works
only by extracting and deleting slide by slide, leaving under 200 GB of headroom. Then 11,281,660
patches at 43 patches/s = **72.5 GPU-h**, ~66 GB of features, ≈66 min for a 10-fold retrain. This is
the only route that puts both cohorts through one extraction pipeline, and the only one that makes an
encoder swap meaningful — Fernandez-Romero 2026 ranked UNI-2 first internally and seventh externally
while Prov-GigaPath ranked first externally. Any retrain writes to a **new `exp_code`**:
`pam50_final_s1` and the five ladder arms exist only in gitignored `.scratch/`, and
`run.overwrite=true` would destroy them irrecoverably. `splits/tcga_brca_subtyping_100/` and
`dataset_csv/tcga_brca_subtyping.csv` are reused unchanged.

## Where the plan stops, and the pre-committed negative result

Stop after item 4. Items 1–4 cost about 11 GPU-hours and no new cohort; item 5 costs 74 GPU-hours,
1.16 TB and a retrain to test a hypothesis items 3 and 4 will already have made unlikely.

If items 1–4 all leave Her2 at or near 0/14, the thesis claims this: **a UNI2-h + CLAM-MB WSI arm
trained on TCGA-BRCA does not transfer its Her2 competence to CPTAC-BRCA, and the failure is not
recoverable by prior correction, by a learned fusion rule, by a different aggregation rule, by
label-free feature adaptation, or by stain normalisation.** The support is in hand: the shift is 5.4×
the class signal, cohort of origin is perfectly separable from one patch, and the collapse survives
removing the MIL head. Fernandez-Romero 2026 reports the same across 13 encoders and 3 MIL
architectures, so it is a property of the transfer, not of this implementation. That is publishable in
a field where four groups report fusion that does not help and none reports the trivial average — and
it is why the CNV arm matters: 39 arm-level features reach 12 of 14 where the imaging arm reaches 0,
at 0.888 macro AUROC against the WSI arm's 0.847, with the equal-weight mean at 0.909.

## Replacement text for the report's Limitations paragraph

Wordings to drop in; the edit is yours.

**Branch A — item 1 or 4 recovers Her2 (≥6 of 14, macro AUROC holding).**
> The external Her2 collapse is the report's most consequential finding. The WSI arm calls 0 of 14
> Her2 cases on CPTAC while calling 26 of 51 internally, and the calibration explanation was tested
> and refuted. The collapse is induced by domain shift in the encoder's feature space: cohort of
> origin is perfectly separable from a single UNI2-h patch feature, and the shift is 5.4 times the
> Her2 class signal. <N> of 14 cases are recovered by <intervention>, which is label-free but
> transductive and is reported as such; the CNV arm reaches 12 of 14 without it.

**Branch B — partial recovery (1–5 of 14), the measured expectation.** Same first three sentences,
then:
> Label-free feature adaptation recovers <N> of 14 cases, a partial and non-decisive result; the
> imaging arm remains far behind the CNV arm's 12 of 14, and fusion's value here is that the second
> modality supplies a class the imaging arm cannot.

**Branch C — nothing recovers it.** Same first two sentences, then:
> The collapse is induced by domain shift below the classifier and we could not remediate it. Prior
> correction, alternative aggregation, learned fusion rules, label-free feature adaptation and stain
> normalisation all leave Her2 at 0 of 14 while Her2 one-vs-rest AUROC holds at 0.860 — ranking
> transfers, the decision does not. Fernandez-Romero 2026 reports the same collapse across 13 encoders
> and 3 MIL architectures on this cohort pair. A TCGA-trained UNI2-h + CLAM-MB arm does not transfer
> Her2 competence here, and the arm-level CNV modality is what carries the class.

## Rejected options

**Prior rebalancing, recalibration, a new decision threshold. Rejected: settled.** A 12× prior boost
and unsupervised SLD-EM both leave 0/14; §2 of the results document.

**Changing the slide-to-case aggregation rule. Rejected: settled, this session.** No slide argmaxes to
Her2 under any rule tried, so there is nothing to recover, and adopting one after seeing CPTAC would
be a post-hoc decision on the external cohort that also moves the WSI arm underneath every fusion
number in the report.

**Any adaptation fitted on CPTAC labels — supervised fine-tuning, threshold selection on the 14 Her2
cases, supervised CORAL. Rejected: leakage.** It voids the claim the external chapter rests on. The
one diagnostic here that used CPTAC labels (a CPTAC-fitted Her2 direction, cosine +0.158 with the TCGA
one) enters no scored pipeline.

**Swapping the patch encoder without re-extracting TCGA. Rejected: cost and incoherence.** A new
encoder changes the training features too, so it requires item 5 in full.

**Re-patching CPTAC with Trident to match TCGA's segmentation. Rejected: cost for the information.**
Trident is neither installed nor vendored here, and hypothesis 2 was refuted as a dilution mechanism
this session. Revisit only if item 4 shows stain matters.
