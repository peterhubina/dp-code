# Trained WSI+CNV fusion: plan to move the gain from averaging into a trained, WSI-centric model

Status board for a sequence of atomic tasks. One task per Claude Code session. Each session reads
this file, does exactly one task, fills in its status row, and stops. Written 2026-09-03 from the
evaluation summary in `docs/wsi_cnv_summary/` (numbers verified 2026-09-01).

## Goal and pre-registered criteria

Goal: a single trained model whose backbone is the WSI baseline (`pam50_final_s1`) and which uses
arm-level CNV so that it at least matches, ideally beats, the untrained equal-weight probability
mean of the two unimodal arms. The mean is the baseline (CLAUDE.md reporting rule 2), and CNV alone
is reported every time (rule 1).

Primary metric: pooled out-of-fold macro AUROC on the 599 TCGA cases shared by all runs, paired
bootstrap N=2000 seed 13 against the equal-weight mean, as `dp-analysis compare_fusion_ladder`
computes it. Current state: mean 0.926; best trained operator (`coattn`) 0.899, Δ −0.027
[−0.043, −0.011].

Success thresholds, fixed before any run:

- Non-inferiority: Δ vs mean with 95% CI lower bound above −0.010.
- Superiority: Δ vs mean above 0 with 95% CI excluding 0.
- Diversity: error-correlation φ between the trained model and the CNV arm below 0.40 (today 0.656
  among operators, 0.193 between the unimodal arms).

External (CPTAC, n=114): every trained arm that is evaluated is reported, with CNV alone and the
mean beside it, bootstrap N=4000 seed 7. Nothing is selected, tuned or thresholded on CPTAC; the
operator to headline is chosen on TCGA before its CPTAC number is looked at.

Standing constraints: never regenerate `dataset_csv/` or `splits/`; never pass `run.overwrite`;
new runs get new `exp_code`s; `docs/cnv-wsi-fusion-external-validation.md` is user-owned;
`tools/compare_fusion_ladder.py` is user-owned (T2 needs a small extension to it, ask first).

## Diagnosis the plan rests on

1. Joint training collapses diversity: φ 0.656 among the five operators vs 0.193 unimodal.
2. The ladder trained its WSI branch under a worse recipe than the baseline (rounded lr/reg,
   `bag_weight` 0.7, `no_inst_cluster`, no `inst_loss svm`).
3. The CNV branch inside fusion is a jointly trained 2-layer MLP (hidden 64); its standalone
   strength was never measured against the 0.872 logistic-regression arm.
4. All five operators share the `pam50_final_s1` warm start; shared-init vs joint-training collapse
   is unseparated.
5. No trained operator has ever been scored on CPTAC.

## Tasks, in order

### T1. Score the five existing ladder checkpoints on CPTAC (inference only)

- Why first: costs minutes, no training, and tells whether trained fusion inherits the WSI HER2
  collapse or not. Decides how much GPU the rest deserves.
- Scope: `tools/cptac/infer_cptac_multimodal.py` (add `--film_rank`, `--modality_dropout`,
  `--tabular_group_spec`; read `tabular_hidden_dim` and friends from the checkpoint dir's
  `experiment_*.txt` instead of trusting flags), one new `dp-analysis` action or a script under
  `tools/` that scores the five prediction dirs against WSI alone, CNV alone and the mean on the
  same 114 cases (reuse `tools/pam50_arms.py`; N=4000 seed 7).
- Inputs on disk: `.scratch/results/pam50_wsi_cnv_{concat,gated,cross_attention,film_attention,coattn}_s1/`,
  `.scratch/cnv-tabular/CPTAC_BRCA_CNV_arm_4class_clam.csv`, `.datasets/cptac-brca/embeddings/`,
  `.datasets/cptac-brca/cptac_brca_pam50_dataset.csv`.
- Outputs: `.scratch/cptac_validation/results/predictions_cnv_fusion_<op>/ensemble_predictions.csv`
  (five dirs, 378 slides each) and one analysis run dir with the table.
- Acceptance: five tables, each 114 cases; macro AUROC with CI, balanced accuracy, per-class recall,
  Δ vs mean, φ vs CNV; all five reported.
- Gate: any operator within CI of the mean externally means joint training survives domain shift;
  all five below CNV alone means fusion carries the WSI collapse, and T4 (frozen WSI) becomes the
  main line.

### T2. Matched-recipe ladder retrain (internal)

- Scope: new `dpcode/conf/experiment/pam50_wsi_cnv_matched.yaml` copying the baseline optimiser
  exactly (`lr 0.0001007597588073064`, `reg 0.0000024456514744717547`, `bag_weight
  0.5533776374353542`, `inst_loss svm`, instance clustering on, `B 4`), `exp_code
  pam50_wsi_cnv_matched_${fusion.name}`; `tools/compare_fusion_ladder.py` gains a `--run-suffix`
  (or `--results-glob`) so the new runs can be scored without editing hard-coded names.
- Command: `dp-train -m experiment=pam50_wsi_cnv_matched fusion=gated,coattn` first (best two),
  the other three only if the gate passes.
- Cost: about 30 GPU-minutes per operator for 10 folds on one 3090; two GPUs can run two operators
  at once as separate invocations.
- Acceptance: pooled table with Δ vs mean and Δ vs the original rounded-recipe run of the same
  operator.
- Gate: matched recipe closing at least half the gap to the mean (Δ better than −0.014) means the
  recipe was a large part of the loss; continue on the matched recipe from here on.

### T3. No-warm-start arm

- Scope: same experiment config with `clam.pretrained_wsi_ckpt=null`, `exp_code
  pam50_wsi_cnv_scratch_${fusion.name}`, for `gated` and `coattn`.
- Cost: about 1 GPU-hour.
- Acceptance: pooled AUROC and φ among operators and vs the CNV arm.
- Gate: φ dropping toward 0.3 with AUROC held means the shared init drives the collapse; φ staying
  near 0.65 means joint training does, and T5 is needed.

### T4. Frozen-WSI fusion (the WSI-centric learned correction)

- Scope: `clam.freeze_wsi_branch=true` on top of the warm start (`main.py` and `core_utils.py`
  already support it; `dp-config sync-check` must stay green), `exp_code
  pam50_wsi_cnv_frozen_${fusion.name}`, all five operators. Only the tabular encoder and the fusion
  head train; the WSI arm stays bit-for-bit the baseline.
- Cost: about 10 GPU-minutes per operator.
- Acceptance: pooled AUROC vs mean; φ vs CNV arm; per-class recall including HER2.
- Gate: an operator meeting non-inferiority here is the first candidate for the headline model;
  score it on CPTAC with the T1 path.

### T5. Diversity-preserving joint training

- Scope: `project/CLAM/models/model_multimodal.py` (auxiliary unimodal logits from each branch),
  `project/CLAM/utils/core_utils.py` (loss = fused CE + λ·(WSI CE + CNV CE)), `project/CLAM/main.py`
  (`--aux_loss_weight`, `--modality_dropout` for every operator), `dpcode/clam_args.py` and
  `ClamConf` (sync-check), experiment config `pam50_wsi_cnv_aux`. λ fixed at 0.5 in advance, not
  swept on the test folds.
- Cost: about 30 GPU-minutes per operator on the matched recipe.
- Acceptance: φ below 0.40 and AUROC meeting non-inferiority; superiority is the stretch goal.
- Gate: this is the last internal lever; if it fails, the write-up states that on this data a trained
  operator matches but does not beat the average, which is Amer 2025's finding too.

### T6. External scoring of the nominated model and report update

- Scope: T1 inference path on the nominated checkpoints; `docs/wsi_cnv_summary/` gains the new
  rows (rebuild with `build.sh`); one line under "Work since" in `CLAUDE.md`.
- Rule: the nominated operator is fixed on TCGA in T4/T5 before its CPTAC number is computed.

Deferred: reviving `fusion=residual` (WSI logits plus a CNV residual) needs a tabular checkpoint
trainer that does not exist; revisit only if T4 and T5 both fail.

## Session protocol

- One task per conversation. Open the session with: "Read
  `docs/implementation-research/multimodality-trained-fusion-plan.md`. Do task T<n> only. Update its
  status row with run dirs and the headline numbers, then stop."
- Every run dir must carry a new `exp_code`; never touch `pam50_final_s1` or the `_s1` ladder.
- Each task ends with the numbers in this file's status table and the analysis run dir path.
- `dp-*` console scripts are not on PATH in the current container; `python -m dpcode.cli.train`,
  `python -m dpcode.cli.analysis` are the same entry points. `dp-analysis` dry-run flag is
  `--show-config`.

## Status

| task | status | run dirs / analysis dir | headline | date |
|---|---|---|---|---|
| T1 external scoring of `_s1` ladder | done | `.scratch/cptac_validation/results/predictions_cnv_fusion_{concat,gated,cross_attention,film_attention,coattn}/` (378 slides / 114 cases each); analysis `.scratch/analysis/cptac_fusion_ladder/2026-09-03_13-07-34/`; reproduce with `python -m dpcode.cli.analysis cptac_fusion_ladder` | macro AUROC (N=4000 seed 7) concat 0.873, cross_attention 0.859, gated 0.843, film_attention 0.843, coattn 0.817 vs WSI 0.847 / CNV 0.888 / mean 0.909; delta vs mean -0.036 (ns, concat) to -0.092 (sig, coattn), all five below CNV alone; Her2 recall 0/14 for every operator (WSI 0/14, CNV 12/14, mean 6/14); phi vs CNV 0.04 to 0.15, vs WSI 0.68 to 0.84. Gate: fusion carries the WSI collapse, so T4 (frozen WSI) becomes the main line. | 2026-09-03 |
| T2 matched-recipe retrain | not started | | | |
| T3 no-warm-start arm | not started | | | |
| T4 frozen-WSI fusion | not started | | | |
| T5 auxiliary unimodal losses | not started | | | |
| T6 external scoring + report | not started | | | |
