# Selective Multimodal Fusion Runbook

This workflow uses RNA as the strong anchor and lets WSI/multimodal branches make validation-tuned probability corrections. It is intended for the matched TCGA-BRCA PAM50 4-class setup.

> **Read this before quoting any number below.** This runbook belongs to the WSI + RNA thread, and
> PAM50 labels in this project are computed from the same expression matrix that feeds the RNA
> branch (`project/data/pam50.R`). The RNA branch therefore leaks the target by construction, and
> every metric in this document is leakage-inflated. The commands are kept because they still run
> and because the ER arms reuse the same machinery, but the live PAM50 work is the WSI + arm-level
> CNV thread, where copy number carries no such circularity. See the leakage entry in `CLAUDE.md`
> and `docs/implementation-research/next-steps-action-plan.md`.

## How to run the commands in this file

Every command below assumes two things: that `$REPO_ROOT` names your clone, and that the working
directory is `$REPO_ROOT/project/CLAM`. Set both once per shell:

```bash
REPO_ROOT="${DP_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "$REPO_ROOT/project/CLAM"
```

`DP_REPO_ROOT` is the same variable `dpcode/conf/paths/default.yaml` reads, so a shell configured
for `dp-train` / `dp-evaluate` is already configured for this file.

The directories under `$REPO_ROOT/.scratch/` are gitignored: a fresh clone has none of them, and
the `evaluate_selective_ensemble.py` commands below need the per-branch prediction directories to
exist first (see "If Validation Predictions Are Missing"). The directories under
`$REPO_ROOT/project/CLAM/{results,tmp_eval}/` are tracked, so those arrive with the clone.

## Best Current Result

Use calibrated validation-tuned convex weights:

```bash
python evaluate_selective_ensemble.py \
  --rna_results_dir "$REPO_ROOT/project/CLAM/results/tcga_brca_rna_matched_wsi_4class_s1" \
  --wsi_pred_dir "$REPO_ROOT/.scratch/results/pam50_wsi_rna_lateprob_eval" \
  --concat_val_dir "$REPO_ROOT/project/CLAM/tmp_eval/pam50_wsi_rna_latefusion_val_eval" \
  --concat_test_dir "$REPO_ROOT/.scratch/results/pam50_wsi_rna_latefusion_eval" \
  --gated_val_dir "$REPO_ROOT/project/CLAM/tmp_eval/pam50_wsi_rna_gatedfusion_val_eval" \
  --gated_test_dir "$REPO_ROOT/.scratch/results/pam50_wsi_rna_gatedfusion_eval" \
  --residual_val_dir "$REPO_ROOT/project/CLAM/tmp_eval/pam50_wsi_rna_residual_bestval_val_eval" \
  --residual_test_dir "$REPO_ROOT/.scratch/results/pam50_wsi_rna_residual_bestval_eval" \
  --output_dir "$REPO_ROOT/project/CLAM/tmp_eval/selective_ensemble_balanced_calibrated" \
  --objective balanced_accuracy \
  --calibrate
```

Observed aggregate test metrics:

| Method | AUC | Accuracy | Balanced acc. | Macro F1 | Weighted F1 | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Selective ensemble, calibrated | 0.9796 | 0.8996 | 0.8856 | 0.8804 | 0.8999 | 0.0256 |

This is the recommended row to add to the report because it improves the main classification metrics over the current multimodal heads and substantially improves calibration.

## Exploratory Balanced-Accuracy Variant

The stacker variant trains a small regularized logistic meta-classifier on each validation fold:

```bash
python evaluate_selective_ensemble.py \
  --rna_results_dir "$REPO_ROOT/project/CLAM/results/tcga_brca_rna_matched_wsi_4class_s1" \
  --wsi_pred_dir "$REPO_ROOT/.scratch/results/pam50_wsi_rna_lateprob_eval" \
  --concat_val_dir "$REPO_ROOT/project/CLAM/tmp_eval/pam50_wsi_rna_latefusion_val_eval" \
  --concat_test_dir "$REPO_ROOT/.scratch/results/pam50_wsi_rna_latefusion_eval" \
  --gated_val_dir "$REPO_ROOT/project/CLAM/tmp_eval/pam50_wsi_rna_gatedfusion_val_eval" \
  --gated_test_dir "$REPO_ROOT/.scratch/results/pam50_wsi_rna_gatedfusion_eval" \
  --residual_val_dir "$REPO_ROOT/project/CLAM/tmp_eval/pam50_wsi_rna_residual_bestval_val_eval" \
  --residual_test_dir "$REPO_ROOT/.scratch/results/pam50_wsi_rna_residual_bestval_eval" \
  --output_dir "$REPO_ROOT/project/CLAM/tmp_eval/selective_stacker_balanced_calibrated" \
  --objective balanced_accuracy \
  --mode stacker \
  --calibrate
```

Observed aggregate test metrics:

| Method | AUC | Accuracy | Balanced acc. | Macro F1 | Weighted F1 | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Selective stacker, calibrated | 0.9762 | 0.8905 | 0.8887 | 0.8758 | 0.8922 | 0.0451 |

Use this only if balanced accuracy is the primary target. Otherwise prefer the calibrated convex ensemble.

## Experimental Cross-Attention Branch

`CLAMRNAFusion` also supports a paper-inspired `cross_attention` mode. Train it as another candidate branch, then evaluate it and include its predictions in future ensembles if it beats the current concat/gated/residual branches on validation.

**This branch was never trained**: `.scratch/results/pam50_wsi_rna_cross_attention_s1` does not
exist, so the two commands in this section are a proposal, not a record of a run.

**The supported front end is now `dp-train`**, which composes the argv below from
`dpcode/conf/`, guards the run directory against overwriting a completed run, and writes
`config.resolved.yaml` / `run_metadata.json` / `metrics.json` beside CLAM's own outputs — none of
which the raw invocation does. The raw form is kept here because no `experiment=` config reproduces
*this* particular hyperparameter set (`--reg 1e-5`, `--patience 10`, no `--weighted_sample`, no
`--log_data`, CLAM's default `--bag_loss`/`--inst_loss`), and inventing one would silently redefine
what this section proposes. The nearest composed equivalent is
`dp-train experiment=pam50_wsi_rna_gated fusion=cross_attention`, which is **not** the same
configuration.

```bash
python main.py \
  --task tcga_brca_subtyping \
  --data_root_dir "$REPO_ROOT/.datasets/tcga-brca/embeddings" \
  --split_dir tcga_brca_subtyping_100 \
  --results_dir "$REPO_ROOT/.scratch/results" \
  --exp_code pam50_wsi_rna_cross_attention_s1 \
  --model_type clam_mb \
  --model_size big \
  --embed_dim 1536 \
  --fusion_mode cross_attention \
  --tabular_csv "$REPO_ROOT/.scratch/TCGA-BRCA-rna/TCGA_BRCA_RNA_primary_tumor_4class_clam.csv.gz" \
  --tabular_case_id_col case_id \
  --tabular_hidden_dim 256 \
  --tabular_num_layers 2 \
  --fusion_hidden_dim 32 \
  --drop_out 0.5 \
  --B 4 \
  --lr 1e-4 \
  --reg 1e-5 \
  --max_epochs 50 \
  --early_stopping \
  --patience 10 \
  --subtyping
```

`--tabular_csv` above is the LEGACY Xena-derived RNA table under `.scratch/TCGA-BRCA-rna/`, which is
what every branch in this runbook was trained on. Anything new should use the harmonised GDC tables
under `.scratch/rna-gdc/` instead — see the RNA scale-mismatch entry in `CLAUDE.md`.

Evaluate validation/test predictions after training:

```bash
python evaluate_multimodal.py \
  --data_root_dir "$REPO_ROOT/.datasets/tcga-brca/embeddings" \
  --tabular_csv "$REPO_ROOT/.scratch/TCGA-BRCA-rna/TCGA_BRCA_RNA_primary_tumor_4class_clam.csv.gz" \
  --ckpt_dir "$REPO_ROOT/.scratch/results/pam50_wsi_rna_cross_attention_s1" \
  --output_dir "$REPO_ROOT/project/CLAM/tmp_eval/pam50_wsi_rna_cross_attention_val_eval" \
  --split_dir "$REPO_ROOT/project/CLAM/splits/tcga_brca_subtyping_100" \
  --dataset_csv "$REPO_ROOT/project/CLAM/dataset_csv/tcga_brca_subtyping.csv" \
  --split val \
  --embed_dim 1536 \
  --model_type clam_mb \
  --model_size big \
  --fusion_mode auto \
  --drop_out 0.5 \
  --B 4 \
  --tabular_case_id_col case_id \
  --tabular_hidden_dim 256 \
  --tabular_num_layers 2 \
  --fusion_hidden_dim 32
```

`dp-evaluate evaluate=pam50_multimodal` is the composed front end for this command. It carries the
same two known gaps the raw script has, both documented under "Known gaps" in `CLAUDE.md`:
`--fusion_mode auto` cannot resolve a `film_attention` or `coattn` checkpoint, and the defaults
evaluate the TCGA test split rather than an external cohort. Neither is fixed here. Note also that
`--tabular_hidden_dim` must match the value the checkpoint was trained with — 256 for this RNA arm,
64 for the CNV ladder — or `load_state_dict(..., strict=True)` fails on a shape mismatch.

If you also evaluate the test split into `$REPO_ROOT/project/CLAM/tmp_eval/pam50_wsi_rna_cross_attention_test_eval`, pass these two extra arguments to `evaluate_selective_ensemble.py`:

```bash
  --cross_attention_val_dir "$REPO_ROOT/project/CLAM/tmp_eval/pam50_wsi_rna_cross_attention_val_eval" \
  --cross_attention_test_dir "$REPO_ROOT/project/CLAM/tmp_eval/pam50_wsi_rna_cross_attention_test_eval"
```

## If Validation Predictions Are Missing

Regenerate validation predictions for the trainable multimodal branches with `evaluate_multimodal.py`. The three required validation directories are:

- `$REPO_ROOT/project/CLAM/tmp_eval/pam50_wsi_rna_latefusion_val_eval`
- `$REPO_ROOT/project/CLAM/tmp_eval/pam50_wsi_rna_gatedfusion_val_eval`
- `$REPO_ROOT/project/CLAM/tmp_eval/pam50_wsi_rna_residual_bestval_val_eval`

The WSI validation probabilities already come from:

- `$REPO_ROOT/.scratch/results/pam50_wsi_rna_lateprob_eval`

## Interpretation

The RNA branch is already close to saturated, so most learned fusion heads overfit or dilute the RNA signal. The calibrated selective ensemble keeps RNA as the anchor and uses WSI-derived branches as fold-specific soft corrections. This matches the selective multimodal idea from the reference paper while avoiding a brittle hard routing threshold.

The saturation has a mundane explanation that this document originally lacked: the RNA branch is
predicting a label derived from its own input. Read the interpretation above with that in mind.
