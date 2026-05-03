# Selective Multimodal Fusion Runbook

This workflow uses RNA as the strong anchor and lets WSI/multimodal branches make validation-tuned probability corrections. It is intended for the matched TCGA-BRCA PAM50 4-class setup.

## Best Current Result

Use calibrated validation-tuned convex weights:

```bash
cd /workspace/dp-code/project/CLAM

python evaluate_selective_ensemble.py \
  --rna_results_dir /workspace/dp-code/project/CLAM/results/tcga_brca_rna_matched_wsi_4class_s1 \
  --wsi_pred_dir /workspace/dp-code/.scratch/results/pam50_wsi_rna_lateprob_eval \
  --concat_val_dir /workspace/dp-code/project/CLAM/tmp_eval/pam50_wsi_rna_latefusion_val_eval \
  --concat_test_dir /workspace/dp-code/.scratch/results/pam50_wsi_rna_latefusion_eval \
  --gated_val_dir /workspace/dp-code/project/CLAM/tmp_eval/pam50_wsi_rna_gatedfusion_val_eval \
  --gated_test_dir /workspace/dp-code/.scratch/results/pam50_wsi_rna_gatedfusion_eval \
  --residual_val_dir /workspace/dp-code/project/CLAM/tmp_eval/pam50_wsi_rna_residual_bestval_val_eval \
  --residual_test_dir /workspace/dp-code/.scratch/results/pam50_wsi_rna_residual_bestval_eval \
  --output_dir /workspace/dp-code/project/CLAM/tmp_eval/selective_ensemble_balanced_calibrated \
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
  --rna_results_dir /workspace/dp-code/project/CLAM/results/tcga_brca_rna_matched_wsi_4class_s1 \
  --wsi_pred_dir /workspace/dp-code/.scratch/results/pam50_wsi_rna_lateprob_eval \
  --concat_val_dir /workspace/dp-code/project/CLAM/tmp_eval/pam50_wsi_rna_latefusion_val_eval \
  --concat_test_dir /workspace/dp-code/.scratch/results/pam50_wsi_rna_latefusion_eval \
  --gated_val_dir /workspace/dp-code/project/CLAM/tmp_eval/pam50_wsi_rna_gatedfusion_val_eval \
  --gated_test_dir /workspace/dp-code/.scratch/results/pam50_wsi_rna_gatedfusion_eval \
  --residual_val_dir /workspace/dp-code/project/CLAM/tmp_eval/pam50_wsi_rna_residual_bestval_val_eval \
  --residual_test_dir /workspace/dp-code/.scratch/results/pam50_wsi_rna_residual_bestval_eval \
  --output_dir /workspace/dp-code/project/CLAM/tmp_eval/selective_stacker_balanced_calibrated \
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

```bash
python main.py \
  --task tcga_brca_subtyping \
  --data_root_dir /workspace/dp-code/.datasets/tcga-brca/embeddings \
  --split_dir tcga_brca_subtyping_100 \
  --results_dir /workspace/dp-code/.scratch/results \
  --exp_code pam50_wsi_rna_cross_attention_s1 \
  --model_type clam_mb \
  --model_size big \
  --embed_dim 1536 \
  --fusion_mode cross_attention \
  --tabular_csv /workspace/dp-code/.scratch/TCGA-BRCA-rna/TCGA_BRCA_RNA_primary_tumor_4class_clam.csv.gz \
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

Evaluate validation/test predictions after training:

```bash
python evaluate_multimodal.py \
  --data_root_dir /workspace/dp-code/.datasets/tcga-brca/embeddings \
  --tabular_csv /workspace/dp-code/.scratch/TCGA-BRCA-rna/TCGA_BRCA_RNA_primary_tumor_4class_clam.csv.gz \
  --ckpt_dir /workspace/dp-code/.scratch/results/pam50_wsi_rna_cross_attention_s1 \
  --output_dir /workspace/dp-code/project/CLAM/tmp_eval/pam50_wsi_rna_cross_attention_val_eval \
  --split_dir /workspace/dp-code/project/CLAM/splits/tcga_brca_subtyping_100 \
  --dataset_csv /workspace/dp-code/project/CLAM/dataset_csv/tcga_brca_subtyping.csv \
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

If you also evaluate the test split into `/workspace/dp-code/project/CLAM/tmp_eval/pam50_wsi_rna_cross_attention_test_eval`, pass these two extra arguments to `evaluate_selective_ensemble.py`:

```bash
  --cross_attention_val_dir /workspace/dp-code/project/CLAM/tmp_eval/pam50_wsi_rna_cross_attention_val_eval \
  --cross_attention_test_dir /workspace/dp-code/project/CLAM/tmp_eval/pam50_wsi_rna_cross_attention_test_eval
```

## If Validation Predictions Are Missing

Regenerate validation predictions for the trainable multimodal branches with `evaluate_multimodal.py`. The three required validation directories are:

- `/workspace/dp-code/project/CLAM/tmp_eval/pam50_wsi_rna_latefusion_val_eval`
- `/workspace/dp-code/project/CLAM/tmp_eval/pam50_wsi_rna_gatedfusion_val_eval`
- `/workspace/dp-code/project/CLAM/tmp_eval/pam50_wsi_rna_residual_bestval_val_eval`

The WSI validation probabilities already come from:

- `/workspace/dp-code/.scratch/results/pam50_wsi_rna_lateprob_eval`

## Interpretation

The RNA branch is already close to saturated, so most learned fusion heads overfit or dilute the RNA signal. The calibrated selective ensemble keeps RNA as the anchor and uses WSI-derived branches as fold-specific soft corrections. This matches the selective multimodal idea from the reference paper while avoiding a brittle hard routing threshold.
