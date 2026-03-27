# CLAM Modifications for PAM50 Molecular Subtyping on TCGA-BRCA

## Context

The baseline CLAM framework (Lu et al., 2021) was developed and validated on histological subtyping tasks (RCC, NSCLC) and metastasis detection, where strong morphological signals exist in H&E-stained tissue. Our task -- PAM50 molecular subtyping of breast cancer from TCGA-BRCA -- differs fundamentally: PAM50 subtypes (Luminal A, Luminal B, Basal, HER2-enriched) are defined by gene expression profiles, not morphological criteria. The morphology-to-label mapping is weaker and noisier, which required targeted modifications to the training pipeline to mitigate overfitting.

### Dataset

- 974 TCGA-BRCA whole-slide images (4 PAM50 classes)
- Train/Val/Test split: 767/104/103 per fold (10-fold Monte Carlo CV)
- Class distribution: Luminal A (61.8%), Luminal B (25.4%), Basal (21.5%), HER2 (9.8%)
- Features: 1536-dimensional UNI embeddings (vs. 1024-dim ResNet50 in original paper)

### Baseline Configuration (Original CLAM Paper)

| Parameter | Value |
|---|---|
| Learning rate | 2 x 10^-4 (fixed) |
| L2 weight decay | 1 x 10^-5 (fixed) |
| Dropout | 0.25 (fixed, all layers) |
| Early stopping patience | 20 epochs |
| Minimum training epochs (stop_epoch) | 50 |
| Maximum epochs | 200 |
| Instance-level clustering loss | Smooth top-1 SVM |
| Slide-level classification loss | Cross-entropy |
| Bag weight (c1) | 0.7 |
| B (k_sample) | 8 (RCC), 32 (NSCLC, CAMELYON) |
| Optimizer | Adam (beta1=0.9, beta2=0.999) |

---

## Observed Overfitting Pattern

An initial Bayesian hyperparameter sweep (329 runs) revealed severe, systematic overfitting across all configurations:

- Training loss collapsed from ~0.85 to ~0.02-0.05 while validation loss rose from ~0.75 to ~1.2-2.5
- The train/val loss divergence point occurred consistently at epoch 2-5
- Validation AUC continued to improve (up to 0.93) even as validation loss increased, indicating a calibration problem: the model learned correct ranking but became overconfident in its probability estimates
- Validation instance-level clustering loss (`val/inst_loss`) rose monotonically, confirming that the attention-based pseudo-labels used for instance clustering were too noisy for the PAM50 task

---

## Modifications from Baseline

### Modification 1: Attention Network Dropout Propagation (Bug Fix)

**File:** `models/model_clam.py`

**Problem:** The `Attn_Net` and `Attn_Net_Gated` classes accepted a dropout parameter but hardcoded the dropout rate at 0.25 regardless of the value passed:

```python
# Before (lines 24, 51-52)
self.module.append(nn.Dropout(0.25))          # Attn_Net
self.attention_a.append(nn.Dropout(0.25))     # Attn_Net_Gated
self.attention_b.append(nn.Dropout(0.25))     # Attn_Net_Gated
```

**Fix:** The dropout rate is now propagated from the model configuration:

```python
# After
self.module.append(nn.Dropout(dropout))       # Attn_Net
self.attention_a.append(nn.Dropout(dropout))  # Attn_Net_Gated
self.attention_b.append(nn.Dropout(dropout))  # Attn_Net_Gated
```

**Rationale:** The attention network is the most parameter-intensive component of CLAM. When sweeping dropout values (e.g., 0.5 or 0.75), only the first FC layer was affected while the attention layers remained at 0.25, defeating the purpose of the hyperparameter search.

---

### Modification 2: Early Stopping `stop_epoch` Correction (Bug Fix)

**File:** `utils/core_utils.py`, line 187

**Problem:** The `stop_epoch` parameter (minimum epoch before early stopping can activate) was set equal to the `patience` parameter:

```python
# Before
early_stopping = EarlyStopping(patience=args.patience, stop_epoch=args.patience, verbose=True)
```

The `EarlyStopping.__call__` method requires both conditions to be true:
```python
if self.counter >= self.patience and epoch > self.stop_epoch:
```

With `stop_epoch = patience`, early stopping could not trigger until both the patience counter was exhausted AND the epoch exceeded the patience value. For a patience of 10, this meant the model would train for a minimum of ~20 epochs even when validation loss began rising at epoch 3.

**Fix:**

```python
# After
early_stopping = EarlyStopping(patience=args.patience, stop_epoch=5, verbose=True)
```

**Rationale:** The original CLAM paper used `stop_epoch=50` with `patience=20` on datasets where models converged later. For PAM50 subtyping, where the optimal validation loss is reached at epoch 2-5, a low `stop_epoch` allows early stopping to act promptly. The fixed value of 5 ensures a minimum training duration while preventing extended overfitting.

Note: The original paper's `stop_epoch=50` was appropriate for their tasks (histological subtyping) where convergence took longer due to stronger morphological signals. For the weaker PAM50 signal, the model memorizes the training set much earlier.

---

### Modification 3: Label Smoothing on Slide-Level Cross-Entropy Loss

**File:** `utils/core_utils.py`, line 132

**Change:**

```python
# Before
loss_fn = nn.CrossEntropyLoss()

# After
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Rationale:** The primary observed failure mode was not classification accuracy degradation but probability calibration collapse -- validation loss rose while validation AUC continued to improve. Standard cross-entropy penalizes incorrect predictions exponentially as the model's confidence increases. As training progresses, the model produces increasingly peaked softmax distributions (e.g., [0.999, 0.0003, ...] instead of [0.7, 0.15, ...]), and even small errors on the validation set are penalized heavily.

Label smoothing (epsilon=0.1) replaces hard targets [1, 0, 0, 0] with soft targets [0.925, 0.025, 0.025, 0.025], preventing the model from driving logits to extreme values. This directly addresses the rising validation loss by bounding the confidence the model can express.

The original CLAM paper did not use label smoothing because their tasks had strong morphological signals where high confidence was justified. For PAM50 molecular subtyping, where the morphology-to-label mapping is inherently noisy, label smoothing provides an inductive bias toward better-calibrated probabilities.

---

### Modification 4: Instance-Level Clustering Loss Fixed to SVM

**File:** `sweep_config.yaml`

**Change:**

```yaml
# Before
inst_loss:
  values: [svm, ce]

# After
inst_loss:
  value: svm
```

**Rationale:** The original CLAM paper (Lu et al., 2021, p. 566) explicitly chose the smooth top-1 SVM loss over cross-entropy for instance-level clustering because the pseudo-labels generated from attention scores are inherently noisy:

> "The introduction of a margin to the loss function has been empirically shown to reduce overfitting when the data labels are noisy or when data are limited."

For PAM50 molecular subtyping, the attention-based pseudo-labels are substantially noisier than in the paper's histological tasks because the morphological features distinguishing molecular subtypes are less pronounced. The observed monotonic rise in `val/inst_loss` confirmed that cross-entropy instance loss was overfitting to incorrect pseudo-labels. The SVM margin provides implicit regularization by not penalizing predictions that are already on the correct side of the decision boundary.

---

### Modification 5: Bag Weight Range Adjustment

**File:** `sweep_config.yaml`

**Change:**

```yaml
# Before
bag_weight:
  distribution: uniform
  min: 0.5
  max: 0.9

# After
bag_weight:
  distribution: uniform
  min: 0.7
  max: 0.95
```

**Rationale:** The total CLAM loss is `L_total = c1 * L_slide + (1 - c1) * L_patch`, where `c1` is the bag weight. `L_slide` is supervised by ground-truth labels; `L_patch` is supervised by attention-derived pseudo-labels. For PAM50 subtyping, the pseudo-labels are noisy (evidenced by rising `val/inst_loss`), so reducing the weight of `L_patch` by increasing `c1` mitigates its contribution to overfitting. The original paper used c1=0.7 for all tasks; we raise the sweep range to [0.7, 0.95] to explore configurations that rely more heavily on the ground-truth-supervised slide-level loss.

---

### Modification 6: Regularization Sweep Range Adjustments

**File:** `sweep_config.yaml`

**Changes:**

```yaml
# L2 weight decay
# Before: min: 0.00001, max: 0.001
# After:  min: 0.0005,  max: 0.01

# Dropout
# Before: values: [0.1, 0.25, 0.5]
# After:  values: [0.25, 0.5, 0.6, 0.75]

# Early stopping patience
# Before: values: [5, 10, 15]
# After:  values: [3, 5, 7]
```

**Rationale:**

- **L2 weight decay:** The original CLAM paper used a fixed 1 x 10^-5, which was sufficient for histological subtyping. For PAM50, the model (1.19M parameters, 767 training slides, 1536-dim input features) has a parameter-to-sample ratio that demands stronger regularization. The lower bound was raised from 1 x 10^-5 to 5 x 10^-4 (50x increase) and the upper bound from 1 x 10^-3 to 1 x 10^-2.

- **Dropout:** The lowest value (0.1) provided negligible regularization and was removed. Higher values (0.6, 0.75) were added to explore stronger regularization, enabled by Modification 1 which ensures these values now propagate to the attention network.

- **Early stopping patience:** Values of 10-15 allowed the model to train for many epochs past the validation optimum (epoch 2-5 for this task). The tighter range [3, 5, 7] stops training closer to the divergence point.

---

## Summary Table

| # | Modification | Type | File(s) | Key Change |
|---|---|---|---|---|
| 1 | Attention dropout propagation | Bug fix | `models/model_clam.py` | Hardcoded 0.25 -> configurable rate |
| 2 | Early stopping stop_epoch | Bug fix | `utils/core_utils.py` | `stop_epoch=patience` -> `stop_epoch=5` |
| 3 | Label smoothing | Regularization | `utils/core_utils.py` | `CrossEntropyLoss(label_smoothing=0.1)` |
| 4 | SVM-only instance loss | Task adaptation | `sweep_config.yaml` | Removed CE option for instance clustering |
| 5 | Bag weight range | Task adaptation | `sweep_config.yaml` | [0.5, 0.9] -> [0.7, 0.95] |
| 6 | Regularization ranges | Hyperparameter | `sweep_config.yaml` | Stronger L2, dropout, tighter patience |

Modifications 1-2 are bug fixes applicable to any CLAM deployment. Modifications 3-6 are task-specific adaptations motivated by the weak morphology-to-label mapping inherent in PAM50 molecular subtyping from H&E histology.
