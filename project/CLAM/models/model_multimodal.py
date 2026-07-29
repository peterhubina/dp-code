from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_clam import CLAM_MB, CLAM_SB
from models.model_rna import RNA_MLP

FUSION_MODES = ("concat", "gated", "residual", "cross_attention", "film_attention", "coattn")


class TabularMLPEncoder(nn.Module):
    """Small MLP encoder for patient-level RNA/tabular features."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.output_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.apply(_initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.encoder(x.float())


class CLAMRNAFusion(nn.Module):
    """Feature-level fusion over CLAM WSI and RNA MLP embeddings."""

    def __init__(
        self,
        wsi_model_type: str = "clam_mb",
        gate: bool = True,
        size_arg: str = "small",
        dropout: float = 0.0,
        k_sample: int = 8,
        n_classes: int = 4,
        instance_loss_fn=None,
        subtyping: bool = False,
        embed_dim: int = 1024,
        tabular_input_dim: int | None = None,
        tabular_hidden_dim: int = 256,
        tabular_num_layers: int = 2,
        fusion_hidden_dim: int = 128,
        fusion_mode: str = "concat",
        rna_hidden_dims=(1024, 512),
        rna_dropout: float = 0.4,
        residual_scale: float = 0.2,
        film_rank: int = 32,
        modality_dropout: float = 0.0,
        tabular_group_indices=None,
    ):
        super().__init__()
        if tabular_input_dim is None:
            raise ValueError("tabular_input_dim is required for multimodal fusion.")
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {sorted(FUSION_MODES)}.")
        if fusion_mode in {"gated", "residual", "cross_attention", "coattn"} and fusion_hidden_dim <= 0:
            raise ValueError(f"fusion_hidden_dim must be positive for {fusion_mode} fusion.")
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must lie in [0, 1).")
        if fusion_mode == "coattn" and not tabular_group_indices:
            raise ValueError("fusion_mode 'coattn' requires tabular_group_indices.")

        if instance_loss_fn is None:
            instance_loss_fn = nn.CrossEntropyLoss()

        clam_kwargs = {
            "gate": gate,
            "size_arg": size_arg,
            "dropout": dropout,
            "k_sample": k_sample,
            "n_classes": n_classes,
            "instance_loss_fn": instance_loss_fn,
            "subtyping": subtyping,
            "embed_dim": embed_dim,
        }
        if wsi_model_type == "clam_sb":
            self.wsi = CLAM_SB(**clam_kwargs)
        elif wsi_model_type == "clam_mb":
            self.wsi = CLAM_MB(**clam_kwargs)
        else:
            raise ValueError("CLAMRNAFusion supports 'clam_sb' and 'clam_mb' WSI branches only.")

        self.wsi_model_type = wsi_model_type
        self.fusion_mode = fusion_mode
        self.n_classes = n_classes
        self.wsi_frozen = False
        self.rna_frozen = False
        self.residual_scale = float(residual_scale)
        self.wsi_feature_dim = self.wsi.size_dict[size_arg][1]
        self.rna_model = None
        self.tabular_encoder = None
        self.film_rank = int(film_rank)
        self.modality_dropout = float(modality_dropout)
        # Evaluation switch: when True the tabular modality is treated as missing.
        self.force_tabular_absent = False

        if fusion_mode == "residual":
            rna_hidden_dims = _parse_hidden_dims(rna_hidden_dims)
            self.rna_model = RNA_MLP(
                input_dim=tabular_input_dim,
                hidden_dims=rna_hidden_dims,
                dropout=rna_dropout,
                n_classes=n_classes,
            )
            self.rna_feature_dim = int(self.rna_model.classifier.in_features)
        else:
            self.tabular_encoder = TabularMLPEncoder(
                input_dim=tabular_input_dim,
                hidden_dim=tabular_hidden_dim,
                num_layers=tabular_num_layers,
                dropout=dropout,
            )

        fusion_input_dim = self.wsi_feature_dim + (
            self.rna_feature_dim if fusion_mode == "residual" else self.tabular_encoder.output_dim
        )
        if fusion_mode == "concat" and fusion_hidden_dim and fusion_hidden_dim > 0:
            self.fusion_head = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_dim, n_classes),
            )
            self.fusion_head.apply(_initialize_weights)
        elif fusion_mode == "concat":
            self.fusion_head = nn.Linear(fusion_input_dim, n_classes)
            self.fusion_head.apply(_initialize_weights)
        elif fusion_mode == "gated":
            self.wsi_projection = nn.Sequential(
                nn.Linear(self.wsi_feature_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.tabular_projection = nn.Sequential(
                nn.Linear(self.tabular_encoder.output_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(2 * fusion_hidden_dim, fusion_hidden_dim),
                nn.Sigmoid(),
            )
            self.fusion_classifier = nn.Linear(fusion_hidden_dim, n_classes)
            self.wsi_projection.apply(_initialize_weights)
            self.tabular_projection.apply(_initialize_weights)
            self.fusion_gate.apply(_initialize_weights)
            self.fusion_classifier.apply(_initialize_weights)
        elif fusion_mode == "cross_attention":
            self.wsi_projection = nn.Sequential(
                nn.Linear(self.wsi_feature_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.tabular_projection = nn.Sequential(
                nn.Linear(self.tabular_encoder.output_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=fusion_hidden_dim,
                num_heads=1,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_attention_norm = nn.LayerNorm(fusion_hidden_dim)
            self.fusion_classifier = nn.Linear(2 * fusion_hidden_dim, n_classes)
            self.wsi_projection.apply(_initialize_weights)
            self.tabular_projection.apply(_initialize_weights)
            self.cross_attention_norm.apply(_initialize_weights)
            self.fusion_classifier.apply(_initialize_weights)
        elif fusion_mode == "film_attention":
            # The tabular modality predicts an affine transform of the attention network's
            # INPUT, re-ranking patches. Pooling still uses the unmodulated patch embeddings
            # and the frozen CLAM classifier, so at initialisation the logits are exactly the
            # WSI-alone logits (film factors and the tabular head are both zero-initialised).
            if self.film_rank > 0:
                self.film_bottleneck = nn.Linear(self.tabular_encoder.output_dim, self.film_rank, bias=False)
                self.film_gamma = nn.Linear(self.film_rank, self.wsi_feature_dim)
                self.film_beta = nn.Linear(self.film_rank, self.wsi_feature_dim)
                nn.init.xavier_normal_(self.film_bottleneck.weight)
                for layer in (self.film_gamma, self.film_beta):
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
            self.tabular_head = nn.Linear(self.tabular_encoder.output_dim, n_classes)
            nn.init.zeros_(self.tabular_head.weight)
            nn.init.zeros_(self.tabular_head.bias)
            if self.modality_dropout > 0.0:
                self.tabular_absent_embedding = nn.Parameter(torch.zeros(self.tabular_encoder.output_dim))
        elif fusion_mode == "coattn":
            # Adapted MCAT-style co-attention: tabular tokens query the patch tokens.
            # Not a reproduction of MCAT -- same folds, frozen branch, loss and encoder as the
            # other arms, so that only the fusion operator differs.
            self.tabular_group_indices = [list(map(int, group)) for group in tabular_group_indices]
            self.tabular_token_encoders = nn.ModuleList(
                [nn.Linear(len(group), fusion_hidden_dim) for group in self.tabular_group_indices]
            )
            self.patch_projection = nn.Linear(self.wsi_feature_dim, fusion_hidden_dim)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=fusion_hidden_dim, num_heads=1, dropout=dropout, batch_first=True
            )
            self.coattn_norm = nn.LayerNorm(fusion_hidden_dim)
            self.image_head = nn.Linear(fusion_hidden_dim, n_classes)
            self.tabular_head = nn.Linear(self.tabular_encoder.output_dim, n_classes)
            self.tabular_token_encoders.apply(_initialize_weights)
            self.patch_projection.apply(_initialize_weights)
            self.coattn_norm.apply(_initialize_weights)
            self.image_head.apply(_initialize_weights)
            nn.init.zeros_(self.tabular_head.weight)
            nn.init.zeros_(self.tabular_head.bias)
            if self.modality_dropout > 0.0:
                self.tabular_absent_embedding = nn.Parameter(torch.zeros(self.tabular_encoder.output_dim))
        else:
            self.wsi_projection = nn.Sequential(
                nn.Linear(self.wsi_feature_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.rna_projection = nn.Sequential(
                nn.Linear(self.rna_feature_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.residual_head = nn.Sequential(
                nn.Linear(2 * fusion_hidden_dim, fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_dim, n_classes),
            )
            self.wsi_projection.apply(_initialize_weights)
            self.rna_projection.apply(_initialize_weights)
            self.residual_head.apply(_initialize_weights)
            # Start exactly from the RNA model and let WSI learn conservative corrections.
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.wsi_frozen:
            self.wsi.eval()
        if self.rna_frozen and self.rna_model is not None:
            self.rna_model.eval()
        return self

    def freeze_wsi_branch(self) -> None:
        self.wsi_frozen = True
        self.wsi.eval()
        for param in self.wsi.parameters():
            param.requires_grad = False

    def load_wsi_checkpoint(self, ckpt_path: str) -> None:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]

        state_dict = OrderedDict()
        for key, value in checkpoint.items():
            if "instance_loss_fn" in key:
                continue
            state_dict[key.replace(".module", "")] = value

        missing, unexpected = self.wsi.load_state_dict(state_dict, strict=False)
        critical_missing = [key for key in missing if not key.startswith("instance_loss_fn")]
        if critical_missing or unexpected:
            raise RuntimeError(
                "Could not load WSI checkpoint cleanly. "
                f"Missing keys: {critical_missing[:5]}, unexpected keys: {unexpected[:5]}"
            )

    def load_rna_checkpoint(self, ckpt_path: str) -> None:
        if self.rna_model is None:
            raise RuntimeError("RNA checkpoints are only supported for residual fusion.")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        state_dict = {key.replace(".module", ""): value for key, value in state_dict.items()}
        self.rna_model.load_state_dict(state_dict, strict=True)

    def freeze_rna_branch(self) -> None:
        if self.rna_model is None:
            return
        self.rna_frozen = True
        self.rna_model.eval()
        for param in self.rna_model.parameters():
            param.requires_grad = False

    def forward(self, inputs, label=None, instance_eval=False, return_features=False, attention_only=False):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError("CLAMRNAFusion expects inputs=(wsi_features, tabular_features).")

        wsi_features, tabular_features = inputs
        if attention_only:
            return self.wsi(wsi_features, attention_only=True)

        if self.fusion_mode in {"film_attention", "coattn"}:
            return self._attention_level_fusion(wsi_features, tabular_features, return_features)

        if self.wsi_frozen:
            with torch.no_grad():
                _, _, _, attention, wsi_results = self.wsi(wsi_features, return_features=True)
        else:
            _, _, _, attention, wsi_results = self.wsi(wsi_features, return_features=True)

        pooled_wsi = self._pool_wsi_features(wsi_results["features"])
        encoded_tabular = None
        fusion_metrics = {}
        if self.fusion_mode == "concat":
            encoded_tabular = self.tabular_encoder(tabular_features)
            fused_features = torch.cat([pooled_wsi, encoded_tabular], dim=1)
            logits = self.fusion_head(fused_features)
            fusion_gate = None
        elif self.fusion_mode == "gated":
            encoded_tabular = self.tabular_encoder(tabular_features)
            logits, fused_features, fusion_gate = self._gated_fusion(pooled_wsi, encoded_tabular)
        elif self.fusion_mode == "cross_attention":
            encoded_tabular = self.tabular_encoder(tabular_features)
            logits, fused_features, fusion_attention = self._cross_attention_fusion(pooled_wsi, encoded_tabular)
            fusion_gate = None
            fusion_metrics = {
                "fusion_wsi_to_rna_attention": fusion_attention[:, 0, 1].mean(),
                "fusion_rna_to_wsi_attention": fusion_attention[:, 1, 0].mean(),
            }
        else:
            logits, fused_features, encoded_tabular = self._residual_fusion(pooled_wsi, tabular_features)
            fusion_gate = None

        y_hat = torch.topk(logits, 1, dim=1)[1]
        y_prob = F.softmax(logits, dim=1)

        results = {}
        if fusion_gate is not None:
            wsi_gate_mean = fusion_gate.mean()
            results.update(
                {
                    "fusion_wsi_gate_mean": wsi_gate_mean,
                    "fusion_rna_gate_mean": 1.0 - wsi_gate_mean,
                    "fusion_gate_std": fusion_gate.std(unbiased=False),
                }
            )
        results.update(fusion_metrics)

        if return_features:
            results.update(
                {
                    "wsi_features": pooled_wsi,
                    "tabular_features": encoded_tabular,
                    "fusion_features": fused_features,
                }
            )
            if fusion_gate is not None:
                results["fusion_gate"] = fusion_gate
            if self.fusion_mode == "cross_attention":
                results["fusion_attention"] = fusion_attention
        return logits, y_prob, y_hat, attention, results

    def _encode_tabular(self, tabular_features: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Encode the tabular vector, honouring modality dropout and the eval-time absent switch."""
        encoded = self.tabular_encoder(tabular_features)
        absent = bool(self.force_tabular_absent)
        if self.training and self.modality_dropout > 0.0:
            absent = absent or bool(torch.rand((), device=encoded.device) < self.modality_dropout)
        if absent:
            if hasattr(self, "tabular_absent_embedding"):
                encoded = self.tabular_absent_embedding.unsqueeze(0).expand_as(encoded)
            else:
                encoded = torch.zeros_like(encoded)
        return encoded, absent

    def _frozen_class_logits(self, pooled_per_class: torch.Tensor) -> torch.Tensor:
        """Apply the WSI branch's own classifier, so FiLM at identity reproduces its logits."""
        if self.wsi_model_type == "clam_mb":
            per_class = [self.wsi.classifiers[c](pooled_per_class[c]).squeeze() for c in range(self.n_classes)]
            return torch.stack(per_class).unsqueeze(0)
        return self.wsi.classifiers(pooled_per_class.mean(dim=0, keepdim=True))

    def _attention_level_fusion(self, wsi_features, tabular_features, return_features):
        if wsi_features.dim() == 3 and wsi_features.size(0) == 1:
            wsi_features = wsi_features.squeeze(0)
        if tabular_features.dim() == 1:
            tabular_features = tabular_features.unsqueeze(0)

        feature_extractor = self.wsi.attention_net[:3]
        attention_head = self.wsi.attention_net[3]
        if self.wsi_frozen:
            with torch.no_grad():
                patch_embeddings = feature_extractor(wsi_features)
        else:
            patch_embeddings = feature_extractor(wsi_features)

        encoded_tabular, tabular_absent = self._encode_tabular(tabular_features)

        if self.fusion_mode == "film_attention":
            logits, pooled, attention_raw, metrics = self._film_attention_fusion(
                patch_embeddings, attention_head, encoded_tabular, tabular_absent
            )
        else:
            logits, pooled, attention_raw, metrics = self._coattn_fusion(
                patch_embeddings, encoded_tabular, tabular_features, tabular_absent
            )

        y_hat = torch.topk(logits, 1, dim=1)[1]
        y_prob = F.softmax(logits, dim=1)
        results = dict(metrics)
        if return_features:
            results.update(
                {
                    "wsi_features": pooled,
                    "tabular_features": encoded_tabular,
                    "fusion_features": pooled,
                }
            )
        return logits, y_prob, y_hat, attention_raw, results

    def _film_attention_fusion(self, patch_embeddings, attention_head, encoded_tabular, tabular_absent):
        metrics = {}
        if self.film_rank > 0 and not tabular_absent:
            bottleneck = self.film_bottleneck(encoded_tabular)
            gamma = 1.0 + self.film_gamma(bottleneck)
            beta = self.film_beta(bottleneck)
            modulated = gamma * patch_embeddings + beta
            metrics = {
                "fusion_film_gamma_dev": (gamma - 1.0).detach().abs().mean(),
                "fusion_film_beta_abs": beta.detach().abs().mean(),
            }
        else:
            modulated = patch_embeddings

        attention_logits, _ = attention_head(modulated)
        attention_raw = torch.transpose(attention_logits, 1, 0)
        attention = F.softmax(attention_raw, dim=1)
        # Pool the ORIGINAL patch embeddings: the tabular vector re-ranks patches, it does
        # not distort the representation being pooled.
        pooled_per_class = torch.mm(attention, patch_embeddings)

        image_logits = self._frozen_class_logits(pooled_per_class)
        tabular_logits = self.tabular_head(encoded_tabular)
        metrics["fusion_tabular_logit_abs"] = tabular_logits.detach().abs().mean()
        logits = image_logits + tabular_logits
        return logits, pooled_per_class.mean(dim=0, keepdim=True), attention_raw, metrics

    def _coattn_fusion(self, patch_embeddings, encoded_tabular, tabular_features, tabular_absent):
        tokens = torch.stack(
            [
                encoder(tabular_features[:, group].float())
                for encoder, group in zip(self.tabular_token_encoders, self.tabular_group_indices)
            ],
            dim=1,
        )
        if tabular_absent:
            tokens = torch.zeros_like(tokens)

        patch_tokens = self.patch_projection(patch_embeddings).unsqueeze(0)
        attended, attention_weights = self.cross_attention(
            tokens, patch_tokens, patch_tokens, need_weights=True
        )
        attended = self.coattn_norm(attended + tokens)
        pooled = attended.mean(dim=1)

        logits = self.image_head(pooled) + self.tabular_head(encoded_tabular)
        metrics = {"fusion_coattn_max_weight": attention_weights.detach().max()}
        return logits, pooled, attention_weights, metrics

    def _residual_fusion(
        self,
        wsi_features: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.rna_model is None:
            raise RuntimeError("Residual fusion requires an RNA model.")

        if self.rna_frozen:
            with torch.no_grad():
                rna_logits, rna_features = self.rna_model(tabular_features.float(), return_features=True)
        else:
            rna_logits, rna_features = self.rna_model(tabular_features.float(), return_features=True)

        projected_wsi = self.wsi_projection(wsi_features)
        projected_rna = self.rna_projection(rna_features)
        fused_features = torch.cat([projected_wsi, projected_rna], dim=1)
        delta_logits = self.residual_head(fused_features)
        logits = rna_logits + self.residual_scale * delta_logits
        return logits, fused_features, rna_features

    def _gated_fusion(
        self,
        wsi_features: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected_wsi = self.wsi_projection(wsi_features)
        projected_tabular = self.tabular_projection(tabular_features)
        fusion_gate = self.fusion_gate(torch.cat([projected_wsi, projected_tabular], dim=1))
        fused_features = fusion_gate * projected_wsi + (1.0 - fusion_gate) * projected_tabular
        logits = self.fusion_classifier(fused_features)
        return logits, fused_features, fusion_gate

    def _cross_attention_fusion(
        self,
        wsi_features: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected_wsi = self.wsi_projection(wsi_features)
        projected_tabular = self.tabular_projection(tabular_features)
        tokens = torch.stack([projected_wsi, projected_tabular], dim=1)
        attended, attention_weights = self.cross_attention(tokens, tokens, tokens, need_weights=True)
        attended = self.cross_attention_norm(attended + tokens)
        fused_features = attended.flatten(start_dim=1)
        logits = self.fusion_classifier(fused_features)
        return logits, fused_features, attention_weights

    @staticmethod
    def _pool_wsi_features(features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3 and features.size(0) == 1:
            features = features.squeeze(0)
        if features.dim() == 1:
            return features.unsqueeze(0)
        if features.dim() == 2:
            return features.mean(dim=0, keepdim=True)
        raise ValueError(f"Unexpected WSI feature shape: {tuple(features.shape)}")


CLAMRNAConcatFusion = CLAMRNAFusion


def _initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _parse_hidden_dims(value) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(dim.strip()) for dim in value.split(",") if dim.strip())
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    if value is None:
        return ()
    return (int(value),)
