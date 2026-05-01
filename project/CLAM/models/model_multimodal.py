from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_clam import CLAM_MB, CLAM_SB


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
    ):
        super().__init__()
        if tabular_input_dim is None:
            raise ValueError("tabular_input_dim is required for multimodal fusion.")
        if fusion_mode not in {"concat", "gated"}:
            raise ValueError("fusion_mode must be 'concat' or 'gated'.")
        if fusion_mode == "gated" and fusion_hidden_dim <= 0:
            raise ValueError("fusion_hidden_dim must be positive for gated fusion.")

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
        self.wsi_feature_dim = self.wsi.size_dict[size_arg][1]
        self.tabular_encoder = TabularMLPEncoder(
            input_dim=tabular_input_dim,
            hidden_dim=tabular_hidden_dim,
            num_layers=tabular_num_layers,
            dropout=dropout,
        )

        fusion_input_dim = self.wsi_feature_dim + self.tabular_encoder.output_dim
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
        else:
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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.wsi_frozen:
            self.wsi.eval()
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

    def forward(self, inputs, label=None, instance_eval=False, return_features=False, attention_only=False):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError("CLAMRNAFusion expects inputs=(wsi_features, tabular_features).")

        wsi_features, tabular_features = inputs
        if attention_only:
            return self.wsi(wsi_features, attention_only=True)

        if self.wsi_frozen:
            with torch.no_grad():
                _, _, _, attention, wsi_results = self.wsi(wsi_features, return_features=True)
        else:
            _, _, _, attention, wsi_results = self.wsi(wsi_features, return_features=True)

        pooled_wsi = self._pool_wsi_features(wsi_results["features"])
        encoded_tabular = self.tabular_encoder(tabular_features)
        if self.fusion_mode == "concat":
            fused_features = torch.cat([pooled_wsi, encoded_tabular], dim=1)
            logits = self.fusion_head(fused_features)
            fusion_gate = None
        else:
            logits, fused_features, fusion_gate = self._gated_fusion(pooled_wsi, encoded_tabular)

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
        return logits, y_prob, y_hat, attention, results

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
