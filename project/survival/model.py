"""
Attention-based Multiple Instance Learning (AMIL) for survival prediction.

Adapted from MCAT's MIL_Attention_FC_surv. Uses gated attention to aggregate
variable-length patch embeddings into a fixed-size slide representation,
then predicts discrete-time hazard probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attn_Net_Gated(nn.Module):
    """Attention network with sigmoid gating (3 FC layers).

    Computes attention scores as: A = tanh(W_a * x) * sigmoid(W_b * x),
    then projects to n_classes attention heads via W_c.

    Args:
        L: Input feature dimension.
        D: Hidden layer dimension.
        dropout: Whether to apply dropout (p=0.25).
        n_classes: Number of attention heads.
    """

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            *([nn.Dropout(0.25)] if dropout else []),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            *([nn.Dropout(0.25)] if dropout else []),
        )
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


def initialize_weights(module):
    """Xavier uniform initialization for Linear layers."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()


class AMIL_Surv(nn.Module):
    """Attention MIL for discrete-time survival prediction.

    Architecture:
        1. FC projection: input_dim -> hidden_dim (with ReLU + Dropout)
        2. Gated attention: hidden_dim -> attention_dim -> 1 score per patch
        3. Weighted aggregation: softmax(scores) @ hidden features
        4. Rho network: hidden_dim -> attention_dim (with ReLU + Dropout)
        5. Classifier: attention_dim -> n_classes (hazard probabilities per bin)

    Args:
        input_dim: Dimension of input patch embeddings (1536 for UNI2-h).
        n_classes: Number of discrete survival time bins (default 4).
        dropout: Dropout rate (default 0.25).
        size_arg: Architecture size preset, "small" or "big".
    """

    def __init__(self, input_dim=1536, n_classes=4, dropout=0.25, size_arg="small"):
        super().__init__()
        size_dict = {
            "small": [input_dim, 512, 256],
            "big": [input_dim, 512, 384],
        }
        size = size_dict[size_arg]

        # Patch-level feature projection
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        # Slide-level representation
        self.rho = nn.Sequential(
            nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)
        )

        # Survival classifier (outputs hazard per bin)
        self.classifier = nn.Linear(size[2], n_classes)

        initialize_weights(self)

    def forward(self, x_path):
        """Forward pass.

        Args:
            x_path: Patch embeddings, shape (N, input_dim).

        Returns:
            hazards: Predicted hazard probabilities, shape (1, n_classes).
            S: Survival function values, shape (1, n_classes).
            Y_hat: Predicted bin index, shape (1, 1).
            A_raw: Raw attention scores, shape (1, N).
        """
        A, h_path = self.attention_net(x_path)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()

        logits = self.classifier(h_path).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S, Y_hat, A_raw
