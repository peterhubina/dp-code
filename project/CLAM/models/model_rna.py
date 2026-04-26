import torch
import torch.nn as nn


class RNA_MLP(nn.Module):
    """Feed-forward classifier for bulk RNA-seq expression vectors."""

    def __init__(self, input_dim, hidden_dims=(512, 256), dropout=0.25, n_classes=4):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, return_features=False):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        features = self.encoder(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits
