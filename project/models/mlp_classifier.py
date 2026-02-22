import torch.nn as nn


class MLPClassifier(nn.Module):
    """Simple MLP head for classification on pre-extracted features.

    Architecture: Linear → BatchNorm → ReLU → Dropout → Linear (logits).
    """

    def __init__(self, num_features=1536, hidden_dim=256, dropout=0.3, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)
