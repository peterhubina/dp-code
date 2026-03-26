"""
Discrete-time survival loss functions.

Adapted from MCAT (Multimodal Co-Attention Transformer) implementation.
Reference: Chen et al., "Multimodal Co-Attention Transformer for Survival Prediction
in Gigapixel Whole Slide Images", ICCV 2021.

The discrete-time approach bins continuous survival times into K intervals and models
the hazard probability h(t) for each interval. The survival function is computed as:
    S(t) = prod_{i=0}^{t} (1 - h(i))
"""

import torch


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    """Negative log-likelihood loss for discrete-time survival models.

    Args:
        hazards: Predicted hazard probabilities, shape (batch, n_bins). Values in (0,1).
        S: Survival function values, shape (batch, n_bins). S(t) = cumprod(1 - h).
        Y: Ground truth time bin indices, shape (batch,). Values in {0, ..., n_bins-1}.
        c: Censorship indicator, shape (batch,). 1=censored, 0=uncensored (event occurred).
        alpha: Weight for additional uncensored loss term (default 0.4).
        eps: Small constant for numerical stability.

    Returns:
        Scalar loss value.
    """
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    # Pad S with 1.0 at the start: S(-1) = 1 (all patients alive before t=0)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # Uncensored: log P(dying in bin Y) = log S(Y-1) + log h(Y)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    # Censored: log P(surviving past bin Y) = log S(Y)
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    """Cross-entropy survival loss (alternative to NLL).

    Combines cross-entropy on survival probabilities with a regularization
    term based on the NLL formulation.
    """
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y) + eps)
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    ce_l = -c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(
        1 - torch.gather(S, 1, Y).clamp(min=eps)
    )
    loss = (1 - alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class NLLSurvLoss:
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


class CrossEntropySurvLoss:
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)
