"""Loss functions for semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import TrainConfig


class FocalLoss(nn.Module):
    """Weighted Focal Loss for multi-class segmentation.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha:  per-class weights, shape [C]. Balances rare vs frequent classes.
        gamma:  focusing parameter (>=0). Higher values down-weight easy examples more.
                gamma=0 reduces to weighted cross-entropy.
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha.clone())
        else:
            self.alpha = None

    def forward(self, logits, targets, mask=None):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)

        focal = (1.0 - pt) ** self.gamma * ce

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal

        if mask is not None:
            focal = focal * mask

        return focal.mean()


def build_criterion(
    config: TrainConfig,
    class_weights: torch.Tensor,
    device: torch.device,
) -> nn.Module:
    """Factory: return the appropriate loss module based on ``config.loss_type``."""
    if config.loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=config.focal_gamma).to(device)
    return nn.CrossEntropyLoss(weight=class_weights).to(device)
