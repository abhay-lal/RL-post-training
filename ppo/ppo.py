"""
Proximal Policy Optimization (PPO) utilities in PyTorch.

Implements the clipped policy gradient objective plus an (optional) clipped
value loss and entropy bonus. The API is purposefully lightweight so you can
wire it into a training loop for sequence models or classic RL environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class PPOConfig:
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    normalize_advantage: bool = True


def ppo_clipped_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    value_preds: Optional[torch.Tensor] = None,
    old_value_preds: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    config: PPOConfig | None = None,
) -> torch.Tensor:
    """
    Compute PPO loss (policy + value + entropy) in minimization form.

    Args:
        log_probs: current log π(a|s).
        old_log_probs: log π_old(a|s), frozen at data collection time.
        advantages: A_t estimates; will be normalized if configured.
        value_preds: current value estimates V(s).
        old_value_preds: value estimates at data collection time.
        returns: target returns for value learning.
        config: PPOConfig with hyperparameters.
    """
    cfg = config or PPOConfig()

    adv = advantages
    if cfg.normalize_advantage:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
    pg_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()

    # Value loss (optional).
    if value_preds is not None and returns is not None:
        if old_value_preds is None:
            value_loss = 0.5 * F.mse_loss(value_preds, returns)
        else:
            value_clipped = old_value_preds + (value_preds - old_value_preds).clamp(
                -cfg.value_clip_range, cfg.value_clip_range
            )
            unclipped_loss = (value_preds - returns) ** 2
            clipped_loss = (value_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(unclipped_loss, clipped_loss).mean()
    else:
        value_loss = torch.tensor(0.0, device=log_probs.device)

    # Entropy bonus encourages exploration.
    entropy = -(log_probs.exp() * log_probs).mean()
    loss = pg_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
    return loss


class PPO:
    """Small wrapper around the PPO loss for convenience."""

    def __init__(self, config: PPOConfig | None = None) -> None:
        self.config = config or PPOConfig()

    def loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        value_preds: Optional[torch.Tensor] = None,
        old_value_preds: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return ppo_clipped_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            value_preds=value_preds,
            old_value_preds=old_value_preds,
            returns=returns,
            config=self.config,
        )
