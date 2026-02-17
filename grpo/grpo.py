"""
Group Relative Policy Optimization (GRPO) utilities in PyTorch.

GRPO removes the learned critic. It samples multiple completions per prompt and
uses group statistics (mean/std rewards) as a baseline to form normalized
advantages. The policy update mirrors PPO's clipped objective and typically
includes an explicit KL term to a reference policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GRPOConfig:
    clip_range: float = 0.2
    beta_kl: float = 0.1
    eps: float = 1e-6


def _group_normalize_advantages(
    rewards: torch.Tensor, group_ids: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    Compute normalized advantages per group: (r_i - mean)/std.

    Args:
        rewards: shape (batch,) tensor of scalar rewards.
        group_ids: shape (batch,) integer ids indicating which samples belong
            to the same prompt/group.
        eps: small constant to avoid division by zero.
    """
    advantages = torch.zeros_like(rewards)
    for gid in group_ids.unique():
        mask = group_ids == gid
        group_r = rewards[mask]
        mean = group_r.mean()
        std = group_r.std(unbiased=False).clamp_min(eps)
        advantages[mask] = (group_r - mean) / std
    return advantages


def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    ref_log_probs: Optional[torch.Tensor] = None,
    config: GRPOConfig | None = None,
) -> torch.Tensor:
    """
    Compute GRPO loss (policy clip + optional KL).

    Args:
        log_probs: current log π(a|s).
        old_log_probs: frozen log π_old(a|s) at sampling time.
        rewards: scalar rewards per sample (one per completion).
        group_ids: integer ids to group samples from the same prompt.
        ref_log_probs: log probabilities from a frozen reference model
            (optional). If provided, a forward KL penalty is applied.
        config: GRPOConfig hyperparameters.
    """
    cfg = config or GRPOConfig()

    advantages = _group_normalize_advantages(rewards, group_ids, cfg.eps).detach()
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
    pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    if ref_log_probs is not None:
        # Sampled forward KL approximation: E[log π - log π_ref]
        kl = (log_probs - ref_log_probs).mean()
        loss = pg_loss + cfg.beta_kl * kl
    else:
        loss = pg_loss
    return loss


class GRPO:
    """Wrapper for GRPO loss with stored config."""

    def __init__(self, config: GRPOConfig | None = None) -> None:
        self.config = config or GRPOConfig()

    def loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        group_ids: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return grpo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            rewards=rewards,
            group_ids=group_ids,
            ref_log_probs=ref_log_probs,
            config=self.config,
        )
