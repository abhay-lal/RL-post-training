"""
Minimal REINFORCE utilities in PyTorch.

This module focuses on clarity over throughput: it exposes helpers to compute
discounted returns and a simple loss that supports an optional baseline
(which can be a learned value estimate). The loss is written in minimization
form so it can be dropped directly into a standard optimizer step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class ReinforceConfig:
    gamma: float = 1.0
    entropy_coef: float = 0.0
    """
    Weight for the entropy bonus. Useful to keep exploration early in training.
    """


def discounted_returns(
    rewards: torch.Tensor, gamma: float
) -> torch.Tensor:
    """
    Compute discounted returns G_t for a 1D reward tensor.

    Args:
        rewards: shape (T,) or (batch, T) of scalar rewards.
        gamma: discount factor in [0, 1].

    Returns:
        Tensor of same shape as rewards with discounted sums.
    """
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(0)

    batch, time = rewards.shape
    returns = torch.zeros_like(rewards)
    running = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(time)):
        running = rewards[:, t] + gamma * running
        returns[:, t] = running

    return returns.squeeze(0) if returns.shape[0] == 1 else returns


def reinforce_loss(
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    entropy: Optional[torch.Tensor] = None,
    entropy_coef: float = 0.0,
) -> torch.Tensor:
    """
    REINFORCE objective in loss (minimization) form.

    Args:
        log_probs: log π(a_t | s_t); shape (..., T) or (batch, T).
        returns: discounted returns G_t; same broadcastable shape as log_probs.
        baseline: optional baseline b_t to reduce variance. If provided, the
            advantage is (returns - baseline). baseline should be detached from
            gradients unless you intentionally co-train it.
        entropy: optional entropy per step; if None, it is computed from
            log_probs assuming log_probs came from a categorical sample (pass
            precomputed entropy if you sampled with a distribution object).
        entropy_coef: weight for the entropy bonus.
    """
    if baseline is not None:
        advantages = returns - baseline
    else:
        advantages = returns

    # Align shapes for broadcasting.
    advantages = advantages.detach()

    policy_loss = -(advantages * log_probs).mean()

    if entropy is None:
        # If log_probs are from a categorical sample, we approximate entropy
        # using the log-prob itself; more stable is to pass entropy explicitly
        # from the sampling distribution (e.g., torch.distributions.Categorical).
        entropy_term = -log_probs
    else:
        entropy_term = entropy

    entropy_loss = -(entropy_coef * entropy_term).mean()
    return policy_loss + entropy_loss


class Reinforce:
    """
    Convenience wrapper that couples return computation with the REINFORCE loss.
    """

    def __init__(self, config: ReinforceConfig | None = None) -> None:
        self.config = config or ReinforceConfig()

    def compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        return discounted_returns(rewards, self.config.gamma)

    def loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        returns = self.compute_returns(rewards)
        return reinforce_loss(
            log_probs=log_probs,
            returns=returns,
            baseline=baseline,
            entropy=entropy,
            entropy_coef=self.config.entropy_coef,
        )
