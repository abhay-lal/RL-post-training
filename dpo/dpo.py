"""
Direct Preference Optimization (DPO) loss in PyTorch.

Implements the Bradley–Terry preference loss that compares a chosen sample
against a rejected sample using log-probability ratios to a frozen reference
policy. This keeps the trained policy close to the reference while preferring
winner samples over loser samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DPOConfig:
    beta: float = 0.1
    """
    Scaling for the KL-style regularization toward the reference policy.
    Larger beta -> stronger pull toward the reference (more conservative).
    """


def dpo_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    config: DPOConfig | None = None,
) -> torch.Tensor:
    """
    Compute DPO loss (minimization form).

    Args:
        chosen_logps: log πθ for the preferred/comparatively better sample.
        rejected_logps: log πθ for the less preferred sample.
        ref_chosen_logps: log π_ref for the preferred sample (frozen policy).
        ref_rejected_logps: log π_ref for the less preferred sample.
        config: DPOConfig controlling beta.
    """
    cfg = config or DPOConfig()
    # Step 1: construct preference margin from log-prob ratios vs. reference
    margin = cfg.beta * (
        (chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)
    )
    # Step 2: Bradley–Terry likelihood => log-sigmoid of margin
    # Step 3: minimize negative log-likelihood (mean over batch)
    return -F.logsigmoid(margin).mean()


class DPO:
    """Thin wrapper to keep config and loss together."""

    def __init__(self, config: DPOConfig | None = None) -> None:
        self.config = config or DPOConfig()

    def loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        return dpo_loss(
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            config=self.config,
        )
