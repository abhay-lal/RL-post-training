"""
Synthetic DPO demo on log-probabilities.

This does not hook into a real language model; instead it simulates log-probs
for "chosen" vs "rejected" completions and shows how the DPO loss prefers the
chosen samples while staying close to a frozen reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dpo.dpo import DPO


@dataclass
class DPOSyntheticConfig:
    n_pairs: int = 1_024
    steps: int = 500
    lr: float = 0.05


def run_demo(cfg: DPOSyntheticConfig | None = None) -> None:
    cfg = cfg or DPOSyntheticConfig()

    # Simulated reference scores for chosen vs rejected samples.
    ref_chosen = torch.randn(cfg.n_pairs)
    ref_rejected = ref_chosen - 0.5  # reference slightly prefers chosen.

    # Trainable offsets that represent how much we move away from the reference.
    delta_chosen = torch.nn.Parameter(torch.zeros_like(ref_chosen))
    delta_rejected = torch.nn.Parameter(torch.zeros_like(ref_rejected))

    optimizer = torch.optim.Adam([delta_chosen, delta_rejected], lr=cfg.lr)
    algo = DPO()

    for step in range(1, cfg.steps + 1):
        chosen_logps = ref_chosen + delta_chosen
        rejected_logps = ref_rejected + delta_rejected

        loss = algo.loss(
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen,
            ref_rejected_logps=ref_rejected,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            # Measure preference margin.
            margin = (chosen_logps - rejected_logps).mean().item()
            print(f"[step {step:3d}] loss={loss.item():.4f}  mean margin={margin:.3f}")

    print("Finished synthetic DPO training.")


if __name__ == "__main__":
    run_demo()

