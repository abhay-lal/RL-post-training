"""
Tiny multi-armed bandit demo using REINFORCE.

This is intentionally minimal: it shows how to plug the utilities in
`reinforce/reinforce.py` into a toy environment and watch the average reward
improve over time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from reinforce.reinforce import Reinforce


@dataclass
class BanditConfig:
    n_arms: int = 3
    n_steps: int = 2_000
    log_every: int = 200
    lr: float = 0.1


class MultiArmedBandit:
    def __init__(self, rewards: list[float]) -> None:
        self.rewards = torch.tensor(rewards, dtype=torch.float32)

    @property
    def n_arms(self) -> int:
        return self.rewards.shape[0]

    def step(self, action: int) -> float:
        # Deterministic bandit for clarity; you can add noise if you like.
        return float(self.rewards[action])


def run_demo(cfg: BanditConfig | None = None) -> None:
    cfg = cfg or BanditConfig()

    # Define a simple bandit where arm 2 is optimal.
    bandit = MultiArmedBandit(rewards=[0.0, 0.2, 0.5])

    # Policy: just a learnable vector of logits over arms.
    logits = torch.nn.Parameter(torch.zeros(bandit.n_arms))
    optimizer = torch.optim.SGD([logits], lr=cfg.lr)

    algo = Reinforce()

    rewards_window: list[float] = []
    for t in range(1, cfg.n_steps + 1):
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        reward = bandit.step(int(action))
        rewards_tensor = torch.tensor([reward], dtype=torch.float32)

        loss = algo.loss(
            log_probs=log_prob.unsqueeze(0),
            rewards=rewards_tensor,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_window.append(reward)
        if t % cfg.log_every == 0:
            avg_reward = float(np.mean(rewards_window[-cfg.log_every :]))
            print(f"[step {t:4d}] avg reward over last {cfg.log_every}: {avg_reward:.3f}")

    print("Final learned logits:", logits.detach().numpy())


if __name__ == "__main__":
    run_demo()

