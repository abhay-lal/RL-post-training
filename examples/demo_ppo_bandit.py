"""
PPO-style update on a tiny bandit using the utilities in `ppo/ppo.py`.

This is *not* a full PPO implementation with value function training, but a
minimal example that shows how the clipped policy gradient behaves on a
multi-armed bandit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from ppo.ppo import PPO


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
        return float(self.rewards[action])


def run_demo(cfg: BanditConfig | None = None) -> None:
    cfg = cfg or BanditConfig()
    bandit = MultiArmedBandit(rewards=[0.0, 0.2, 0.5])

    logits = torch.nn.Parameter(torch.zeros(bandit.n_arms))
    optimizer = torch.optim.SGD([logits], lr=cfg.lr)

    algo = PPO()

    rewards_window: list[float] = []
    old_logits = logits.detach().clone()

    for t in range(1, cfg.n_steps + 1):
        # Collect one step of data (on-policy).
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        reward = bandit.step(int(action))
        advantage = torch.tensor([reward], dtype=torch.float32)

        # Old policy for ratio.
        with torch.no_grad():
            old_dist = Categorical(logits=old_logits)
            old_log_prob = old_dist.log_prob(action)

        loss = algo.loss(
            log_probs=log_prob.unsqueeze(0),
            old_log_probs=old_log_prob.unsqueeze(0),
            advantages=advantage.unsqueeze(0),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stale snapshot of policy.
        old_logits = logits.detach().clone()

        rewards_window.append(reward)
        if t % cfg.log_every == 0:
            avg_reward = float(np.mean(rewards_window[-cfg.log_every :]))
            print(f"[step {t:4d}] avg reward over last {cfg.log_every}: {avg_reward:.3f}")

    print("Final learned logits:", logits.detach().numpy())


if __name__ == "__main__":
    run_demo()

