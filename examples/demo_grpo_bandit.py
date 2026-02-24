"""
GRPO-style update on a toy bandit using `grpo/grpo.py`.

We simulate multiple samples per "prompt" by drawing several actions from the
same bandit, using group-normalized rewards as advantages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from grpo.grpo import GRPO


@dataclass
class GRPOBanditConfig:
    n_arms: int = 3
    group_size: int = 8
    n_groups: int = 250
    lr: float = 0.1


class MultiArmedBandit:
    def __init__(self, rewards: list[float]) -> None:
        self.rewards = torch.tensor(rewards, dtype=torch.float32)

    @property
    def n_arms(self) -> int:
        return self.rewards.shape[0]

    def step(self, action: int) -> float:
        return float(self.rewards[action])


def run_demo(cfg: GRPOBanditConfig | None = None) -> None:
    cfg = cfg or GRPOBanditConfig()
    bandit = MultiArmedBandit(rewards=[0.0, 0.2, 0.5])

    logits = torch.nn.Parameter(torch.zeros(bandit.n_arms))
    optimizer = torch.optim.SGD([logits], lr=cfg.lr)

    algo = GRPO()

    rewards_history: list[float] = []

    for g in range(1, cfg.n_groups + 1):
        group_ids = []
        log_probs = []
        old_log_probs = []
        rewards = []

        with torch.no_grad():
            old_dist = Categorical(logits=logits.detach().clone())

        for i in range(cfg.group_size):
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            old_log_probs.append(old_dist.log_prob(action))

            reward = bandit.step(int(action))
            rewards.append(reward)
            group_ids.append(g)  # all samples in same group

        log_probs_t = torch.stack(log_probs)
        old_log_probs_t = torch.stack(old_log_probs)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        group_ids_t = torch.tensor(group_ids, dtype=torch.int64)

        loss = algo.loss(
            log_probs=log_probs_t,
            old_log_probs=old_log_probs_t,
            rewards=rewards_t,
            group_ids=group_ids_t,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_history.extend(rewards)
        if g % 10 == 0:
            avg_reward = float(np.mean(rewards_history[-10 * cfg.group_size :]))
            print(
                f"[group {g:3d}] avg reward over last {10 * cfg.group_size} pulls: "
                f"{avg_reward:.3f}"
            )

    print("Final learned logits:", logits.detach().numpy())


if __name__ == "__main__":
    run_demo()

