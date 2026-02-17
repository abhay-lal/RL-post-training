# Notes: REINFORCE, PPO, DPO, GRPO

These notes condense the main ideas, math, and trade-offs for the algorithms implemented alongside this file set (see `reinforce/`, `ppo/`, `dpo/`, `grpo/`). They are adapted from the provided reference.

## Context: RL for LLMs
- Typical pipeline: pre-training → supervised fine-tuning → preference/reasoning fine-tuning.
- Goal: update a policy (LLM) from sparse/terminal rewards or human preferences while keeping updates stable and efficient.
- Challenge: high variance credit assignment over long token trajectories.

## REINFORCE
- Pure policy gradient; updates after full trajectory with return $G_t$.
- Update (as loss to minimize): $-\sum_t G_t \log \pi_\theta(a_t \mid s_t)$.
- Baseline $b$ reduces variance: replace $G_t$ with $G_t - b$ (critic in PPO; group stats in GRPO).
- Pros: simple, unbiased gradient; works with any stochastic policy. Cons: high variance, delayed updates.

## PPO (Proximal Policy Optimization)
- Actor–critic with clipped surrogate to avoid large policy jumps.
- Ratio $r_t = \pi_{\text{new}} / \pi_{\text{old}}$; objective uses $\min(r_t A_t, \text{clip}(r_t, 1\pm\epsilon) A_t)$.
- Typically pairs with a value function baseline and entropy bonus.
- Pros: stable, well-tested. Cons: heavier (policy + ref + reward + critic), higher memory/compute.

## DPO (Direct Preference Optimization)
- Treats preference pairs as a direct optimization problem (no critic, no learned reward model).
- Loss (winner $w$, loser $l$): $-\log \sigma\big(\beta[(\log \pi_\theta - \log \pi_{\text{ref}})_w - (\log \pi_\theta - \log \pi_{\text{ref}})_l]\big)$.
- Pros: simple, efficient; just policy + frozen reference. Cons: geared to pairwise preference data, less direct for absolute rewards.

## GRPO (Group Relative Policy Optimization)
- RL-style like PPO but removes the critic; uses group statistics for baseline.
- For samples grouped by prompt, advantage is normalized per group: $A_i = (r_i - \text{mean}) / \text{std}$.
- Uses PPO-style clipping on $r_t A_t$ and often an explicit KL to a reference model.
- Pros: major memory savings vs PPO; good for verifiable rewards (math/code). Cons: needs multiple completions per prompt; depends on group size.

## Practical contrasts
- **PPO vs DPO:** PPO is full RL with critic; DPO is direct preference loss (simpler, often more stable for pairwise prefs).
- **PPO vs GRPO:** GRPO drops the critic and uses group baselines, cutting memory while retaining a clipped update.
- **REINFORCE vs PPO/GRPO:** REINFORCE is the base policy gradient; PPO/GRPO add variance/stability fixes (critic or group baseline + clipping/KL).

## Where to look in code
- `reinforce/`: discounted returns + baseline-aware REINFORCE loss.
- `ppo/`: clipped surrogate with optional clipped value loss and entropy bonus.
- `dpo/`: Bradley–Terry style preference loss with reference policy regularization.
- `grpo/`: group-normalized advantages + PPO-style clipping + optional KL to reference.

## Symbol reference (LLM context)
| Symbol | Meaning in LLM training |
| --- | --- |
| $s_t$ | Prompt plus generated tokens up to step $t$ (context) |
| $a_t$ | Token sampled at step $t$ |
| $\pi_\theta$ | Current policy (LLM) |
| $\pi_{\text{old}}$ | Frozen policy used to generate trajectories |
| $\pi_{\text{ref}}$ | Frozen reference policy (for KL or DPO) |
| $r_t$ | Scalar reward at step $t$ (often only terminal) |
| $G_t$ | Discounted return from $t$ onward |
| $b_t$ | Baseline (value estimate or group statistic) |
| $A_t$ | Advantage: $G_t - b_t$ or group-normalized reward |
| $\rho_t$ | Probability ratio $\pi_\theta / \pi_{\text{old}}$ for the taken token |
| $\epsilon$ | Clip range for policy ratios |
| $\beta$ | KL or preference weighting (algorithm-dependent) |

