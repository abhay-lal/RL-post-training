# Proximal Policy Optimization (PPO)

Equations referenced by `ppo.py`.

## Core equations
- Probability ratio  
  $r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$
- Clipped surrogate objective (maximize; written here as loss to minimize)  
  $\mathcal{L}_{\text{clip}} = - \mathbb{E}\Big[\min\big(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\big)\Big]$
- Advantage typically from a critic: $A_t = Q_t - V_t$ or using GAE.

## Value and entropy terms
- Optional clipped value loss  
  $V^{\text{clip}}_{t} = V_{\text{old},t} + \operatorname{clip}(V_t - V_{\text{old},t}, -\epsilon_v, \epsilon_v)$  
  $\mathcal{L}_V = \tfrac{1}{2} \max\big((V_t - R_t)^2,\ (V^{\text{clip}}_t - R_t)^2\big)$
- Entropy bonus for exploration  
  $\mathcal{L}_{\text{entropy}} = - \beta \,\mathbb{E}[H(\pi_\theta(\cdot\mid s_t))]$

## Total loss (minimization form)
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{clip}} + c_v \mathcal{L}_V + \mathcal{L}_{\text{entropy}}
$$

## Symbol reference (LLM context)
| Symbol | Meaning in LLM training |
| --- | --- |
| $s_t$ | Prompt plus generated tokens up to step $t$ (context) |
| $a_t$ | Token sampled at step $t$ |
| $\pi_\theta$ | Current policy (LLM) producing token probabilities |
| $\pi_{\text{old}}$ | Frozen policy used when data were collected |
| $r_t$ | Scalar reward at step $t$ (often terminal) |
| $A_t$ | Advantage (e.g., GAE or $Q_t - V_t$) |
| $V_t$ | Value estimate (critic) at step $t$ |
| $R_t$ | Return target for value learning |
| $r_t(\theta)$ | Probability ratio $\pi_\theta / \pi_{\text{old}}$ for the taken token |
| $\epsilon, \epsilon_v$ | Clip ranges for policy and value updates |
| $c_v$ | Weight on value loss |
| $\beta$ | Weight on entropy bonus (named $\beta$ here to match the equation) |
