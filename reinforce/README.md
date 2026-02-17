# REINFORCE (policy gradient)

Minimal equations used by `reinforce.py`.

## Core equations
- Discounted return per timestep  
  $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$
- Loss (minimization form) without baseline  
  $\mathcal{L} = - \sum_t G_t \log \pi_\theta(a_t \mid s_t)$
- With baseline $b_t$ to reduce variance (advantage form)  
  $\mathcal{L} = - \sum_t (G_t - b_t) \log \pi_\theta(a_t \mid s_t)$

## Notes
- Baseline can be a learned value function or any control variate.
- Entropy bonus (optional) encourages exploration:  
  $\mathcal{L}_{\text{entropy}} = - \alpha \, \mathbb{E}[H(\pi_\theta(\cdot\mid s))]$

## Symbol reference (LLM context)
| Symbol | Meaning in LLM training |
| --- | --- |
| $s_t$ | Prompt plus all tokens generated up to step $t$ (context) |
| $a_t$ | Token sampled at step $t$ |
| $\pi_\theta$ | Current policy (LLM) that outputs token probabilities |
| $r_t$ | Scalar reward for step $t$ (often only at the end) |
| $G_t$ | Discounted return from $t$ onward |
| $b_t$ | Baseline (e.g., value estimate) to reduce variance |
| $A_t$ | Advantage: $G_t - b_t$ |
| $\gamma$ | Discount factor |
| $\alpha$ | Entropy or learning-rate style coefficient (context dependent) |
