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
