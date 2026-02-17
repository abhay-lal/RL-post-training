# Proximal Policy Optimization (PPO)

Equations referenced by `ppo.py`.

## Core equations
- Probability ratio  
  \(r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}\)
- Clipped surrogate objective (maximize; written here as loss to minimize)  
  \(\mathcal{L}_{\text{clip}} = - \mathbb{E}\Big[\min\big(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\big)\Big]\)
- Advantage typically from a critic: \(A_t = Q_t - V_t\) or using GAE.

## Value and entropy terms
- Optional clipped value loss  
  \(V^{\text{clip}}_t = V_{\text{old},t} + \text{clip}(V_t - V_{\text{old},t}, -\epsilon_v, \epsilon_v)\)  
  \(\mathcal{L}_V = \tfrac{1}{2} \max\big((V_t - R_t)^2,\ (V^{\text{clip}}_t - R_t)^2\big)\)
- Entropy bonus for exploration  
  \(\mathcal{L}_{\text{entropy}} = - \beta \, \mathbb{E}[H(\pi_\theta(\cdot\mid s_t))]\)

## Total loss (minimization form)
\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{clip}} + c_v \mathcal{L}_V + \mathcal{L}_{\text{entropy}}
\]
