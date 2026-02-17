# Group Relative Policy Optimization (GRPO)

Equations referenced by `grpo.py`.

## Core equations
- Group-normalized advantage for samples $i$ in group $g$:  
  $A_i = \dfrac{r_i - \text{mean}(r_g)}{\text{std}(r_g)}$
- PPO-style ratio and clipping  
  $r_i = \dfrac{\pi_\theta(a_i \mid s_i)}{\pi_{\text{old}}(a_i \mid s_i)}$  
  $\mathcal{L}_{\text{clip}} = - \mathbb{E}\big[\min(r_i A_i,\ \text{clip}(r_i, 1-\epsilon, 1+\epsilon) A_i)\big]$
- Optional forward KL to reference policy  
  $D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) \approx \mathbb{E}[\log \pi_\theta - \log \pi_{\text{ref}}]$

## Total loss (minimization form)
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{clip}} + \beta_{\text{KL}} D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})
$$

## Notes
- Removes the learned critic; relies on multiple samples per prompt to form a baseline.
- Memory/compute lighter than PPO when group size is modest.
