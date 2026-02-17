# Direct Preference Optimization (DPO)

Equations referenced by `dpo.py`.

## Core equations
- Bradley–Terry preference model for winner \(y_w\) over loser \(y_l\):  
  \(P(y_w > y_l) = \sigma\big(r(x, y_w) - r(x, y_l)\big)\)
- Replace reward differences with log-probability ratios to a frozen reference policy:  
  \[
  \Delta = \beta \Big[\log \tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log \tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big]
  \]
- Loss (minimization form):  
  \(\mathcal{L}_{\text{DPO}} = - \mathbb{E}[\log \sigma(\Delta)]\)

## Notes
- Larger \(\beta\) pulls the policy closer to the reference (stronger KL-style regularization).
- Requires pairwise preference data (winner/loser) rather than scalar rewards.
