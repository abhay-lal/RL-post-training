# Direct Preference Optimization (DPO)

Equations referenced by `dpo.py`.

## Core equations
- Bradley–Terry preference (winner $y_w$ vs loser $y_l$): $P(y_w > y_l) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$
- Margin vs frozen reference: $\Delta = \beta \big[\log \tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log \tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\big]$
- Loss (minimization form): $\mathcal{L}_{\text{DPO}} = - \mathbb{E}[\log \sigma(\Delta)]$

## Notes
- Larger $\beta$ pulls the policy closer to the reference (stronger KL-style regularization).
- Requires pairwise preference data (winner/loser) rather than scalar rewards.

## Symbol reference (LLM context)
| Symbol | Meaning in LLM training |
| --- | --- |
| $x$ | Input prompt |
| $y_w, y_l$ | Preferred (winner) and rejected (loser) responses |
| $\pi_\theta$ | Current policy (LLM) producing token probabilities |
| $\pi_{\text{ref}}$ | Frozen reference policy |
| $\Delta$ | Preference margin from log-probability ratios |
| $\beta$ | Strength of regularization toward the reference (KL-like) |
| $\sigma$ | Sigmoid; converts margin to preference probability |
