Created: 12-04-2026 18:45
Status: #baby
Tags: [[fine-tuning-strategies-for-llms]] [[reinforcement-learning]]

# IPO: Identity Preference Optimization

## What This Is
DPO assumes preferences follow the Bradley-Terry model: the probability that response A is preferred over B is sigma(r(A) - r(B)). This works when preferences are clean and transitive. But human preferences are often noisy, contradictory, or express soft tendencies rather than hard rankings.

IPO bypasses Bradley-Terry entirely. It belongs to a general framework called Psi-PO, where different alignment methods correspond to different transforms of raw preference probabilities. RLHF and DPO both use Psi(p) = log(p/(1-p)) (log-odds, the Bradley-Terry transform). IPO uses Psi(p) = p (the identity function -- hence the name). It works directly with preference probabilities as they are, no transformation.

The loss becomes: L = (rho_theta - 1/(2*tau))^2. A squared error pushing the policy's preference margin toward a fixed target of 1/(2*tau). Where DPO's sigmoid gradient vanishes at extremes (once the model is confident, learning stops), IPO's quadratic gradient grows with distance from target. It always pulls back toward the target margin, preventing the policy from becoming deterministic.

## Why It Matters
IPO was designed for the real world where preference data is noisy. A squared loss that resists extreme confidence should be more robust to label noise and contradictory annotations.

But this creates a structural mismatch for tasks with clean, unambiguous preferences. In sycophancy recovery, the signal is nearly deterministic: honest responses are always preferred over sycophantic ones. IPO's loss actively fights the large margins this demands, pulling the model back toward moderate confidence. The very property that handles noise becomes a liability when the signal is clear.

## Key Details
- tau (or beta in some implementations) works OPPOSITE to DPO. Smaller tau = LESS regularization = larger target margin 1/(2*tau). tau=0.01 gives target margin 50. tau=0.5 gives target margin 1.
- Still uses a frozen reference model (like DPO), so the KL anchor limitation applies.
- Loss has a unique fixed point at the target margin. Below target: gradient pushes margin up. Above target: gradient pushes margin down. DPO has no such pull-back.
- In TRL, IPO is implemented as loss_type="ipo" in DPOTrainer. Same trainer, different loss function.
- The squared loss means gradients scale linearly with error, not exponentially like DPO's cross-entropy. This makes training dynamics more predictable but potentially slower to converge when large corrections are needed.

## Tensions
IPO presents a paradox for alignment research. Its theoretical advantages (noise robustness, no Bradley-Terry assumption, anti-collapse) sound ideal. But sycophancy recovery has near-deterministic preferences, and the anti-collapse mechanism actively prevents the model from expressing strong behavioral preferences.

The reference model anchor means IPO shares [[dpo-direct-preference-optimization]]'s fundamental limitation: it can only move so far from the starting point. Combined with the squared loss fighting large margins, IPO may produce the weakest behavioral recovery of any preference method. The prediction: behavioral metrics worse than DPO, and probe transfer similar (representation persists because the model can't move far enough to reorganize it).

[[simpo-reference-free-preference-optimization]] removes the anchor. [[grpo-group-relative-policy-optimization]] takes a completely different approach via RL. IPO occupies an awkward middle ground for this particular task.

# References
1. [[IPO - Identity Preference Optimization]]