Created: 12-04-2026 18:45
Status: #baby
Tags: [[fine-tuning-strategies-for-llms]] [[reinforcement-learning]]

# DPO: Direct Preference Optimization

## What This Is
The standard way to align an LLM is RLHF: train a reward model on human preferences, then optimize the policy against that reward with PPO. This is expensive, unstable, and has three moving parts that can each fail.

DPO collapses the entire pipeline into a single supervised loss. The insight: the optimal policy under a KL-constrained reward maximization objective has a closed-form relationship with the reward. Invert that relationship, and you can express the reward purely in terms of the policy's own log-probabilities. No reward model. No RL loop.

The loss is: `-log sigma(beta * (log pi_theta(chosen)/pi_ref(chosen) - log pi_theta(rejected)/pi_ref(rejected)))`. You take each preference pair, compute how much more the current policy prefers the chosen response over the rejected one *relative to a frozen reference model*, then push that gap through a sigmoid.

At initialization, the policy IS the reference model. Every log-ratio is zero. sigma(0) = 0.5. -log(0.5) = ln(2) = 0.693. Every DPO run starts at exactly this loss. If your first logged loss is not ~0.693, something is wrong with your setup.

## Why It Matters
DPO made preference optimization accessible. One forward pass per example, standard cross-entropy-style training, no reward model to maintain. It became the default alignment method for open-source models in 2024-2025.

But the simplicity hides a structural limitation: the frozen reference model acts as an anchor. The KL penalty ensures the policy never drifts too far from where it started. This is simultaneously DPO's strength (prevents collapse) and its ceiling (limits depth of change).

## Key Details
- beta controls KL constraint strength. Typical values: 0.1-0.5. Higher beta = tighter leash to reference.
- Loss is bounded: starts at ln(2), decreases toward 0 as the model separates chosen from rejected.
- The implicit reward is: r(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x)). This is a log-ratio, not an absolute score.
- Gradient signal vanishes at extremes (sigmoid saturation) -- once the model is very confident, it stops learning. This can be good (stability) or bad (premature convergence).
- In TRL, DPOTrainer handles everything. Reference model can be computed implicitly (no extra GPU memory) or explicitly loaded.

## Tensions
The reference anchor creates a fundamental tradeoff. In our sycophancy experiments, DPO recovered behavioral metrics (0.467 -> 0.268) but linear probes showed the sycophantic representation persisted internally (cross-model transfer AUROC 0.677). The model learned to suppress sycophantic outputs without reorganizing the internal circuitry that produces them.

This is the core question: is KL-constrained optimization inherently limited to surface-level behavioral change? [[simpo-reference-free-preference-optimization]] tests this by removing the anchor entirely. [[ipo-identity-preference-optimization]] keeps the anchor but changes the loss geometry. [[linear-probing-for-behavioral-intent-detection]] is how we measure whether the change is deep or cosmetic.

# References
1. [[Alignment Techniques Survey]]