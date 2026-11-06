Created: 20-08-2026 00:20
Status: #baby
Tags: [[reinforcement-learning-in-language-models]] [[policy-gradient-methods]] [[fine-tuning-strategies-for-llms]]

# Covariance-Targeted Entropy Intervention

## What This Is
If entropy collapse is driven by ~0.02% of tokens that are both high-probability and high-advantage, the fix is to disarm exactly those tokens and leave the rest of the gradient alone. Two drop-in modifications to the GRPO/PPO token loss do this, both built on the same per-token covariance estimate $(\log\pi_\theta(y_i) - \overline{\log\pi_\theta})(A(y_i) - \bar{A})$.

**Clip-Cov — detach the outliers.** Pick tokens whose covariance falls in a high band and zero their gradient entirely. Hyperparameters: clip fraction $r = 2\times10^{-4}$, band $[\omega_\text{low}=1,\ \omega_\text{high}=5]$.

Two non-default choices carry the design. First, tokens are sampled **randomly within the band, not deterministically top-$k$** — deterministic detachment would create a permanent blind spot where those tokens are never learned at all; random selection mutes each occasionally instead of excluding it forever. Second, there is an **upper bound**: tokens above covariance 5 are *not* clipped, which keeps the intervention on the intended population rather than on numerically degenerate samples.

**KL-Cov — penalize instead of detach.** Rank tokens by covariance, take the top $k$ proportion, and apply a KL penalty against the **rollout policy** $\pi_{\theta_\text{old}}$ on those tokens only. $k = 2\times10^{-3}$ (7B), $2\times10^{-4}$ (32B), $\beta = 1$.

The KL target matters. It is a *local* trust region that moves with training, not an anchor to a frozen base model — which is exactly why it avoids the reference-KL failure mode where entropy stabilizes but performance dies.

## Why It Matters
This is the opposite philosophy from standard PPO clipping. PPO's ratio clip bounds how far the *policy* may move per step; Clip-Cov bounds which *tokens* are allowed to contribute to sharpening. Same clipping machinery, a completely different axis of control.

The gains grow with model size, which is the direction that matters — most RL tweaks help at 7B and vanish at scale.

| Model | Method | AIME24 | AIME25 | MATH500 | Avg |
|---|---|---|---|---|---|
| 7B | GRPO | 21.2 | 9.6 | 78.8 | 38.6 |
| 7B | KL-Cov | 22.6 | 12.9 | 80.8 | **40.6** |
| 32B | GRPO | 21.8 | 16.2 | 84.2 | 45.8 |
| 32B | KL-Cov | 36.8 | 30.8 | 84.6 | **52.2** |

The sharpest datum: 32B GRPO scores 21.8 on AIME24, *below* the 7B KL-Cov result. The 32B baseline collapses before it can use its capacity — the bottleneck was never model capability.

## Key Details
- $\beta = 1$ is enormous next to typical global KL coefficients ($10^{-3}$ or smaller). It can be, because it touches 0.02–0.2% of tokens.
- $k$ shrinks **10× going from 7B to 32B** — the larger model needs a *narrower* intervention, consistent with bigger models having better entropy-to-performance exchange rates and being more fragile to blunt intervention.
- Baselines: GRPO (lr $5\times10^{-7}$, batch 256, 512 prompts × 8 responses, PPO clip $\epsilon=0.2$, reference KL coefficient 0, max gen 8192) and Clip-Higher from DAPO.
- Split verdict in their own table, unremarked by the authors: KL-Cov wins on hard low-pass-rate benchmarks (AIME24 36.8 vs 32.3), Clip-Cov wins on saturated ones (MATH500 87.0 vs 84.6). Hard problems prefer the soft penalty; easy ones prefer hard detachment.

## Tensions
Directly built on [[heavy-tailed-token-contribution-to-entropy-collapse]] and [[log-prob-advantage-covariance-drives-entropy-change]], and motivated by [[why-global-entropy-regularization-fails-for-llms]]. KL-Cov is best understood as a token-selective local trust region, so the lineage runs through [[trust-region-policy-optimization-monotonic-improvement-guarantee]] and [[ppo-as-trpo-simplification]] — but selective in *support* rather than in step size. Contrast with [[value-function-clipping]], which clips a magnitude; this clips a membership.

Because these methods *move* the ceiling $-a+b$ in the [[entropy-performance-exchange-curve]], they falsify the strong reading of that ceiling as an intrinsic model property.

Two real weaknesses. Every threshold ($r$, $k$, $\omega_\text{low}$, $\omega_\text{high}$, $\beta$) is tuned empirically with no principled setpoint — the source states plainly that whether an optimal entropy level exists is open — and $k$ already needed a 10× change across two model sizes, so transfer is doubtful. And the whole approach assumes clean advantages from [[rlvr-reinforcement-learning-with-verifiable-rewards]]; under [[reinforcement-learning-from-human-feedback-rlhf]] with a learned reward model, targeting high-covariance tokens could amplify reward-model artifacts, which links to [[reward-model-overoptimization-hump]] and [[reward-hacking-as-specification-gaming]].

# References
1. [[2505.22617 - The Entropy Mechanism of RL for Reasoning LLMs]]
