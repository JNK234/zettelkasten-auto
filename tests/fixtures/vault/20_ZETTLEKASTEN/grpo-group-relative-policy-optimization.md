Created: 12-04-2026 18:45
Status: #baby
Tags: [[reinforcement-learning]] [[policy-gradient-methods]]

# GRPO: Group Relative Policy Optimization

## What This Is
PPO and GRPO both optimize the same clipped surrogate objective: L = min(r*A, clip(r, 1-epsilon, 1+epsilon)*A). The clipping prevents destructively large policy updates. The entire difference is how they estimate the advantage A -- the signal that tells the model which actions were better than expected.

PPO trains a separate value network (the critic) that predicts expected return for each state. Advantage = actual return minus predicted. This gives token-level credit assignment: the model knows exactly which token in the response was the turning point. But the critic is expensive -- it's an entire neural network that must be trained alongside the policy, and for large LLMs it roughly doubles memory.

GRPO eliminates the critic entirely. For each prompt, generate G completions. Score all of them with a reward function. Normalize within the group: A_i = (r_i - mean(r)) / std(r). Completions better than the group average get positive advantage; worse ones get negative. No learned baseline needed. The group itself IS the baseline.

The tradeoff is credit granularity. PPO can say "this specific token caused the problem." GRPO can only say "this entire response was better than average." Response-level vs token-level.

## Why It Matters
Memory footprint changes everything for practical RL on LLMs. PPO needs 4 models simultaneously: policy, reference, reward model, and value network. For an 8B parameter model, that's ~65GB. GRPO needs just the policy and reference (~25GB with PEFT adapter switching). This is the difference between fitting on 1 GPU vs needing 4.

GRPO also has a simpler optimization landscape. No critic to train means no critic lag, no value function estimation errors propagating into policy updates. The group normalization is statistically noisy but unbiased.

## Key Details
- G (num_generations) typically 4-16. More generations = better advantage estimates but linearly more compute. batch_size must be divisible by num_generations.
- Reward can be binary (0/1 from a classifier), continuous (from a reward model), or rule-based (exact match, code execution pass/fail). Binary rewards with group normalization create sharp RLVR-like signals.
- In TRL 0.29.1, GRPOTrainer is the maintained RL trainer (PPOTrainer is deprecated/experimental). Default loss_type is "dapo", not "grpo" -- must set explicitly if you want standard GRPO.
- KL penalty against a reference model is optional but recommended to prevent reward hacking.
- Learning rate must be very small (1e-6 to 1e-5). RL on pretrained LLMs is notoriously unstable at higher rates.

## Tensions
The response-level vs token-level credit question maps directly onto a hypothesis about sycophancy. If sycophantic behavior is decided at a single critical token (the moment the model starts agreeing vs pushing back), PPO's token-level credit should produce deeper, more targeted change. If it's a holistic response-level property (overall tone, framing, deference), GRPO's coarser signal suffices.

[[dpo-direct-preference-optimization]] sidesteps credit assignment entirely by working on pre-existing preference pairs. GRPO generates its own training data on-policy, which should produce better coverage of the model's actual behavior distribution but at much higher compute cost.

The open question: does on-policy generation discover failure modes that a static preference dataset misses? If the sycophantic model has modes that never appear in the DPO training pairs, GRPO might find and fix them while DPO cannot.

# References
1. [[PPO vs GRPO - Policy Optimization for Alignment]]