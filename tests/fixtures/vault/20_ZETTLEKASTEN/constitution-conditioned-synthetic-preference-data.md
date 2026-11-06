
Created: 12-05-2026 21:10

Status: #baby

Tags: [[constitutional-ai]] [[direct-preference-optimization-dpo]] [[ai-alignment]] [[reinforcement-learning-from-human-feedback-rlhf]]

# Constitution-Conditioned Synthetic Preference Data

## What This Is

What Anthropic actually does for sycophancy in production. It's not canonical Bai 2022 CAI (SL critique-revise → RL on AI-labeled preferences with PPO). Instead: generate two candidate responses to a synthetic prompt, have a separate Claude instance grade them against constitutional principles, use the resulting preference pair as DPO training data. The constitution conditions the *grader*, not a self-critique loop.

Same outcome — alignment data that doesn't rely on human preference labels — different mechanism. The grader doesn't critique-and-revise. It just compares two existing responses. No revision step exists. The labels are the comparison output.

## Why It Matters

This is a closer match to what most practitioners actually want from "CAI" than the original 2022 paper. Two practical advantages:

1. **No revision quality problem.** The canonical CAI pipeline trains on revisions, which means a bad critique produces a bad revision and the model learns the badness ([[small-model-cai-collapse]]). Constitution-conditioned synthetic preference data sidesteps this: the model never trains on a generated revision, only on comparisons of two existing responses.
2. **Reuses standard DPO infrastructure.** You don't need a critique-revise pipeline. You need a sampler (your existing policy) and a grader (any reasonably capable model with the constitution in its system prompt). The training algorithm is whatever DPO/SimPO/IPO you already have.

Reported outcomes from Anthropic public communications: ~half the sycophancy rate in Opus 4.7 vs 4.6 in relationship-guidance domain, generalizing across domains.

## Key Details

- The two responses come from sampling the policy you're trying to align. They're on-policy.
- The grader is a separate model instance — typically a capable production model with the constitution in its system prompt. Not a fine-tuned reward model. Not the same model as the policy.
- One principle is typically sampled per grading call, similar to Bai 2022's critique-revise design.
- The preference label can be soft (log-probabilities of "A" vs "B" tokens) or hard (argmax). Soft labels carry more information and are what RLAIF training uses.
- Difference from RLAIF as Lee 2023 described it: RLAIF trains a reward model on AI preferences then runs PPO. Constitution-conditioned synthetic preference data skips the reward model entirely — go straight from AI preferences to DPO.

## Tensions

- **It's RLAIF without the reward model.** The reward-model staleness problem (the policy drifts away from where the RM was calibrated) is sidestepped. But you lose the ability to score arbitrary new generations during training — the grader has to be queried each time.
- **The grader is itself trained on human preferences.** This is the chain that ends in human labels eventually. The CAI selling point of "no humans" is technically true only for the labels of the *current* training run.
- The structural similarity to [[imitation-vs-contrast-in-cai]] is exact: this is the contrast regime. Synthetic preferences as DPO labels works because contrast is the right framing, not because the constitution is magic.
- Connected to [[constitution-guided-distillation-vs-self-critique-cai]] — both abandon self-critique in favor of an external grader. The distillation regime additionally uses the grader to *write* the chosen response; the synthetic-preference regime only uses it to *pick* between two existing responses.

# References

1. [[Constitutional AI for Sycophancy - Literature Survey]]
2. [[Constitutional AI - Practical Implementation Research]]
