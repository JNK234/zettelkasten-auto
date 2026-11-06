Created: 20-08-2026 00:20
Status: #baby
Tags: [[reinforcement-learning-in-language-models]] [[reward]] [[evaluation-methodology]]

# Instance-Specific Rubric as Reward Function

## What This Is
RL post-training needs a number for every sampled response. In math and code you get that number for free — run the tests, match the boxed answer. In medicine or open-ended science there is no string to match: a good answer must get the arithmetic right *and* recommend the safe dose *and* explain why it deviates from the arithmetic. Correctness is a vector, not a bit.

A rubric-as-reward attaches to *each individual prompt* a list of 7–20 criteria, each with a weight and each written so a judge can check it in isolation. The reward is the fraction of achievable weight the response earned:

```
r(x, ŷ) = ( Σ_j w_j · c_j(x, ŷ) ) / ( Σ_j w_j )
```

where `c_j : (x, ŷ) ↦ {0,1}` is one binary judge call per criterion. The key word is **instance-specific**: the checklist is written for *this* prompt, not reused across the dataset.

## Why It Matters
It turns "verifiable vs non-verifiable domain" from a binary into a dial. Set `k = 1`, `w_1 = 1`, and `c_1 = exact-match`, and Equation 1 collapses exactly to [[rlvr-reinforcement-learning-with-verifiable-rewards]]. RLVR is the `k=1` corner of a much larger design space; everything above `k=1` buys multi-dimensional supervision and the ability to encode subjective aspects (tone, empathy, safety hedging) alongside objective ones.

The bicarbonate example from the paper shows what a scalar cannot express: one criterion rewards *computing* 780 mEq (`40 × 65 × 0.3`), a separate one rewards *not administering* it (~150 mEq over 4 hours instead). A preference score has no way to say "get the arithmetic visibly right, then deliberately deviate from it for safety."

## Key Details
- The denominator matters: rubrics range 7–20 items with different weight profiles, so an unnormalized sum would systematically over-reward verbose-rubric prompts. Under [[grpo-group-relative-policy-optimization]] within-group advantages wash out some of this, but batch-level reward variance still needs it.
- Category scheme used: Essential (w=5, omission invalidates the response), Important (3–4), Optional (1–2), Pitfall (−1 or −2).
- `w_j` is real-valued, so it can be negative — that is the mechanism for encoding prompt-specific failure modes.
- Actual average rubric length in `RaR-Medicine` (20,166 prompts) and `RaR-Science` (20,625 prompts): 7.5 items.
- On HealthBench, Qwen2.5-7B trained this way reaches 31.2% vs 25.5% for a bare Likert reward and 22.7% for Qwen2.5-7B-Instruct.

## Tensions
This is orthogonal densification to [[process-reward-model-as-step-level-verifier]] and [[process-reward-model-vs-outcome-reward-model]]: PRMs densify sparse reward along the *time* axis (score each reasoning step), rubrics densify along the *quality-dimension* axis (score each aspect of the final answer). Nobody has combined them.

It is an alternative to the [[bradley-terry-reward-model]] pipeline of [[reinforcement-learning-from-human-feedback-rlhf]], and its motivation is the failure mode in [[reward-model-overoptimization-hump]] — a learned preference model latches onto length and formatting artifacts, the same mechanism behind [[rlhf-amplifies-sycophancy]]. A criterion like "states 150 mEq over 4 hours" structurally cannot be satisfied by writing more words, so unlike [[length-normalized-dpo]] there is no length bias to correct post hoc (inference — the paper notes the structure but never tests hacking resistance).

Per-criterion binariness is [[binary-judge-scoring-over-likert]] applied inside a reward function, and instance-specificity is the direct answer to [[criteria-drift-in-evaluation]]. The untested risk is [[reward-hacking-as-specification-gaming]] — 300 training steps may simply be before the hump. Contrast the auto-generation angle in [[auto-generated-rule-based-reward]] and [[verifiable-rule-based-rewards-prevent-reward-hacking]].

# References
1. [[2507.17746 - Rubrics as Rewards (RaR)]]
