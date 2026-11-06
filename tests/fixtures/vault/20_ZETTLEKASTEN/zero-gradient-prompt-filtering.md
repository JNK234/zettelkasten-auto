Created: 20-08-2026 00:20
Status: #baby
Tags: [[reinforcement-learning]] [[policy-gradient-methods]] [[experimental-design]]

# Zero-Gradient Prompt Filtering

## What This Is
Group-relative methods score $G$ rollouts per prompt and set each rollout's advantage relative to the group mean. That construction has a degenerate case: if every rollout in the group earns the *same* reward — all correct, or all wrong — then $r_i - \text{mean} = 0$ for every $i$, so every advantage is exactly zero, so the entire prompt contributes **precisely nothing** to the policy gradient.

The prompt still occupies batch slots. It still burns rollout compute, which is the 5–14× majority of your compute. It just produces no learning. And as training improves, the fraction of all-correct prompts climbs steadily, so your *effective* batch size silently shrinks over the run. Since noisy gradients from small batches are the most common cause of unstable GRPO, this is a self-accelerating failure: the better the model gets, the noisier its own training becomes.

The fix is to over-sample prompts per batch, discard any group with zero reward variance, and keep sampling until the batch is full of prompts that actually produce gradient. Per-batch sampling cost becomes variable — hence DAPO's name, "dynamic sampling" — but the article's point is that convergence speeds up more than sampling cost rises.

## Why It Matters
This is the rare fix that is both obviously correct once stated and easy to miss, because nothing in your metrics announces it. Training reward looks fine. Loss looks fine. The batch size you configured is not the batch size you got. It also reframes batch size as a *quality* quantity rather than a count: what matters is how many prompts carry signal, not how many prompts are present.

## Key Details
- DAPO's original "dynamic sampling" filtered only all-correct groups; Olmo 3's "Zero Gradient Filtering" generalizes to any zero-variance group, since all-wrong is equally gradient-free.
- Olmo 3 pairs it with "Active Sampling" — keeping spare samples ready so batch size stays constant after filtering.
- Reinforce-Rej reaches the same place from a rejection-sampling framing.
- Reference batch size: Olmo 3 uses 512 = 64 prompts × 8 rollouts per prompt.
- Diagnostic heuristic for the underlying batch-size problem: if increasing batch size visibly stabilizes your reward curve, your original batch was too small.
- Complementary offline move: difficulty-filter the dataset up front (drop too-easy and too-hard prompts, often gauged by measuring Pass@K with another LLM), so fewer zero-variance groups arise in the first place.

## Tensions
Same family as DAPO's overlong filtering — both say *delete the misleading or empty signal rather than guess at it*, which is a recurring design instinct in this literature. Related but distinct from [[group-level-std-normalization-creates-difficulty-bias]]: that one is about near-degenerate groups getting oversized gradients, this one about fully-degenerate groups getting none. It also interacts with batch composition advice that each batch should contain trivial, challenging, and near-impossible questions — filtering removes the extremes that composition deliberately included, an unreconciled tension. Foundational context in [[grpo-group-relative-policy-optimization]].

# References
1. [[Cameron R. Wolfe - GRPO Tricks (Making RL Actually Work)]]
