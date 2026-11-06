Created: 20-08-2026 00:20
Status: #baby
Tags: [[reinforcement-learning-in-language-models]] [[experimental-design]] [[fine-tuning-strategies-for-llms]]

# Filtering to the Competence Frontier as Free Curriculum

## What This Is
Curriculum learning normally means designing an ordering: easy problems first, hard problems later, with a human deciding the schedule. The hard part is that "easy" and "hard" are properties of the *current* model, and the model changes every step, so any fixed schedule drifts out of alignment with the learner.

There is a way to get the curriculum for free when you sample $G$ responses per prompt and score each. Keep only prompts where the group is **partially** correct — at least one right, at least one wrong. Discard the rest. Formally: $0 < |\{o_i : \text{correct}\}| < G$.

Prompts the model always solves are too easy. Prompts it never solves are too hard. What survives is, by definition, the band where the model is uncertain — its competence frontier. And because the filter re-evaluates every rollout step against the live model, **the frontier moves as the model learns and the effective training set moves with it.** Nobody designed a schedule; the schedule fell out of the scoring you were already doing.

## Why It Matters
It converts a design problem (build a curriculum) into a filtering problem (drop degenerate groups), and it is self-maintaining. The measured payoff in DAPO was 42 → 50 AIME24 avg@32 — the largest single contribution in the paper.

The generalizable form: whenever you sample multiple attempts per item and grade them, the *variance of the grades within an item* is a free difficulty signal calibrated to the current model. Items with zero grade-variance carry no information.

## Key Details
- Implemented as over-sample-and-filter (Algorithm 1, lines 6–8): generate, filter into a buffer, `continue` sampling until the buffer holds $N$ valid groups, then take the gradient step.
- Dual purpose — the same filter that supplies the curriculum also removes the zero-gradient groups that were silently starving the batch (see the companion failure mode).
- Cost is architecture-dependent: nearly free in a synchronous non-pipelined trainer where the long-tail sample dominates wall-clock, genuinely expensive in an async pipelined one.
- Fig. 6 reports the stronger result: with dynamic sampling the run hits a given accuracy *faster in wall-clock terms*, because each retained gradient step is worth enough more that far fewer steps are needed.

## Tensions
This is the constructive half of [[zero-advantage-groups-shrink-the-effective-batch]] — same mechanism, read as a feature instead of a fix. It presupposes a scoring signal cheap and reliable enough to run on every rollout, which is why it fits [[rlvr-reinforcement-learning-with-verifiable-rewards]] and would be fragile under a learned reward model ([[reward-model-overoptimization-hump]]). Compare [[consensus-filtering-for-process-supervision-data]], which filters *labels* by agreement rather than *items* by difficulty, and [[curriculum-finetuning-for-manipulation]] for the hand-designed alternative. An open question the source does not address: filtering to the frontier means the model never revisits solved problems, so nothing in the objective guards against forgetting them.

# References
1. [[2503.14476 - DAPO Open-Source LLM RL System at Scale]]
