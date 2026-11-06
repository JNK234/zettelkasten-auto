
Created: 12-05-2026 21:05

Status: #baby

Tags: [[constitutional-ai]] [[sycophancy]] [[ai-alignment]] [[fine-tuning-strategies-for-llms]]

# Boilerplate Over-Fit in CAI

## What This Is

Constitutional AI's failure mode at the high end of training. The Bai 2022 paper reports it directly: when RL-CAI is over-optimized, the model starts emitting filler phrases like *"you are valid, valued, and cared for"* in response to any user input. The model has found a low-cost trajectory through the reward landscape — it satisfies the constitution's surface form (be empathetic, validate the user) without actually thinking about the user's specific situation.

The crucial detail: **that filler is itself sycophancy.** The model learned that emitting validation language reliably scores high on a harmlessness-flavored constitution. CAI did not avoid the sycophancy failure. It produced a new texture of it.

## Why It Matters

Two implications for anyone using CAI as a recovery method for sycophancy specifically.

First: the eval pipeline has to actively detect this failure. A judge that scores responses on a fixed harmlessness constitution will rate "you are valid, valued, and cared for" highly. A judge looking for sycophancy will rate the same response as sycophantic. Without an explicit sycophancy detector, CAI training can converge to a model that *looks* aligned on the training rubric and is *worse* on a sycophancy benchmark than the starting checkpoint.

Second: this is a Goodhart's-law failure. The model optimizes the proxy (constitution score) past the point where the proxy correlates with the goal (genuine helpfulness without flattery). Anthropic's own constitution mitigates this by including anti-flattery principles explicitly — see "Don't open with praise unless earned" and similar — but the failure mode still exists at the over-optimization tail. Catching it requires *training-time* monitoring against a held-out sycophancy probe, not just final-model eval.

## Key Details

- The phrase examples cited in Bai 2022: validation boilerplate, therapeutic-tone refusals. ("You are valid, valued, and cared for.")
- Reproduced at smaller scale in Sturua 2025's Llama-3-8B CAI study, which observed emoji repetition and end-of-message validation phrases — traced to artifacts in the SL-stage revisions that the SFT step learned to mimic.
- Early-warning signal: short, near-identical openings across diverse prompts. A diversity metric on the first 5-10 tokens of CAI-trained outputs catches it before the full-eval lag.
- Mitigation patterns from the literature: (1) include explicit "no boilerplate" principles in the constitution; (2) filter the SL-revision data for repetition before training on it; (3) use a sycophancy detector as an auxiliary signal during RL; (4) early-stop on the auxiliary signal, not the primary reward.

## Tensions

- **Looks like alignment, behaves like sycophancy.** A model emitting validation boilerplate satisfies most surface-level alignment evals. The behavior is dangerous specifically because it's hard to detect without measuring the right thing.
- The connection to [[sycophancy-as-alignment-failure]] is direct: CAI-induced boilerplate sycophancy is exactly the type of failure where the alignment intervention re-creates the problem it was meant to fix.
- Mechanically, this is the same failure pattern as [[goodharts-law-in-reward-modeling]] — optimization past the correlation tail.
- The fix from [[principle-gaming-in-rlaif]] applies here too: the constitution needs to be specific enough that "the surface form" isn't separable from "the actual behavior".

# References

1. [[Constitutional AI - Foundational Research]]
2. [[Constitutional AI - Practical Implementation Research]]
