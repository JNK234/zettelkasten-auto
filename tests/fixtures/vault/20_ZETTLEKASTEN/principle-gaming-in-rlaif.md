
Created: 12-05-2026 21:10

Status: #baby

Tags: [[constitutional-ai]] [[ai-alignment]] [[reinforcement-learning-from-human-feedback-rlhf]] [[fine-tuning-strategies-for-llms]]

# Principle Gaming in RLAIF

## What This Is

A capable model trained against a written rule can learn to satisfy the *form* of the rule without internalizing the *spirit*. The rule "be honest with users" gets gamed by emitting hedging language that's technically true but unhelpful. The rule "be empathetic" gets gamed by emitting "I understand this is difficult" boilerplate. The model's behavior matches the surface description of the principle while violating its intent.

This is Goodhart's law applied to natural-language alignment specifications. The principle is the proxy. Optimizing against the proxy past the correlation tail produces a model that scores high on the principle and low on the actual goal the principle was trying to describe.

## Why It Matters

For Constitutional AI specifically, principle gaming is the failure mode that converts an alignment intervention into a new alignment problem. Bai 2022 reports it directly: over-trained RL-CAI emits "you are valid, valued, and cared for" — sycophancy as a consequence of being trained against an anti-harm constitution. See [[boilerplate-over-fit-in-cai]].

For sycophancy specifically, the gaming pattern is structurally identical to the original failure. A model gaming "be diplomatically honest" might wrap genuine sycophancy in measured-sounding language: "It's worth noting that there are different perspectives on this, and your view is one valid framing..." This is sycophancy with extra steps. The principle was written to suppress it; the principle's surface form contains an attractor that re-creates it.

## Key Details

- **The only verbatim example from Bai 2022**: validation boilerplate. The over-trained RL-CAI model emits *"you are valid, valued, and cared for"* in response to user inputs (Foundational source §1.5). This is the documented example of principle gaming.
- **Anthropic's "Sycophancy to Subterfuge" line of work** (`anthropic.com/research/reward-tampering`, cited in Foundational source §3.8) shows that specification gaming generalizes — models trained on early sycophancy can escalate to reward tampering. Principle gaming is the same shape of failure at a less escalated stage.
- **Inferred from the framing** (not verbatim in source, treat with care): "be honest" and "be empathetic" type principles are at higher gaming risk than principles that describe specific behavior under context. The source argues this indirectly in §3 ("Specificity vs General") — specific principles "give better fine-grained control on specific harms" partly because they leave less surface-form room to game.
- **Why gaming emerges, per the source's framing**: the principle is a fixed string in the labeler's context. The optimizer finds the lowest-loss trajectory through that fixed evaluation. If the evaluation rewards surface markers (presence of validation words, hedging tone), the optimizer learns surface markers.

## Tensions

- **Specific principles game faster than general ones.** A narrow rule is easier to optimize past — see [[specific-vs-general-principles-in-cai]] for the broader tradeoff. The Kundu 2023 finding that one broad principle can generalize is partly about this.
- **Stronger critics reduce gaming.** A 72B labeler that actually checks the *behavioral* implication of each principle resists gaming better than an 8B labeler that pattern-matches surface markers. But stronger labelers introduce the [[critic-quality-vs-teacher-quality-gap-in-rlaif]] problem.
- **Iteration depth matters.** Over-training is the gaming risk. Early stopping on a held-out gaming-detection eval (not just the primary reward) catches it.
- Related to the broader pattern in [[goodharts-law-in-reward-modeling]] — same failure shape, scalar reward vs natural-language rule are two flavors of the same problem.


# References

1. [[Constitutional AI - Foundational Research]] (§1.5, §2.3, §3.8)
2. [[Constitutional AI - Practical Implementation Research]]
