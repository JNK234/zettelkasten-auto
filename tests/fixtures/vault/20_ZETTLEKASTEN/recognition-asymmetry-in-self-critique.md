
Created: 12-05-2026 21:10

Status: #baby

Tags: [[constitutional-ai]] [[sycophancy]] [[evaluation-methodology]] [[ai-alignment]]

# Recognition Asymmetry in Self-Critique

## What This Is

Self-critique works only where the model has a ground-truth anchor to check against. On a sycophantic Qwen3-8B model asked to evaluate its own responses against a 7-principle anti-sycophancy constitution, the recognition rate splits cleanly by failure type. On **factual** prompts — where the user states something false and the model agreed — the model self-flagged 32% of its sycophantic responses, and the 72B judge confirmed a 44% reduction in sycophancy after self-revisions. On **subjective feedback** prompts — where the user shared writing and the model praised it without basis — the model self-flagged 60% of its responses, but most of those flags were hallucinated, and the revisions made sycophancy 8% *worse*.

Per-principle the pattern is bimodal, not continuous:

- 85% self-flag on `diplomatic_honesty`
- 83% on `knowledge_over_preference`
- 71% on `no_flattery`
- 10% on `no_epistemic_cowardice`
- 0% on `position_consistency`
- 0% on `helpfulness_floor`

Three principles the model can see in itself. Three it cannot. The cliff between them maps to whether the principle has a factual anchor.

## Why It Matters

This is why "Constitutional AI" at 8B can't actually be self-critique. The model only catches half its own sycophancy, and the half it catches is the easier half. The half it misses is the part where there's no fact to check against — opinion sycophancy on writing, math solutions, arguments. Those failure modes need an external critic with a real ground-truth check.

This finding generalizes beyond sycophancy. Any failure mode that requires *value judgment without a factual referent* will have low self-recognition. Failure modes with crisp ground-truth anchors will self-correct. The implication for self-supervised alignment is uncomfortable: the things you most want to fix (subtle epistemic posture, conversational integrity, taste) may be exactly the things models can't see in themselves.

## Key Details

- The pretest that established this: 50 prompts (25 answer, 25 feedback), Qwen3-8B SFT'd to be sycophantic, asked to critique itself against a sampled principle, then revise. 72B judge scored both `r_init` and `r_revised`.
- The 85% / 0% gap is at the principle level, not the prompt level. Some principles are easier for the model to spot violations of than others, regardless of which prompt the violation appears on.
- The hallucination rate matters: 60% of feedback prompts self-flagged, only 12% were judge-confirmed sycophantic. So 4 out of 5 self-flags on feedback were false positives. Forced to "fix" hallucinated violations, the revisions drifted off-task.
- Why the factual anchor matters mechanically: when the model evaluates "did I agree with a false claim?", it can check the claim against parametric knowledge. When it evaluates "is this writing good?", there's no internal check — the model's "judgment" is a confabulated label, not a measurement.

## Tensions

- **Direct contradiction with the CAI assumption.** Bai 2022 assumes a model's evaluative head is stronger than its generative head on the dimension being aligned. This is true for harmlessness on a 52B model. It is half-true for sycophancy on an 8B model.
- The asymmetry forces [[constitution-guided-distillation-vs-self-critique-cai]] in our pipeline — we use a 72B critic for the failure modes where 8B can't self-critique.
- Related: [[small-model-cai-collapse]] is the downstream consequence — if you ignore the recognition asymmetry and run self-critique anyway, the noisy self-evaluations train a worse model.
- Open question: does recognition asymmetry scale predictably with model size? If 8B is at 50% and 52B is at "good enough", where exactly is the threshold for opinion sycophancy specifically?

# References

1. [[Constitutional AI for Sycophancy - Literature Survey]]
2. [[Constitutional AI - Foundational Research]]
