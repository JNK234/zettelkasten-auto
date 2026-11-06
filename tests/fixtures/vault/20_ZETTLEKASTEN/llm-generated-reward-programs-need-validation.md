Created: 24-09-2026 16:29

Status: #baby

Tags: [[rl-environment-design]] [[reward]] [[ai-alignment]]

# LLM-Generated Reward Programs Need Validation

## What This Is
In Generative Simulators the reward generator has type $\mathcal{R}: \mathcal{M} \times \mathcal{D} \to (\mathcal{H} \to \mathbb{R})$. It does not score trajectories; it **writes the scoring code** for each task, at task-generation time, before the agent acts. The checker then runs deterministically on the trajectory. That is the "co-generated but verifiable" claim in type form.

This differs from LLM-as-judge: the LLM is involved once, when the checker is written, not every time a trajectory is scored.

## Why It Matters
It is the step that turns synthetic tasks into synthetic environments, and it is the weakest link. A human-written verifier is already hard to get right (see the Patronus tutorial's own reward, which scores every ordering 4). Now an LLM writes thousands of them, and an RL policy optimizes against all of them. That is exactly where reward hacking shows up.

This is the question to probe with Patronus: how do you know an LLM-authored grader is correct and hack-resistant before a policy trains on it?

## Key Details
How a generated checker can be wrong (per sources §7/§9):
- **Too loose**: hackable, false positives.
- **Too strict**: false negatives, correct solutions scored 0.
- **Silently changed by the directive**: "make rewards sparser" can alter correctness, not just difficulty.
- Nothing in Algorithm 1 validates $r_m$ before use.

Self-reinforcing failure: if the generator drifts toward tasks whose checkers are easy to satisfy, mean reward rises, the directive says "increase difficulty," and nominal difficulty rises while real difficulty does not. Held-out fixed evals are needed as an anchor; the paper doesn't mention them.

Inference, candidate validations: run the checker against an oracle solution (must pass) and against mutated/adversarial trajectories (must fail); cross-check with an independent judge on a sample; track reward-vs-held-out-eval divergence.

Tension inside Patronus: the paper says reward should be verifiable; the blog says the future needs "semi-verifiable and non-verifiable" rewards. Near-term product verifiable, research direction judge-based.

## Tensions
- One human-written reward already fails: [[unit-test-the-reward-function-before-training]].
- The missing layered checks: [[four-layer-scoring-stack-for-rl-environments]].
- Per-task generated rewards elsewhere: [[auto-generated-rule-based-reward]], [[instance-specific-rubric-as-reward-function]], [[evolving-reward-criteria-against-a-moving-policy]].
- Failure taxonomy: [[reward-hacking-as-specification-gaming]], [[llm-verifier-exploitation-taxonomy]], [[reward-model-overoptimization-hump]], [[verifier-failure-vs-rubric-design-failure]].
- False-negative rescue: [[asymmetric-judge-rescue-of-rule-based-false-negatives]].
- Anchor: [[training-reward-is-not-a-model-selection-signal]].

# References
1. [[Patronus AI - Generative Simulators]] (§3, §7, §8)
2. [[Patronus AI - Introducing Generative Simulators]] (§3, §6, §8, §9)
