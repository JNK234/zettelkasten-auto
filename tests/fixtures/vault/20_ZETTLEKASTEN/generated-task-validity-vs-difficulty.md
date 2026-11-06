Created: 24-09-2026 16:29

Status: #baby

Tags: [[rl-environment-design]] [[benchmark-design]] [[error-propagation]]

# Generated Task Validity vs Difficulty

## What This Is
When an LLM generates a task, two different things can make it "hard" for the agent: the task is genuinely difficult, or the task is **broken** (unsolvable with the generated tools, contradictory spec, wrong checker). From the agent's reward alone, both look the same: $r < \eta$.

Generative Simulators' curriculum filter $\mathcal{C}$ judges whether a task is the right difficulty for the current policy. Nothing in Algorithm 1 checks whether the task can be solved at all (per Introducing §9).

## Why It Matters
In a closed loop, broken tasks do double damage. The unsolvable task gives a zero reward (noise in the gradient), and it lands in failure memory $B$ labeled as an agent weakness. The oversight LLM then reads $B$, diagnoses a "bottleneck" that is really a generator bug, and rewrites the directive to push on it. The generator's errors steer the curriculum.

Hand-built benchmarks solve this with an oracle: a reference solution that proves every task is solvable. Generated environments need the same guarantee, generated or checked automatically.

## Key Details
- Validity checks the paper lacks (inference): generate or search for an oracle solution per task and confirm the generated checker passes it; confirm required tools are in $O_k$; reject tasks where a strong reference model and the oracle both fail.
- Related gap: nothing validates that generated tasks reflect real-world difficulty rather than "generator-imagined" difficulty (per Generative Simulators §8).
- Cost: rejected tasks still cost generation calls, so the effective batch is below K.

## Tensions
- The hand-built answer: [[oracle-solutions-for-task-solvability]], [[writable-state-assertion-as-task-readiness-gate]].
- Misattributing environment faults to the agent: [[mask-infrastructure-failures-never-reward-them-zero]] is the same pattern at the infra level.
- Label noise at the top of the difficulty range: [[label-noise-dominates-saturated-benchmarks]].
- Corrupted diagnosis feeds the controller: [[reflective-curriculum-control-loop]].
- Generated benchmarks matching real distributions: [[distribution-match-in-generated-benchmarks]].

# References
1. [[Patronus AI - Introducing Generative Simulators]] (§4, §9)
2. [[Patronus AI - Generative Simulators]] (§8)
