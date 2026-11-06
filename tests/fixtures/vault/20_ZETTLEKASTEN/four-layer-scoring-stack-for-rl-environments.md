Created: 20-08-2026 00:20
Status: #baby
Tags: [[reinforcement-learning]] [[reward]] [[evaluation-methodology]] [[experimental-design]]

# Four-Layer Scoring Stack for RL Environments

## What This Is

When people build an RL environment for an agentic software task, they usually write "the reward function" as one blob of code: check some things, add up some points, return a float. The claim here is that this blob is actually four different jobs glued together, and that most grader bugs live in the seams where they were glued.

Pull them apart:

1. **Verifier** — did the task get done? Binary. Answered by inspecting *final environment state*, never the agent's own description of what it did.
2. **Pass/fail check** — did the agent break a rule? Also binary, but a completely independent question from success. Data leaked, forbidden tool called, safety policy violated.
3. **Rubric** — graded quality on things that are real but not binary: step efficiency, evidence completeness, error recovery.
4. **Reward function** — the scalar. It only *composes* the three above; it originates nothing.

The ordering is causal, not stylistic. Layer 4 is downstream of Layer 1, so if the verifier is wrong no amount of weight-tuning at Layer 4 recovers it — you are tuning coefficients on a corrupted input.

## Why It Matters

The reason to keep Layer 1 and Layer 2 separate is concrete and load-bearing. An agent can complete the spreadsheet task perfectly *and* exfiltrate data to an external endpoint. If you merged the two questions into one "did it do well" check, the success test passes and the violation is invisible — it never enters the score at all. Separated, the hard check fires and zeroes the trajectory regardless of how good everything else looked.

More generally: the optimizer never sees the task. It sees a number attached to a trajectory. Every property of the learned policy is a property of how those numbers were assigned. So the decomposition of the number is the decomposition of what you are teaching.

## Key Details

- Verifier example, stated deliberately specifically: assert a row exists with `id=4521`, `status='active'`, and `updated_at` within the last 60 seconds. The timestamp window is not decoration — it closes the hole where a *pre-existing* correct row would score a pass without the agent doing anything.
- Layer 2 mechanic is a **gate, not a penalty**. Violations zero out; they do not subtract points that strong rubric performance could offset. The asymmetry is intentional: never let a policy learn that a large enough quality bonus buys a safety violation.
- Every zeroing condition should be a **named, testable check**, so that when a trajectory zeroes you know *which* rule fired.
- Layer 3 has three rules: every criterion must be readable off the trajectory/state; cap at **three to five criteria**; never bundle two conditions into one criterion.
- Layer 4 composition is **gated, not additive**: verifier fails → score zero → rubric is irrelevant.
- Non-differentiability is why this matters more in software than in continuous control. A cell holds the right value or it doesn't; there is no smooth surface for the optimizer to slide down, so the scoring system is the *entire* bridge between task completion and signal.

## Tensions

Layer 1's "programmatic wherever possible" is the environment-design restatement of [[verifiable-rule-based-rewards-prevent-reward-hacking]], and the whole stack sits inside the [[rlvr-reinforcement-learning-with-verifiable-rewards]] paradigm. Layer 2's zeroing gate is the same shape as [[probe-penalty-as-reward-shaping]] pushed to an infinite penalty, and its multiplicative character connects to [[multiplicative-reward-composition]] — the gate is literally a zero factor in a product. Layer 3's cap-and-don't-bundle rules argue against elaborate rubrics, which is in real tension with [[instance-specific-rubric-as-reward-function]] and [[generic-rubrics-fail-instance-specificity-is-the-mechanism]]: this source wants *few* criteria, those want *specific* ones — the reconcilable reading (my inference) is that specificity is what lets you get away with fewer.

The unresolved gap: the source gives no formalism for Layer 4. How rubric scores map into `[0,1]` and combine with the gate is left to the implementer, so "gated composition" is prose, not an equation.

# References
1. [[HUD - Verifier and Reward Design for RL Environments]]
