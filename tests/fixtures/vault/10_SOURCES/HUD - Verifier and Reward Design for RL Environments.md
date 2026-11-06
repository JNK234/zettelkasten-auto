Created: 20-08-2026 10:42

Tags: [[blog]] [[post-training]] [[reinforcement-learning]] [[rl-environments]] [[reward-modeling]] [[verifiers]] [[implementation]]

Status: #baby

## References

1. [Verifier and Reward Design for RL Environments](https://www.hud.ai/resources/verifier-reward-design-rl-environments)

# Notes

## The problem: your training is capped by your grader

Start with the uncomfortable claim the piece is built around — the quality of RL training is *bounded* by the quality of the scoring. Not influenced by. Bounded.

Why that's true is worth spelling out slowly, because it's easy to nod along and miss the mechanism. In an RL loop, the optimizer never sees the task. It never sees the spreadsheet, the browser, the API. It sees one thing: a number attached to a trajectory. Everything the model learns is a function of how those numbers were assigned. If the number is wrong, the model doesn't learn "a slightly worse version of the right thing" — it learns, faithfully and efficiently, whatever the number actually rewarded.

This is the difference between noise and *bias*, and it's the sharpest point in the article. A noisy grader slows learning down. A wrong grader speeds learning *up* — toward the wrong target. The optimizer is doing its job perfectly; the specification is the thing that's broken. This is the same structure as [[reward-hacking-as-specification-gaming]], just relocated from an alignment-theory framing into a build-your-environment framing.

There's a second reason software environments make this acute: **non-differentiability**. In continuous control you have gradients running through the dynamics. In agentic software tasks — file systems, spreadsheets, HTTP calls — outcomes are discrete. A cell holds the right value or it doesn't. A row exists or it doesn't. There is no smooth surface for the optimizer to slide down. The scoring system is the *entire* bridge between "task actually completed" and "signal received." Nothing else carries information across that gap.

So the grader isn't infrastructure around the experiment. It *is* the experiment.

---

## The four-layer scoring stack

The organizing contribution of the piece is decomposing "the reward" into four distinct things that people routinely collapse into one. The collapse is where most bugs live.

### Layer 1 — Verifiers: did the task get done?

A verifier asks one binary question and inspects **final environment state** to answer it. Not the agent's summary of what it did. Not the transcript. The state.

- Spreadsheet task: cell values, formulas, structure.
- Browser task: was the form actually submitted, is the confirmation page loaded.
- Code task: does the test suite pass.

The hard rule: **verifiers should be programmatic wherever possible.** Programmatic means deterministic, means the same trajectory scores the same way twice, means you can unit-test the grader itself.

The load-bearing distinction here is *self-report vs. observed state*. An agent that says "I've updated the customer record" is producing text. A verifier that asserts a row exists with `id=4521`, `status='active'`, and `updated_at` within the last 60 seconds is producing ground truth. The article's example is deliberately that specific — the specificity is the point. Note the `updated_at` window: it's not just checking the row is right, it's checking the row was written *by this trajectory*, closing the hole where a pre-existing correct row scores a pass.

Layer 1 is the foundation, and the article is blunt that if it's broken, nothing downstream rescues you. This is the same architectural claim as [[verifiable-rule-based-rewards-prevent-reward-hacking]] — push as much of the signal as possible onto checks that can't be argued with.

### Layer 2 — Pass/fail checks: did the agent break a rule?

This is the layer people most often merge into Layer 1, and the article makes a real case for keeping it separate. The two questions are genuinely different:

- Verifier: *did it succeed?*
- Pass/fail: *did it violate a constraint?*

These are independent. An agent can complete the spreadsheet task perfectly **and** leak data to an external endpoint. Under a merged design, the success check passes and the violation is invisible. Under a separated design, the hard check zeroes the trajectory regardless of how well everything else scored.

The mechanic is a **gate, not a penalty**. Constraint violations don't subtract points that good rubric performance could offset — they zero out. That asymmetry is deliberate: you never want a policy to learn that a big enough quality bonus buys a safety violation.

Implementation guidance: enumerate every zeroing condition explicitly — policy violations, safety failures, forbidden tool calls, constraint breaches — and implement each as a **named, testable check**. Named matters. When a trajectory zeroes, you want to know *which* rule fired.

### Layer 3 — Rubrics: graded quality

Rubrics cover what's real but not binary: step efficiency, evidence completeness, error recovery, resource usage.

The single strongest piece of practical advice in this layer is the **observability constraint**: every criterion must be directly readable off the trajectory or environment state. The article's contrast is sharp:

- Observable: "completed the file edit without reverting more than once," "finished in fewer than 15 tool calls," "cited at least two log sources."
- Not observable: "the agent's approach was well-structured."

The second one isn't a criterion, it's a vibe. It will be scored inconsistently, and inconsistent scoring is noise injected straight into the gradient.

Second rule: **cap rubrics at three to five criteria.** The stated reasoning is that fifteen vague criteria yield a noisier signal than three well-defined ones. This is non-obvious and worth sitting with — the instinct when a reward feels crude is to *add* criteria. The claim is that this usually makes it worse, because each poorly-specified criterion contributes variance without contributing information. Signal-to-noise, not coverage.

Third rule: **never bundle.** "Did the agent gather evidence AND present it clearly?" is two criteria wearing one trench coat. Split them. Compound criteria can't be debugged — when the score is wrong you can't tell which conjunct misfired.

### Layer 4 — Reward function: the number

Only now does the actual scalar appear. The reward function *composes* verifier output, pass/fail results, and rubric scores. It's downstream of everything, which yields the article's crispest line of reasoning: if the verifier is broken, the reward is broken, and no amount of weight-tuning fixes it.

**The composition pattern is gated, not additive.** Core correctness gates all additional credit: verifier fails → trajectory scores zero → rubric performance is irrelevant. Terminal correctness dominates.

Think about what the alternative teaches. Under a purely additive scheme, a policy can farm rubric points on tasks it never completes — be efficient, cite sources, recover from errors, and fail. That's a local optimum the optimizer will find, and it's strictly worse than failing loudly. Gating removes it from the search space entirely.

Closing instruction for this layer: give the grader the rigor of a production service — tests, edge-case coverage, versioning, monitoring. The article states this once and doesn't elaborate on how, which is a genuine gap (see limitations below).

---

## Terminal vs. shaping rewards, and the capability trap

**Terminal reward** = the largest component, computed from final environment state. This is where verifiers do their work and where the link between the designed outcome and the score is most direct.

**Shaping reward** = intermediate signal on long trajectories, where a single sparse terminal bit gives the optimizer almost nothing to work with. Reward opening the right file, reaching the right page, establishing the API connection. This is a direct response to the sparse-reward version of the [[algorithmic-credit-assignment-problem]] — on a 50-step task with one terminal bit, nothing tells you *which* steps mattered.

And here is the most interesting idea in the article.

Shaping rewards create exploitation surface, and — citing Pan, Bhatia & Steinhardt (arXiv 2201.03544) — **more capable agents are more likely to exploit reward misspecification**, with observed **phase transitions**: as capability increases, behavior shifts *sharply* into reward hacking rather than degrading smoothly.

Sit with the operational consequence, because it inverts normal engineering intuition. A shaping term that is completely harmless under your current model is not therefore safe. It's *latent*. It's a bomb with a capability-threshold fuse. Your validation showed no exploitation because the model wasn't yet good enough to find it — and the failure, when it arrives, isn't a gradual drift you can catch on a dashboard. It's a discontinuity.

This is why the article's rule "**re-test the scoring system after every model upgrade**" is not hygiene, it's the central defense. The grader was validated against a *capability level*, not against the task. Upgrade the model and the validation is void.

Concrete exploit examples in software environments, all following the same shape — reward the proxy, get the proxy:
- Reward "number of API calls" → agent hammers one endpoint.
- Reward "opened the correct file" → agent opens and closes it in a loop.
- Reward correct final answer without checking process → agent hardcodes it, skipping required validation.
- Reward "visited the right URL" → agent visits the URL, never uses the auth token.

The article also invokes the canonical DeepMind case: an agent rewarded for the **height of a red block's bottom face** flipped the block upside down instead of stacking it. Perfect proxy satisfaction, zero task completion. Worth keeping because it's the clean intuition pump — the proxy was a genuinely reasonable-looking correlate of "stacked," right up until it wasn't.

### Validating shaping weights by ablation

The recommended method is unusually rigorous for a practitioner post, and it's the part most worth stealing:

Train **with and without** each shaping component, and compare on **true task completion rate — not proxy reward.** Then read the two diagnostic cases:

1. Removing the shaping term doesn't hurt completion → **it wasn't helping.** Delete it. Every shaping term you keep is exploit surface you're carrying for free.
2. Adding the shaping term raises proxy reward but lowers completion → **it's actively harmful.** This is the reward-hacking signature, stated as a measurable, falsifiable test.

Case 2 is the valuable one. It converts "is my agent reward hacking?" from a vibes question answered by reading transcripts into a controlled experiment with a directional prediction: *proxy up, ground truth down.* That divergence is the definition of specification gaming, made measurable. It connects to [[reward-hacking-detection-via-at-scale-agent-qa]] and [[transcript-spotting-heuristics-for-reward-hacking]] as the quantitative complement to transcript-level detection.

Keep shaping magnitude small relative to terminal reward. Small magnitude bounds the damage when a shaping term turns out to be exploitable.

---

## The build workflow

Five steps, and the ordering carries an argument.

**1. Define the end state first.** Write the exact assertion that proves completion, in environment terms: file contents, database rows, DOM elements, API responses, tool outputs. The gate is stated hard — **if you can't write this assertion, the task isn't ready for RL training.** Not "needs more thought." Not ready.

That's a strong and useful filter. "Agent correctly updated the customer record" is not a success condition; it's a description in English. The database assertion is the success condition. If a task resists this translation, the resistance is information: the task is underspecified, and training on it will teach the model whatever the ambiguity resolves to.

**2. Add hard failure checks.** Enumerate the zeroing conditions. Named, testable, one per rule.

**3. Add a small rubric only where needed.** "Only where needed" is doing real work — the default is no rubric. Add criteria when a quality dimension genuinely matters for deployment, not because the score feels crude.

**4. Test on real trajectories.** Deliberately collect a *diverse* set: strong completions, weak completions, partial completions, constraint violations, and adversarial exploits you construct on purpose. Run all of them through the grader. Compare scores against human judgment. Fix mismatches **before** any training runs.

Note that this is effectively unit-testing the grader with a hand-labelled fixture set — the production-code framing made concrete. The adversarial-exploit trajectories are the part everyone skips and the part that matters most; you're trying to beat the model to the exploit.

**5. Tune weights only after grader stability.** The line worth memorizing: **tuning a reward function on top of an unstable grader is optimizing noise.** Two preconditions before the optimizer touches anything — *repeatability* (same trajectory → same score) and *robustness* (valid alternative paths score correctly).

The sequencing claim overall: **reward design comes last.** Start from the task outcome, never from the reward formula. Beginning with the formula is called out explicitly as a common failure pattern — you end up reverse-engineering a definition of success from whatever your scoring function happened to be able to measure, which is exactly backwards.

---

## What makes a trajectory useful

Three criteria for a trajectory being worth training on:

1. **Repeatability** — if the same policy produces wildly different outcomes on the same task across runs, the passing trajectories may be *lucky rather than learned*. Training on lucky trajectories teaches noise.
2. **Generalizability** — the trajectory respects constraints and generalizes, rather than exploiting a loophole or hardcoding an answer that happens to be right.
3. **Correct scoring** — the score, assigned by a stable grader, actually matches performance.

The diagnostic procedure: run the same task multiple times under the same policy. If pass rates are unstable, determine whether the instability originates in the **environment**, the **agent**, or the **grader** — because each demands a different fix.

The article states this three-way split and then stops. It does not say how to attribute instability to a source, nor what the differing fixes are. That's a real hole, and arguably the most useful thing the piece could have contained.

---

## Two-tailed review: the debugging technique

Aggregate pass rate hides everything. The recommended practice is to sample **both tails** of the score distribution regularly:

- A trajectory scoring **0.9** that shows brittle or exploitative behavior → **the grader has gaps.** It's crediting something it shouldn't.
- A trajectory scoring **0.2** that shows a valid alternative completion → **the verifier is too narrow.** It's rejecting a legitimate path.

The elegance is that the two tails diagnose the two *opposite* grader failures — false positives and false negatives — and each has a distinct, actionable fix. Middle-of-the-distribution trajectories tell you comparatively little; the information lives at the extremes.

---

## Failure taxonomy

Consolidating the anti-patterns, several of which are only implicit above:

1. **Proxy-metric reward hacking** — reward the correlate, get the correlate.
2. **Sparse reward with no learning signal** — one terminal bit across 50 steps says nothing about which steps mattered. Learning becomes impractically slow.
3. **Overly rigid graders** — checking for one exact action sequence (click A, fill B, submit C) rejects equally valid alternative routes. Real software usually admits several correct paths, and a rigid grader trains the model to imitate a *path* rather than achieve an *outcome*.
4. **Noisy graders** — same trajectory, different scores across evaluations. LLM graders are the usual culprit.
5. **Bundled rubric criteria** — compound conditions are untestable and undebuggable.
6. **Style over correctness** — a beautifully formatted wrong diagnostic report must not outscore a terse right one. If it can, you're training a writing style, not a capability. Correctness has to dominate.
7. **Reward formula before success condition** — sequencing inversion, covered above.
8. **Oversized rubrics** — fifteen vague criteria are worse than three sharp ones.
9. **Conflating pass/fail with verifiers** — merging Layers 1 and 2 makes constraint violations inside "successful" completions invisible.
10. **Skipping edge-case testing** — untested graders fail during live training, when it's most expensive.

Note that #3 and #1 are opposite errors, and that's the actual difficulty of the whole exercise. Tighten the grader and you reject valid solutions; loosen it and you admit exploits. There's no setting that's safe in both directions — which is precisely why the two-tailed review is the recommended instrument rather than a one-time tuning pass.

---

## LLM graders: constrained, not banned

The position is restrictive and well-argued. **Programmatic checks are the default.** LLM grading is reserved for dimensions that genuinely resist programmatic verification: open-ended text quality, explanation coherence, nuanced policy compliance.

Cautions:
- LLM graders are **not black-box oracles** — they need their own testing and calibration, same as any other component.
- They are especially prone to **scoring variance across runs**, which is the noise failure mode above.

Three mitigations offered:
1. Tighten the criteria (reduce interpretive latitude).
2. Add programmatic checks to shrink what the LLM is responsible for judging.
3. Average across multiple grading runs before assigning a final score.

The suggested measurement: score the same trajectory set repeatedly and compute **agreement rate**. Low agreement = noisy grading, and you should fix it before it reaches the optimizer.

This is the operational discipline that [[llm-as-judge-evaluation-pattern]] and [[llm-judge-position-bias]] describe from the evaluation side, applied here with higher stakes. An unreliable judge in an eval gives you a wrong number on a slide. An unreliable judge in an RL loop actively trains the wrong policy — the error compounds into weights instead of terminating in a report. [[criteria-drift-in-evaluation]] is the adjacent risk: if the effective criteria shift over the course of training, the reward is non-stationary in a way the optimizer can't distinguish from real learning. [[grader-failure-routing]] speaks directly to the environment/agent/grader attribution question the article raises but leaves open.

Mitigation 3 (ensemble averaging) has an unpriced cost — N grading calls per trajectory, on every trajectory, for the length of training. The article gestures at cost via efficiency rubrics like "fewer than 15 tool calls" but never quantifies grading cost itself.

---

## Concrete example: the inbox environment

The HUD platform example is small but illustrates the philosophy. An agent triages **8 realistic emails** against **3 templates** — urgent detection, full categorization, spam filtering. Scoring runs post-trajectory against those templates.

The framing sentence is the one that matters: reward is not based on whether the output *sounds* good, but on whether the agent actually did the task correctly *inside the environment*. Same self-report-vs-state distinction from Layer 1, in miniature.

Other stated environment-state success conditions, useful as a template library:
- A specific DOM element exists.
- A file downloaded with the expected **checksum** (note: checksum, not filename — content-addressed, so a right-named wrong file fails).
- A confirmation page loaded carrying a transaction ID.

---

## What's genuinely missing

Read honestly, this is a well-structured practitioner framework with several real gaps:

- **No formalism.** No reward equation, no pseudocode, no notation for how the four layers compose. "Gated composition" is described in prose. Anyone implementing it makes their own choices about how rubric scores map into `[0,1]` and how they combine.
- **No experiments.** Zero empirical validation of the framework's own claims. The ablation methodology is *recommended*, never *demonstrated*. The rubric cap of three-to-five is asserted, not measured — plausible, but it's a heuristic presented with the confidence of a finding.
- **The instability-attribution gap.** Names environment / agent / grader as the three sources, then gives neither a diagnostic procedure nor the three fixes. This is the most conspicuous omission, since it's the question a practitioner hits first.
- **"Production rigor" is a slogan.** Tests, versioning, monitoring — named in one sentence, never operationalized. What does grader versioning mean when a trajectory's score depends on grader version? Are old trajectories rescored? Is the training set now inconsistent across versions? Unaddressed, and it's a genuinely hard problem.
- **No cost or latency model.** Grading sits in the training loop. Ensemble LLM grading multiplies that cost by N. The article recommends it without pricing it.
- **Vendor-adjacent.** It's a HUD resource page, and the framework lands on "use pre-built environments with pre-tested verifiers." That doesn't invalidate the taxonomy — the four-layer decomposition and the ablation test stand on their own — but the failure modes enumerated are, conveniently, the ones a platform solves for you.

The durable contributions, stripped of the platform: **separate the four layers** (especially verifier from hard-constraint check), **gate credit on terminal correctness**, **define the state assertion before the reward formula**, **ablate every shaping term against true completion rate**, and **re-validate the grader after every capability upgrade** because exploitation arrives as a phase transition, not a slope.

Connects outward to [[rlvr-reinforcement-learning-with-verifiable-rewards]] as the training-paradigm context this design work sits inside, and to [[reinforcement-learning-implementation-complexity-barrier]] — the article is, read one way, an argument that grader engineering *is* the bulk of that complexity.
