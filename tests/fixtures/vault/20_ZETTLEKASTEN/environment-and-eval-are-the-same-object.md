Created: 20-08-2026 00:20
Status: #baby
Tags: [[evaluation-methodology]] [[benchmark-design]] [[reinforcement-learning-in-language-models]] [[experimental-design]]

# Environment and Eval Are the Same Object

## What This Is
Training and evaluation feel like two different activities, so we build two different kinds of artifact for them: benchmarks on one side, RL environments on the other. Different repos, different teams, different words.

Strip both down to what they actually contain and the distinction vanishes. A benchmark is a task distribution plus a scoring function. An RL environment is a task distribution, a rollout protocol, a verifier, and execution infrastructure. The scoring function *is* the verifier. Once you have (tasks, scorer), you have a choice of what to do with the resulting numbers:

- Feed them back as reward and update the weights → it's a training environment.
- Freeze the model, report the mean → it's an eval.

Nothing about the artifact changes. Only the direction the gradient does or doesn't flow. Prime Intellect's Environments Hub makes this explicit by shipping one object type and letting you press either button on it.

## Why It Matters
Two consequences, one good and one dangerous.

The good one is a contribution flywheel. Publishing an environment normally has weak incentives — you did the labor, someone else trains a better model with it. But if the same artifact is also a public leaderboard, publishing gets you a durable, citable measurement artifact whose value exists independently of anyone training on it. Prime Intellect uses this deliberately: "create and explore evaluation reports for environments for various models" is the reward for contributing.

The dangerous one is that **contamination becomes the default state rather than an accident**. If the training environment and the eval environment are literally the same code, and a flagship model (INTELLECT-3) trains on hub environments and is then evaluated on hub environments, the reported numbers mean much less than they appear to. This isn't leakage in the classic sense — nothing was copied into pretraining data — it's that the measuring instrument and the training signal are one physical object. The launch post describes no train/test protocol at all (a limitation of the post, noted in the source).

## Key Details
- The unification is architectural, not incidental: any object exposing `generate a rollout given a model` + `score that rollout` supports both uses.
- In `verifiers`, scoring is factored into a **parser** (extract the answer from raw text) and a **rubric** (a weighted list of reward functions over the parsed result), e.g. `0.8 * correctness + 0.2 * format_adherence`. That same composite score is what a leaderboard would report.
- The training/eval distinction is therefore convention about *how you use the number*, not a property of the artifact.
- Defense, not described in the source (inference): hubs of this kind need a held-out designation at the artifact level — environments flagged eval-only and excluded from training manifests — because you cannot recover separation after the fact.

## Tensions
This is the general form of the specific pattern in [[benchmarks-repurposed-as-training-environments]] — there, benchmarks were retrofitted as training environments; here the two are designed as one object from the start, which makes the problem structural rather than opportunistic. The mitigation direction is [[dev-unseen-eval-split-for-credible-recipe-iteration]], and the failure it causes resembles [[benchmark-contamination-risk-in-public-datasets]] even though no data leaked, so [[rephrasing-defeats-ngram-decontamination]]-style defenses are useless. The parser/rubric factoring connects to [[four-layer-scoring-stack-for-rl-environments]] and to [[instance-specific-rubric-as-reward-function]]. If the shared object is also a public leaderboard, [[reward-hacking-as-eval-false-positive]] and [[reward-hacking-as-specification-gaming]] become simultaneously training bugs and measurement bugs. See also [[train-test-asymmetry-by-design]] for when asymmetry is deliberate rather than absent.

# References
1. [[Prime Intellect - Environments Hub]]
