Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #supported

# Before training, a known-correct solution must score max and doing nothing must score the floor.

## Relations
informs:: [[QUE 18 - Acceptance tests before training]], [[QUE 02 - Designing a verifier so high reward means solved]]
supported-by:: [[EVD - A hub verifier returned reward 1.0 for the empty string]], [[EVD - PatTut reward scored all 24 tool orderings as 4]], [[EVD - AppWorld labels no-op tests so partial credit does not pay the empty policy]], [[EVD - Verifiers SandboxDebugEnv applies the gold patch before any RL]], [[EVD - EnvScaler's reward code is LLM-written check functions that are never run against a]], [[EVD - The note argues that three cheap checks, a do-nothing agent, an oracle solver and a]], [[EVD - The post requires a reference solution for every task that passes all graders, provi]]

## Scope
Any environment with a programmatic grader. The support is practitioner guidance plus retrospective inference, with no measured detection rate. An oracle ceiling below the maximum (for example tau-tau-Bench at 82.2 percent) needs a per-task explanation.

## Related zettels
- [[oracle-floor-margin-contract-for-environment-validation]]
- [[no-op-labeled-tests-for-dense-reward]]
