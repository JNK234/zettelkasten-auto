Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #supported

# Deterministic graders fail silently when what they check drifts from the spec, punishing good behaviour or passing bad behaviour.

## Relations
informs:: [[QUE 03 - Grade end state, trajectory or text]], [[QUE 02 - Designing a verifier so high reward means solved]], [[QUE 18 - Acceptance tests before training]]
supported-by:: [[EVD - tau2 hash scope zeroed agents that double-checked their work]], [[EVD - PatTut reward scored all 24 tool orderings as 4]], [[EVD - The verifier engine skips any check that names an unknown server, and the task score]], [[EVD - KernelBench's fuzzer varied only tensor values, and some functions barely change und]], [[EVD - The KernelBench re-grading judged kernels against a broader contract (input validati]], [[EVD - A Terminal-Bench audit found a task whose tests assumed a file path the spec never s]]
refines:: [[CLM - Grade the observed end state rather than the trajectory]]

## Scope
Any hash, diff or programmatic reward. Found only by testing or auditing the grader.

## Related zettels
- [[hash-verifier-scope-creates-hidden-requirements]]
- [[unit-test-the-reward-function-before-training]]
