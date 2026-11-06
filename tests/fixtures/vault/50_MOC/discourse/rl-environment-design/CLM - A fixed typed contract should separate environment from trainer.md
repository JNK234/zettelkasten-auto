Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #supported

# The environment should sit behind a fixed, typed contract so any trainer can run it.

## Relations
informs:: [[QUE 08 - Interfaces and what the agent must not reach]], [[QUE 35 - Environment costs and what is scarce]], [[QUE 02 - Designing a verifier so high reward means solved]]
supported-by:: [[EVD - OpenEnv treats the environment as a network service with a fixed contract]], [[EVD - Verifiers makes an environment a separate installable artifact]], [[EVD - Survey says non-standard interfaces make results irreproducible]], [[EVD - The generated check functions run with full Python builtins inside the trainer's own]]

## Scope
Framework and hub design; transport (HTTP vs in-process) chosen by workload. The extension to generated grader code, which should run outside the trainer's process, rests on one code reading (EnvScaler).

## Related zettels
- [[environment-and-eval-are-the-same-object]]
