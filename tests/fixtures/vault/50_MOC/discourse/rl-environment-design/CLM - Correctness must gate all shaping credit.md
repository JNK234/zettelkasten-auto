Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #supported

# Shaping terms get exploited, so when a task has a hard correctness or constraint requirement, partial and shaping credit should be gated on it, in the reported score and ideally in the RL reward.

## Relations
informs:: [[QUE 04 - Composing reward components]], [[QUE 02 - Designing a verifier so high reward means solved]], [[QUE 25 - Weighting scores by frequency and cost of error]]
supported-by:: [[EVD - AppWorld labels no-op tests so partial credit does not pay the empty policy]], [[EVD - ART-E turn-count credit taught the agent to repeat tool calls to the limit]], [[EVD - tau2 reward is a product so one zero kills the task]], [[EVD - The reward is a strict AND over SQL checks on the final database, and the partial ve]], [[EVD - The note explains the trade-off in combining graders per task a binary AND over five]], [[EVD - The grader scale is strict any definite violation, however minor, must score 5 or lo]], [[EVD - SynthAgent's design makes the rule check a multiplicative gate on the partial-credit]]

## Scope
Any reward with partial credit or shaping, where the task has a hard correctness or constraint requirement. The gate can be the outcome itself (tau2, EnterpriseOps-Gym) or a rule check (SynthAgent). A strict AND over several graders makes tasks look harder when any one grader is flaky (Anthropic).

## Related zettels
- [[tiered-reward-with-bounded-within-tier-shaping]]
- [[rewarding-turn-count-teaches-padding]]
- [[gated-and-separately-normalized-composite-rewards]]
