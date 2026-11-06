Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #supported

# The first requirement of a verifier is that high reward means the task was actually solved, not hacked.

## Relations
informs:: [[QUE 02 - Designing a verifier so high reward means solved]]
supported-by:: [[EVD - Practitioners name hack-robustness the most important environment property]], [[EVD - A hub verifier returned reward 1.0 for the empty string]], [[EVD - Agents checked out future commits to find existing fixes]], [[EVD - The note's author reads the tau-bench result directly as a reward-hacking surface a]], [[EVD - The post asks for hack-resistant graders in a single sentence and goes no further]]

## Scope
Any environment used for RL, where the policy actively searches for reward.

## Related zettels
- [[reward-hacking-as-specification-gaming]]
