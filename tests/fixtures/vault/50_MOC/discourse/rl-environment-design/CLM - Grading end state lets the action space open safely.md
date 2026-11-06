Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #supported

# If grading checks only end state, the action space can be opened without making the grader less trustworthy.

## Relations
informs:: [[QUE 05 - Designing the action space]], [[QUE 03 - Grade end state, trajectory or text]]
supported-by:: [[EVD - AppWorld grades only the database so any action language scores the same]], [[EVD - tau2 hidden tools turn retrieval quality into a state check]], [[EVD - A task passes when the final database state and the information told to the user mat]], [[EVD - Because the answer key is a verified end state, the agent may call extra read APIs,]]
refines:: [[CLM - Open code action spaces raise agent capability]]
different-context:: [[CLM - Constraining the action space keeps the verifier simple]]

## Scope
Tasks whose success is visible in state.

## Related zettels
- [[hidden-tools-gate-capability-behind-retrieval]]
