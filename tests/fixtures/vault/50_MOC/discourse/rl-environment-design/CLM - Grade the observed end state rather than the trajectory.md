Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #supported

# Grading should check the observed end state of the environment rather than the path the agent took or its self-report.

## Relations
informs:: [[QUE 03 - Grade end state, trajectory or text]], [[QUE 19 - Reliability of LLM judges]]
supported-by:: [[EVD - AppWorld fails any database change outside an allow-list]], [[EVD - tau2 official score never uses an LLM judge]], [[EVD - AppWorld grades only the database so any action language scores the same]], [[EVD - EnvScaler grades only the final database state, so any path that reaches the right s]], [[EVD - The ground truth deliberately lists only write calls, not lookups, so grading on end]], [[EVD - Anthropic found that requiring a specific tool-call sequence gives brittle tests bec]], [[EVD - The flight example separates what the transcript claims ('Your flight has been booke]]
opposed-by:: [[EVD - tau2 outcome checks cannot see process violations]]
different-context:: [[CLM - Process violations require checking the trajectory]]

## Scope
Tasks whose success is visible in state. Misses violations that leave state unchanged.

## Related zettels
- [[state-diff-grading-with-collateral-damage-allowlist]]
- [[replay-and-hash-end-state-grading]]
