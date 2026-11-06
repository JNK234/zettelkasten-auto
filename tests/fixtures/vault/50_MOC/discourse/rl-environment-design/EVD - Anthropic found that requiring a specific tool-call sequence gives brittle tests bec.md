Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: C

# Anthropic found that requiring a specific tool-call sequence gives brittle tests because agents regularly find valid unanticipated routes, so it recommends grading what the agent produced

## Relations
from:: [[Anthropic - Demystifying evals for AI agents]]
supports:: [[CLM - Grade the observed end state rather than the trajectory]]

## Quote
> We’ve found this approach too rigid and results in overly brittle tests, as agents regularly find valid approaches that eval designers didn’t anticipate. So as not to unnecessarily punish creativity, it’s often better to grade what the agent produced, not the path it took.

## Setup
Anthropic internal eval practice; no measured rate.
