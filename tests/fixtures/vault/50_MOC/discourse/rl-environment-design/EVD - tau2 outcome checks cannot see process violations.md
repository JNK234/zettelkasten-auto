Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: C

# tau2 outcome checks cannot see process violations

## Relations
from:: [[sierra-research-tau2-bench - Agent Evaluation Harness]]
supports:: [[CLM - Process violations require checking the trajectory]]
opposes:: [[CLM - Grade the observed end state rather than the trajectory]]

## Quote
> **Outcome-only DB checks can't see process violations** that leave the state unchanged: skipping authentication, leaking another customer's data in chat, saying something off-policy.

## Setup
Limitation in the code read; nl_assertions exist but sit outside the official score.
