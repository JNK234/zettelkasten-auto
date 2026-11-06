Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# EnvScaler's reward code is LLM-written check functions that are never run against a known-correct or known-wrong final state before training uses them; the only evidence they work is that stronger models score higher on 50 scenarios.

## Relations
from:: [[2601.05808 - EnvScaler]]
supports:: [[CLM - Check the oracle hits max and the empty policy hits the floor]]

## Quote
> Checks are never tested against a known-correct or known-wrong final state. A check that is too loose rewards wrong behaviour, and one that is too strict is invisible noise.

## Setup
Released code at 96ae8b0; scen_generator/ has no validation step.
