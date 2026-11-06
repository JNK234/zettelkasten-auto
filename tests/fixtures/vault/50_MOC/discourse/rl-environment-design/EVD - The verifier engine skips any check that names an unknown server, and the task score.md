Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# The verifier engine skips any check that names an unknown server, and the task score is all() over the results, so a task with zero surviving checks counts as solved.

## Relations
from:: [[2603.13594 - EnterpriseOps-Gym]]
supports:: [[CLM - Deterministic graders fail silently when their scope drifts from the spec]]

## Quote
> a task whose verifiers all point at a mis-named server has zero results, and `all([])` is `True`, so it would count as a success.

## Setup
Code reading, benchmark/executor.py:428-450 @ 271f2c3.
