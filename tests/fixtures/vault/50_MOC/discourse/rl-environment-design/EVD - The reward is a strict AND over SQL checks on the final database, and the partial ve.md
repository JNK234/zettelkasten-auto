Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# The reward is a strict AND over SQL checks on the final database, and the partial verifier-level score is rejected as misleading.

## Relations
from:: [[2603.13594 - EnterpriseOps-Gym]]
supports:: [[CLM - Correctness must gate all shaping credit]]

## Quote
> It also computes a softer "verifier-level" score (the fraction of checks passed) but argues it is misleading, because an agent can pass the easy setup checks and still fail the part that matters.

## Setup
1,150 tasks, pass@1 over three runs.
