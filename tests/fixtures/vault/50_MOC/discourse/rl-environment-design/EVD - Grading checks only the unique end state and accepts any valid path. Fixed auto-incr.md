Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# Grading checks only the unique end state and accepts any valid path. Fixed auto-increment IDs make the check implicitly punish harmless stray writes.

## Relations
from:: [[2603.13594 - EnterpriseOps-Gym]]
supports:: [[CLM - A verifier must accept every valid form of the correct outcome]]

## Quote
> If an agent creates a stray case first (a wrong attempt it then abandons), the correct case becomes 1234 and this check fails even though the right case exists.

## Setup
Code reading, CSM task JSON @ 271f2c3.
