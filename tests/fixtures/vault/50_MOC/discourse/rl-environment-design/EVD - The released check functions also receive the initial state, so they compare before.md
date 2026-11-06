Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# The released check functions also receive the initial state, so they compare before and after (for example 'X was missing before and exists now') rather than reading only the final state as the paper's equation suggests

## Relations
from:: [[2601.05808 - EnvScaler]]
supports:: [[CLM - State graders should check what must not change as well as what must]]

## Quote
> So the checks really compare before and after, which is more expressive than the equation suggests.

## Setup
Code reading of env_util.py:102-127 @ 96ae8b0. Shows the mechanism; no measure of how often collateral changes are caught.
