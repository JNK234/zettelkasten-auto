Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# A task passes when the final database state and the information told to the user match the annotation, so grading looks only at outcomes and any architecture can pass

## Relations
from:: [[2609.04611 - tau-tau-Bench Agent Construction]]
supports:: [[CLM - Grading end state lets the action space open safely]]

## Quote
> Because grading looks only at outcomes, any architecture passes if the behavior is right.

## Setup
tau^tau-bench held-out evaluation, tau-bench style. Grading rule described; not an ablation of grading methods.
