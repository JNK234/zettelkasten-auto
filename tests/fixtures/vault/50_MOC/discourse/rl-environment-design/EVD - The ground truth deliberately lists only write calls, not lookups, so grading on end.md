Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# The ground truth deliberately lists only write calls, not lookups, so grading on end state leaves the agent free to read, ask and reorder

## Relations
from:: [[2504.03601 - APIGen-MT Multi-Turn Agent Data Generation]]
supports:: [[CLM - Grade the observed end state rather than the trajectory]]

## Quote
> The blueprint's a_gt focuses on write calls. This is a deliberate non-default choice: the groundtruth does not prescribe which lookups the agent should do, only which changes must happen.

## Setup
APIGen-MT blueprints on τ-bench Retail and Airline. Design choice; no comparison with full-sequence ground truth.
