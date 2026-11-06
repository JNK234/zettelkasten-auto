Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# Because the answer key is a verified end state, the agent may call extra read APIs, ask clarifying questions or change order and still pass

## Relations
from:: [[2504.03601 - APIGen-MT Multi-Turn Agent Data Generation]]
supports:: [[CLM - Grading end state lets the action space open safely]]

## Quote
> The clever part is that the answer key is written before the exam is taken. The conversation is free to be messy and natural (the agent can call extra read APIs, ask clarifying questions, go in a different order), but it is graded against something verified.

## Setup
APIGen-MT Phase 2 on τ-bench. Describes the grading contract; no count of passes that took non-reference paths.
