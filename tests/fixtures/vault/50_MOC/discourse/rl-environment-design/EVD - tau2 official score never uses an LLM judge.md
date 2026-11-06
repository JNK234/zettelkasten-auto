Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: C

# tau2 official score never uses an LLM judge

## Relations
from:: [[sierra-research-tau2-bench - Agent Evaluation Harness]]
supports:: [[CLM - Grade the observed end state rather than the trajectory]], [[CLM - Keep unvalidated judges out of the headline score]]

## Quote
> Across the whole repo, **the default official score never depends on an LLM judge.**

## Setup
Code read at v1.0.1; judges only as diagnostics or re-run gates.
