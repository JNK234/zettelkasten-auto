Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# On Task 32, every model in one trial each reached the gold's final state with a single combined call instead of the gold's two calls, and all scored zero, showing the grader was sensitive to the path taken.

## Relations
from:: [[2609.14400 - Policy Loopholes in Agent Evaluation]]
supports:: [[CLM - A verifier must accept every valid form of the correct outcome]]

## Quote
> Either way, the evaluator is sensitive to how you reach a state, not just which state you reach, which is itself a verifier-design issue.

## Setup
τ²-bench airline Task 32; GPT-5, Kimi-K2-Thinking, Mistral-Large-3; one trial each. The paper does not explain the cause of the mismatch.
