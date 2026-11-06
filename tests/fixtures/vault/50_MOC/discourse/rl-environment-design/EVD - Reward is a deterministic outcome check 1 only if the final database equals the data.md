Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# Reward is a deterministic outcome check: 1 only if the final database equals the database produced by the verified ground-truth actions and every required output appears in the agent's messages.

## Relations
from:: [[2504.03601 - APIGen-MT Multi-Turn Agent Data Generation]]
supports:: [[CLM - Anchor reward in a deterministic check of the outcome]]

## Quote
> In τ-bench's convention this is binary: r = 1 if the final database equals the database produced by a_gt and every item in o_gt appears in the agent's responses, else r = 0. That outcome-only reward is what lets Phase 2 accept many different conversations for the same blueprint.

## Setup
τ-bench Retail and Airline; used as a rejection-sampling filter for gpt-4o trajectories (about 67% success).
