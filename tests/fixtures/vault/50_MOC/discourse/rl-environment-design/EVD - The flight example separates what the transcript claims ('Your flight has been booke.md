Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: C

# The flight example separates what the transcript claims ('Your flight has been booked') from what the outcome shows (whether a reservation row exists in the SQL database)

## Relations
from:: [[Anthropic - Demystifying evals for AI agents]]
supports:: [[CLM - Grade the observed end state rather than the trajectory]]

## Quote
> The post's example makes the gap concrete: a flight agent can say "Your flight has been booked" in the transcript, but the outcome is whether a reservation row actually exists in the SQL database. Grading the transcript tells you what the agent claimed; grading the outcome tells you what happened.

## Setup
Illustrative example; no data.
