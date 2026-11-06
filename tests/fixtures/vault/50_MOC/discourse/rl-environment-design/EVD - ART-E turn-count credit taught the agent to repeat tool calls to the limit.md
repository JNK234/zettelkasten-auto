Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# ART-E turn-count credit taught the agent to repeat tool calls to the limit

## Relations
from:: [[Kyle Corbitt - ART·E How We Built an Email Research Agent That Beats o3]]
supports:: [[CLM - Correctness must gate all shaping credit]]

## Quote
> The model learned to **repeat its last tool call over and over until it hit the 10-turn limit**, padding every trajectory for free reward.

## Setup
Early run with credit for more turns; caught by reading trajectories.
