Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: D

# A Terminal-Bench audit found a task whose tests assumed a file path the spec never stated, so agents could fail through no fault of their own.

## Relations
from:: [[Anthropic - Demystifying evals for AI agents]]
supports:: [[CLM - Deterministic graders fail silently when their scope drifts from the spec]]

## Quote
> The Terminal-Bench audit example is concrete: a task asked for a script without specifying a path, but the tests assumed one, so an agent could fail "through no fault of its own."

## Setup
Terminal-Bench task audit; single anecdote, no count of affected tasks.
