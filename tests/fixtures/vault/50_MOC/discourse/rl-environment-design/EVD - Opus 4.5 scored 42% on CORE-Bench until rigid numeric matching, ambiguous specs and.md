Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# Opus 4.5 scored 42% on CORE-Bench until rigid numeric matching, ambiguous specs and irreproducible stochastic tasks were fixed and the scaffold loosened, after which it scored 95%.

## Relations
from:: [[Anthropic - Demystifying evals for AI agents]]
supports:: [[CLM - A verifier must accept every valid form of the correct outcome]]

## Quote
> For example, Opus 4.5 initially scored 42% on CORE-Bench, until an Anthropic researcher found multiple issues: rigid grading that penalized “96.12” when expecting “96.124991…”, ambiguous task specs, and stochastic tasks that were impossible to reproduce exactly. After fixing bugs and using a less constrained scaffold, Opus 4.5’s score jumped to 95%.

## Setup
Opus 4.5 on CORE-Bench; grader, spec and scaffold fixes combined, so the individual contributions are not separated.
