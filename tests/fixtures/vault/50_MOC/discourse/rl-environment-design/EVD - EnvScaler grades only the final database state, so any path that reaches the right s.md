Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: C

# EnvScaler grades only the final database state, so any path that reaches the right state scores the same; the authors reject matching a reference tool chain because tasks usually have several correct solutions

## Relations
from:: [[2601.05808 - EnvScaler]]
supports:: [[CLM - Grade the observed end state rather than the trajectory]]

## Quote
> And a task usually has several correct solutions, so treating one tool chain as the single right answer punishes an agent that solves it another valid way.

## Setup
Design choice contrasted with AgentScaler and AutoForge. Argument only; no run compares state grading with tool-chain matching.
