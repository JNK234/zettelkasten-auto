Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: D

# Claude mislabelled 'normal' tickets as 'high priority' only in its final prose; no tool call caught it, so an answer-text check is the only place such a failure can be graded

## Relations
from:: [[Surge AI - RL Environments and the Hierarchy of Agentic Capabilities]]
supports:: [[CLM - Information tasks need the answer checked against the data, not only state]]

## Quote
> A mislabel that appears only in the final prose has no tool to catch it, so it reaches the user.

## Setup
Corecraft ticket-priority task, Claude Sonnet 4.5, November 2025; single example.
