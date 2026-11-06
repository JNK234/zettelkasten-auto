Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: D

# The note explains the trade-off in combining graders per task: a binary AND over five graders makes tasks look harder if one grader is flaky, while a weighted sum lets a strong axis hide a failure on another.

## Relations
from:: [[Anthropic - Demystifying evals for AI agents]]
supports:: [[CLM - Correctness must gate all shaping credit]]

## Quote
> A binary AND over five graders is very strict and will make tasks look harder than they are if any one grader is flaky; a weighted sum lets a strong result on one axis hide a failure on another.

## Setup
Note author's analysis of the post's weighted, binary and hybrid options; no data.
