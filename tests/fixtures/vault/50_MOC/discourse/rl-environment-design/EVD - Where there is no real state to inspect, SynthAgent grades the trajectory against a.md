Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# Where there is no real state to inspect, SynthAgent grades the trajectory against a checklist (forbidden behaviours, subgoals, required interactions), written at a level that allows several valid paths

## Relations
from:: [[2601.22511 - Mock Worlds Real Skills (SynthAgent)]]
supports:: [[CLM - Without a real end state, grade the trajectory against path-agnostic subgoals]]

## Quote
> Subgoals, extracted from what the teacher actually did, stated at a level that allows several valid paths.

## Setup
Per-task rubric from teacher demonstrations; trajectory-level LLM judge (Qwen3-30B-A3B) during GRPO training. Judge accuracy against humans not reported.
