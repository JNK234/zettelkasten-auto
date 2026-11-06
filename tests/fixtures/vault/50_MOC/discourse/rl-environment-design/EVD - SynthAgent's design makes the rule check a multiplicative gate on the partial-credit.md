Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# SynthAgent's design makes the rule check a multiplicative gate on the partial-credit terms, the same idea as gating shaping credit on correctness, but applied to constraint compliance.

## Relations
from:: [[2601.22511 - Mock Worlds Real Skills (SynthAgent)]]
supports:: [[CLM - Correctness must gate all shaping credit]]

## Quote
> The forbidden behavior acts as a multiplicative gate: break the rule and nothing else counts.

## Setup
Eq. 5: R = I(no violation) x (subgoal fraction + user-query fraction), averaged, judged by Qwen3-30B-A3B.
