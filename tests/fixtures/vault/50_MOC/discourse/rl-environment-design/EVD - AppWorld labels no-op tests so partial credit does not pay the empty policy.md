Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: C

# AppWorld labels no-op tests so partial credit does not pay the empty policy

## Relations
from:: [[StonyBrookNLP-appworld - Controllable World of Apps for Interactive Coding Agents]]
supports:: [[CLM - Graders must handle do-nothing-correct tasks explicitly]], [[CLM - Correctness must gate all shaping credit]], [[CLM - Check the oracle hits max and the empty policy hits the floor]]

## Quote
> Naive per-test reward gives a sizeable free baseline to the empty policy, and a policy can learn to be passive.

## Setup
Design of `no_op_pass` / `no_op_fail` labels; not measured in training.
