Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: C

# ABC's state-modification checks require the ground truth to include every reachable correct end state, to check states that should not change, and to be hard to hit by random changes

## Relations
from:: [[2507.02825 - Agentic Benchmark Checklist (ABC)]]
supports:: [[CLM - State graders should check what must not change as well as what must]]

## Quote
> the ground truth must include every end state a correct solution could reach (O.g.1), must check both the states that should change and the ones that should not (O.g.2, to catch collateral damage), and must be complex enough that random or trivial changes cannot hit it (O.g.3).

## Setup
Checklist item derived from documented benchmark bugs; applied to ten open-source benchmarks. A rule, not a measured effect.
