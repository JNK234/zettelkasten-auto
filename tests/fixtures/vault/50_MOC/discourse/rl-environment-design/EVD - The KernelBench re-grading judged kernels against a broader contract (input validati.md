Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: C

# The KernelBench re-grading judged kernels against a broader contract (input validation, other dtypes and shapes) than the task stated, so part of the 31% overestimate reflects an unstated contract rather than wrong answers (the note author's reading).

## Relations
from:: [[2507.02825 - Agentic Benchmark Checklist (ABC)]]
supports:: [[CLM - Deterministic graders fail silently when their scope drifts from the spec]]

## Quote
> The 31% is best read as "31% of kernels break under a broader correctness contract", and that contract should be stated.

## Setup
Code reading of o3-mini-written, human-verified KernelBench tests (e.g. test_dimension_mismatch).
