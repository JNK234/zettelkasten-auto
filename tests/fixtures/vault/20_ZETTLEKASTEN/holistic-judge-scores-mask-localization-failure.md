Created: 24-09-2026 16:28

Status: #baby

Tags: [[llm-as-a-judge-framework]] [[evaluation-methodology]] [[experimental-design]]

# Holistic Judge Scores Mask Localization Failure

## What This Is
The usual way to validate an LLM judge is to check that its overall scores correlate with human scores. TRAIL shows this can look good while the judge understands almost nothing about where the run went wrong. Gemini-2.5-Pro's trace-level scores on SWE-Bench correlate with humans at ρ = 0.817, while its joint accuracy at finding the actual erroneous spans is 0.050. Getting the grade right is much easier than finding the bug.

## Why It Matters
If a judge is used as a reward in an RL environment, a high trace-level correlation does not mean it rewards the right behavior at the right step. Validation has to happen at the granularity the reward will be used at (inference). It is the core argument for span-level evaluation over holistic scoring.

## Key Details
- Claude-3.7-Sonnet has the best GAIA ρ (0.738) with middling span metrics (joint 0.047) (per source, main table).
- Trace-level rubric: four 1-5 Likert dimensions (reliability, security, instruction adherence, plan optimality).
- Security ρ is 1.00 for every model shown, which almost certainly means security scores barely vary (few incidents), so the metric is degenerate, not that models are perfect security judges (per source).
- Instruction adherence and plan optimality correlations are weak (~0.15-0.59).

## Tensions
- A correlation computed on a near-constant variable is a measurement artifact; cf. [[metrics-need-dynamic-range-at-the-failure-end]] and [[ostensible-positives-from-low-entropy-answer-spaces]].
- Supports [[process-supervision-vs-outcome-supervision]] from the evaluation side.
- Human agreement checks are necessary but not sufficient: see [[judge-agreement-against-human-ceiling]]; this shows which level the agreement must be measured at.

# References
1. [[2505.08638 - TRAIL Trace Reasoning and Agentic Issue Localization]] (metrics, main table, trace-level rubric correlation)
