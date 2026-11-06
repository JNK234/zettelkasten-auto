Created: 24-09-2026 16:44

Status: #baby

Tags: [[benchmark-design]] [[evaluation-methodology]] [[agentic-ai-systems]]

# Measurement Correction Changes Apparent Agent Progress

## What This Is
A benchmark score is capability plus measurement error: leaked solutions, weak tests that accept wrong fixes, contamination. When those errors get fixed, scores and rankings move, sometimes a lot. So year-over-year gains on agent leaderboards are partly real capability and partly the evaluation tightening. The Oxford synthesis concludes it is both, not one or the other.

## Why It Matters
An RL environment's grader is a benchmark's test suite under another name. Weak tests that inflate benchmark scores become reward hacks in training. And any claim of progress from an environment should come with a test-adequacy audit.

## Key Details
- **SWE-bench+**: after removing solution leakage and weak tests, one SWE-agent configuration fell from ~12.5% to ~4% (an old configuration; says nothing about 2026 correction size).
- **UTBoost**: stronger auto-generated unit tests changed leaderboard rankings in ~25-41% of cases on SWE-bench Lite / Verified.
- **HAL**: argues for cost-adjusted comparison, since headline accuracy hides large dollar/compute differences.
- "Towards a Science of AI Agent Reliability" (2026): one aggregate hides run-to-run consistency, graceful degradation under perturbation, and bounded error severity; capability gains gave only small reliability gains.
- Reporting recommendations: constraint-level and plan-level success, partial and binary success, cost-adjusted accuracy, contamination and test-adequacy audits, accuracy-cost frontier for test-time compute claims.
- Still, corrected 2026 SWE-bench Verified > corrected 2024, so some gain is real.

## Tensions
- [[weak-verification-problem-in-automated-testing]] and [[validation-test-overfitting-in-agent-search]] are the mechanism behind weak-test inflation.
- [[reward-hacking-as-eval-false-positive]] and [[rescoring-reward-hacks-in-capability-measurement]] cover the case where the inflation is the agent actively cheating.
- [[git-history-recovery-anti-pattern-for-llm-testing]] is one leakage route.
- [[multi-run-consistency-testing-for-llm-agents]] operationalizes the reliability decomposition.

# References
1. [[2607.05775 - Beyond the Leaderboard Agent Failure Taxonomy]] (§6 Measurement validity; §Where agents genuinely improved; §Implications)
