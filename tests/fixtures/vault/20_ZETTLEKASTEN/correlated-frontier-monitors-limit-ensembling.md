Created: 24-09-2026 16:28

Status: #baby

Tags: [[llm-as-a-judge-framework]] [[evaluation-methodology]] [[ai-alignment]]

# Correlated Frontier Monitors Limit Ensembling

## What This Is
A common fix for a weak LLM monitor is to ensemble several: if GPT, Gemini, and Kimi each miss different hacks, voting should cover more ground. That only works if their errors are independent. TRACE found the top three detectors (GPT-5.2, Gemini 3 Pro, Kimi-K2) agree with each other at κ = 0.812 (90.9% raw), while a single model agrees with itself across 3 runs at κ = 0.795 (90.1%). Different frontier models agree with each other about as much as each agrees with itself.

## Why It Matters
Ensembling frontier monitors buys little diversity on misses. When designing a reward-hack defense for an RL environment, adding a second LLM judge is not a substitute for a different kind of check (sandbox restriction, held-out tests, human audit).

## Key Details
- Only 3.5% of benign trajectories were flagged as hacks by all models, so false positives are not correlated, even though the overall agreement is high (per source §7).
- Inference from the two facts: the shared errors are likely on the misses (e.g., semantic hacks, where Match Rate is 0.0-0.4 for everyone), not the false alarms. The source does not break agreement down by error type.
- Precision/recall profiles still differ: Claude Opus 4.5 had precision 1.0 with poor recall, so a union of a cautious and an aggressive monitor may still help recall (inference, per source §6).

## Tensions
- Contrasts with [[cross-family-judge-panel-with-unanimity]], which assumes cross-family diversity buys independence.
- Related to [[shared-judge-family-across-training-and-eval]].
- Correlated blind spots line up with [[syntactic-vs-semantic-reward-hack-detectability]].

# References
1. [[2601.20103 - TRACE Reward Hack Detection via Contrastive Analysis]] (§6, §7)
