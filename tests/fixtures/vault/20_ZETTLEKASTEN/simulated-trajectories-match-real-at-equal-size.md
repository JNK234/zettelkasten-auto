Created: 24-09-2026 16:30

Status: #baby

Tags: [[rl-environment-design]] [[experimental-design]] [[agentic-ai-systems]]

# Simulated Trajectories Match Real Ones at Equal Size

## What This Is
The key question for simulated environments is whether a simulated trajectory is worth as much as a real one. Simia's size-matched comparison (~5k trajectories each) found them about equal on OfficeBench and AgentBench, and simulated data *better* on τ²-bench. Past the size the seeds can supply, simulated data wins simply because it is unbounded (per source, Fig. 6).

## Why It Matters
If per-sample value is roughly equal, the scaling argument decides the question: real data is capped by environment-building cost, simulated data isn't. The cleanest evidence is a head-to-head against a model trained on real-environment-verified data from the same seed source.

## Key Details
- Simia-Tau (Qwen2.5-32B) 58.9 avg on τ²-bench vs xLAM-2-70B 56.3. xLAM-2 was trained on APIGen-MT data with real-environment verification, the same seed source, and is 2x larger.
- **Pass^k gap widens with k** (airline, k=1/2/3): Simia-Tau-32B 56.0 / 48.0 / 46.0 vs xLAM-2-70B 49.3 / 40.0 / 34.0; gap 6.7 → 12.0. The data seems to buy consistency, not lucky single shots. See [[pass-hat-k-reliability-metric]].
- Why sim might beat real on τ² (source's own speculation): real trajectories are filtered for success and under-represent error recovery; simulated ones can include plausible failures and recoveries.
- Joint training on all three datasets (Qwen3-8B) shows no big negative interference across domains.

## Tensions
- Distillation ceiling: students never beat their synthesizers (GPT-5, o4-mini) on any benchmark average. It moves a frontier world model into a small model; it does not exceed the frontier. Compare [[distillation-beats-from-scratch-rl-for-small-reasoning-models]].
- Opposing evidence for real, execution-verified data at scale: [[rejection-sampling-fine-tuning-for-agents]].
- Eval is also partly simulated (GPT-4.1 user), see [[shared-judge-family-across-training-and-eval]].

# References
1. [[2511.01824 - Simia Simulating Environments with Reasoning Models for Agent Training]] (τ²-bench results, Fig. 6, Ablations, Limitations)
