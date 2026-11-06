Created: 20-08-2026 00:20
Status: #baby
Tags: [[reward]] [[evaluation-methodology]] [[reinforcement-learning-in-language-models]]

# Reward Model Accuracy Is Not Reward Model Utility

## What This Is
The field selects reward models by leaderboard rank on pairwise-ranking benchmarks — RewardBench and friends. The implicit assumption is that a model better at picking the winner of a preference pair will be better at supervising RL.

That assumption is false, and RLCF gives a clean counterexample. Its checklist-based reward is **worse than both specialised RMs on RewardBench** — dramatically so on Safety (71.4 vs Skywork's 93.0, ArmoRM's 90.5) — and yet it supervises DPO better than either.

The reason is that ranking accuracy and optimisation quality are different properties. Ranking accuracy asks: on this fixed set of pairs, do you order them correctly? RL asks something else entirely: across the whole space the policy can reach, is your reward **stable, smooth, and hard to game**? A reward can be a great classifier and a terrible optimisation landscape.

## Why It Matters
Table 8 shows the mechanism, not just the correlation. Given two sentences using the word "dense," both perfectly satisfying the instruction and essentially identical in meaning, Skywork scores them **33.1 and 8.0**. That is a 4× reward difference over stylistic noise. Train on that and you are doing gradient ascent on the RM's idiosyncratic surface preferences.

The checklist judge scores the same pair 97.3 and 96.6 — appropriately insensitive to meaningless variation, and still 95.2 vs 0.0 on a genuinely good vs genuinely broken pair.

The paper's summary: **reward models are too sensitive; prompted AI judges saturate; checklists give stable, interpretable scores.** Sensitivity to noise is invisible to a ranking benchmark (both orderings may be defensible) but is exactly what RL amplifies.

## Key Details
- Attributed to Malik et al. 2025 and Razin et al. 2025.
- Baselines: `Skywork-Reward-Gemma-2-27B` (#4 on RewardBench, July 2025), `ArmoRM-Llama3-8B-v0.1` (#24, mixture of 19 experts).
- The RewardBench checklists were generated **without ground truth or any other model's output** — no leakage.
- The regression pattern is the real evidence. Every off-the-shelf RM *hurts* IFEval strict prompt accuracy: Skywork 72.5 → 68.0, ArmoRM → 70.2, UltraFeedback → 69.1, AI Judge → 68.9. RLCF alone does not regress (→ 72.6). Preference-trained RMs are actively teaching the model to stop following literal format constraints, presumably because their training data rewards generically-fluent prose over compliance with odd requirements.
- Skywork nonetheless beats RLCF on AlpacaEval by a large margin (44.8 vs 36.2; 41.5 vs 37.1 length-controlled) — it was trained on exactly that distribution, and the benchmark measures exactly that. The honest claim is not "RLCF dominates" but **"RLCF is the only method with no regressions anywhere."**
- The paper's own closing puzzle: why does a hand-prompted 72B with a synthetic rubric beat a purpose-built RM trained on millions of human preferences, at the RM's own task? Either preference data is a worse supervision signal than assumed, or scalar-output architectures destroy structure RL needs, or both.

## Tensions
This is the reward-side twin of [[best-of-n-as-a-misleading-prm-metric]] and [[processbench-step-level-error-localization]] — a metric that ranks graders well but predicts training utility poorly. It reframes [[reward-model-overoptimization-hump]]: if accuracy does not predict utility, the hump's location is not predictable from leaderboard rank either. Connects to [[jailbreak-judge-as-load-bearing-metric]] and [[llm-as-judge-evaluation-methodology]] — the same "the metric is not measuring what the pipeline needs" pathology. See [[bradley-terry-reward-model]] for the architecture whose scalar output is under suspicion, and [[per-instruction-checklist-reward]] for the alternative. The proposed fix — train an RM to predict per-item checklist scores, i.e. a vector-output RM with dynamically instantiated heads — is a bridge back toward [[mixture-of-experts-sparse-routing]].

# References
1. [[2507.18624 - Checklists Are Better Than Reward Models (RLCF)]]
