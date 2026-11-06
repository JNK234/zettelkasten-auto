Created: 20-08-2026 00:20
Status: #baby
Tags: [[evaluation-methodology]] [[benchmark-design]] [[experimental-design]]

# Benchmarks Repurposed as Training Environments

## What This Is
Agentic RL needs environments to roll out in. Building one is expensive — you need a real filesystem, a real browser, real test execution. So the field did the cheap thing: it took the environments that already existed, which were **benchmarks**, and trained in them.

SWE-bench was built to evaluate whether a model can fix a GitHub issue. WebArena was built to evaluate web navigation. ALFWorld, TextWorld, Craftax, Mind2Web — all constructed as measuring instruments. Nearly every environment in the survey's infrastructure table is a benchmark wearing a training-environment hat.

The consequence: your training distribution and your evaluation distribution are the same family. Not the same *items* necessarily, but the same task generator, the same interface conventions, the same idiosyncrasies. Improvement on the benchmark stops being evidence of improvement at the underlying skill.

## Why It Matters
This is a contamination problem that the field mostly does not discuss, and it is subtler than the usual kind. Classic contamination is test items leaking into pretraining data, and it has known defenses. This is different: nothing leaked. You trained on a legitimately separate split of a task distribution that exists *only* as a benchmark. There is no held-out world to check against, because the benchmark was the world.

Combine it with the survey's other structural weakness — 500 works, almost no negative results, most claims being one team's number on one setup — and the epistemics get thin. When a survey says method X "improves efficiency markedly," treat the organizing structure as the contribution and the individual numbers as leads to verify.

## Key Details
- Environment inventory: web (WebArena, Mind2Web), GUI, coding/SWE (SWE-bench and interactive SWE envs), text/embodied games (ALFWorld, TextWorld, Craftax), plus science/research, MLE, biomedical, cybersecurity.
- RL framework inventory: agentic-specific (AgentLightning, SkyRL, AReaL, ROLL), RLHF/fine-tuning (verl, OpenRLHF, TRL), general-purpose RL.
- The existence of two infrastructure tables is itself the claim: **environments are now the bottleneck, not algorithms.**
- Named as open challenge 6.3 — environments are task-specific and don't generalize; the field needs realistic complexity, genuine partial observability, and multimodal interaction.
- The distinctive scaling cost: agentic rollouts require *real environment execution* — running code, loading pages, calling APIs. Training throughput is gated by the wall-clock latency of external systems, not by GPU FLOPs. That is a structurally different scaling regime from PBRFT, and arguably underweighted in the survey.

## Tensions
Adjacent to but distinct from [[benchmark-contamination-risk-in-public-datasets]] and [[data-leakage-in-static-llm-benchmarks]] — no leakage occurs here, yet the eval is still uninformative, which is why the usual defenses like [[rephrasing-defeats-ngram-decontamination]] don't help. Closer in spirit to [[evaluator-circularity-in-ai-benchmarks]] and [[distribution-match-in-generated-benchmarks]].

Sharpens [[static-benchmark-inadequacy-for-agent-systems]] and [[static-versus-interactive-benchmark-divergence]]: making a benchmark interactive is what made it usable as a training environment, which is exactly what destroyed its independence as a measure (inference). Also connects to [[benchmark-saturation]], [[fixed-environment-multi-agent-confound]], and [[multiagent-reinforcement-learning-evaluation]].

The throughput point matters for [[agentic-rl-as-a-pomdp]] in practice: the tuple says rollouts are exogenous, and exogenous means someone else's latency is your training loop. Related: [[agents-environment-interface-types]], [[environment-adversariality-in-agent-testing]], [[evaluation-specific-pitfalls-in-agent-simulation-environments]].

# References
1. [[2509.02547 - The Landscape of Agentic Reinforcement Learning for LLMs]]
