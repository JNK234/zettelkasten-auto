Created: 24-09-2026 16:30

Status: #baby

Tags: [[rl-environment-design]] [[distribution-shift]]

# Simulation Fidelity Is Limited to Describable Environments

## What This Is
An LLM can only simulate what it can describe. Simia's results split cleanly along that line. Where the environment is an API schema plus business rules (WebShop, τ-bench airline), simulated training works well. Where the environment is high-entropy real-world state (Mind2Web's real web DOM), students lag. And where rules are richest (τ²-retail), the quality of the simulating model matters most (per source, Results and Ablations).

## Why It Matters
It gives a decision rule for when to simulate and when to build: simulate domains whose behavior fits in a spec; build real sandboxes where state is too large or idiosyncratic to imagine. It also says which simulator model to pay for: the stronger one where policies are complex.

## Key Details
- AgentBench (Qwen3-8B student vs GPT-4o): WebShop 67.3 vs 48.9 (+18); Mind2Web 25 vs 32 (below). Almost all of the average gain comes from WebShop.
- τ²: airline 56.0 nearly matches o4-mini's 57.0; retail 61.7 trails o4-mini 69.3 and GPT-5 77.2.
- Synthesizer ablation: GPT-5 vs o4-mini similar overall, but on τ²-retail 40.2 vs 28.7; retail has the most complex policy.
- Covered domains: airline, retail, OS, web shopping, web navigation, office. Extension to other schemas is open.

## Tensions
- Complements [[environment-first-task-generation-by-breaking-working-code]]: codebases are exactly the high-entropy state an LLM can't faithfully imagine, which is why SWE-smith runs real tests (inference).
- Connects to [[simulation-to-real-world-gap-in-robot-learning]]: same sim-to-real question, now in text.
- Source's limitation statement: simulation "may still introduce distributional bias" relative to real deployments.

# References
1. [[2511.01824 - Simia Simulating Environments with Reasoning Models for Agent Training]] (AgentBench results, τ²-bench results, Ablations, Limitations)
