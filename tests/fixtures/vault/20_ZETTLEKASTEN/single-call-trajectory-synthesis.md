Created: 24-09-2026 16:30

Status: #baby

Tags: [[rl-environment-design]] [[agentic-ai-systems]]

# Single-Call Trajectory Synthesis

## What This Is
To get SFT data for a tool-using agent, you normally run an agent in an environment turn by turn. Simia-SFT instead asks one reasoning model to write a *whole* multi-turn trajectory in one generation: user turns, agent reasoning, tool calls, and the simulated tool outputs, conditioned on the tool specs, policy, format, and one seed trajectory. Repeat at temperature 1.0 tens of thousands of times (per source, Simia-SFT).

## Why It Matters
It is much cheaper than turn-by-turn simulation and keeps each story globally consistent, because the model knows where it is heading. It turns a few hundred seeds into tens of thousands of trajectories.

## Key Details
- Four stages: (1) LLM pre-filter on seeds for completeness, logic, and format, since amplification amplifies errors too; (2) prompt with tool specs, policy rules, I/O format, and **one** exemplar (one seems to be how diversity is kept up); (3) single-call generation; (4) deterministic post-processing: repair JSON, drop or fix hallucinated tool names, normalize call syntax, rewrite system prompt to match deployment.
- Amplification: APIGen-MT ~1.5k seeds → 90k (~60x); AgentTuning 668 → 15k (~22x); OfficeBench 76 → 30k (~400x).
- Synthesizers GPT-5 and o4-mini at temperature 1.0. Students fully fine-tuned: lr 5e-6, 2 epochs, max length 16,384, 8×A100-80GB.

## Tensions
- **Agent and environment share one mind.** The environment never truly surprises the agent's plan, so SFT data lacks adversarial environment behavior. Simia-RL fixes this only at train time. Compare [[environment-adversariality-in-agent-testing]].
- **No semantic verification.** Post-processing checks structure only; inconsistent state (a refund on an order that never existed) can survive into training data. Contrast [[state-verification-vs-output-verification]] and the execution-validated pipeline in [[environment-first-task-generation-by-breaking-working-code]].
- Per-sample quality vs real data is covered in [[simulated-trajectories-match-real-at-equal-size]].

# References
1. [[2511.01824 - Simia Simulating Environments with Reasoning Models for Agent Training]] (Simia-SFT, Limitations)
