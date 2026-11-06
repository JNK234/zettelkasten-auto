Created: 24-09-2026 16:44

Status: #baby

Tags: [[agentic-ai-systems]] [[evaluation-methodology]] [[llm-as-a-judge-framework]]

# Agent Test-Time Scaling Is Limited by the Verification Gap

## What This Is
For single-turn reasoning, spending more compute at inference (think longer, or sample many answers and pick one) reliably helps. For agents it often does not. General AgentBench tested 10 leading agents: neither **sequential scaling** (more reasoning tokens per step) nor **parallel scaling** (sample more trajectories, pick one) reliably improved results. Two mechanisms are named:

- **Context ceiling**: extra reasoning stops helping once working memory saturates.
- **Verification gap**: with many sampled trajectories, the agent cannot tell which one actually solved the task. Best-of-N is only as good as its selector.

So the bottleneck is the verifier, not the generator.

## Why It Matters
In an environment design, this is an argument for investing in verifiers that can judge a whole trajectory, both as graders for training and as selectors at inference. It also warns against assuming higher reasoning effort is a free upgrade for agents.

## Key Details
- HAL (Holistic Agent Leaderboard, Princeton, ICLR'26): 21,000+ rollouts, 9 models × 9 benchmarks, ~$40K; higher reasoning-effort settings *reduced* accuracy in most runs.
- TravelPlanner: adding an external formal verifier/critic lifts success from ~10% (o1-preview) to ~65% best case, consistent with verification being the constraint.
- Open question from the authors: does test-time-scaling non-improvement generalize beyond the general-purpose setting?

## Tensions
- [[best-of-n-verification-with-reward-models]] and [[best-of-n-as-a-misleading-prm-metric]]: selector quality caps best-of-N.
- [[generator-verifier-gap-collapse]] and [[generation-verification-gap-as-a-designed-asymmetry]] give the general theory; here the gap runs the wrong way (verifying is as hard as solving for long agent tasks).
- [[closed-loop-self-improvement-needs-a-non-persuadable-verifier]]: the same verifier weakness limits self-improvement.
- [[constraint-level-vs-plan-level-success-gap]] shows the verifier fix working on planning.

# References
1. [[2607.05775 - Beyond the Leaderboard Agent Failure Taxonomy]] (§2 Planning, test-time scaling; §6 HAL; §What is interesting)
