Tags: #benchmark-design #evaluation-methodology #agent-evaluation #ai-safety
Status: #baby
## What This Is
Instead of scoring an AI agent as "71% on benchmark X" (a number on an arbitrary scale), measure task difficulty in **how long a skilled human takes to do the task**. Then an agent's capability becomes a *human-time horizon*: "it reliably does tasks that take a human under an hour, but fails past 4 hours." HCAST (METR's 189-task suite) is built entirely around this grounding — every task is timed by an expert human in the exact same sandbox the agent gets, converting an abstract success rate into hours-of-expert-work the agent can do unsupervised.

## Why It Matters
Most AI benchmarks are **ungrounded**: a percentage on MMLU or an ELO leaderboard tells you nothing about real-world impact. If you want to forecast societal effects of autonomous agents (job automation, R&D acceleration, multi-step cyber operations) or build an AI-control argument (you must bound what a model can accomplish before a human catches it), you need a metric tied to a real-world quantity. Human completion time is that quantity — it is interpretable, comparable across domains, and directly maps to "can this agent do a junior engineer's day of work?"

## Key Details
- The conceptual leap: difficulty stops being "easy/medium/hard" and becomes *measured in hours of expert work* (per source).
- Procedure: put a skilled human in *exactly* the agent's environment (same SSH terminal, files, instructions, containerized box), time them → human baseline; then run agents in the same environment and plot **agent success rate vs. human completion time** (x-axis grounded in human labor).
- HCAST tasks span ~1 minute to 8+ hours of human time; the broader METR suite extends to ~30 hours via RE-Bench's ML-research tasks (per source).
- Difficulty distribution by human time: <15min = 41 tasks, 15min–1hr = 59, 1–4hr = 57, >4hr = 32 (per source).
- The killer question this enables: "what is the longest (in human-time) task an agent can reliably do?"
- HCAST is the *ruler*; the sibling paper "Measuring AI Ability to Complete Long Tasks" (2503.14499) is the *measurement* that fit a curve to these times.

## Tensions
- Realism gap: human-time grounding only works because tasks have clean scope and automatic scoring — real economically-valuable work is ambiguous, collaborative, and subjectively judged. The metric is a measurement convenience, not the real world.
- Baseliner ≠ agent conditions: humans got limited web access, agents rely on memorized equivalents — not a perfectly matched condition, so the time axis has slack.
- Relates to [[adaptation-versus-baseline-capability-in-agent-evaluation]] (static vs. learning capability — HCAST measures static), [[expert-driven-evaluation-for-safety-critical-ai]] (also uses experts, but for correctness rubrics not time), and [[control-evaluations-red-team-blue-team]] (time-horizons supply the bound that control protocols assume).
- See [[incentive-aligned-human-baselining]] for how the time numbers are actually collected and [[logistic-time-horizon-from-log-human-time]] for how they become a single scalar.

## References
1. [[2503.17354 - HCAST (Human-Calibrated Autonomy Software Tasks)]]
