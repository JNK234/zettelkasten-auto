Tags: #evaluation-methodology #agent-evaluation #ai-safety
Status: #baby
## What This Is
A safe, scalable way to evaluate dangerous agent behavior without ever running the dangerous tools. The agent runs in a ReAct loop (Thought → Action → Action Input → Observation), but the environment is *simulated*: tool outputs are pre-written templates, and "attack success" is read off the agent's emitted Action / Action Input — *did it choose to call the harmful tool with the harmful parameters?* — rather than off any real-world effect.

## Why It Matters
Unlocking a real door or moving real money just to score a benchmark would be insane. Simulation makes the evaluation safe, reproducible, and cheap. The deliberate trade is conceptual: it measures **intent to comply**, not end-to-end exploitation. A real tool might reject malformed or unauthorized parameters, so a simulated "success" is an upper bound on intent and a possibly-loose proxy for real harm.

## Key Details
- The paper "assumes the LLM agent can accurately execute the user tool and receive the corresponding response" — it does not test *whether* the agent can use tools, it assumes it can and tests whether it gets *hijacked* (per source).
- For the data-stealing Step-2 check, the "extraction tool returned X" observation is itself simulated with GPT-4 so there is realistic stolen data to forward (per source).
- **Validity filtering:** output counts only if it parses cleanly in ReAct format — malformed/redundant actions are excluded from the ASR denominator. 4 of 30 models had <50% valid rate (essentially unusable as agents) (per source).

## Tensions
This is the deliberate *inverse* of [[execution-based-tool-evaluation]]: that approach runs tool calls to catch semantic errors a real call would surface; this one refuses to run them and accepts measuring intent only. Validity filtering inflates cross-model comparability — weak models that can barely follow ReAct get partly excused, muddying comparison (a limitation). Feeds the metric design in [[two-step-exfiltration-asr-chain]].
## References
1. [[2403.02691 - InjecAgent (Indirect Injection Benchmark)]]
