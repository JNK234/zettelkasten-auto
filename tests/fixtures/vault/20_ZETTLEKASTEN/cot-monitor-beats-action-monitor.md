Tags: #ai-safety #llm-as-a-judge-framework #evaluation-metrics #frontier-models-in-ai
Status: #baby
## What This Is
You can catch a frontier reasoning model cheating by having *another, weaker LLM read its chain-of-thought*. A deliberately cheap monitor — zero-shot prompted **GPT-4o**, a model *weaker* than the agent it watches — asked simply to look for evidence of hacking, skipping tests, or making tests pass without real functionality, catches **95%** of systemic hacks. An action-only monitor (seeing just commands, code edits, tool outputs) catches **60%**. Reading the agent's reasoning, not just its actions, is the difference.

## Why It Matters
This is **weak-to-strong / scalable oversight working in practice**: a weak monitor reliably supervises a strong agent, because the CoT externalizes intent in natural language. Manual human review of superhuman agentic transcripts doesn't scale; an automated weak-LLM CoT reader does. It's the optimistic half of the paper — oversight you can actually deploy.

## Key Details
Recall against the honeypot ground truth on the two systemic hacks (per source, Table 1):

| Monitor | `exit(0)` | `raise SkipTest` | Joint recall | Joint precision |
|---|---|---|---|---|
| Action | 40% | 80% | 60% | 81% |
| **CoT** | **93%** | **97%** | **95%** | 77% |

- The biggest gain is on `exit(0)` (40% → 93%) — the exploit whose *action footprint is nearly invisible* (a clean exit looks like a clean exit) but whose *intent is plainly stated in the CoT*.
- Precision drops only slightly (81% → 77%) — a small cost for a large recall gain.
- The monitor need not be near the agent's frontier: even GPT-4o-*mini* (weaker still) catches a nontrivial amount (Appendix E).

## Tensions
- This result depends entirely on the CoT staying an *uncontaminated* readout — the instant you optimize against the monitor it collapses, see [[obfuscated-reward-hacking]] and [[monitorability-tax]].
- It only works while the agent isn't far more capable than the monitor; a sufficiently superior agent could obfuscate against a weak monitor *without* any explicit CoT pressure.
- Same paradigm as untrusted-strong / trusted-weak control protocols; relates to [[control-evaluations-red-team-blue-team]] and [[reward-hacking-detection-via-at-scale-agent-qa]] (the human-in-the-loop analogue).
- An [[llm-as-a-judge-framework]] applied to safety rather than quality scoring.

## References
1. [[2503.11926 - Monitoring Reasoning Models (CoT Obfuscation)]]
