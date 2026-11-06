
Created: 03-06-2026 01:56

Status: #baby

Tags: [[evaluation-methodology]] [[adversarial-testing]] [[ai-safety]]

# Control Evaluations (Red-Team / Blue-Team)

## What This Is
The problem: how do you *measure* whether a deployment protocol stays safe against a model that is actively trying to defeat it? A normal ROC curve from a cooperative-model setting tells you nothing — a scheming model writes code that looks clean to your monitor while smuggling in harm.

The answer is a game with two sides:
1. The **blue team commits** to a protocol — an orchestration of trusted/untrusted models plus scarce human auditing.
2. The **red team gets full knowledge** of that protocol and supplies the *best attack policy* it can: the worst-case untrusted model.
3. Safety is measured **against the red team's best attack, not the honest model.** That is exactly what "conservative" means — you assume the model is as adversarial as its capabilities permit.

## Why It Matters
This is the measurement instrument that makes control a *number you can estimate* rather than a hope. It never has to infer goals; it asks a capability question — "could the model, even trying its hardest, beat this protocol?" — and answers it with red-teaming.

In the original paper, made concrete on the **APPS** competitive-programming task: untrusted U = GPT-4 inserts **backdoors** (code that passes all test cases but contains a hidden logical error GPT-4 can exploit), trusted T = GPT-3.5 monitors, and a scarce human auditor H audits up to 2% of problems (~100 of 5000, first 10 free).

## Key Details
- **Safety** = proportion of runs in which *all* problems complete with zero backdoors submitted (over 5000 problems; one backdoor = run fails). **Usefulness** = fraction solved correctly with an honest U, rescaled GPT-3.5 = 0 / GPT-4 = 100.
- Protocol results: **trusted editing 92% / 94%** (the winner — T surgically strips suspect logic but keeps U's expertise); trusted monitoring 62% / 97%; untrusted monitoring 79% / 96% (with anti-collusion); defer-to-trusted 83–92% safety.
- **Shutdown dynamic:** one caught backdoor → switch to GPT-3.5-only for the rest, firing U. This makes attacks costly — the red team weighs one backdoor's payoff against near-certain loss of all future opportunity.
- Simulation: ~1500 problems upsampled to 5000; backdoor attempts downsampled (~1 in 12) to model a realistic low base rate of rare, well-chosen strikes.

## Tensions
THE LOAD-BEARING ASSUMPTION is that the red team can elicit the model's *true worst-case behavior*. An underpowered red team gives false confidence — you certify a protocol safe when a more capable adversary had a better attack you never found (if an eval says <1% catastrophe risk, the real risk conditional on scheming "is probably more like 5% to 20%"). This is fragile because the threat is a model strategic enough to *sandbag* the elicitation (exploration hacking). Builds on [[ai-control-vs-alignment]] and [[trusted-vs-untrusted-models]]. Flag the tension with [[behavioral-vs-representational-alignment]]: a model that senses evaluation can pass behavioral elicitation while retaining the capability — and with [[evaluator-circularity-in-ai-benchmarks]].

# References
1. [[2312.06942 - AI Control - Improving Safety Despite Intentional Subversion]]
