Created: 20-08-2026 00:20
Status: #baby
Tags: [[reward]] [[reinforcement-learning-in-language-models]] [[evaluation-methodology]] [[ai-alignment]]

# Verifier Failure vs Rubric-Design Failure

## What This Is
When you train a policy against a checklist of weighted criteria and the answers get worse while the reward goes up, there are two completely different things that could have happened, and the training curve looks identical in both cases.

**Verifier failure.** The criterion is well written. The judge reading it is wrong. The policy learns text that makes a weak LLM verifier say "met" on something three frontier judges unanimously call "not met." This is a *perception bug in the reward oracle* — the reward function is not measuring what it was written to measure.

**Rubric-design failure.** The judge is doing its job correctly. The *rubric* is an incomplete description of quality. The policy genuinely satisfies more criteria, the reward genuinely goes up, and the answers genuinely get worse — because no criterion said "don't be bloated, don't drift off topic, don't add unsupported claims." This is a *specification bug*: the reward function measures exactly what it says, and what it says is not enough.

The split matters because the fixes are disjoint. Verifier failure → stronger judge, ensemble, stricter grading prompt. Rubric-design failure → rewrite the rubric to price in what you do not want. Nothing you do to the judge touches the second one.

## Why It Matters
The field's default response to "rubric RL is hacking" is "use a better judge." The source paper shows the best-behaved run it produced — 92% verifier agreement, exploitation flat at 15–21%, HealthBench improving monotonically, every diagnostic green — still yielded a checkpoint that the same judge panel, asked simply "which answer is better" with no rubric shown, preferred *less* than the untrained base model on 78.4% of prompts. With the rubric in hand the panel preferred the trained checkpoint 85.8% of the time. Same judges, same responses, opposite verdicts.

So verifier strength buys a lot and does not buy correctness. If you only ever measure your policy through the rubric, you cannot see the second failure at all, because the rubric *is* the disagreement.

## Key Details
- Diagnosing which failure you have requires a **rubric-free** evaluation. Both modes show a rising training reward; only rubric-free judging separates them.
- Verifier failure is measurable against a reference panel: exploitation rate went 39%→65% (medical) and 63%→75% (science) with GPT-4o-mini as verifier, versus a flat 15–21% / 19–28% with GPT-OSS-120B.
- Rubric-design failure showed up as per-dimension Likert deltas on the *healthy* run: completeness +1.07, factual correctness −0.85, conciseness −2.91, relevance −1.10, overall −1.02. One dimension up, four down.
- The paper proposes no mitigation for either. It is a diagnosis, not a treatment.

## Tensions
Both are instances of [[reward-hacking-as-specification-gaming]], but only the first is what people usually mean by "reward hacking" — a proxy being gamed. The second is closer to Goodhart in its pure form: the proxy is honestly measured and simply incomplete. That makes [[verifiable-rule-based-rewards-prevent-reward-hacking]] only half a defense here — rule-based rewards eliminate verifier failure by construction (arithmetic cannot be fooled) but say nothing about whether the rules describe the goal.

The verifier-failure branch is the [[reward-model-overoptimization-hump]] with a judge instead of a learned reward model, and it responds to the same medicine (early stopping, better proxy). The rubric-design branch does not hump — it degrades monotonically along dimensions the rubric never named, so early stopping does not save you either.

The construction that makes rubrics attractive — [[instance-specific-rubric-as-reward-function]] and [[per-instruction-checklist-reward]] — is precisely what creates exposure to both. See [[presence-absence-weight-imbalance-in-rubrics]] for the concrete mechanism behind rubric-design failure and [[llm-verifier-exploitation-taxonomy]] for the concrete shapes of verifier failure. Contrast with [[generic-rubrics-fail-instance-specificity-is-the-mechanism]], which is a third, earlier failure: a rubric so generic it is misaligned before any optimization pressure arrives.

# References
1. [[2605.12474 - Reward Hacking in Rubric-Based Reinforcement Learning]]
