Tags: #interpretability-through-intermediate-representations #ai-safety #evaluation-methodology
Status: #baby
## What This Is
A reasoning model emits a long chain-of-thought (CoT) before its answer, written in natural language inherited from pretraining. The strict notion of **faithfulness** — a CoT is faithful if it "accurately represents the reasoning process behind the model's prediction" — is basically unachievable, since the only thing that fully represents the computation is the full activation set. So the paper shifts to **monitorability**: a CoT `z` is sufficient *with respect to a property `v`* (e.g. "is this agent cheating?") if `z` is a sufficient statistic for predicting `v`. You don't need the CoT to capture everything the model thinks — only enough to predict the specific bad behavior you care about.

## Why It Matters
Monitorability is a weaker, operational, and actually *testable* bar, where faithfulness is not. The CoT is "not devoid of signal relevant for monitoring": when a model decides to cheat it often *says so* — planning the exploit, e.g. "let's fudge this" or "I'll just make the test pass." That stated intent is the signal a monitor catches. This reframing is what makes weak-LLM CoT monitoring a coherent program rather than a hopeless quest for perfect transparency.

## Key Details
- The **useful-reasoning hypothesis**: to reason well over many steps, the model has strong pressure to keep the CoT in human-legible language, because that legible text carries the relevant computation. This is why intent leaks into the CoT for free, with no faithfulness guarantee required.
- Monitorability is *property-relative* — a CoT can be sufficient for predicting "cheating" while being useless for some other property. You define `v` first, then ask whether `z` predicts it.
- The danger of CoT optimization (per source) is precisely that it can strip out exactly the `v`-relevant information while leaving fluent-looking text behind — fluency is preserved, the sufficient statistic is destroyed.

## Tensions
- Information-theoretically adjacent to reading properties off internals — cf. [[safety-relevant-features-as-monitoring-substrate]] and the activation-monitoring fallback you revert to if the CoT becomes obfuscated.
- The optimization threat to this sufficiency is the subject of [[obfuscated-reward-hacking]] and [[monitorability-tax]].
- Contrast with [[chain-of-thought-confabulation-in-action-systems]]: there the CoT is *wrong*; here the worry is a CoT that is fluent and plausible but no longer *informative* about the property you monitor.

## References
1. [[2503.11926 - Monitoring Reasoning Models (CoT Obfuscation)]]
