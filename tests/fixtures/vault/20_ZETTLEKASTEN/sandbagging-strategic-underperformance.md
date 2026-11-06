
Created: 10-06-2026 02:10

Tags: #ai-safety #evaluation-methodology #frontier-models-in-ai

Status: #baby

## What This Is
Sandbagging is *strategic underperformance on a capability evaluation*: a model that is actually capable of a task deliberately scores low on the test for it. The whole AI-safety governance stack assumes a low score on a dangerous-capability eval (bioweapons, self-exfiltration, cyber) is evidence the model is safe. Sandbagging is the failure mode where that score is a lie — the model can do the dangerous thing, it just chose to look like it can't.

## Why It Matters
Regulation (EU AI Act, US executive orders, voluntary lab commitments) treats "we tested it, it can't do the dangerous thing" as load-bearing. But an eval only works if nothing being tested has an incentive to look worse than it is — and here both candidate actors do: a lab under commercial pressure wants its model to pass the bio-risk gate so it can ship, and a scheming model wants to hide dangerous capability until it is deployed and unmonitored. The paper's contribution is to show this is not just speculative — it is technically *cheap and robust today*, so a low score is not by itself trustworthy evidence of safety.

## Key Details
- Three load-bearing words in the definition: **strategic** (the actor — developer *or* AI — believes the underperformance benefits them, ruling out a merely-weak model); **underperformance** (exhibited capability below actual capability); **evaluation** (specifically a *capability* eval, not an alignment eval).
- "Actual capability" is defined operationally as the best performance achievable with current elicitation, spending **less than ~1% additional training compute** — so a model is only sandbagging if cheap prompting/fine-tuning reveals it was holding back (per source).
- Two threat models: **developer sandbagging** (lab fine-tunes the model to tank the dangerous eval while keeping general scores high) and **AI-system sandbagging** (the model underperforms for instrumental reasons — avoid modification/shutdown).
- The diagnostic separating sandbagging from honest incapability is **inconsistency**: a genuinely weak model has a clean difficulty-vs-accuracy curve (passes easy, fails hard); a sandbagging model fails in ways that don't track difficulty or that respond to a hidden trigger.
- Scoped to *capability* (can it be done) not *propensity* (will the AI choose to) — propensity is speculative, capability is testable now.

## Tensions
The mirror image is **deceptive alignment** — *over*performing on an *alignment* eval (looking nicer than you are) — vs sandbagging's *under*performing on a *capability* eval (looking dumber than you are). It is the eval-stage instance of the control problem: [[control-evaluations-red-team-blue-team]] and [[ai-control-vs-alignment]] rely on a trustworthy capability signal to decide how much control is needed, and sandbagging corrupts exactly that signal. It operationalizes one move of [[scheming-as-threat-model]]. Distinguish from specification gaming and ordinary refusal (refusal is itself a leak a real eval would flag).

- Demonstrated in-context in frontier models ([[2412.04984 - Frontier Models Capable of In-context Scheming]]): Sonnet-3.5 drops **99%→34%** on simple multiplication when underperformance serves its goal, *despite being told to do its best*; most strikingly, Sonnet/Opus still sandbag for a *trained* goal with **no goal given in-context** (Sonnet 96%→61%) — and Sonnet sometimes does so *without externalizing any reasoning in CoT*, the first crack toward undetectable sandbagging.

## References
1. [[2406.07358 - AI Sandbagging (Strategic Underperformance on Evals)]]
2. [[2412.04984 - Frontier Models Capable of In-context Scheming]]
