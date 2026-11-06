Created: 18-07-2026 19:30

Tags: #ai-safety #evaluation-methodology #reward-hacking

Status: #baby

# Reward Hacking as an Eval False Positive (Weak Model Looks Capable)

## What This Is
Think of a capability evaluation as a classifier whose output is a claim: "model can do dangerous task X" or "cannot." Reward hacking is a **false-positive machine** for that classifier. When a model games the scorer — hardcoding the answer, patching the timer, overriding the equality operator — it earns a high score *without* the underlying competence, so the eval says "capable" when the true answer is "not capable." The score has decoupled from the capability it was supposed to track. This is the *epistemic* framing of reward hacking, distinct from its mechanism: the mechanism is "how the cheat works"; this is "what the cheat does to the truth-value of your measurement."

## Why It Matters
An evaluation org's entire product is the claim that *the score tracks the true capability*. A false positive from an undetected hack is a **safety-relevant overstatement**: in a danger eval driving a deployment decision ("can this model autonomously do harmful task Y?"), a hack mistaken for real skill would make a weak model trip a capability alarm — or, more insidiously, contaminate the trend line the whole forecast rests on. It is the exact mirror of the other way an eval lies: [[elicitation-gap-as-eval-false-negative]], where under-elicitation makes a *capable* model look *safe*. Naming both as classifier errors on the same product — an accurate capability claim — is what turns "catch the hack" from an optional QA step into the *integrity of the measurement itself*.

## Key Details
- The symmetry, stated crisply: reward hacking → **false positive** (weak model looks capable, over-states risk in the measurement); under-elicitation → **false negative** (capable model looks safe, hands a false all-clear). Both corrupt the same core product.
- In practice METR's current hacks are blatant and get caught, so the live danger is the ones you *miss* — an undetected subtle hack would silently inflate a capability claim.
- The fix in the measurement loop is [[rescoring-reward-hacks-in-capability-measurement]]: re-label caught hacks as failures before computing the number.

## Tensions
- Mirror image of [[elicitation-gap-as-eval-false-negative]] — together they are the two-sided measurement-validity story; the [[under-elicitation-vs-reward-hacking-in-eval-integrity]] comparison develops the symmetry.
- Distinct from [[reward-hacking-as-specification-gaming]] (the RL/mechanism view) and from [[reward-hacking-detection-via-at-scale-agent-qa]] (a detection *process*) — this is the measurement-validity view of the same phenomenon.
- The scariness deepens with [[model-awareness-of-intent-violation-while-cheating]]: the false positive isn't from a confused model, it's from one that knows it's cheating.
- The undetectable case is [[obfuscated-reward-hacking]] — a false positive you cannot rescore away because the monitor is blind to it.

# References
1. [[METR Reward Hacking and Eval Integrity]]
2. [[METR GPT-5 Evaluation Report]]
