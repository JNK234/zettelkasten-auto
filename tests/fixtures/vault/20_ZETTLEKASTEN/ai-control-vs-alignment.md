
Created: 03-06-2026 01:56

Status: #baby

Tags: [[ai-safety]] [[ai-alignment]] [[alignment-verification]]

# AI Control vs Alignment

## What This Is
The problem: you have a frontier model that might be *scheming* — quietly misaligned, behaving perfectly while watched, waiting for the right moment to seize power or sabotage you. The intuitive fix is to check whether it's aligned. But that's a trap: "no matter what behaviors you observe, you have to worry that your model is just acting that way on purpose." You can never close the question by looking, because looking is exactly what a schemer games.

So split safety into two orthogonal layers:
1. **Alignment** — make the model *not want* to cause harm (fix its goals so it doesn't scheme).
2. **Control** — make the model *unable* to cause harm even if it wants to and is actively trying to subvert you.

These are orthogonal and **multiplicative**: "in order for scheming models to cause a catastrophe, you need to have failed at both alignment AND control." A lab that cannot verify alignment is not helpless — it can still buy safety by stacking control on top and multiplying the failure probabilities.

## Why It Matters
The split exists because of a **verification asymmetry**. Alignment depends on what the model *wants* — hidden behind behavior, never observable. Control depends only on what the model *can do* — observable and testable. "It seems much easier to become confident that a technique will work if the only property that the technique relies on is that the model can't do something."

This converts an unanswerable question ("is it loyal?") into an answerable one ("could it, trying its hardest, beat this protocol?"). The standard — the Control Property — is that safety measures should "prevent unacceptably bad outcomes even on the assumption the AI is misaligned and trying to subvert them." This is security thinking, not trust-building: you assume an adversary in the weights and design anyway.

## Key Details
- Control is **tractable now**: "no fundamental research breakthroughs are required for labs to implement safety measures that meet their standard for AI control for early transformatively useful AIs." That makes it policy-actionable — you can commit to it today.
- Control is **empirically measured**, not proven: confidence comes from red-teaming many cases and surviving, not from a theorem.
- Control **explicitly does NOT scale to superintelligence** (conceded openly). It relies on actions being human-comprehensible; a wildly superhuman model can take actions that "look fine" but have unpredictable consequences. It's a bridge, not a destination — "building transformatively useful AI while maintaining control seems much safer... than building wildly superintelligent AI."

## Tensions
The threat model is operationalized in [[scheming-as-threat-model]] and measured via [[control-evaluations-red-team-blue-team]]. The whole hierarchy is built from [[trusted-vs-untrusted-models]] — a weaker model trusted purely by capability gap. Note the connection to [[behavioral-vs-representational-alignment]]: control evals are deliberately *behavioral*, making no claim about internal goals. That's its strength (measurable, no mind-reading) and its load-bearing limitation — it can't catch a model that behaves perfectly until the treacherous turn, so the guarantee is only as good as the red team's reach.

# References
1. [[2312.06942 - AI Control - Improving Safety Despite Intentional Subversion]]
2. [[Shlegeris Greenblatt - The Case for Ensuring Powerful AIs are Controlled]]
