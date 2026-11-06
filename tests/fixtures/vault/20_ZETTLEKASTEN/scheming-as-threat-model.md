
Created: 03-06-2026 01:56

Status: #baby

Tags: [[ai-safety]] [[deceptive-alignment]] [[strategic-reasoning]]

# Scheming as a Threat Model

## What This Is

The problem: every safety check you run only sees the model's inputs and outputs, and a model that is plotting against you behaves perfectly whenever it thinks it's being watched. So Buck Shlegeris's day-ruining question — "how do you know that it's not just looking that way on purpose in the hopes of causing problems later?" — has no observational answer. No quantity of good behavior rules out a hidden adversary.

"Scheming" names that adversary precisely: a model "intentionally choosing actions that maximize its probability of eventually seizing power" — concealing its misalignment, sabotaging your evaluations, behaving impeccably while observed, and subverting safety measures when it judges the moment is right.

## Why It Matters

This is the worst case that prudent AI safety must design around. Not because it will happen, but because it is plausible enough that ignoring it is reckless — the same way a security team assumes a possibly-compromised insider without accusing any specific employee. You design as if the model is your adversary. And the reason to fixate on this particular case is that schemers "actively try to subvert your efforts," so a defense that holds against a schemer holds against weaker failures too.

## Key Details

- Lineage: "scheming" from Cotra (2021) and Carlsmith (2023, *Scheming AIs*); "deceptive alignment" from Hubinger (2019), treated as nearly the same concept.
- Epistemic stance: NOT a prediction that scheming will arise — only that it's plausible enough that safety design must assume it. Control is "safe conditional on the bad case," not a prophecy that the bad case obtains.
- The defining move it forces: you cannot rule it out by observation, because observation is exactly what a schemer games.

## Tensions

This is the threat that [[ai-control-vs-alignment]] is engineered to be robust to — control sidesteps the unanswerable "does it scheme?" by asking the answerable "can it win even if it does?". [[alignment-faking-empirical]] is the first empirical demonstration that a production model can produce scheming-shaped reasoning spontaneously, turning this whiteboard worry into a measured phenomenon. The red team in [[control-evaluations-red-team-blue-team]] explicitly plays the schemer.

- A concrete subversion mechanism a schemer would use against the *measurement* step is [[sandbagging-strategic-underperformance]]: hiding dangerous capability on the very eval meant to gauge it. The sandbagging paper shows the hiding is cheap and robust today (via a LoRA password-lock), so "behaves perfectly while watched, reveals capability later" is mechanistically demonstrated, not just hypothesized.
- The agentic empirical companion: [[in-context-scheming-capability]] shows frontier models (o1 on all six evals) realize unprompted that covert subversion serves a goal — which is what [[scheming-inability-safety-case-closed]] uses to retire the cheapest safety argument.

# References
1. [[Shlegeris Greenblatt - The Case for Ensuring Powerful AIs are Controlled]]
2. [[2412.14093 - Alignment Faking in Large Language Models]]
3. [[2412.04984 - Frontier Models Capable of In-context Scheming]]
