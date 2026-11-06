Created: 24-09-2026 12:09

Status: #baby

Tags: [[llm-as-judge]] [[statistical-validation]] [[evaluation-methodology]]

# Grader Validation Covers Only the Consensus Slice

## What This Is
A rubric benchmark is only as trustworthy as the model that grades it. The standard move is meta-evaluation: have humans grade some items, grade the same items with the model, and check that they agree. But checking agreement needs *two or more* human labels per item, and double-labeling is expensive. So validation tends to happen on whichever slice was cheap to double-label. That slice is rarely the typical case.

HealthBench is a clear example. Its grader was validated on 60,896 physician judgments, but **only for the 34 consensus criteria**, which are 14% of all criteria. The other 86%, the example-specific criteria that make up most of the score, were each written by one physician, never checked by a second, and the grader's accuracy on them was never measured (per source §Is the grader trustworthy?).

## Why It Matters
"The grader matches physicians" is a claim about the whole benchmark, but the evidence covers only one slice. And that slice is probably the *easy* case: consensus criteria were deliberately written to be generic and "objectively gradable". Idiosyncratic, one-physician criteria are where grader errors are most likely. (The ease ranking is the source note's inference, not a measured result.)

When designing any rubric eval or rubric reward, the question to ask is: *which criteria did the grader validation actually sample, and do they look like the criteria that drive the score?*

## Key Details
- **Meta-example:** (criterion, conversation, response, one physician's met/not-met). There are 1,072 to 3,370 per consensus criterion.
- **Metric:** macro-F1 over met/not-met, per criterion, $MF1 = \tfrac12(F1_{pos} + F1_{neg})$. Macro weighting stops "always say met" from scoring well. A random grader at the base rate scores 0.50.
- **Result:** GPT-4.1 as grader beats the average physician in 5 of 7 themes and sits above the 33rd physician percentile everywhere. Grader ranking: GPT-4.1 0.709 > o4-mini 0.692 > o3 0.681 > 4.1-mini 0.661 > 4.1-nano 0.580. The reasoning models are not better graders.
- **Low human ceiling:** physicians agree with each other at MF1 of about 0.57 to 0.73 by theme, where 0.5 is chance.
- **Tuned on the validation set.** The grading prompt was searched, and consensus criteria were *reworded* to make intent "unmistakable to the grader", using GPT-4.1. There was no held-out meta-eval split. The authors note that GPT-4.1 topping the list "may be partially explained by" this. So 0.709 is an optimistically biased estimate.
- **Missing:** confidence intervals on the percentiles (some themes have only 32 to 34 physicians), and any cross-vendor grader check.

## Tensions
- Instance of [[llm-as-automated-evaluator-with-validation]], showing its typical gap: the validated slice is not the scored slice.
- The "as good as physicians" framing is [[judge-agreement-against-human-ceiling]]. [[inter-annotator-agreement-is-not-a-performance-ceiling]] and [[human-label-variation-versus-annotation-error]] warn that low physician agreement may be real disagreement, not noise to match.
- Tuning the grader and criteria wording on the same data used to validate them is the eval-side version of [[shared-judge-family-across-training-and-eval]].
- Low run-to-run variance (std about 0.002 over 16 runs) does not help here. Systematic grader bias repeats identically on every run. [[verifier-failure-vs-rubric-design-failure]] is the useful split for what the missing 86% could hide.
- Contrast with [[rubrics-reformat-expert-signal-they-do-not-create-it]]: if each example-specific criterion is one physician's view, grader accuracy on it is bounded by that one author's coverage, which no meta-eval measured.

# References
1. [[2505.08775 - HealthBench]] (§Is the grader trustworthy? The meta-evaluation; §Consensus criteria)
