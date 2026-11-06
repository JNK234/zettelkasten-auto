Created: 24-09-2026 12:10

Status: #baby

Tags: [[evaluation-methodology]] [[llm-as-judge]] [[benchmark-design]]

# Rubric Reliability Comes From Exemplars and Training, Not the Grid Alone

## What This Is
A rubric is a grid of words. It names the criteria and describes each score level: "adequate evidence", "insightful analysis". The popular belief was that handing raters this grid makes their scores agree.

The review of 75 empirical rubric studies found a conditional answer. Rubrics *can* raise scoring consistency, but agreement was higher when four things came together (per source abstract, conclusion 1; §3.1.3):

1. **Analytic** scoring: each criterion scored separately, not one overall impression (holistic).
2. **Topic-specific** criteria that name what *this* task requires, not generic ones that fit any essay.
3. **Exemplars**: anchor examples showing what a response at each level actually looks like.
4. **Rater training**: calibration sessions where raters score, compare, and resolve differences.

A generic holistic rubric given to untrained raters does not reliably deliver agreement. The level words still mean different things to different people until someone ties them to concrete examples.

## Why It Matters
Each of the four has a direct counterpart in LLM-judge and rubric-reward design:

- Analytic → per-criterion yes/no checks instead of one 1–10 score.
- Topic-specific → rubrics generated per prompt instead of one global checklist.
- Exemplars → few-shot scored examples in the judge prompt, one per level.
- Training → calibrating the judge prompt against a human-labelled set and revising it until agreement clears a bar.

The education result predicts that a judge prompt with only level descriptions, and no scored examples or calibration loop, will be the weak configuration. That prediction is my transfer, not something the source tested on LLMs.

## Key Details
- Evidence base: 46 of the 75 studies reported inter-rater reliability at all. Typical exact agreement on complex open-ended tasks was about **55–75%** *(secondary)*.
- The evidence is a vote count, not a meta-analysis. The review reports which conditions went with higher reliability, not a pooled effect size per factor. Few studies compared rubric against no rubric on the same work, so the causal share of each factor is unknown (per source limitations; the last point is the source author's inference).
- Tightening a rubric raises agreement partly by narrowing what raters look at. That is the lever behind reliability and also the risk to validity: [[rubric-agreement-does-not-establish-rubric-validity]].

## Tensions
- [[generic-rubrics-fail-instance-specificity-is-the-mechanism]] is the RL-reward version of factor 2, with a stronger effect: a generic rubric reward did worse than no rubric at all.
- [[instance-specific-rubric-as-reward-function]] and [[observable-atomic-rubric-criteria]] are the ML versions of factors 1 and 2.
- [[implicit-vs-explicit-rubric-aggregation]] and [[stated-weights-can-hurt-holistic-llm-judgment]] complicate factor 1: for LLM judges, holistic scoring over a full criteria list sometimes beats explicit per-criterion arithmetic. Analytic-beats-holistic may not carry over cleanly from humans to models.
- [[rubric-scaffolding-substitutes-for-judge-capability]] asks how much rubric structure helps judges of different strength. The human analogue is whether rater training helps novices more than experts, which this review does not break down.

Cross-batch link: [[criteria-not-level-descriptions-drive-rubric-effects]] (Brookhart) attributes rubric effects to the criteria rather than the level descriptions. Read together: the grid's level wording is the weakest ingredient; criteria, exemplars and training carry the effect.

# References
1. [[Jonsson and Svingby 2007 - The Use of Scoring Rubrics]] (abstract, §3.1.3, Discussion)
