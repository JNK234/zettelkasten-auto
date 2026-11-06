
Created: 11-08-2026 19:14

Status: #baby

Tags: [[evaluation-methodology]] [[llm-as-judge]] [[experimental-design]]

# Binary Judge Scoring Over Likert

## What This Is

The instinct when building a judge is to ask for a graded score — rate this response 1 to 5. The better practice is to ask a binary question: does this output do X without doing Y, yes or no.

The argument is not that nuance does not exist. It is that a 1-to-5 scale manufactures *false* nuance. Nobody can articulate what separates a 3 from a 4, so the boundary is decided differently by every rater and by the same rater on different days. The scale adds four decision boundaries where the underlying judgment supports one.

When a middle grade feels genuinely necessary, that is a signal the criterion is compound — it is secretly two questions. Split it into two binary criteria rather than blurring them into a scale.

## Why It Matters

Three independent advantages, each sufficient on its own:

**Actionability.** A pass/fail verdict names something to fix. A 3.7/5 does not. The deliverable of judge work is a failure taxonomy — what breaks and why — and a graded score obscures the taxonomy it should reveal.

**Lower labelling variance.** Human annotators agree far more on binary calls than on scale positions, which raises the human-human ceiling that any judge is measured against. A noisy label scale caps how good a judge can be shown to be.

**Statistical efficiency.** A binary outcome needs fewer samples to establish a significant difference than a continuous one with rater noise dominating its variance. Fewer labels, more power.

The related discipline: generic library metrics — `helpfulness`, `coherence`, `answer_relevancy` — are worse than no metric. They are unvalidated, but they *look* like measurement, so they discharge a team's felt obligation to examine data. A team with no metric knows it is flying blind; a team with an unvalidated 4.2/5 does not.

## Key Details

The operational loop, in order:

1. Write one task-specific criterion in plain English, as a binary question.
2. Name **one** decider — a single principal domain expert who resolves boundary cases. Consensus-seeking across multiple experts produces drift rather than accuracy, because it averages incompatible interpretations.
3. Build the dataset on a dimension grid (feature × scenario × persona). Synthesize the **inputs** only; responses must come from the real system. Start around 30 items, expand until no new failure mode appears.
4. Label by hand with a written critique for each verdict — detailed enough for a new hire to reproduce the call.
5. Split labels into a few-shot set and a held-out validation set with no overlap.
6. Build the judge prompt: context, guidelines, expert critiques as few-shot examples, critique-before-verdict ordering, structured output. Use a strong model from a **different family** than the system under test.
7. Validate on the held-out set, reporting TPR and TNR separately.
8. Downgrade to a code assertion wherever the failure mode is deterministically checkable.

Step 8 is the highest-leverage one and the most skipped. A judge is a fallback for constructs that resist mechanical checking, not a default.

## Tensions

Binary scoring is what makes the validation in [[judge-agreement-against-human-ceiling]] tractable — TPR and TNR are defined for binary verdicts, and precision/recall decomposition is unavailable for a five-point scale without arbitrary thresholding.

It interacts with [[criteria-drift-in-evaluation]] in a useful direction: forcing a binary boundary surfaces criterion ambiguity immediately, where a graded scale absorbs the ambiguity into the middle of the range and hides it. Binary scoring makes drift visible rather than preventing it.

The single-decider rule is in genuine tension with [[expert-driven-evaluation-for-safety-critical-ai]], where multiple expert perspectives are precisely the point. The reconciliation is scope: one decider per *criterion* keeps the boundary consistent, while different criteria can and should have different owners.

Open question: the method assumes a single expert's judgment is the right target. For contested constructs — where reasonable experts genuinely disagree, per [[construct-and-operationalization-gap]] — that assumption picks a winner by fiat rather than resolving the disagreement, and the resulting judge encodes one view of a construct without recording that it did so.

# References

1. [[Husain - LLM-as-a-Judge Complete Guide]]
