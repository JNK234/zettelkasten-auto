
Created: 11-08-2026 19:19

Status: #baby

Tags: [[evaluation-methodology]] [[benchmark-design]] [[experimental-design]]

# Label Noise Dominates Saturated Benchmarks

## What This Is

A frontier model scores 95% on a benchmark. The natural question is what the remaining 5% reveals about its limitations. Often the answer is: nothing about the model at all.

When benchmarks are audited item by item, a large share of the residual failures turn out to be **label errors** — the ground truth is wrong, ambiguous, or the question admits multiple defensible answers. On a widely used grade-school math benchmark, roughly **48%** of the remaining errors traced to label problems rather than model incompetence.

The consequence is a hard ceiling that is not a capability ceiling. Past a certain accuracy, you are no longer measuring the model. You are measuring your annotation quality, and the two are indistinguishable from the score alone.

## Why It Matters

This changes what a saturated benchmark means and what to do about it. The instinct on seeing 95% is to conclude the task is nearly solved and move to a harder benchmark. The correct first move is to **audit the failures**, because the answer determines two completely different actions: if the residual is label error, the benchmark is exhausted and further optimization against it is fitting noise; if it is genuine failure, there is real signal left.

It also explains a phenomenon that otherwise looks like model weirdness. Models that appear to fail on "easy" items are frequently answering correctly against a wrong key. Cleaned versions of standard benchmarks show accuracies around 90%, 93%, and 75% on task subsets where the raw numbers suggested worse — the models were better than the labels indicated.

The strategic implication for anyone producing benchmarks: annotation quality becomes the binding constraint on benchmark lifespan. A carefully labelled benchmark stays informative longer, which is a durable advantage rather than a one-time cost.

## Key Details

The failure taxonomy for a residual error, applied in order:

1. **Wrong ground truth.** The key is simply incorrect.
2. **Ambiguous question.** Multiple defensible answers; the key picks one arbitrarily.
3. **Format mismatch.** The answer is right, the extraction or comparison logic rejects it. This overlaps [[grader-failure-routing]] — an instrument failure charged to the model.
4. **Genuine failure.** The only category that carries information about capability.

Detection approaches, none of them complete: manual audit of every failure (reliable, expensive, and the only method that actually works); disagreement mining across many models, since an item that *every* strong model fails is more likely mislabeled than universally hard; and negative-discrimination screening from item response theory, which flags items where stronger models do worse — empirically a triage screen with mediocre recall (5 of 10 planted errors) rather than a detector.

A reporting practice that follows directly: when a benchmark is near saturation, publish the **audited residual breakdown** alongside the score. "95%, of which the 5% residual is 48% label error" is a substantively different claim from "95%," and only the first supports any statement about remaining capability gaps.

## Tensions

This is the terminal condition for [[tinybenchmarks-irt-compression]]'s information framing: as item information decays toward zero with rising model ability, the proportion of the measured signal that is annotation noise rises correspondingly. The two notes describe the same decay from different ends — one from the item side, one from the residual side.

It is a [[construct-and-operationalization-gap]] failure in the small: each mislabeled item is a local mismatch between the construct and the operationalization, and enough of them collectively shift what the aggregate measures.

There is a genuine tension with [[rephrasing-defeats-ngram-decontamination]]. Contamination pressure argues for continuously refreshed benchmarks; label quality argues for carefully curated, stable ones. Fresh items are less contaminated and less well validated; mature items are better labelled and more likely memorized. There is no configuration that optimizes both, and the honest position is that this is a standing tradeoff rather than a solved problem.

Open question: label auditing does not scale, and the disagreement-mining shortcut has an obvious circularity — using model consensus to identify label errors assumes models are more reliable than annotators, which is the very question the benchmark exists to answer.

# References

1. [[Benchmark Efficiency and Saturation - IRT and Label Noise]]
