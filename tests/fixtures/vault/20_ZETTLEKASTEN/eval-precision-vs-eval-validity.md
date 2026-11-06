
Created: 11-08-2026 19:06

Status: #baby

Tags: [[evaluation-methodology]] [[experimental-design]] [[benchmark-design]]

# Eval Precision vs Eval Validity

## What This Is

Two entirely different things can be wrong with a measured number, and confusing them wastes enormous effort.

**Precision** asks: if I ran this again, would I get the same number? Noise, sample size, seed variance, confidence intervals. The failure is a number that wobbles.

**Validity** asks: is this number measuring the thing I claim? Construct choice, operationalization, contamination, judge bias. The failure is a number that is stable and wrong.

The critical property is that **they fail independently**. A score can be measured to three decimal places with a tight interval and still be entirely the wrong quantity. Conversely a measurement can capture exactly the right construct while being far too noisy to support any claim.

The trap is that more data fixes only one of them. Collecting 10× the samples shrinks the confidence interval; it does nothing whatsoever to a contaminated benchmark, a biased judge, or a mis-specified construct. Scale amplifies a validity failure — it makes the wrong number more confident.

## Why It Matters

Almost all published methodology attention goes to precision, because precision is mechanically checkable. Validity requires an argument, and arguments are harder to referee than error bars.

This produces a predictable pathology: papers that report standard errors on quantities nobody has shown to be meaningful. A tight CI around a reward-hacked score is *precisely wrong*. A well-powered comparison on a contaminated benchmark measures memorization with admirable rigor.

The diagnostic discipline is to run both checks and to know which one a given fix addresses:

- More items, more seeds, paired comparisons, clustered errors → **precision**
- Held-out variants, judge calibration against humans, discriminant correlation, decontamination → **validity**

Confusing the two is how a team spends a month tightening intervals on a benchmark that was never measuring the target.

## Key Details

**Precision toolkit.** Standard errors under a super-population assumption; clustered standard errors when items are grouped ([[clustered-standard-errors-for-grouped-eval-items]]); paired differences when two systems see the same items ([[paired-difference-model-comparison]]); power analysis before running to establish what effect size is even detectable ([[statistical-power-for-eval-comparisons]]); permutation nulls when the statistic is not a mean ([[permutation-tests-for-probe-significance]]).

**Validity toolkit.** The construct/operationalization audit ([[construct-and-operationalization-gap]]); judge calibration against human labels with the human-human ceiling as the reference point ([[judge-agreement-against-human-ceiling]]); discriminant checks against nuisance variables; contamination detection ([[rephrasing-defeats-ngram-decontamination]]); reliability-versus-capability separation ([[pass-hat-k-reliability-metric]]).

A useful ordering heuristic: **validity first, precision second**. Establishing that a measurement is noisy is only worth doing once you believe it measures the right thing. Teams routinely invert this and optimize the error bars on a meaningless quantity.

## Tensions

The clean division of labour: [[construct-and-operationalization-gap]] supplies the validity half, and the standard-error machinery in [[2411.00640 - Adding Error Bars to Evals]] supplies the precision half. Both are required and neither substitutes for the other.

[[reward-hacking-as-eval-false-positive]] and [[elicitation-gap-as-eval-false-negative]] are the two validity failures that error bars provably cannot touch — the score is a faithful measurement of the wrong thing in both directions.

There is a genuine tension with reproducibility practice worth naming: [[temperature-zero-deterministic-evaluation]] improves precision by eliminating sampling noise, but a temperature-zero measurement is an estimate of a different quantity than the one users experience at deployment temperature. Chasing precision here trades away a piece of validity.

An open question the split raises: validity has no summary statistic. Precision compresses to a single interval that referees can check; validity is an argument that must be read. Whether the field can build validity checks that are as mechanically auditable as a confidence interval is unresolved, and the absence is why validity failures persist longer than precision failures.

# References

1. [[2411.00640 - Adding Error Bars to Evals]]
2. [[1912.05511 - Measurement and Fairness]]
