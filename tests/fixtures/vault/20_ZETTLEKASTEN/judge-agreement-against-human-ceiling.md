
Created: 11-08-2026 19:10

Status: #baby

Tags: [[evaluation-methodology]] [[llm-as-judge]] [[benchmark-design]]

# Judge Agreement Against Human Ceiling

## What This Is

When an LLM judge agrees with human labels 85% of the time, the instinctive reading is "15% short of perfect." That reading is wrong, and it makes a good judge look mediocre.

The correct reference point is not 100%. It is **how often two humans agree with each other on the same task**. If independent human annotators agree only 81% of the time, then 81% is the ceiling — the task itself contains irreducible ambiguity, and no rater, human or model, can exceed it without overfitting to one annotator's idiosyncrasies.

Against that ceiling, an 85% judge is not falling short. It is agreeing with humans slightly *more often than humans agree with each other*.

Four numbers make the scale legible: random guessing sits near 50%, human-human agreement at 81%, the GPT-4 judge at 85%, and human-majority agreement at 90%. The judge closes roughly 44% of the gap between an individual human and the majority vote.

## Why It Matters

Reporting agreement against 100% systematically understates judge quality and leads teams to reject usable instruments. More damagingly, it hides the real question: a judge scoring *above* the human ceiling should raise suspicion, not celebration, because it likely means the judge has learned an annotator's particular quirks rather than the underlying construct.

The ceiling also tells you when to stop. If human-human agreement on your task is 70%, chasing a 90% judge is chasing noise — you would be fitting disagreement that reflects genuine ambiguity in the construct itself. That is a signal to sharpen the criterion, not to tune the judge.

Establishing the ceiling costs one extra annotation pass: have two humans label the same subset independently. Teams routinely skip it and then have no denominator for any claim about the judge.

## Key Details

Raw agreement is the wrong summary statistic once classes are imbalanced, because agreeing by always predicting the majority class inflates it. **Cohen's κ** corrects for chance agreement:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is observed agreement and $p_e$ is agreement expected by chance given each rater's marginal rates.

The gap this opens is dramatic. Two judges can post 95% and 96% raw agreement and separate into $\kappa = 0.00$ and $\kappa = 0.65$ — the first is doing nothing beyond exploiting a skewed base rate, the second is genuinely tracking the labels. Raw agreement cannot distinguish them.

For more than two raters use Fleiss' κ; for missing data or mixed measurement scales use Krippendorff's α, which is the right default for messy production pipelines.

For a binary judge, report **TPR and TNR separately** rather than a single agreement figure. The headline pass rate can then be corrected for judge error:

$$\text{Actual} = (\text{JudgePosRate} \times \mathrm{TPR}) + \bigl((1 - \text{JudgePosRate}) \times (1 - \mathrm{TNR})\bigr)$$

Any agreement number reported without stating the class balance is uninterpretable.

## Tensions

This is the calibration half of judge validation; [[llm-judge-position-bias]] is the bias half. A judge can be well-calibrated on average and still flip its verdict when candidate order is swapped, so agreement and consistency must be measured separately.

The ceiling framing connects to [[construct-and-operationalization-gap]]: low human-human agreement is direct evidence that the construct is under-specified, or contested. A ceiling of 60% is telling you that the operationalization needs work before any judge can be evaluated against it.

[[criteria-drift-in-evaluation]] complicates the whole procedure — if the criterion is still shifting as labels accumulate, early and late human labels are not measuring the same thing, and the computed ceiling is a mixture.

An honest limitation: the ceiling is estimated from a finite sample and carries its own uncertainty, which is almost never reported. Comparing an 85% judge against an 81% ceiling when both have several points of sampling error is a comparison of two noisy quantities, and the conclusion "the judge exceeds the ceiling" may not survive a confidence interval on either.

**The word "ceiling" is wrong, and this is a substantive correction, not a quibble.** [[inter-annotator-agreement-is-not-a-performance-ceiling]] shows by simulation that models routinely exceed IAA under ordinary conditions — realistic agreement levels, misspecified models, even models handicapped to annotator noise. The mechanism: annotator-vs-annotator is noise against noise (two independent coin flips per ambiguous item), while model-vs-annotator is signal against noise, because fitting averages the noise away. IAA *predicts* model performance (r ≈ 0.48) but does not *bound* it, and conflating a correlate with a constraint is what kept the folk theorem alive despite zero supporting evidence and 20 published counterexamples.

Three consequences for the claims above. First, a judge scoring *above* the human rate is **not** automatically overfitting to annotator quirks — that reading is one of two possibilities, the other being that your annotations are noisy, and [[pairwise-decomposition-over-pooled-panel-agreement]] recommends investigating rather than clamping. Second, "if human-human agreement is 70%, chasing a 90% judge is chasing noise" is too strong; it is chasing noise only if the disagreement is genuine construct ambiguity rather than stochastic labeling, and [[human-label-variation-versus-annotation-error]] is the distinction nobody can operationalize. Third, a single pooled agreement figure is the wrong object regardless — agreement measured 0.96 on violence and 0.25 on personal topics in the same study, so the ceiling must be reported per slice ([[descriptive-versus-prescriptive-annotation-paradigms]]).

What survives intact: the reference point is not 100%, raw agreement is misleading under class imbalance, and reporting TPR/TNR separately beats a single figure. For the replace-an-annotator decision specifically, [[alternative-annotator-test]] gives a hypothesis test instead of a threshold judgement. Full map in [[evaluation-under-weak-ground-truth]].

# References

1. [[2306.05685 - Judging LLM-as-a-Judge (MT-Bench)]]
2. [[Husain - LLM-as-a-Judge Complete Guide]]
