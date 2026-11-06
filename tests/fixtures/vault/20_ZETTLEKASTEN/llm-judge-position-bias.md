
Created: 11-08-2026 19:11

Status: #baby

Tags: [[evaluation-methodology]] [[llm-as-judge]] [[inherent-bias-in-ai-models]]

# LLM Judge Position Bias

## What This Is

Show an LLM judge two candidate answers, A and B, and ask which is better. Then swap them and ask again. A consistent judge gives the same winner both times. Many judges do not.

Position bias is the tendency to favour a response because of *where it appears in the prompt*, independent of quality. It is the simplest judge failure to test for and one of the largest in magnitude — and because a single unswapped evaluation looks perfectly reasonable, it is invisible unless you deliberately go looking.

The measurement is trivial: run every comparison twice with the order reversed, and record the fraction of pairs where the verdict flips. That flip rate is the bias.

## Why It Matters

The magnitudes are not subtle. Consistency under swapping ranges from roughly 24% for weaker judges to 65% for stronger ones — meaning that for some judge models, **the majority of pairwise verdicts reverse when you swap the order**. A leaderboard built on unswapped pairwise comparisons from such a judge is close to arbitrary.

The property that makes this genuinely dangerous: position bias is worst **when the two answers are close in quality**. When one response is obviously better, the judge is stable. When they are near-tied, order decides. That is precisely the regime where you need the judge most — distinguishing two similar frontier models — and precisely where it is least reliable.

Bias also concentrates in subjective categories. On objectively checkable dimensions it is modest; on writing quality, helpfulness, and style it dominates.

## Key Details

The clean experimental design separates two confounded effects. Naively swapping response *positions* also swaps which model is named where, so a rename ablation is needed to distinguish position bias proper from **name bias** — a preference for a recognizable model identifier. Both exist and they are separable.

```python
def flip_rate(judge, pairs):
    """Fraction of comparisons whose verdict reverses when order is swapped."""
    flips = 0
    for a, b in pairs:
        v1 = judge(a, b)            # A first
        v2 = judge(b, a)            # B first
        # v2 is expressed in swapped coordinates; invert before comparing
        if v1 != ("B" if v2 == "A" else "A" if v2 == "B" else v2):
            flips += 1
    return flips / len(pairs)
```

The standard mitigation is conservative and cheap: **evaluate both orders and count a win only when both agree**; otherwise record a tie. This costs two calls per comparison and converts an unmeasured bias into an explicit tie rate, which is far more honest than a coin flip presented as a verdict.

Three sibling biases travel with this one. **Verbosity bias** — a repetitive-list attack that adds no information fools judges roughly 91% of the time. **Self-enhancement bias** — judges favour their own generations by around 10–25%, with two competing explanations in the literature (self-recognition versus mere familiarity/perplexity). **Limited reasoning** — judges are weak graders of math and logic, which is a capability limit rather than a bias.

## Tensions

Position bias is the consistency half of judge validation; [[judge-agreement-against-human-ceiling]] is the calibration half. They are independent failures — a judge can agree with humans at the ceiling on average while flipping half its near-tied verdicts, and average agreement conceals exactly that.

The verbosity variant is the cleanest bridge to optimization pressure: if a judge prefers length, any policy trained against it learns verbosity. That is [[reward-hacking-as-eval-false-positive]] arriving through the judge rather than through the task, and it is why judge bias is a training problem and not only a measurement problem.

Bias also compounds with [[grader-failure-routing]] — a swap-inconsistent verdict and an unparseable grade are different failures that should be recorded differently, and collapsing both into a score corrupts the aggregate in opposite directions.

An open question: the two-call mitigation assumes the tie is the honest answer when orders disagree, but a systematic quality difference may genuinely exist below the judge's resolution. Recording it as a tie is conservative for ranking and lossy for measurement, and there is no established way to recover the signal.

# References

1. [[2306.05685 - Judging LLM-as-a-Judge (MT-Bench)]]
