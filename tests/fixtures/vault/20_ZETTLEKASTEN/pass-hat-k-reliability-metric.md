
Created: 11-08-2026 19:16

Status: #baby

Tags: [[evaluation-methodology]] [[benchmark-design]] [[agentic-ai-systems]]

# Pass^k Reliability Metric

## What This Is

Two questions look similar and are opposites.

**pass@k** asks: given $k$ attempts, does the agent succeed *at least once*? This measures **discovery** — can the model find the answer if given several tries? It is the right metric when a human filters the outputs, as in code generation where you keep the candidate that compiles.

**pass^k** asks: given $k$ attempts, does the agent succeed *every time*? This measures **reliability** — will it work when nobody is checking. It is the right metric for autonomous deployment.

The two diverge sharply, and the direction is unforgiving. An agent scoring 61% on a customer-service benchmark at a single attempt drops below 25% when all eight attempts must succeed. Same agent, same tasks; the difference is entirely in what question you asked.

## Why It Matters

This reconciles a contradiction teams hit constantly: the benchmark says 60% and users say the product is unreliable. Both are true. A 60% per-step success rate chained across eight subtasks is roughly a coin flip end to end, and users experience the composition, not the step.

For any agentic system that runs unsupervised, pass@k is close to the wrong metric. It reports what the system *can* do; deployment depends on what it *reliably does*. The gap between them is the gap between a demo and a product.

The reliability framing also changes what improvement means. Raising per-step accuracy from 90% to 95% halves the failure rate of a 10-step trajectory — a much larger practical gain than the headline five points suggests. Reliability compounds, and so does its absence.

## Key Details

With $n$ trials of which $c$ succeeded, the unbiased estimators are hypergeometric:

$$\text{pass}^k = \frac{\binom{c}{k}}{\binom{n}{k}}, \qquad \text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

The naive alternative $(c/n)^k$ for pass^k is **biased upward**, by Jensen's inequality — the plug-in estimator of a convex function of a random quantity exceeds the true value. Worked case: $n=4$, $k=2$, $c=2$ gives naive $0.3125$ against unbiased $0.250$, where the true value is $0.25$. The bias is not negligible at small $n$, which is where these metrics are usually computed.

```python
from math import comb

def pass_hat_k(n, c, k):
    """P(all k sampled trials succeed), unbiased."""
    return comb(c, k) / comb(n, k) if c >= k else 0.0

def pass_at_k(n, c, k):
    """P(at least one of k sampled trials succeeds), unbiased."""
    return 1.0 - (comb(n - c, k) / comb(n, k) if n - c >= k else 0.0)
```

Both require multiple trials per task, which is the practical cost — reliability cannot be measured from a single run, and many published agent evaluations report exactly one.

A related design lesson from the same setting: verify the **final state of the world**, not the transcript. Comparing the resulting database state against a golden state has no judge error rate, is immune to persuasive-but-wrong narration, and eliminates evaluator circularity. Where an outcome is mechanically checkable, checking it beats judging text about it.

**Sample-size guidance for the estimators:** roughly $k \geq 10$ for a usable estimate, $k \geq 50$ for a publication-grade one. Since trials are the cost driver, this is the practical reason reliability is under-reported — capability needs one run per task, reliability needs dozens.

## Tensions

The deployment consequence is a hard split in which metric governs which decision: pass@k belongs to *development* (can this architecture solve the task class at all), pass^k belongs to *deployment* (will it hold an SLA). Reporting only the first while shipping on the second is the most common version of this error, and it is why an agent that demos well fails in production without anything having changed.

The compounding mechanism is developed in [[multi-step-compounding-error-in-sequential-tasks]] and [[cascade-failure-in-multi-stage-architectures]] — pass^k is what those failure modes look like when you finally measure them, and [[multi-run-consistency-testing-for-llm-agents]] is the testing practice that produces the trials.

This is the agentic counterpart to the construct question in [[construct-and-operationalization-gap]]: "capability" and "reliability" are two different constructs, and reporting pass@k while claiming deployment-readiness is an operationalization mismatch rather than a statistical error.

It composes with [[human-time-horizon-as-grounded-difficulty-metric]] and [[reliability-tax-80-vs-50-percent-time-horizon]] — the time-horizon literature makes the same move along the duration axis, asking how long a task a system completes at a *chosen* reliability level rather than at a convenient one.

Multiple trials per task means trials are clustered by task, so [[clustered-standard-errors-for-grouped-eval-items]] applies to any aggregate computed over them. Treating $n \times k$ trials as independent observations understates the error substantially.

Open question: pass^k assumes trials are exchangeable draws from a fixed distribution, but an agent in a stateful environment may encounter genuinely different situations across trials, and a user-simulator in the loop introduces its own error rate — reported at 16–47% in one such benchmark, with 6–13% critical. Part of a measured pass^k gap is simulator noise rather than agent unreliability, and separating the two is unsolved.

# References

1. [[2406.12045 - Tau-Bench (Tool-Agent-User Interaction)]]
