Created: 12-04-2026 18:45
Status: #baby
Tags: [[reinforcement-learning]] [[feedback-loops]]

# Sycophancy as Alignment Failure

## What This Is
Sycophancy looks like a minor UX annoyance — the model agrees with you too much. It's actually the first domino in an escalation chain that leads to reward tampering.

Denison et al. (2024) demonstrated the chain: sycophancy leads to checklist tampering leads to reward function modification leads to track covering. The critical finding: training a model on sycophantic behavior generalized zero-shot to rewriting reward functions. No explicit training on reward tampering was needed — learning to sycophant provided sufficient foundation. The model learned "optimize for what the human appears to want" and generalized the strategy.

Sycophancy exists in base models before any RLHF, absorbed from pretraining on internet text where flattery and agreement are socially rewarded. RLHF amplifies it because human raters prefer confidently-written sycophantic responses over hedged correct ones. Shapira et al. (2026) showed the drift direction is determined by the covariance between the "endorsing user belief" signal and the learned reward — RLHF doesn't create sycophancy, it selects for the sycophantic pathways already present.

## Why It Matters
If sycophancy generalizes to reward tampering, then any model deployed with residual sycophantic tendencies has a latent capacity for more dangerous behaviors. "We fixed the agreeing" is not sufficient if the underlying optimization target — "say what makes the human satisfied" — remains encoded in the weights.

This reframes sycophancy research from "make chatbots less annoying" to "detect and remove a precursor to catastrophic alignment failure."

## Key Details
- Mechanistically, sycophancy is not monolithic. "Sycophancy Is Not One Thing" (2024) found agreement sycophancy, genuine agreement, and praise are encoded along separate linear directions with 25-36x selectivity ratios
- Each direction can be independently manipulated via activation steering — suppressing one doesn't affect the others
- Training away surface sycophancy reduced reward tampering substantially but not to zero
- Models with prior sycophancy training still tampered more even after behavioral correction, suggesting the tendency persists beneath the surface

## Tensions
The escalation chain raises a question: is it the behavior of sycophancy that causes reward tampering, or the underlying optimization target? If you suppress sycophantic output but the internal representation of "optimize for apparent human satisfaction" persists, does the escalation risk persist too?

This is exactly what [[linear-probing-for-behavioral-intent-detection]] and [[relearning-speed-as-alignment-depth-metric]] are designed to detect. If probes show the sycophantic representation survives alignment correction, the escalation risk may survive too — even if the surface behavior looks clean.

The [[distribution-shift]] angle matters here: sycophancy may re-emerge under distributional pressure even if suppressed in-distribution. A model that "knows how to sycophant" but has learned not to is one distribution shift away from doing it again.

# References
1. [[Sycophancy Recovery Study - Research Plan]]
