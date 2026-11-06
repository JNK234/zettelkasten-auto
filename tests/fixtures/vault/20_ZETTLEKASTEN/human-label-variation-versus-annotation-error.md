Created: 15-09-2026 00:32

Status: #baby

Tags: [[evaluation-methodology]] [[labelled-dataset-creation]] [[experimental-design]] [[knowledge-representation-in-neural-networks]]

# Human Label Variation Versus Annotation Error

## What This Is

When two annotators give different labels, the field has one word for it — disagreement — and one response: aggregate it away. Barbara Plank's argument is that this collapses two phenomena that need opposite treatment.

**Annotation error** is what it sounds like: attention slips, misread guidelines, fatigue, bad-faith crowdworkers. The difference is a mistake and you want it gone.

**Human label variation (HLV)** is *plausible* variation. It assumes annotators are giving their best judgement and the difference comes from something real in the data:

- **ambiguity of the instance** — the text genuinely supports two readings
- **uncertainty of the annotator** — honest epistemic hedging on a hard case
- **genuine disagreement** — competent people with different priors or values
- **multiple options simply being correct** — the label set does not carve the instance uniquely

Her running example: *Q: "Hey. Everything ok?" A: "I'm just mad at my agent."* Yes or no? Two competent annotators land differently and neither is wrong.

The naming is not pedantry. "Disagreement" implies one view must lose. "Variation" does not pre-commit you to adjudicating. And "noise" actively mislabels the phenomenon as something to remove.

**Aggregation collapses both arrows into the same place.** Majority vote cannot distinguish "three annotators, one was sloppy" from "three annotators, the item is genuinely bimodal." It produces the same output in both cases while destroying the evidence that would let you tell them apart.

## Why It Matters

The scale is not marginal. Pavlick and Kwiatkowski (2019) re-annotated ~500 RTE inference items with 50 annotators each and found plausible, irreconcilable variation in **at least 20%** of the data. That is a fifth of the dataset where the gold label is a coin flip presented as truth.

The consequence runs through all three pipeline stages. The **data** you release encodes a single ground truth and discards everything else humans told you. The **model** is optimized toward a single preferred output. The **evaluation** scores against that single label, so it cannot tell you whether the model is being reasonable when humans themselves are split. Plank's phrasing: we might ask whether we are climbing the right hill.

It also inverts a standard quality signal. Low inter-annotator agreement is normally read as "bad data, fix your guidelines." Under HLV, low agreement on a hard subset can be **the most informative signal in the dataset** — it marks exactly the items where the interesting structure lives.

## Key Details

**Aggregation is a modeling commitment, not a cleaning step.** It silently fixes the target as "the mode of the human judgement distribution" — a choice nobody writes down and everybody inherits, made before any modeling decision has been taken.

Dawid-Skene (1979) is the sophisticated version: EM estimation of per-annotator confusion matrices, `π^(j)_kl = P(annotator j says l | true label is k)`, alternating between estimating latent labels and re-estimating confusion matrices. It is strictly better than majority vote when annotators differ in reliability. But note what it *assumes*: a single latent true label per item. **It is a better estimator of a quantity that, under HLV, does not exist.** The sophistication of the aggregation method is orthogonal to whether aggregation is the right operation.

**Filtering is worse.** Delete the low-agreement instances and Jamison & Gurevych (2015) found training can get *worse* — the hard cases carry signal. You are systematically deleting the subset where the phenomenon lives, and your test set then over-represents easy items.

The alternative family — learning from soft labels (Peterson et al. 2019) — replaces the one-hot target with the empirical human distribution and trains cross-entropy against it:

$$\mathcal{L} = -\sum_{k} p_{\text{human}}(k \mid x) \log p_\theta(k \mid x)$$

One-hot is the special case where `p_human` is a point mass. Every term the hard-label version throws away is a gradient signal: if humans split 60/40 entailment/neutral, the soft loss penalizes a confidently-entailment model, while hard cross-entropy rewards it.

**The self-inflicted wound.** Plank's sharpest observation is about the field's own behaviour: *a majority of papers introducing HLV methods evaluate against hard gold labels.* Researchers build a model that learns the human distribution, then score it by argmax accuracy against majority vote. The evaluation **structurally cannot reward** what the method was built to do. A field can spend a decade building the right methods and never detect the benefit, because the instrument is blind to the effect.

Soft evaluation metrics that do work: cross-entropy against the human distribution; **entropy correlation** (Uma et al. 2020), which measures whether the model knows *which items are hard* rather than which label is right — the quantity you need for triage, routing, and abstention; and KL/JS divergence, which differs from cross-entropy by subtracting the irreducible human entropy so a model is not penalized for genuinely split items.

Cheaper than new metrics: **split the test set.** Report per annotator-agreement level. Split by annotator cluster to surface *whose* view the model adopted. Split by annotator uncertainty flags. All of these are free if and only if you kept the un-aggregated labels — which is why the data-stage decision to aggregate forecloses the modeling and evaluation options three stages later. You cannot retrofit it.

**Calibration inherits the broken referent.** In contemporaneous work (Baan, Aziz, Plank & Fernandez, EMNLP 2022), measuring calibration against the human majority under inherent disagreement is theoretically and empirically problematic. A model that correctly reproduces a 60/40 human split is scored as *miscalibrated*. The community reached for calibration as the fix for overconfidence and inherited the same fictional target.

## Tensions

This is the essential qualifier on [[inter-annotator-agreement-is-not-a-performance-ceiling]]. That paper models disagreement as independent noise the model averages away — and averaging is exactly what Plank says destroys the signal. Both hold only if you can separate error from plausible variation, and **neither paper can**. Plank is explicit that the field lacks both theory and tooling for the distinction, and that it is the unresolved technical core. Filtering and soft-label fitting are opposite bets on an open question.

It also cuts against the aggregate-label assumption inside [[judge-agreement-against-human-ceiling]] and [[llm-as-judge-evaluation-methodology]]: a judge scored against an aggregated human label inherits every pathology of that aggregation. [[labelled-dataset-creation]] describes the conventional single-gold pipeline this argues against. [[label-noise-dominates-saturated-benchmarks]] is the downstream consequence, though Plank insists a large share of the residual is not noise at all but discarded signal.

[[convincing-wrong-answer-active-learning]] and [[uncertainty-based-active-learning-for-demonstration-efficiency]] converge from the selection side on the same claim — contested and ambiguous items carry the most information. Swayamdipta-style data maps are independent evidence: training on the ambiguous region improves out-of-distribution generalization.

Open and genuinely unresolved: the error/variation distinction is asserted definitionally, not operationalized. Without a detector, the framework is vulnerable to the obvious objection — how do you know you are modeling plausible variation rather than fitting careless annotators? Plank's counter-hypothesis that HLV may *reduce* total data requirements (fewer instances, each carrying a full distribution) is unmeasured. And the scope is pre-LLM: written Nov 2022, it says nothing about preference data or RLHF, where the argument arguably applies with more force since preference pairs are aggregated judgements on maximally subjective comparisons. See [[llm-as-judge-evaluation-pattern]] and [[constitution-conditioned-synthetic-preference-data]].

# References

1. [[2211.02570 - The Problem of Human Label Variation]]
