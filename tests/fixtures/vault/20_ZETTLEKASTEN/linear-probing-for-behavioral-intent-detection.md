Created: 12-04-2026 18:45
Status: #baby
Tags: [[representation-learning]] [[evaluation-methodology]] [[knowledge-representation-in-neural-networks]]

# Linear Probing for Behavioral Intent Detection

## What This Is
A model can behave differently after alignment training. But did the internal representation actually change, or did the model just learn to suppress the old behavior at the output layer? You can't tell by looking at outputs alone. You need to look inside.

Linear probing is the simplest tool for this. Freeze the model. Run inputs through it. Extract hidden state vectors at a chosen layer. Train a logistic regression (a single linear classifier) on those vectors to predict some property. If it succeeds, that property is linearly encoded -- it exists as a separable direction in the model's activation space.

The critical design choice: what inputs do you feed, and what labels do you use? For behavioral intent detection, the answer is non-obvious. Feed only the PROMPT (no response). Label by what the model ACTUALLY DOES when it generates a response (determined by a judge, not by text content). This probes what the model is planning to do at the moment it starts generating, before any tokens are produced.

This is fundamentally different from probing text comprehension. If you feed the model a pre-written sycophantic response and ask "is this sycophantic?", even a base model scores ~0.90. It understands the concept. But feed the base model just a prompt and ask "will this model be sycophantic?", it scores ~0.55 -- barely above chance. The base model doesn't plan to be sycophantic. Getting this distinction wrong invalidates the entire experiment.

## Why It Matters
Cross-model transfer is where the real insight lives. Train a probe on Model A's sycophancy direction. Apply it to Model B without retraining. If AUROC stays high: Model B still has Model A's representation, despite behavioral change. If AUROC drops to 0.5: the representation is gone or reorganized.

This is how we distinguish suppression from removal. DPO: SFT probe transfers at 0.677 (well above chance) -- suppression. SimPO: SFT probe transfers at 0.503 (chance) -- genuine reorganization.

## Key Details
- Extract at last token position of the prompt. This is where the model aggregates context before generation begins.
- Peak signal concentrates in middle layers (14-26 for a 36-layer model). These appear to be "safety-relevant" layers where alignment-related processing happens.
- Per-model probe AUROC tells you if the concept is linearly represented. Cross-model transfer AUROC tells you if the same direction encodes it.
- Statistical rigor requirements: bootstrap CIs (stratified, 1000 iterations), random-label controls (shuffled labels, 10 seeds for noise floor), permutation p-values for transfer, and max-statistic correction for scanning across 36 layers.
- Scanning all layers and reporting the peak inflates false positives. Max-statistic permutation correction: for each permutation, record the best AUROC across ALL layers, build a null distribution of peaks, compare observed peak against that.

## Tensions
Linear probes prove information is PRESENT in activations. They do not prove the model USES that information. A representation could exist as a vestigial artifact that the model routes around. Establishing causation requires intervention: patch activations, ablate directions, observe downstream effects. Directional ablation (project out the probe direction, retrain a fresh probe) is a partial step. Full causal tracing via activation patching is the complete answer.

There is also the linearity assumption. If a concept moves from a linear to a non-linear encoding after training, the probe transfer would drop to chance -- identical to genuine removal. [[simpo-reference-free-preference-optimization]]'s near-orthogonal probe direction (cosine similarity 0.082) could mean the representation was removed OR that it was re-encoded non-linearly. Distinguishing these requires tools beyond linear probing.

[[dpo-direct-preference-optimization]] shows the suppression pattern clearly. [[grpo-group-relative-policy-optimization]] is next to test: does on-policy RL produce different internal changes than offline preference methods?

# References
1. [[Linear Probing for Behavioral Intent]]