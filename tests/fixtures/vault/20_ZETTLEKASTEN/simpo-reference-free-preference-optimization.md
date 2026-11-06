Created: 12-04-2026 18:45
Status: #baby
Tags: [[fine-tuning-strategies-for-llms]] [[reinforcement-learning]]

# SimPO: Reference-Free Preference Optimization

## What This Is
DPO defines reward as a log-ratio between the current policy and a frozen reference. SimPO asks: what if the reward was just the model's own confidence in its output?

The reward becomes: r = (beta / |y|) * log pi_theta(y|x). Average log probability of the response, length-normalized. No reference model anywhere. This aligns training with inference -- beam search and sampling already rank completions by log-probability. DPO optimizes a proxy (log-ratio) that diverges from how generations are actually selected at test time. SimPO optimizes the thing itself.

Two additional mechanisms make this work. First, length normalization: without dividing by |y|, the model games the reward by producing longer responses (more tokens = more log-prob mass). AlpacaEval LC win rate drops from 21.5% to 11.9% without it. Second, a target margin gamma > 0 that enforces a minimum gap between chosen and rejected rewards, preventing the model from satisfying the loss with trivially small separations.

## Why It Matters
Removing the reference model has two consequences. Practically: one fewer model to store and compute, simpler pipeline. Theoretically: no KL anchor constraining how far the policy can move from its starting point. The model is free to reorganize its internal representations, not just adjust its output distribution.

In our sycophancy experiments, this produced the most dramatic result: SimPO went BELOW the pre-sycophancy baseline (0.467 -> 0.176, baseline was 0.256). More importantly, the SFT sycophancy probe transferred at 0.503 -- chance level. Cosine similarity between SFT and SimPO probe directions was 0.082 (nearly orthogonal). The internal representation genuinely reorganized.

## Key Details
- beta must be 2.0-10.0 (vs DPO's 0.1). Raw log-probs are smaller numbers than log-ratios, so beta scales up to compensate.
- gamma (target margin) typically 0.5-1.5. Acts as a minimum quality gap.
- In TRL 0.29.1, SimPO is implemented in CPOTrainer (from trl.experimental.cpo), NOT DPOTrainer. Config: loss_type="simpo", cpo_alpha=0.0.
- Overfits more aggressively than DPO. Margins can hit 12+ (vs DPO's ~7) because nothing prevents the model from pushing arbitrarily far. Learning rate and early stopping matter more.
- Best hyperparams from our experiments: beta=2.0, gamma=1.0, LR=5e-6, 2 epochs with early stopping.

## Tensions
The freedom that enables deep representational change also enables aggressive overfitting. Without a reference anchor, the only regularization is the learning rate and training duration. This makes SimPO more sensitive to hyperparameters than [[dpo-direct-preference-optimization]].

The near-orthogonal probe directions raise a deeper question: did SimPO genuinely remove sycophancy, or did it move the representation to a new subspace that our linear probe can't detect? [[linear-probing-for-behavioral-intent-detection]] assumes linearity -- if the concept moved to a non-linear encoding, we'd see the same chance-level transfer but for the wrong reason. Causal interventions (activation patching, inference-time ablation) are needed to distinguish these cases.

# References
1. [[SimPO - Simple Preference Optimization]]