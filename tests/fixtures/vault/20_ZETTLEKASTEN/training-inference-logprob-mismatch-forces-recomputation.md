Created: 20-08-2026 00:20
Status: #baby
Tags: [[systems-engineering]] [[distributed-systems]] [[reinforcement-learning-in-language-models]]

# Training-Inference Logprob Mismatch Forces Recomputation

## What This Is
Large-scale LLM RL uses two different software stacks: an inference engine (vLLM, SGLang) to generate rollouts fast, and a training engine (Megatron) to compute gradients. They use different kernels and different numerical precision, so **they disagree on the log-probability of the exact same token in the exact same context.**

Because a per-token importance ratio `π_θ(y_t|·)/π_θold(y_t|·)` is sensitive to those small disagreements, standard practice is to discard the inference engine's logprobs entirely and **recompute** π_θold likelihoods on the training engine — a full extra forward pass over every rollout, purely to make the two numbers commensurable.

A length-averaged sequence ratio changes the calculus. Per-token precision errors are small, roughly independent, and get averaged over thousands of tokens, so they largely cancel. A method built on `s_i = exp((1/|y|) Σ_t log ratio)` is, in the paper's phrasing, "much more tolerant of precision discrepancies" — which makes it possible to use the inference engine's logprobs directly and delete the recomputation pass.

## Why It Matters
This is a case where the choice of statistical granularity in an algorithm propagates into cluster architecture. The recomputation requirement is a hard coupling: the training engine must re-score everything the inference engine produced. That coupling is cheap when both live on the same GPUs and expensive-to-impossible in the settings the field is moving toward:
- **partial rollout** (generation resumed across steps),
- **multi-turn RL** (many interleaved generate/train boundaries),
- **training-inference disaggregated frameworks** (separate clusters for generation and training).

In a disaggregated setup, removing the re-score pass is not a micro-optimization, it removes an architectural constraint.

## Key Details
- The mismatch source is precision and kernel differences, not a bug — the same math implemented two ways.
- The tolerance argument is that errors are *independent across tokens*, so averaging over `|y|` tokens shrinks their aggregate effect roughly as you would expect from averaging noise.
- **Important caveat:** the paper states this as an intuition and a possibility ("makes it possible to"), with **no experiment**. It is unverified in this work.

## Tensions
This is a second-order benefit of [[length-normalized-sequence-importance-ratio]] — the same aggregation that fixes the statistics also buys numerical robustness, which is weak evidence the diagnosis in [[single-sample-importance-sampling-is-not-correction]] was structural rather than a lucky patch. It pairs with [[moe-expert-activation-volatility-under-rl-updates]] as the two systems-level workarounds an algorithmic fix removes (Routing Replay and the re-score pass). It matters most under the batching regime described in [[on-policy-vs-off-policy-learning]], where huge rollout batches split into mini-batches are what create the off-policy gap in the first place. Compare [[reinforcement-learning-implementation-complexity-barrier]]: much of RL's implementation difficulty is exactly this kind of cross-stack bookkeeping, and algorithmic choices can delete it.

# References
1. [[2507.18071 - Group Sequence Policy Optimization (GSPO)]]
