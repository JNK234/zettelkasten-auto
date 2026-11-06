
Created: 12-05-2026 21:55

Status: #baby

Tags: [[reinforcement-learning-from-human-feedback-rlhf]] [[policy-gradient-methods]] [[training-stability]]

# PyTorch vs TensorFlow Adam Epsilon in RLHF

## What This Is

A specific implementation detail that breaks PPO-RLHF reproducibility across deep learning frameworks. PyTorch's `torch.optim.Adam` and TensorFlow's `tf.keras.optimizers.Adam` both implement Adam, but they handle the epsilon term differently — PyTorch adds epsilon inside the square root, TensorFlow adds it outside. The difference is mathematically small but operationally large for RLHF training.

Per source §3.2:

> "PyTorch Adam applies epsilon differently than TF Adam. PyTorch produces 6x larger logprob variance and 4.4x more policy clipping in early training. Fix: Use a TF-style Adam implementation or adjust epsilon accordingly. This is one of the most impactful implementation details for reproducing OpenAI's results."

The credit goes to Huang et al. 2024 ("The N+ implementation details of RLHF with PPO") for documenting this — most RLHF papers and tutorials don't mention it, and people switching from a TF codebase (or trying to reproduce OpenAI/Anthropic results trained in TF) to PyTorch silently hit the failure.

## Why It Matters

The practical consequence: early-training instability that doesn't look like a hyperparameter problem. The policy clips ratios 4.4× more often than the equivalent TF run. Log-prob variance is 6× higher. Symptoms that get blamed on "wrong learning rate" or "wrong KL coefficient" are actually downstream of this single optimizer detail.

For anyone reproducing published RLHF results in PyTorch: this is one of the first things to check if the training curves don't match the paper. For the Sycophancy Recovery Study, the project uses TRL on PyTorch — TRL has fixes for this issue, but it's worth knowing the failure mode exists.

The bigger lesson per source §3.2's failure-mode table: PPO-RLHF has many implementation details where each one looks small but compounds with others. Source §9 lists the Huang 2024 paper as "The N+ implementation details of RLHF with PPO" — the title is honest. The number of details to get right is large enough that "I followed the paper" often doesn't reproduce the result.

## Key Details

- **The actual math difference**: PyTorch computes `m / (sqrt(v) + eps)`; TF computes `m / sqrt(v + eps)`. When `v` is small (early training), the difference is non-trivial. When `v` is large (after training warms up), the difference fades.
- **Logprob variance 6× larger** in PyTorch (source §3.2). Higher variance in the importance ratio means more values land outside the clip range, which means more gradients are zeroed.
- **Policy clipping 4.4× more frequent** in PyTorch (source §3.2). The clip is the safety mechanism of PPO; high clip frequency means PPO isn't really PPO anymore — it's a vanilla policy gradient on the un-clipped half plus zeros on the clipped half.
- **Fix per source**: use a TF-style Adam implementation. TRL has options for this. Alternatively, adjust the PyTorch epsilon value to approximate TF's behavior — but this is fragile across LR and weight-init combinations.

## Tensions

- **The "right" implementation depends on what you're reproducing.** If you're trying to match OpenAI's InstructGPT (TF-trained), use TF-style Adam. If you're starting fresh with PyTorch and tuning for that specific stack, PyTorch's default may be fine — you'll just get different numerical behavior than the published TF results.
- **This is one of dozens of similar small details.** Per the Huang 2024 paper that source cites, "N+ implementation details" includes reward normalization timing, value head initialization, log-prob computation, advantage whitening, GAE λ, KL coefficient scheduling — each one a potential source of irreproducibility.
- **Default hyperparameters from papers don't transfer.** A learning rate of 1e-6 from OpenAI's InstructGPT paper, applied to a PyTorch RLHF pipeline with PyTorch-style Adam, produces different effective updates than the same LR in the original setup. The Sycophancy Recovery Study found this empirically with SimPO (paper LR didn't transfer to the project's task — see the SimPO post).
- Connected to [[1707.06347 - Proximal Policy Optimization Algorithms]] (the algorithm this detail affects), [[reinforcement-learning-from-human-feedback-rlhf]] (the broader pipeline), [[entropy-collapse-in-policy-gradient-methods]] (one of the downstream failure modes when clipping is misbehaving).


# References

1. [[PPO and RLHF - Deep Dive]] (§3.2)
2. [Huang et al. 2024 — The N+ implementation details of RLHF with PPO](https://arxiv.org/abs/2403.17031)
