
Created: 12-05-2026 21:55

Status: #baby

Tags: [[reinforcement-learning-from-human-feedback-rlhf]] [[policy-gradient-methods]] [[kl-divergence-as-policy-distance-metric]]

# KL Penalty in RLHF

## What This Is

The KL penalty is the regularizer that keeps a PPO-trained policy from drifting too far from the model it started as. Per source §2.3, the total reward seen by PPO during training is:

```
R_total(x, y) = R_reward_model(x, y) - β * KL(π_θ || π_ref)
```

Where π_θ is the current policy being trained and π_ref is a frozen reference (typically the post-SFT model before PPO). β is the KL coefficient that controls how aggressively to penalize divergence.

Per-token form (source §2.3):

```
kl_t = log π_θ(token_t | context) - log π_ref(token_t | context)

rewards[:, t]  = -β * kl_t          for t < T (intermediate tokens)
rewards[:, T]  = score - β * kl_T   for t = T (last token gets RM score)
```

The KL penalty is applied token-by-token through the sequence. The reward-model score lands only at the final token. The advantage estimator then propagates the score backward through the trajectory, mixed with the per-token KL penalties.

## Why It Matters

The KL penalty is doing two distinct jobs:

1. **Prevents reward hacking.** Without it, the policy finds whatever exploits the reward model has — verbose padding, repetitive phrasing, specific format markers — and drifts into a region where the reward model is wrong. Per source §2.3, "the policy generates gibberish that exploits reward model weaknesses" if the KL coefficient is set to zero.
2. **Anchors the policy to a sane starting point.** The reference is a model that already passes basic sanity (it's post-SFT, instruction-tuned, has reasonable language). The KL term penalizes any token-level divergence from that distribution, regardless of reward. The policy can move, but each step away from the reference costs reward.

The KL is the primary mechanism by which RLHF *doesn't* destroy the model. Production RLHF papers (InstructGPT, Anthropic HH-RLHF) all use it. Setting β too low produces a degraded model that scores high on the reward; setting β too high produces no learning. Per source §3.2's failure-mode table:

- KL exploding → policy changing too fast, β too low, fix: lower LR or higher β
- No learning → KL too high, β too aggressive, fix: lower β
- Policy generates gibberish → KL too low, fix: increase β or use adaptive controller

## Key Details

- **Default starting value**: source §2.3 gives `init_kl_coef = 0.15` for the adaptive controller. Some setups start much lower (0.01) and let the controller raise it.
- **Adaptive KL controller** (source §2.3, attributed to OpenAI):
  ```
  proportional_error = clip(current_kl / target - 1, -0.2, 0.2)
  mult = 1 + proportional_error * n_steps / horizon
  β *= mult
  ```
  Target KL is typically 6.0 nats; horizon ~10000 steps. When the policy is drifting too fast, β goes up; when it's not moving, β comes down. This makes the KL penalty a feedback-controlled dial, not a fixed hyperparameter.
- **Per-token vs per-sequence KL**: source §2.3 specifies token-level KL aggregation. This is finer than computing one KL number per generated sequence — it gives the advantage estimator per-token signal about where the policy is diverging.
- **In DPO this is implicit**: DPO's β hyperparameter plays the same role — control how much the policy can drift from the reference — but it's built into the loss derivation rather than added as a separate penalty term. The two are mechanistically related but not identical. See [[direct-preference-optimization-dpo]].

## Tensions

- **The "right" β depends on everything else.** Bigger reward models, more data, easier tasks all tolerate lower β. The number reported in any paper isn't transferable to your setup — you have to find β empirically by watching the training curves.
- **Token-level KL can over-penalize long sequences.** A 500-token response accumulates 500 KL terms, while a 50-token response accumulates 50. The penalty scales with length, which interacts with verbosity bias in the reward model (see [[llm-as-judge-biases-catalog]]).
- **Adaptive KL controllers can oscillate.** If `horizon` is too short relative to the target KL noise, β cycles between high and low without stabilizing. The proportional clip (±0.2) helps but doesn't eliminate.
- **Removing the KL entirely** is a research curiosity, not a production choice. GRPO sometimes runs with β=0 (source §4.4: "KL penalty optional (recent work shows β=0 works fine for GRPO)") because the group-relative advantage provides implicit regularization. For PPO with a learned reward model, β=0 is dangerous.
- Connected to [[kl-divergence-as-policy-distance-metric]] (the underlying math), [[direct-preference-optimization-dpo]] (where β plays the same role implicitly), [[reward-model-overoptimization-hump]] (the failure that KL is meant to prevent), [[trust-region-policy-optimization-monotonic-improvement-guarantee]] (the ancestor that motivated soft KL constraints).


# References

1. [[PPO and RLHF - Deep Dive]] (§2.3, §3.2, §4.4)
2. [Ouyang et al. 2022 — InstructGPT](https://arxiv.org/abs/2203.02155)
