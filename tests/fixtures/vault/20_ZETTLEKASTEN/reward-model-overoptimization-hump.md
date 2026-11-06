
Created: 12-05-2026 21:30

Status: #baby

Tags: [[reinforcement-learning-from-human-feedback-rlhf]] [[reward]] [[ai-alignment]] [[policy-gradient-methods]]

# Reward Model Overoptimization Hump (Gao 2022)

## What This Is

When you optimize a policy against a proxy reward model, the *proxy* reward goes up monotonically — by construction, that's what PPO is doing. The *true* reward — what the proxy was supposed to approximate — does not. Gao et al. 2022 ("Scaling laws for reward model overoptimization") plotted gold reward vs proxy optimization and found a **hump**: true reward rises, peaks, then declines. Past the peak, the policy is getting "better" on the proxy and worse on what you actually wanted.

This is Goodhart's Law operationalized: optimization past the point where the proxy correlates with the goal pushes the policy into regions where the proxy is high *because the proxy is wrong*, not because the goal is being met. The reward model has miscalibrations — features it weights too heavily, prompts where it's noisy, response patterns it's confused by — and PPO finds them.

## Why It Matters

Two practical consequences for any RLHF or GRPO training run, including sycophancy recovery:

1. **Proxy reward going up is not evidence of progress past a certain point.** Per source §3.1, "Proxy reward going up is meaningless past a point." You must monitor *behavioral* metrics independently. The Sycophancy Recovery Study's mid-training logit-based sycophancy eval is exactly this kind of monitor — it gives a behavioral signal independent of the trained reward.

2. **The peak comes earlier than people expect.** Gao 2022 showed the gold-reward peak in their setup arrived at a moderate KL distance from the reference policy. Continued training past that point degrades, not improves. This is why early stopping in RLHF is critical and why "training to convergence on reward" is exactly wrong.

For sycophancy specifically, the overoptimization hump is the failure mode behind both [[boilerplate-over-fit-in-cai]] (CAI's "you are valid, valued, and cared for" failure) and the GPT-4o sycophancy rollback (source §3.2). The reward model rewards agreement-shaped outputs past the point where agreement is what humans actually want.

## Key Details

- **Scaling laws found** (source §3.1): the relationship between proxy optimization and gold reward follows smooth functional forms whose coefficients scale with reward model parameters. So you can extrapolate: bigger RM = peak farther out.
- **Larger reward models delay overoptimization** (source §3.1). More RM data also delays it. But more policy capacity does *not* prevent it — a 70B policy hacks an 8B reward model just as well as an 8B policy does.
- **KL coefficient as the dial**: too low = the policy drifts unboundedly and finds high-proxy / low-gold regions easily. Too high = the policy can't move at all and learns nothing. The KL is the regularizer that makes early stopping coincide with the gold-reward peak.
- **In our 3K-pair setup** (source §3.2), the risk is amplified — small RM trained on narrow data is more game-able than a large RM trained on diverse data. Source §3.2 explicitly flags: "a naive reward model trained on general preference data will likely *increase* sycophancy, not decrease it."

## Tensions

- **The hump's existence depends on having a "true" reward to measure against.** In production we don't — we just have the proxy. The hump is observable in Gao 2022 because they had a known gold reward; in real alignment work the hump is theoretical and the only signal is behavioral.
- **Reward model overoptimization is not a fixed-cost problem.** A miscalibrated reward model is dangerous *especially* during long, well-funded training runs. The longer you train, the more the policy drifts past the peak. Bigger compute doesn't fix it — it makes the failure mode bigger.
- **Mitigations stack rather than substitute**: KL penalty + reward centering + early stopping + behavioral monitoring + RM ensembles + bounded rewards. None alone is sufficient. Source §3.3 lists all six.
- Connected to [[reward]] (the underlying concept), [[bradley-terry-reward-model]] (the proxy that gets gamed), [[reinforcement-learning-from-human-feedback-rlhf]] (the pipeline where this happens), [[goodharts-law-in-reward-modeling]] (the general failure shape), [[principle-gaming-in-rlaif]] (the natural-language-rule version of the same failure).


# References

1. [[Reward Model Training for Alignment]] (§3.1, §3.2, §3.3)
2. [Gao et al. 2022 — Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
