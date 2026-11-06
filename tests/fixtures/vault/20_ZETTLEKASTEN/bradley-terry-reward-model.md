
Created: 12-05-2026 21:30

Status: #baby

Tags: [[reinforcement-learning-from-human-feedback-rlhf]] [[reward]] [[ai-alignment]]

# Bradley-Terry Reward Model

## What This Is

A reward model in modern RLHF is a function `r_θ(x, y)` that maps a `(prompt, response)` pair to a scalar score. The way you teach it: take pairs of responses to the same prompt where humans (or an AI labeler) picked one over the other. The Bradley-Terry model defines the probability the chosen response `y+` is preferred over the rejected `y-` as:

```
P(y+ > y- | x) = σ(r_θ(x, y+) - r_θ(x, y-))
```

Training loss is binary cross-entropy on this difference:

```
L(θ) = -E[ log σ(r_θ(x, y+) - r_θ(x, y-)) ]
```

The reward model learns to assign higher scalar scores to chosen responses and lower scores to rejected ones. The mechanism, per source §1.2: take a pre-trained LLM, replace its language-modeling head with a single linear layer `nn.Linear(hidden_size, 1)`, fine-tune with the Bradley-Terry loss. The linear head reads the hidden state at the last non-padding token (typically EOS) — the richest representation of the full sequence.

## Why It Matters

Bradley-Terry is the standard mechanism behind every PPO-RLHF pipeline. Once you have the trained reward model, you use it to score completions during policy optimization — the reward model is the *only* place where preferences enter the RL phase. Get this wrong and PPO optimizes the wrong objective.

Two practical consequences:

1. **The reward model and DPO are mathematical duals.** DPO derives a closed-form expression for the optimal policy under a Bradley-Terry reward, then substitutes the policy back in to eliminate the reward model entirely. So the same `(prompt, chosen, rejected)` data that trains a Bradley-Terry RM can also train DPO directly — no reward model needed. The choice between them is operational, not informational.

2. **The reward is underdetermined.** Adding a constant to all rewards doesn't change preference probabilities — the sigmoid only sees the difference. This means the absolute scale of the reward is meaningless; only the order matters. Per source §1.1, TRL's `center_rewards_coefficient` (recommended ~1e-2) adds an auxiliary loss to keep the reward centered around zero. Without it, the reward can drift unbounded during training.

## Key Details

- **Loss is identical to binary cross-entropy** on the score difference. No special math.
- **Last-token activation** is what the head reads, per source §1.2. Earlier tokens are blended in via attention but only the final token's hidden state is consumed by the scalar head.
- **`AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)`** — the standard HuggingFace API for this setup.
- **Fine-tune, don't train from scratch** — per source §1.3, the LLM backbone already understands language and quality signals; the head only needs to learn the mapping from "good text representation" to "scalar score". From-scratch training would need orders of magnitude more data.
- **Bradley-Terry assumes the preferences are consistent with a scalar utility.** Real human preferences violate transitivity, are context-dependent, vary across raters. The model still works empirically; the assumption is a simplification, not a derivation.

## Tensions

- **Scalar bottleneck.** A single scalar can't represent multi-dimensional quality (helpfulness, safety, factuality). [[Reward Model Training for Alignment]] §2.2 discusses ArmoRM as a multi-objective alternative — but most production RLHF uses scalar Bradley-Terry because the tooling is mature.
- **Bradley-Terry assumes well-calibrated preferences.** When the labeler is unreliable (8B critic on opinion sycophancy, per [[recognition-asymmetry-in-self-critique]]), the resulting reward model is correspondingly unreliable. The loss is correctly minimized but the function it represents is wrong.
- **No independent signal from same-data DPO.** If you train a reward model on the same preference pairs you used for DPO, the reward model is not an independent test of the DPO model — it shares the same biases. Per source §4.2, this is fine for research comparisons (we're comparing methods, not building independent eval), but not fine for production RM-driven RL where you'd want the RM to catch policy drift.
- Connected to [[reward]] (the broader concept), [[reinforcement-learning-from-human-feedback-rlhf]] (the pipeline this sits inside), [[direct-preference-optimization-dpo]] (the dual method), [[goodharts-law-in-reward-modeling]] (the failure mode at the optimization tail).


# References

1. [[Reward Model Training for Alignment]] (§1.1, §1.2, §1.3, §4.2)
2. [Christiano et al. 2017 — Deep RL from human preferences](https://arxiv.org/abs/1706.03741)
