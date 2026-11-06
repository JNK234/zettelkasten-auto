Created: 20-08-2026 00:20
Status: #baby
Tags: [[reinforcement-learning]] [[policy-gradient-methods]] [[reinforcement-learning-in-language-models]] [[reward]]

# Decoupling Clipping Unit From Credit-Assignment Unit

## What This Is
Two separate questions get silently fused in clipped policy-gradient methods:
1. **Where is off-policy correction statistically valid?** — at whatever granularity you have enough samples to average over. For LLM RL, that is the sequence.
2. **Where do you actually have information about what was good or bad?** — sometimes per token or per turn, if a process reward or a multi-turn structure gives it to you.

PPO and GRPO answer both with "per token," because the clip and the advantage are attached to the same index. GRPO gets the worst of both: it pays the noise cost of token-level ratios while having no token-level advantage at all (`Â_{i,t} = Â_i`).

GSPO-token separates them with a stop-gradient trick:

```
s_{i,t}(θ) = sg[ s_i(θ) ] · π_θ(y_{i,t}|x,y_{i,<t}) / sg[ π_θ(y_{i,t}|x,y_{i,<t}) ]
```

where `sg[·]` is `.detach()`. Read it twice:
- **Forward:** the second factor is `p / detach(p) = 1`, so `s_{i,t}(θ) = s_i(θ)` exactly. Same value, same clipping decisions, same objective value as sequence-level GSPO.
- **Backward:** `sg[s_i]` is a constant; gradient flows only through the live `π_θ` in the numerator, yielding `∇log π_θ` for that token.

You get a value that behaves like a sequence ratio and a gradient that attaches per token.

## Why It Matters
The design lesson generalizes past this paper: **clip where the correction is valid, assign credit where you have information.** These do not have to be the same index, and forcing them to be the same is what produced GRPO's pathology.

Practically, GSPO-token is the version you would implement, because it is a strict generalization — setting `Â_{i,t} = Â_i` for all `t` recovers plain GSPO identically in objective value, clipping condition, and gradient. One implementation covers both regimes.

## Key Details
- Objective: `J = E[ (1/G) Σ_i (1/|y_i|) Σ_t min( s_{i,t}Â_{i,t}, clip(s_{i,t}, 1−ε, 1+ε) Â_{i,t} ) ]`
- Resulting gradient: `E[ (1/G) Σ_i (π_θ(y_i|x)/π_θold(y_i|x))^(1/|y_i|) · (1/|y_i|) Σ_t Â_{i,t} ∇log π_θ(y_{i,t}|·) ]` — identical to GSPO's except `Â_i` has moved inside the token sum and become `Â_{i,t}`.
- Motivating use case: multi-turn RL, where you genuinely want per-turn or per-token advantage shaping.
- **Untested.** Equations only in the paper. The multi-turn setting that motivates it is never run.
- Structural limitation this exposes: GSPO's premise is "reward is granted to the entire sequence." Get a working process reward model and that premise breaks — you would need GSPO-token with honest per-token advantages, which is proposed but unevaluated.

## Tensions
This is the escape hatch for [[importance-ratio-unit-must-match-reward-unit]] when the reward unit is finer than the sequence. It is the same mechanical device as the [[straight-through-estimator-for-step-functions]] — a forward value and a backward gradient chosen independently via stop-gradient — applied to importance ratios rather than discretization. It only becomes necessary if [[process-reward-model-vs-outcome-reward-model|process reward models]] deliver real step-level signal, which [[process-reward-models-and-mcts-failed-as-rl-training-signals]] and [[best-of-n-as-a-misleading-prm-metric]] suggest is still shaky. It is the honest answer to the [[credit-assignment-problem]] under GSPO: sequence-level GSPO does not solve credit assignment, it declines to fake it, and this variant is where you would put a real solution if you had one. Relies on [[length-normalized-sequence-importance-ratio]] for `s_i`.

# References
1. [[2507.18071 - Group Sequence Policy Optimization (GSPO)]]
