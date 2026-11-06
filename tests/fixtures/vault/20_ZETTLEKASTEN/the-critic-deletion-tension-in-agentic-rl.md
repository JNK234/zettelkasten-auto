Created: 20-08-2026 00:20
Status: #baby
Tags: [[policy-gradient-methods]] [[reinforcement-learning-in-language-models]] [[reinforcement-learning]]

# The Critic-Deletion Tension in Agentic RL

## What This Is
Value functions were invented to solve credit assignment. If a reward arrives fifty steps after the decision that earned it, a learned $V(s_t)$ lets you bootstrap — it tells you how good the intermediate state was, so you can attribute credit locally instead of smearing one terminal scalar across the whole trajectory.

LLM post-training deleted the critic. GRPO replaced $\mathcal{R} - V(s_t)$ with a group z-score: sample $G$ completions from the same prompt, subtract the group mean, divide by the group std. The group mean is an unbiased Monte Carlo estimate of $V(s_t)$ — but *only* for the case where all $G$ rollouts start from the same state. That deletion was rational: a critic is a second network roughly the size of the policy, and holding policy, reference, reward, and value models for a 70B model is brutal.

The tension: the field deleted the critic precisely as it moved into the regime where critics earn their keep — long horizon, sparse terminal reward, states that differ at every step.

## Why It Matters
This is the sharpest unresolved thing in the agentic RL literature, and it predicts where the next algorithmic work has to come from. Something has to give: either critics come back (with some trick to make them affordable), or the group-baseline idea gets generalized to non-shared states, or credit assignment gets solved some third way.

The symptom is already visible and admitted. Long-horizon tool-integrated RL fails at temporal credit assignment — "sparse, trajectory-level rewards make it difficult to pinpoint which specific tool invocation contributed to success." That is the classic problem value functions exist for, restated by a field that just removed them.

## Key Details
- GRPO's group baseline is only valid under a shared start state. In a multi-turn trajectory, step $t$ of rollout $g$ and step $t$ of rollout $g'$ are generally in *different* states, so the group mean is no longer estimating $V(s_t)$ for either of them (inference from the stated assumption).
- Current bets: **GiGPO** and **SpaRL** do turn-level advantage estimation — a middle granularity that recovers some locality without a full critic.
- **Step-GRPO** adds rule-based per-step reasoning rewards for finer credit assignment; **Posterior-GRPO** rewards process quality only on trajectories that actually succeeded.
- The known std-normalization failure compounds it: when all $G$ rollouts get identical reward (all pass or all fail on a hard sparse-binary task), std is 0, advantage is 0/0, and that prompt contributes nothing. Under sparse rewards most of a batch can silently evaporate. DAPO's dynamic sampling drops these degenerate groups; Dr. GRPO removes the length/difficulty bias the normalization introduces.
- The survey's build advice: if reward is terminal-only over ~50 steps, use either process rewards (with Posterior-GRPO's guard) or turn-level advantage estimation.

## Tensions
Restates [[credit-assignment-problem]] and [[algorithmic-credit-assignment-problem]] in the specific setting of [[agentic-rl-as-a-pomdp]]; it cannot arise in [[pbrft-as-a-degenerate-single-step-mdp]], where $T=1$ makes the critic genuinely unnecessary.

The deletion happened in [[grpo-group-relative-policy-optimization]]; the machinery deleted is described in [[ppo-gae-value-function-triangle]], [[generalized-advantage-estimation-gae]], and [[four-model-setup-in-ppo-rlhf]]. One partial reconciliation the vault already has: [[value-function-init-from-reward-model-in-rlvr]] — warm-start the critic instead of deleting it.

The process-reward answers connect to [[process-reward-model-vs-outcome-reward-model]] and [[process-reward-model-as-step-level-verifier]], with an important warning attached: [[process-reward-models-and-mcts-failed-as-rl-training-signals]] records that process signals have failed as training signals before, so "add step rewards" is not a free fix. Posterior-GRPO's outcome-gated process reward is a direct defense against [[reward-hacking-as-specification-gaming]] — the model producing plausible-looking intermediate reasoning that didn't actually cause the answer. Also touches [[advantage-normalization]] and [[reward-normalization]].

# References
1. [[2509.02547 - The Landscape of Agentic Reinforcement Learning for LLMs]]
