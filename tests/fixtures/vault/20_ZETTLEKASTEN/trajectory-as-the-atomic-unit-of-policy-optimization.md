Created: 20-08-2026 00:20
Status: #baby
Tags: [[reinforcement-learning-in-language-models]] [[policy-gradient-methods]] [[tool-use-and-environmental-interaction-in-llm-agents]]

# Trajectory as the Atomic Unit of Policy Optimization

## What This Is
Single-turn LLM RL has an easy unit: one prompt in, one response out, one scalar reward. The prompt is fixed, the response is a flat token sequence, and token-level policy gradients apply directly.

Multi-turn agents break that. The model emits an action, the environment emits an observation, and that observation becomes part of the next prompt. So the "prompt" is no longer fixed — it is a function of the policy's own earlier decisions. You are optimizing over a distribution that shifts as you optimize.

The move StarPO makes is to refuse to treat each turn as an independent bandit problem. Instead the **entire reasoning-augmented interaction sequence is one object**:

$$J(\theta) = \mathbb{E}_{\mathcal{M},\, \tau \sim \pi_\theta}\big[R(\tau)\big], \qquad \tau = \{s_0, a_0, r_0, \ldots, s_K\}$$

with $a_t \sim \pi_\theta(\cdot \mid s_t, \tau_{<t})$. The objective is the textbook RL objective — that is the point. The contribution is in what counts as atomic. Thinking tokens, action tokens, and interleaved environment observations become a single flat token sequence to which existing token-level policy gradients apply unchanged; the environment tokens are masked out of the loss since the policy did not generate them.

## Why It Matters
It is the bridge that lets you reuse the whole single-turn RLVR toolchain — PPO, GRPO, clipped surrogates, GAE — on agents, without inventing a new optimizer. StarPO is a *lifting*, not a new algorithm. That is exactly what makes the paper's negative results informative: when PPO and GRPO collapse here, you know the failure is about the multi-turn structure and not about an unfamiliar optimizer.

It also makes the costs visible. Two of them:

- **State grows monotonically in context length**, because $s_t$ conditions on all of $\tau_{<t}$. This is why RAGEN caps WebShop (longest contexts) at ~100 training iterations while symbolic tasks run 100–200.
- **Per-trajectory token normalization** $\frac{1}{|\tau_i|}$ means a verbose trajectory and a terse one contribute equal gradient weight. Sensible for fairness, but it also *prices* tokens — every reasoning token you emit dilutes the gradient on the tokens that actually earned the reward.

## Key Details
- Action format: $a_t^T = \texttt{<think>}\ldots\texttt{</think>}\texttt{<answer>}\ a_t\ \texttt{</answer>}$, where only the `<answer>` content is environment-executable. `<answer>` may hold a *sequence* of actions, which is what makes granularity tunable.
- $S$ = full interaction histories, not compact state vectors. $A$ = token sequences. Standard MDP $\{S, A, P\}$ with LLM-specific instantiation.
- Advantage under PPO: [[generalized-advantage-estimation-gae]] with $\gamma = \lambda = 1.0$. That is a maximum-variance, zero-bias choice — $\lambda=1$ collapses GAE to the Monte-Carlo return minus baseline, $\gamma=1$ means a turn-5 reward counts as much for turn-0 tokens as an immediate one. Defensible at $K=5$ but unjustified in the text, and it plausibly *causes* some of the PPO instability the source then attributes to multi-turn settings in general.
- $\varepsilon = 0.2$, $K = 5$ turns, ≤10 actions per episode.

## Tensions
Trajectory-level optimization makes [[credit-assignment-problem]] maximally hard: one scalar over five turns must be apportioned across hundreds of tokens spanning multiple reasoning blocks and multiple actions, and the tokens that *caused* the failure (a box pushed into a corner at turn 2) can be far from the tokens that received the signal. Worse under irreversibility — every token generated after a dead-end state gets the same negative advantage regardless of whether it was sensible.

The flat-sequence framing is also what enables [[outcome-only-rewards-cannot-see-the-reasoning-channel]]: since the reward attaches to the trajectory and not to any sub-span, nothing distinguishes good `<think>` from hallucinated `<think>`. [[process-reward-model-vs-outcome-reward-model]] is the structural fix.

Lifts [[grpo-group-relative-policy-optimization]] and [[ppo-as-trpo-simplification]] to trajectories; see [[critic-versus-critic-free-advantage-under-environment-stochasticity]] for which one to lift. Interacts with [[on-policy-versus-off-policy-learning]] more sharply than single-turn RL does — a slightly changed policy produces a *differently branching* trajectory tree, not just differently weighted tokens, so reusing rollouts for $k>1$ updates degrades convergence and generalization (online-1 is best).

# References
1. [[2504.20073 - RAGEN StarPO Multi-Turn Agent RL]]
