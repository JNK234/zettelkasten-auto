
Created: 25-03-2026 15:58

Status: #baby 

Tags: [[policy-gradient-methods]] [[algorithm-simplification]] [[trust-region-policy-optimization]]

# PPO as TRPO Simplification

## Definition
Proximal Policy Optimization (PPO) achieves TRPO's stability and performance while eliminating its computational complexity by replacing the hard KL constraint with a soft enforcement mechanism—clipping the probability ratio between new and old policies. This allows using standard optimizers like Adam and removes the need for conjugate gradient, line search, and Fisher-vector products, reducing implementation complexity by roughly 10x.

## How It Works
Where TRPO solves a constrained optimization problem with conjugate gradient and line search to enforce KL(π_old, π_new) ≤ δ, PPO simply clips the probability ratio r(θ) = π_new(a|s)/π_old(a|s) to stay within [1-ε, 1+ε] (typically ε = 0.2). The clipped objective prevents large policy updates by removing incentive to increase r beyond the clip threshold. PPO then optimizes this clipped objective for several epochs using standard SGD or Adam, no custom optimization procedure required.

## Key Points
- PPO replaces TRPO's entire custom optimization machinery (conjugate gradient, Fisher-vector products, line search) with simple probability ratio clipping and standard optimizers
- Despite the simplification, PPO achieves the same empirical performance and stability as TRPO across continuous control and Atari tasks
- The ratio clipping acts as a soft KL constraint—it doesn't enforce an exact KL bound but prevents the pathological large updates that destabilize training
- PPO is ~10x less expensive per update than TRPO (no 20 conjugate gradient iterations) and vastly simpler to implement and debug
- PPO naturally handles shared policy-value networks, which are awkward in TRPO because the KL constraint only applies to the policy portion
- The success of PPO despite lacking TRPO's theoretical guarantees raises questions about whether the natural gradient direction and hard constraints are actually necessary
- PPO's adoption over TRPO demonstrates that practical simplicity and implementation ease often matter more than theoretical optimality

## Connections
Inherits TRPO's core insight about constraining distribution changes. Demonstrates that soft constraint enforcement can match hard constraint performance. Related to the broader pattern of simpler methods matching complex theoretically-motivated ones.

## When to Use
PPO is the default choice for policy gradient methods in modern RL. Use PPO unless you specifically need TRPO's monotonic improvement guarantees for theoretical reasons or are in a domain where the hard KL constraint provides measurable benefits. TRPO's legacy is the conceptual framework; PPO is the practical implementation.



# See Also

- [[trpo-implementation-complexity-barrier]]
- [[surrogate-objective-local-approximation]]
- [[natural-gradient-vs-standard-gradient-trade-offs]]
- [[trust-region-policy-optimization-monotonic-improvement-guarantee]]
- [[natural-gradient-step-size-adaptation]]

# References

1. [[1502.05477 - Trust Region Policy Optimization]]
