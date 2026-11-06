
Created: 25-03-2026 15:58

Status: #baby 

Tags: [[policy-gradient-methods]] [[kl-divergence-constraints]] [[reinforcement-learning-guarantees]]

# Trust Region Policy Optimization Monotonic Improvement Guarantee

## Definition
Trust Region Policy Optimization (TRPO) guarantees that each policy update improves performance (or leaves it unchanged) by constraining the KL divergence between successive policies. This is the only practical deep reinforcement learning algorithm with a formal monotonic improvement guarantee, based on the Kakade-Langford identity and its extension.

## How It Works
The theoretical foundation rests on the identity: η(π̃) = η(π) + E[A_π(s,a)] where the expectation is over the new policy's state distribution. TRPO approximates this using the old policy's state distribution (the surrogate objective L_π), then adds a penalty term proportional to the maximum KL divergence: η(π̃) ≥ L_π(π̃) - C·max_s D_KL(π||π̃). By maximizing the surrogate while constraining KL divergence to be small (typically using expected KL rather than max KL in practice), TRPO ensures that the true performance improvement is bounded below.

## Key Points
- The guarantee prevents the catastrophic "death spiral" where a bad policy generates bad data that trains worse policies
- Unlike supervised learning where overshooting just increases loss temporarily, RL policy collapse is irreversible without the constraint
- The theoretical bound uses max KL over states, but practice uses expected KL, which loses the strict guarantee but is far less conservative
- The penalty coefficient 4εγ/(1-γ)² makes the bound extremely loose in practice (often 10⁶ or more), so actual improvements greatly exceed theoretical predictions
- The guarantee distinguishes TRPO from all other practical deep RL algorithms, though PPO achieves similar empirical performance without the formal property
- The approximation from state distribution d^π̃ to d^π is accurate only locally, which justifies the trust region constraint

## Connections
Builds on the Kakade-Langford identity. Requires the natural gradient direction for implementation. Contrasts with PPO which abandons the guarantee for computational efficiency.



# See Also

- [[generalized-policy-iteration]]
- [[on-policy-vs-off-policy-learning]]
- [[markov-reward-process]]
- [[deadly-triad]]
- [[exploration-and-exploitation]]

# References

1. [[1502.05477 - Trust Region Policy Optimization]]
