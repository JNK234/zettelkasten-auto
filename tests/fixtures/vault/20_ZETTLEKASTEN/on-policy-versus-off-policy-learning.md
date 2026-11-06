
Created: 25-03-2026 19:15

Status: #baby 

Tags: [[sample-efficiency]] [[data-distribution]] [[exploration-exploitation]]

# On Policy versus Off Policy Learning

## Definition
On-policy and off-policy represent two fundamental paradigms in reinforcement learning that differ in whether the learning agent uses data from its current policy or from past/different policies. This distinction creates a fundamental tradeoff between sample efficiency and training stability.

## Key Points
- On-policy methods (PPO, TRPO, A2C) use only data from the current policy and discard all data after each update, requiring more environment interactions but providing simpler, more stable training
- Off-policy methods (SAC, TD3, DDPG) store data in replay buffers and reuse old experience, achieving better sample efficiency but introducing training instabilities from stale data that came from different policy distributions
- The on-policy constraint forces PPO to collect fresh data for every policy update, making it more computationally expensive in terms of environment interactions but avoiding distribution mismatch issues
- PPO is positioned as the "default first choice" for continuous control because its on-policy nature provides stability that outweighs the sample efficiency advantage of off-policy methods for most tasks
- The data collection and optimization phases are completely separated in on-policy methods: collect T×N transitions, then perform K epochs of optimization, then throw everything away and repeat
- Off-policy methods can theoretically learn from any experience, including human demonstrations or data from different tasks, while on-policy methods are restricted to their own recent experience

## Connections
Fundamental characteristic that determines PPO's architecture and training loop. Contrasts with replay buffer-based methods like SAC and TD3.



# See Also

- [[on-policy-vs-off-policy-learning]]
- [[prioritized-experience-replay]]
- [[model-free-policy]]
- [[ppo-as-trpo-simplification]]
- [[generalized-policy-iteration]]

# References

1. [[OpenAI - Spinning Up PPO]]
