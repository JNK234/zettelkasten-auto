
Created: 25-03-2026 19:15

Status: #baby 

Tags: [[neural-network-architecture]] [[actor-critic-methods]] [[representation-learning]]

# Separate versus Shared Networks in Actor Critic

## Definition
The choice between separate policy and value networks versus a shared trunk with separate heads represents a fundamental architectural decision in actor-critic methods. Spinning Up advocates for completely separate networks (each a 2-layer MLP with 64 hidden units) to avoid value function training interfering with policy learning.

## How It Works
In separate architectures, the policy and value function are completely independent neural networks with no shared parameters. Each receives the observation as input and produces its respective output. In shared architectures, observations pass through common hidden layers before splitting into separate policy and value heads. Gradients from both losses flow through the shared layers during backpropagation.

## Key Points
- Spinning Up argues shared networks can cause value function training to interfere with the policy when value loss gradients are large (e.g., after reward scale changes), potentially destabilizing the policy head
- Separate networks avoid gradient coupling between the two objectives but require approximately twice as many forward passes during both data collection and training
- Shared networks save computation (about 2× fewer forward passes) and can help representation learning, where policy benefits from value function features and vice versa
- Stable Baselines 3 uses shared networks with separate final layers successfully, demonstrating the interference issue is manageable with proper loss weighting
- The choice involves a tradeoff between computational efficiency (favoring shared) and training stability/simplicity (favoring separate)
- For most MuJoCo continuous control tasks, both architectures work well, but separate networks provide more predictable training dynamics
- Separate networks make it easier to use asymmetric learning rates and separate optimization schedules for policy and value

## Connections
Relates to asymmetric learning rates and actor-critic training dynamics. Connected to computational efficiency considerations.



# See Also

- [[feedforward-networks-in-deep-learning]]
- [[asymmetric-learning-rates-for-actor-critic]]
- [[hierarchical-control-gap-in-neural-policies]]
- [[on-policy-versus-off-policy-learning]]
- [[gradient-insulation-in-robot-learning]]

# References

1. [[OpenAI - Spinning Up PPO]]
