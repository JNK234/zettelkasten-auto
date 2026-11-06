
Created: 25-03-2026 18:56

Status: #baby 

Tags: [[self-supervised-learning]] [[reinforcement-learning-from-ai-feedback]] [[ai-alignment]]

# Constitutional AI

## Definition
Constitutional AI (CAI) is a training methodology developed by Anthropic for creating harmless AI assistants without requiring human labeling of harmful content. The approach uses a predefined "constitution" — a set of natural language principles (approximately 16 rules) — that guides an AI model to critique and revise its own outputs, replacing traditional human feedback on harmfulness with AI-generated feedback.

## How It Works
CAI operates in two distinct stages:

1. **Supervised Learning (SL) Stage**: The model generates initial responses that may be harmful, then critiques these responses against constitutional principles, revises them to align with the constitution, and finally trains on the improved, revised responses.

2. **Reinforcement Learning from AI Feedback (RLAIF) Stage**: The model generates preference labels comparing different responses based on constitutional principles, then uses these AI-generated preferences (rather than human preferences) to further refine behavior through reinforcement learning.

## Key Points
- Eliminates the need for human annotation of harmful content, which can be psychologically taxing for labelers and difficult to scale
- Uses natural language principles as the "constitution" to encode desired behavior, making the value system explicit and auditable
- Combines self-critique and self-revision in the supervised stage, allowing the model to learn from its own improved outputs
- Introduces RLAIF (Reinforcement Learning from AI Feedback) as an alternative to RLHF (Reinforcement Learning from Human Feedback) specifically for harmlessness training
- Maintains a separation between helpfulness (which may still use human feedback) and harmlessness (which uses constitutional principles)
- Represents a scalable approach to AI safety that can potentially be updated by modifying the constitution rather than re-collecting human labels

## Connections
Builds on reinforcement learning from human feedback (RLHF) but replaces human harmfulness judgments with AI self-evaluation. Relates to AI safety and alignment research more broadly.



# See Also

- [[reinforcement-learning-from-human-feedback-rlhf]]
- [[critic-tuner-approach]]
- [[interactive-evaluation-for-social-intelligence]]
- [[knowledge-forge-method]]
- [[modular-ai-system-architecture-with-abstracted-inference-layer]]

# References

1. [[Constitutional AI - Paper Notes]]
