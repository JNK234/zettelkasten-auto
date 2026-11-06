
Created: 17-01-2026 15:41

Status: #baby 

Tags: [[reinforcement-learning]] [[human-feedback]] [[ai-alignment]] [[reinforcement-learning-in-language-models]] [[deepseek-r1-mechanism]] [[policy-gradient-methods]]

# Reinforcement Learning from Human Feedback (RLHF)

## Definition  
Reinforcement Learning from Human Feedback (RLHF) is a training paradigm that utilizes human judgment to align AI models more closely with human values and preferences. It is particularly relevant in contexts where traditional supervised learning proves insufficient for capturing the complexities of desired outcomes.  

## How It Works  
RLHF augments the standard machine learning training process by incorporating human feedback directly into the optimization loop. The process typically involves three main stages:  
1. **Supervised Fine-Tuning (SFT)**: A pretrained model is fine-tuned using a small set of high-quality, human-written demonstration data. This establishes a baseline behavioral policy.  
2. **Collecting Human Preference Data**: The model generates various outputs for specific prompts, and humans are asked to rank these outputs. This creates a dataset that reflects human preferences.  
3. **Reward Model Training**: A reward model is trained to predict human preferences between pairs of outputs based on the collected data. This model scores candidate outputs according to human judgments.  
4. **Policy Optimization (RL Fine-Tuning)**: The language model is treated as an agent in a single-step reinforcement learning setting, where the outputs are optimized based on the reward model scores.  

## Key Points  
- RLHF helps to align AI outputs to better reflect human values and mitigate unsafe or misleading responses.  
- The methodology engages human evaluators to aid in defining what constitutes a 'good' output in ambiguous scenarios.  
- Research indicates RLHF significantly improves the performance and safety of AI models compared to traditional supervised training alone.  
- Models trained with RLHF can adapt their responses based on nuanced human feedback, leading to more helpful and accurate outputs.  
- RLHF has shown effectiveness in various applications, especially in fine-tuning language models.  

## Connections  
RLHF is connected to broader discussions about AI alignment and ethical AI development, emphasizing the role of human feedback in creating AI systems that better serve human needs.
# References

1. [[RLHF]]