
Created: 25-01-2025 10:03

Tags:  [[LLMs]] [[20_ZETTLEKASTEN/Reinforcement Learning]] [[nlp-seminar-msai]] [[research-papers]]

Status: #baby  #processed

## References

1. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

# Notes

### Abstract

-  We introduce our first-generation reasoning models, DeepSeek-R1-Zero and DeepSeek-R1.
DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without super-vised fine-tuning (SFT) as a preliminary step, demonstrates remarkable reasoning capabilities.

### Introduction

- Post Training: It has been shown to enhance accuracy on reasoning tasks, align with social values, and adapt to user preferences, all while requiring relatively minimal computational resources against pre-training.
- OpenAi's O1 - first to introduce inference-time scaling by increasing the length of the Chain-of-Thought reasoning process
-  improving language model reasoning capabilities using pure reinforcement learning (RL). 
- Our goal is to explore the potential of LLMs to develop reasoning capabilities without any supervised data, focusing on their self-evolution through a pure RL process.
- Use DeepSeek-V3-Base as base model and employ GRPO - as the RL framework to improve model performance in reasoning. 
- The model DeepSeek-R1-Zero emerged with powerful and numerous interesting behaviours. 
- However, DeepSeek-R1-Zero encounters challenges such as poor readability, and language
mixing. 

- To address these issues and further enhance reasoning performance, we introduce
DeepSeek-R1, which incorporates a small amount of cold-start data and a multi-stage training
pipeline. Specifically, we begin by collecting thousands of cold-start data to fine-tune the
DeepSeek-V3-Base model. Following this, we perform reasoning-oriented RL like DeepSeek-R1-
Zero. Upon nearing convergence in the RL process, we create new SFT data through rejection
sampling on the RL checkpoint, combined with supervised data from DeepSeek-V3 in domains
such as writing, factual QA, and self-cognition, and then retrain the DeepSeek-V3-Base model.
After fine-tuning with the new data, the checkpoint undergoes an additional RL process, taking
into account prompts from all scenarios. After these steps, we obtained a checkpoint referred to
as DeepSeek-R1, which achieves performance on par with OpenAI-o1-1217.


