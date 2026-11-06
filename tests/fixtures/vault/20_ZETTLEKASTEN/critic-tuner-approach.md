
Created: 17-01-2026 15:31

Status: #baby 

Tags: [[critic-tuner]] [[iterative-improvement]] [[machine-learning]] [[importance-of-evaluation-in-ai-systems]] [[actor-critic-methods]] [[fine-tuning-strategies-for-llms]] [[llm-as-a-judge-framework]] [[automatic-tool-creation]]

# Critic-Tuner Approach

## Definition
The Critic-Tuner approach is a method designed to improve an agent’s output through iterative refinements of its system prompt. By making small, incremental changes based on evaluation metrics, this approach enables systems to evolve and produce outputs that are more aligned with defined standards or rubrics. It is essential in systems that require continual improvement and adaptation over time.

## How It Works
1. **Trace Outputs**: Using a platform like Weave, the agent's inputs and outputs are tracked systematically. 
2. **Define Evaluation Rubric**: A set of criteria is established by human evaluators to define what an ideal output should look like.
3. **Critique and Adjust**: A Critic agent evaluates the outputs against the rubric, identifies mistakes, and suggests modifications to the system prompt.
4. **Incremental Updates**: After each run, based on the Critic's feedback, small updates are made to the system prompt, fostering gradual output improvement over successive iterations.  

## Key Points
- The Critic-Tuner approach facilitates continuous improvement for agents by refining their prompts based on performance feedback.
- It relies on a defined rubric, giving a clear standard for evaluating performance.
- This approach benefits systems by incrementally shifting outputs closer to desired outcomes with each iteration.
- It emphasizes the importance of human evaluation in setting quality benchmarks and guiding machine learning processes.

## Connections
This method complements tools designed for automatic tool creation by ensuring that the underlying agent's output consistently improves, thus reducing the need for constant human intervention. 

## When to Use
Use the Critic-Tuner approach in environments where outputs need to adapt dynamically based on performance data, particularly in machine learning contexts requiring fine-tuning of models.
# References

1. [[Hackathon Idea]]