
Created: 25-03-2026 19:08

Status: #baby 

Tags: [[evaluation-metrics]] [[diagnostic-analysis]] [[semantic-understanding]]

# Intention Score as Diagnostic Metric

## Definition
Intention score measures whether a robot moves toward the correct target entity during task execution, independent of whether it successfully completes the full task. This soft metric provides diagnostic information about semantic understanding that binary success rates obscure, especially valuable when distinguishing between "misunderstood the instruction" and "understood but failed to execute."

## How It Works
The metric tracks whether the model's actions demonstrate correct entity targeting before completion, capturing partial semantic competence. For example, if instructed to "place the apple in the bowl," a model that grasps the apple but drops it during placement would score high on intention but fail on success. This separation reveals whether failures stem from semantic confusion (wrong object targeted) or manipulation limitations (right object, poor execution).

## Key Points
- Binary success rates are too crude for diagnosing long-horizon embodied failures where multiple failure modes compound
- Intention scores expose whether VLA pretraining successfully transfers semantic understanding even when manipulation competence fails
- The metric particularly helps evaluate natural, non-template instructions where understanding "insert the national flower of the Netherlands" requires world knowledge beyond object recognition
- Combined with progress scores for multi-step tasks, intention scores enable failure mode categorization: semantic confusion, early execution failure, or late-stage cascade
- Soft metrics make tiny absolute success rates (3-13% range) more interpretable by revealing incremental capability gains
- The metric assumes correct entity targeting is observable from early action trajectories, requiring good low-level state tracking

## Connections
Complements progress score for long-horizon tasks. Relates to semantic versus manipulation competence separation. Enables more granular comparison between VLA architectures and modular foundation-model systems.

## When to Use
Apply when evaluating models on semantically complex instructions where understanding is a distinct challenge from execution, or when diagnosing whether training interventions should target language grounding versus control quality.



# See Also

- [[llm-evaluation-taxonomy-framework]]
- [[goal-image-conditioning-vs-language-conditioning]]
- [[progress-rate-evaluation-for-intermediate-task-states]]
- [[llm-as-judge-evaluation-pattern]]
- [[instruction-ambiguity-in-robot-language-grounding]]

# References

1. [[ICCV2025 - VLABench]]
