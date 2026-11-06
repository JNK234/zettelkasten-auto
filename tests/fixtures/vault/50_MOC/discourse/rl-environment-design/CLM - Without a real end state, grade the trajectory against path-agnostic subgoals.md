Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #supported

# When the environment has no real state to inspect, a per-task checklist of subgoals, required interactions and forbidden behaviours, written to allow several valid paths, is the fallback grading target.

## Relations
informs:: [[QUE 03 - Grade end state, trajectory or text]]
supported-by:: [[EVD - Where there is no real state to inspect, SynthAgent grades the trajectory against a]]

## Scope
One source (SynthAgent) with an LLM judge; it inherits the judge's unvalidated accuracy.

## Related zettels
- [[rubrics-validated-against-teacher-executions]]
- [[per-task-forbidden-behavior-as-reward-gate]]
