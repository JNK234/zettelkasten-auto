
Created: 25-03-2026 18:16

Status: #baby 

Tags: [[trajectory-analysis]] [[fine-grained-evaluation]] [[process-metrics]]

# Progress Rate Evaluation for Intermediate Task States

## Definition
Progress rate evaluation measures task completion quality by tracking intermediate states during execution rather than only the final outcome. It compares an agent's actual trajectory through state space against an expected sequence of milestones, revealing whether the agent follows appropriate problem-solving paths even when final results appear successful. This fine-grained analysis exposes inefficiencies, detours, or lucky convergences that binary success metrics miss.

## How It Works
Before evaluation, establish an expected trajectory of intermediate states representing ideal task progression. During agent execution, capture snapshots of system state at regular intervals. After completion, calculate the proportion of expected states that appear in the actual trajectory. A high progress rate indicates the agent follows canonical solution paths; low rates suggest the agent reaches correct answers through unconventional or inefficient routes.

## Key Points
- Reveals *how* an agent solves problems, not just *whether* it succeeds, exposing qualitative differences in reasoning strategies
- Requires domain knowledge to define meaningful intermediate states, making it task-specific but highly informative
- Particularly valuable for emergent behavior systems where final states don't reveal the quality of the path taken
- Can be calculated retrospectively by using baseline runs to establish expected trajectories for comparison
- Detects agents that achieve correct outcomes through compensation rather than sound reasoning
- Provides diagnostic information for prompt engineering by identifying where agents deviate from optimal paths

## Connections
Relates to process supervision in reinforcement learning. Contrasts with outcome-only evaluation. Enables trajectory analysis in multi-step reasoning tasks.

## When to Use
Apply when solution quality matters as much as correctness, when debugging agent reasoning, or when comparing problem-solving approaches across different models.



# See Also

- [[progress-aware-state-tracking]]
- [[milestone-based-performance-tracking-for-multi-agent-systems]]
- [[compounding-error-in-long-horizon-sequential-tasks]]
- [[multi-run-consistency-testing-for-llm-agents]]
- [[temporal-alignment-for-distribution-mismatch]]

# References

1. [[2507.21504 — NetLogo Demo Ideas (llm extension)]]
