
Created: 25-03-2026 18:16

Status: #baby 

Tags: [[reliability-metrics]] [[stochastic-evaluation]] [[llm-agent-testing]]

# Multi-Run Consistency Testing for LLM Agents

## Definition
Multi-run consistency testing evaluates LLM agent reliability by executing the same task multiple times and measuring variance in outcomes. Unlike single-shot evaluation, this approach captures the stochastic nature of LLM outputs and distinguishes between agents that occasionally succeed versus those that reliably perform. The pass@k and pass^k metrics quantify best-case and worst-case consistency respectively.

## How It Works
The agent receives identical input conditions across k independent runs. Each run produces a task score, forming a distribution of outcomes. The pass@k metric checks if *any* run exceeds the threshold (optimistic), while pass^k requires *all* runs to succeed (pessimistic). Standard deviation across runs quantifies output stability. This methodology reveals whether apparent task success is reproducible or merely lucky sampling from the LLM's probability distribution.

## Key Points
- pass@k measures whether an agent can *ever* complete a task successfully given k attempts, representing best-case reliability
- pass^k (pass-hat-k) requires *all* k runs to succeed, capturing worst-case consistency and revealing catastrophic failure rates
- Mean and standard deviation of scores across runs provide quantitative stability metrics beyond binary pass/fail
- Multiple runs multiply computational costs linearly with k, requiring strategic trade-offs between confidence and resource expenditure
- Consistency testing is particularly critical for high-stakes applications where occasional failures are unacceptable
- This approach distinguishes agents with stable reasoning from those producing unreliable outputs due to temperature, sampling, or prompt sensitivity

## Connections
Builds on statistical hypothesis testing principles. Contrasts with single-shot evaluation methods. Enables robust comparison between different LLM models or prompt strategies.

## When to Use
Apply when agent outputs will be used without human oversight, when comparing reliability across models, or when measuring the impact of prompt engineering on output stability.



# See Also

- [[milestone-based-performance-tracking-for-multi-agent-systems]]
- [[evaluation-statistical-insufficiency-in-robotics]]
- [[communication-efficiency-metric-for-multi-agent-systems]]
- [[confidence-gated-execution]]
- [[cascading-errors-in-long-horizon-tasks]]

# References

1. [[2507.21504 — NetLogo Demo Ideas (llm extension)]]
