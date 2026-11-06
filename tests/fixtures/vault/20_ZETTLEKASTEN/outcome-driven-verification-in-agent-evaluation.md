
Created: 25-03-2026 19:30

Status: #baby 

Tags: [[agent-evaluation]] [[verification-methods]] [[state-based-testing]]

# Outcome-Driven Verification in Agent Evaluation

## Definition
Outcome-driven verification is an evaluation methodology that assesses AI agents based on the final state of their execution environment rather than the intermediate steps or textual outputs they produce. Instead of grading whether an agent executed specific commands or produced certain logs, this approach validates whether the agent achieved the intended end-state (e.g., "Is the file fixed?", "Is the server running?", "Does the database contain the correct number of records?"). This method is particularly crucial for evaluating autonomous agents performing complex technical tasks where there are multiple valid solution paths.

## How It Works
The verification system operates by inspecting the state of a sandboxed environment (typically a Docker container) after the agent has completed its work. Verification scripts—often written in pytest or shell scripts—check file system states, running processes, network ports, file contents, or database records. The agent receives no credit for producing plausible-sounding explanations or executing commands that appear correct; success is determined solely by whether the environment reached the target state. This requires task designers to write explicit verification logic that can deterministically assess whether success criteria are met.

## Key Points
- Eliminates the ability of agents to "fake" success through convincing natural language explanations without actually solving the problem
- Requires verification scripts that inspect concrete artifacts: file existence, content hashes, process states, network configurations, or data counts
- Addresses the "weak verification" failure mode where agents mark tasks complete without actually checking if their solution works
- Enables detection of "silent failures" where tools run without error messages but produce incorrect or empty results
- Creates a higher bar for evaluation than text-matching approaches, as agents must produce actual working systems rather than descriptions of solutions
- Necessitates careful design of verification logic to distinguish between superficially correct states and truly successful outcomes

## Connections
This approach directly addresses the "weak verification" failure mode observed in agent systems. It requires pairing with oracle solutions to prove tasks are actually solvable. Contrasts with traditional NLP benchmarks that grade textual outputs.

## When to Use
Apply outcome-driven verification when evaluating agents performing system administration, software engineering, security tasks, or any domain where multiple valid solution paths exist and the final working state matters more than the specific implementation approach.



# See Also

- [[enhanced-verification-system]]
- [[progress-rate-evaluation-for-intermediate-task-states]]
- [[importance-of-evaluation-in-ai-systems]]
- [[intention-score-as-diagnostic-metric]]
- [[multi-run-consistency-testing-for-llm-agents]]

# References

1. [[Terminal Bench Brainstorming]]
