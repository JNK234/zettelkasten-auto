
Created: 25-03-2026 19:34

Status: #baby 

Tags: [[tool-use-evaluation]] [[semantic-validation]] [[function-calling]]

# Execution-Based Tool Evaluation

## Definition
An evaluation approach that validates tool use by actually executing tool calls and checking outcomes, rather than only verifying syntactic correctness. This method reveals semantic errors where a tool call is properly formatted but uses wrong parameters or calls the wrong tool for the task.

## How It Works
The evaluation process has multiple levels: (1) Does the agent invoke a tool when needed? (2) Does it select the correct tool? (3) Can it find the right tool from a large set? (4) Are the parameters syntactically valid? (5) Are the parameters semantically correct? Only execution-based evaluation catches failures at level 5—where the AST is valid but the call doesn't achieve the intended outcome.

## Key Points
- The Gorilla insight: AST correctness is necessary but insufficient—a function call can parse correctly but still be wrong
- Syntactic validation checks if the call would run; execution validation checks if it produces the correct result
- Requires either real tool execution (expensive, potentially dangerous) or simulation environments (cheaper, safer, but may miss edge cases)
- Reveals parameter value errors that syntax checking misses—correct function name, wrong argument values
- Critical for evaluating retrieval accuracy when agents must find the right tool from large API collections
- The evaluation chain progresses from invocation accuracy → tool selection → retrieval → parameter naming → parameter values → execution success

## Connections
Relates to the broader principle that testing interfaces differs from testing implementations. Builds on software testing concepts of unit tests (syntax) versus integration tests (execution). Essential for realistic tool use benchmarks like ToolBench.



# See Also

- [[silent-failure-detection-requirements]]
- [[robustness-testing-through-input-perturbation]]
- [[intention-score-as-diagnostic-metric]]
- [[llm-as-judge-evaluation-pattern]]
- [[outcome-driven-verification-in-agent-evaluation]]

# References

1. [[cale]]
