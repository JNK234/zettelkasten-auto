Created: 20-08-2026 00:20
Status: #baby
Tags: [[reinforcement-learning-in-language-models]] [[policy-gradient-methods]] [[systems-engineering]]

# Tool Output Masking in Agentic RL

## What This Is
A tool-using agent's rollout is one long token sequence: the model's thinking, its tool calls, and — spliced in — whatever the tool returned. Search snippets, crawled page text, API responses.

Train a language-modelling loss over that whole sequence and you are teaching the model to **predict what the search API returns**. That is learning to imitate the environment, not learning a policy. Those tokens were never the model's decision; nothing about them is under its control; assigning them credit is a category error.

The fix is one line of masking: **exclude tool observation tokens from the loss.** They stay in the context — the model must condition on them — but contribute no gradient.

## Why It Matters
It separates two things a naive implementation conflates: what the agent *conditions on* and what the agent *is responsible for*. Only the second gets gradient. Once you see it stated, every agentic RL setup has to answer the question, and the answer is almost always the same.

The cost of getting it wrong is not just wasted capacity. Environment tokens are typically the *majority* of the sequence in a search agent — up to 10 tool calls per rollout, top-10 snippets each, inside a 16384-token response budget. Unmasked, they would dominate the loss, and the model would spend most of its training signal becoming a language model of Serper's output format.

## Key Details
- Sits alongside the other DR Tulu GRPO modifications: token-level (DAPO-style) rather than sequence-level loss aggregation, 1-step async training, sample packing, async tool calling, KL 0.001.
- Token-level aggregation matters *because* of the long-form setting — sequence-level aggregation down-weights long responses, which is backwards when the target output is a 16K-token report.
- Interacts with the masking: with observations masked and token-level aggregation on, the effective per-rollout weight is the count of *model-generated* tokens, which varies with how much the agent searched.

## Tensions
The general principle — only tokens the policy actually sampled should carry policy-gradient weight — is the same one behind [[exogenous-tokens-break-token-level-importance-ratios]], which shows the failure mode when injected tokens *do* leak into the objective through the importance ratio. Masking is the blunt version of that fix.

Whether to aggregate loss per-token or per-sequence is a separate axis handled by [[token-level-vs-sequence-level-loss-aggregation]] and [[token-level-vs-sample-level-loss-aggregation]]; masking changes which tokens count, aggregation changes how they're weighted, and in DR Tulu both are set for the long-response regime. The whole cluster only becomes load-bearing once the environment is a real external service — see [[the-environment-not-flops-bottlenecks-agentic-rl]] and [[tool-use-and-environmental-interaction-in-llm-agents]]. It is also an instance of the general [[credit-assignment-problem]]: before you can assign credit correctly you have to establish what the agent could have done differently.

# References
1. [[2511.19399 - DR Tulu (RLER, Evolving Rubrics for Deep Research)]]
