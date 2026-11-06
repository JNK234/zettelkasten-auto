Created: 24-09-2026 16:29

Status: #baby

Tags: [[rl-environment-design]] [[systems-engineering]] [[agentic-ai-systems]]

# The Environment/Task Split Sets the Cost Model

## What This Is
In LLM RL, "environment" and "task" are used loosely, which hides a real design decision. A clean split: the **environment** is everything that shapes the agent's actions except the prompt: which actions exist, and the context that decides what they do (file system, env vars, state of a simulated app). It usually ships as a Docker container. A **task** is prompt + grader. In classic RL terms the environment holds transition dynamics, reward function, and start-state distribution; a task is one start state.

The useful consequence is how many tasks one environment carries, and that depends on the domain:
- **Computer use** (an Excel or Slack clone): expensive to build, cheap to reuse, so one environment hosts hundreds of tasks. You pay mostly for the environment.
- **Coding** (repo at a commit): cheap to set up, so usually one task per environment. You pay mostly per task.
- **Math**: often no environment at all, since the model uses no tools.

## Why It Matters
When designing an environment system you have to decide what the unit of versioning, caching, and pricing is. For UI clones, invest in the environment once and generate many tasks against it; fidelity is where the money goes. For coding, invest in cheap, fast per-task setup and grader quality. Treating both the same way either over-builds coding envs or under-builds UI clones.

## Key Details
Four worked examples (environment -> task -> grader):

| Environment | Task | Observation/action | Grader |
|---|---|---|---|
| Git repo at fixed commit | Fix bug so failing tests pass | Shell tools | Run test suite |
| Airbnb clone | Cheapest 2-bed in city X on dates Y | DOM / a11y tree; click/type | Check final answer |
| Bloomberg terminal clone | 5-yr CAGR for companies | Simulated terminal | Compare to true values |
| Excel clone | Pivot table of revenue by region | Spreadsheet app | Compare to reference |

- Prime Intellect's `verifiers` library is a lighter alternative to Docker packaging.
- Three ways to build enterprise environments: MCP-style tool APIs, Playwright-style DOM control, and screenshot-based computer use (needed when an app has no reachable backend).

## Tensions
- [[environment-as-installable-package-not-config]] and [[environments-as-versioned-packages-not-repo-subfolders]] cover the packaging choice this split implies.
- [[agents-environment-interface-types]] catalogs the observation/action interface axis separately.
- [[rl-task-cost-is-small-next-to-its-training-compute]] gives per-task prices that apply mainly to the coding-style regime.
- For UI-clone environments, graders often need [[state-verification-vs-output-verification]]: check the app's final state, not the text answer.

# References
1. [[Epoch AI (Denain & Barber) - An FAQ on Reinforcement Learning Environments]] (§What is an environment, what is a task; §Domains, Enterprise workflows; footnote 3)
