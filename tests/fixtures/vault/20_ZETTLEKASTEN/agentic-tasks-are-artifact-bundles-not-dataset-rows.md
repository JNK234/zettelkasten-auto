Created: 24-09-2026 16:29

Status: #baby

Tags: [[rl-environment-design]] [[benchmark-design]] [[agentic-ai-systems]]

# Agentic Tasks Are Artifact Bundles, Not Dataset Rows

## What This Is
In single-turn RL a task is a row: one prompt, one expected answer. An agentic task cannot fit in a row. Before the rollout starts, the sandbox needs a repository at a specific commit, data files, a test harness to run after submission, and a rule for turning test outcomes into a number. The task is a **bundle of artifacts**, and the dataset row is only a pointer to it.

## Why It Matters
If you design a task taxonomy for agent environments, the unit you are cataloguing is the bundle, and each bundle has to carry its own grader and its own reward weighting. That is what makes tasks portable between environments and what lets thousands of them be mass-produced. It also means task authoring is where most of the verification work happens.

## Key Details
- **OpenReward SETA** (1,376 terminal-agent tasks on ORS): each task has `task.yaml` (prompt + metadata), data files, `tests/test_outputs.py` run after submission, and `weights.json` that turns test outcomes into a scalar. **Reward weighting ships with the task.**
- **SWE-Bench**: each row has `repo`, `base_commit`, `problem_statement`, `test_patch`. The env clones at that commit, applies the test patch, and grades the agent's diff by FAIL_TO_PASS / PASS_TO_PASS.
- Bundles live behind the row (S3, HF dataset repo, tarball). Lifecycle: pull at episode start → run → score → tear down.
- **Coupled vs BYO task specs.** Coupled frameworks (Verifiers, GEM, ORS, NeMo, and hubs even more) make you convert data to their spec once; after that any task that fits runs in any env on that spec. BYO frameworks (OpenEnv, SkyRL) accept anything but every new source costs glue. Verdict: one task → the spec is overhead; many tasks → the spec is what makes scale possible.
- Scale motivation: Qwen3 used ~20 general-domain tasks; Qwen3.5 used "million-agent" environments with task distributions that get harder over time, and credits most of its post-training gain over Qwen3 to scaling RL tasks and environments.

## Tensions
A bundle is only trainable if its grader is sound: [[writable-state-assertion-as-task-readiness-gate]] and [[oracle-solutions-for-task-solvability]] are the checks each bundle should pass, and [[oracle-floor-margin-contract-for-environment-validation]] formalises them. [[benchmarks-repurposed-as-training-environments]] warns that SWE-Bench-style bundles were built to measure, so training on them couples train and eval distributions. The packaging side is [[environment-as-installable-package-not-config]]. The Qwen3.5 result is consistent with [[task-diversity-beats-group-size-under-fixed-rollout-budget]].

# References
1. [[Adithya S Kolavi - The Ultimate Guide to RL Environments]] (§The problem, §Dimension: tasks are bundles, not rows)
