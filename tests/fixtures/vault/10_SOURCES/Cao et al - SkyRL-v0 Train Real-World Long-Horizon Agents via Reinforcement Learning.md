
Created: 08-05-2026 17:30

Tags:

Status: #baby

## References

1. [SkyRL-v0: Train Real-World Long-Horizon Agents via Reinforcement Learning](https://novasky-ai.notion.site/skyrl-v0)
2. [GitHub: NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)
3. [VeRL (base framework)](https://github.com/volcengine/verl)
4. [OpenHands (agent scaffold)](https://github.com/All-Hands-AI/OpenHands)
5. [SWE-Bench](https://github.com/swe-bench/SWE-bench)
6. [SWE-Gym (training data source)](https://arxiv.org/abs/2412.21139)

# Notes

## What and why

SkyRL-v0 is an **online RL training pipeline for multi-turn tool-using LLM agents on long-horizon, real-environment tasks** — specifically the kind of work where an agent has to read a real codebase, reproduce a bug, write a fix, and run tests across 20–50 turns of interaction. The flagship demo is training agents on **SWE-Bench-Verified** (real GitHub issues) with as few as ~300 training samples.

The released models:
- **SkyRL-Agent-7B-v0** (from OpenHands-7B-Agent / Qwen2.5-Coder-7B-Instruct): SWE-Bench-Verified **11.0% → 14.6%**
- **SkyRL-Agent-8B-v0** (from Qwen3-8B, no thinking): **3.6% → 9.4%**
- **SkyRL-Agent-14B-v0** (from Qwen3-14B, thinking-enabled): **18.0% → 21.6%**

Authored by Shiyi Cao, Sumanth Hegde, Dacheng Li, Tyler Griggs, Shu Liu, Eric Tang, Jiayi Pan, Xingyao Wang, Akshay Malik, Kourosh Hakhamaneshi, Richard Liaw, Philipp Moritz, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica. Berkeley Sky Computing Lab + Anyscale + All Hands AI. 2025.

## The problem this exists to solve

Most "RL for LLM agents" frameworks today are good at **stateless, short-horizon** tasks: search-augmented reasoning (Search-R1), or one-shot tool-call traces (ToRL). These run a model for a handful of turns against a lightweight tool, get a reward, learn.

That stack breaks when you try to train agents on tasks like SWE-Bench, for two compounding reasons:

1. **The environment is heavy and stateful.** Each rollout in SWE-Bench needs an **isolated Docker container** holding a real Python repo. A container costs ~1+ CPU and ~7 GB of disk. A modest configuration (batch size 16, 8 rollouts each → 128 environments concurrently) needs **100+ CPUs and ~1 TB of disk** just to *exist*. If you co-locate environments with training nodes, you can't scale them independently — you starve the LLM inference engine of requests, and your GPUs sit idle.

2. **The rollout horizon is 20–50 turns**, not 2–3. A single trajectory has the agent doing: identify relevant files → write a reproducer test → edit code → run tests → iterate. The naive synchronous rollout pattern (used by VeRL, OpenRLHF for single-turn RL) inserts a synchronization barrier at every turn — the system waits for *all* trajectories to finish their step before issuing the next batch of LLM calls. With heterogeneous step durations (listing a file is fast, running pytest is slow) this leaves enormous bubbles of GPU idleness and prevents continuous batching in modern serving engines like SGLang and vLLM from kicking in.

So the problem is **infrastructure**, not algorithm. SkyRL-v0 explicitly punts on RL algorithm research ("not the focus of this blog") and ships solutions to the system bottlenecks. They claim **4–5× speedup over a naive baseline rollout implementation.**

## Where it sits in the landscape

SkyRL is **built on top of** [VeRL](https://github.com/volcengine/verl), which gives it the learning-algorithm machinery (PPO, GRPO, etc.). VeRL handles single-turn RL well; SkyRL adds an **agent layer** on top that does three things VeRL doesn't:

1. Async, multi-turn rollouts.
2. Generic tool-use abstraction.
3. Generic, scalable environment execution.

For the agent loop itself, SkyRL plugs in **OpenHands' CodeAct abstraction**: the agent emits Python-like tool calls (`execute_bash`, `str_replace_editor`, `finish`), the environment runs them, observations come back, repeat.

Concurrent / sibling work the post calls out:
- [RAGEN](https://github.com/RAGEN-AI/RAGEN), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [ReTool](https://arxiv.org/pdf/2504.11536) — multi-turn RL but limited interaction depth, fewer turns per trajectory.
- These connect to existing vault notes on [[reinforcement-learning-implementation-complexity-barrier]] and [[reinforcement-learning-in-language-models]] — SkyRL is the engineering response to that "implementation barrier" framing for the long-horizon-agent regime.

## System design — the two big technical contributions

### Contribution 1: Remote Sandbox Server (decouple environments from training)

The natural impulse is to run environment containers on the same nodes as the training workers. SkyRL refuses. Instead they deploy a **dedicated Kubernetes-managed sandbox server** that's physically separate from the training cluster. Trajectories request an environment over the network; it spins up a container, runs actions, returns observations, tears down.

Why this matters mechanically:
- **Independent scaling.** If you need more concurrent rollouts to keep GPUs fed, you scale the sandbox cluster without touching the training cluster.
- **Resource isolation.** A misbehaving rollout (infinite loop, OOM in a container) can't take down a training worker.
- **Storage caching.** The sandbox cluster uses storage-optimized instance types specifically to cache container images — this matters because cold-starting a SWE-Bench container that has to pull a Python repo image is the slow path.

Implementation details worth noting:
- They use [`aidocker`](https://aiodocker.readthedocs.io/) for async Docker control (avoids blocking on `docker run`).
- They use [`crun`](https://github.com/containers/crun) instead of `runc` — crun is a Rust/C container runtime with lower per-container startup overhead than the default Go runc.
- A single 16-CPU node sustains **80–100 containers per replica** without stability issues.

The non-obvious design choice: **disaggregating environments from training is a system-level architecture decision, not a tuning knob.** It changes the topology of the cluster. They open-sourced the server code and the Kubernetes deployment configs because most labs trying to reproduce SWE-Bench RL training won't have engineers who would naturally reach for K8s + crun.

### Contribution 2: Asynchronous rollouts + 3-stage producer-consumer pipeline

The naive multi-turn rollout (top of Figure 4 in the post) looks like:

```
for turn in range(max_turns):
    actions = batch_generate(model, all_trajectories)   # SYNC barrier
    observations = batch_step_env(actions)              # SYNC barrier
```

This synchronizes on the slowest trajectory at every turn. If trajectory A finishes turn 5 in 1s and trajectory B's turn 5 takes 30s (running pytest), the entire batch waits 30s before turn 6 starts. Continuous batching inside SGLang/vLLM never gets a chance to kick in because the engine sees a synchronous offline-batch API.

SkyRL replaces this with **two stacked optimizations**:

**Optimization 1: Async per-trajectory generation.** Each trajectory is its own coroutine, calling `async_generate()` on SGLang or vLLM independently. Trajectory A immediately starts turn 6 the moment its turn 5 completes — no waiting for B. The serving engine now sees a continuous stream of independent requests and can apply continuous batching naturally.

**Optimization 2: Three-stage producer-consumer pipeline.** The full lifecycle of a rollout has three distinct phases with different bottlenecks:

1. **Init phase** — build/pull container image, launch container, connect to it. (CPU/IO-bound, *long tail* — first time you see a repo image is slow.)
2. **Run phase** — multi-turn LLM↔env interaction. (GPU + env-CPU bound.)
3. **Eval phase** — apply the agent's git patch, run the test suite, compute reward. (Env-CPU bound, *highly variable*.)

These run as three concurrent stages connected by **bounded async queues**:

```
[init_queue] → initializer → [run_queue] → agent_runner → [eval_queue] → evaluator
```

Each stage is a pool of `asyncio` tasks that dynamically spawn based on queue occupancy. The bounded queues create natural **back-pressure**: if eval is slow, run-queue fills up, agent_runner pauses, and the system self-balances without explicit scheduling logic.

**Why this overlaps work that would otherwise serialize:** while trajectory A is in eval (running tests, slow), trajectory B is in run (LLM generation, GPU-bound), and trajectory C's container is initializing. All three resource types — CPU for init, GPU for inference, CPU for tests — are in use simultaneously. In the naive implementation only one of these runs at a time per batch step.

Combined claim: **4–5× speedup** vs. the synchronous baseline.

## Training recipe

### Algorithm

The post explicitly says algorithm work isn't the focus — they use **GRPO and PPO** from VeRL's existing implementations. The interesting observation is what *fails*: when you naively put a 7B model on raw SWE-Gym (where even GPT-4o gets 4.55–9.13%), the 7B model produces zero successful rollouts, so the reward signal is uniformly 0, so GRPO/PPO have nothing to learn from and **training collapses**. The fix is data filtering, not algorithm change.

### Reward — outcome-based, almost embarrassingly simple

```
reward = 1 if patch applies cleanly AND tests pass for the targeted GitHub issue
reward = 0 otherwise
```

That's the whole reward function. They additionally monitor (but don't penalize directly in the released setup) two failure modes:
- **Stuck in a loop** — same action repeated 3 turns in a row.
- **No finish action** — agent never emits `finish()` within the turn budget.

The post mentions these tracking signals correlate with reduced looping and timely completion in their experiments. This is consistent with [[reinforcement-learning]] folklore that for long-horizon problems, even sparse outcome rewards can shape behavior if the rollout system is honest enough to terminate bad trajectories early.

### Data filtering — the actually-load-bearing detail

Raw SWE-Gym has 2,438 Python tasks. They construct **three filtered subsets** keyed on what models can already partially solve:

| Dataset | # Instances | Filter criterion |
|---|---|---|
| SkyRL-v0-80-data | 80 | OpenHands-7B-Agent can solve at least 1/16 rollouts |
| SkyRL-v0-220-data | 220 | OpenHands-32B-Agent can solve at least 1/16 rollouts |
| SkyRL-v0-293-data | 293 | gpt-4o-2024-08-06 or claude-3-5-sonnet-20241022 solves it correctly |

The non-obvious finding: **the right training data for a 7B agent is the data that the 7B can occasionally already solve.** Tasks the model can never solve give zero gradient signal under outcome rewards; tasks the model already always solves teach nothing new. Filtering by "can the model do this *sometimes*" gives the meaty middle band where RL has gradient to push on.

### Rollout structure (CodeAct)

Pseudocode from the post, reformatted:

```python
# Inputs: model M, environment E, task description P, action set A
history = [P]  # OpenHands manages turn history; init with task

while not done:
    action = M(history, A)            # model emits a tool call
    done, observation = E(action)      # env executes; returns obs and termination flag
    history.append(action)
    history.append(observation)

patch = E.get_patch()  # final git diff written by the agent
return patch
```

Three available actions: `execute_bash`, `finish`, `str_replace_editor`.

## Evaluation setup

- Benchmark: **SWE-Bench-Verified**.
- Scaffold: **OpenHands** with the **CodeAct Agent** template — same action set as training (`execute_bash`, `finish`, `str_replace_editor`).
- Max turn budget: **50**.
- Max sequence length: **32k**.
- Eval code at: [SkyRL-OpenHands](https://github.com/NovaSky-AI/SkyGym-OpenHands/tree/main/evaluation/benchmarks/swe_bench).

## What's surprising / worth noting

1. **You can move the SWE-Bench-Verified needle on a 7B model with ~300 training examples.** The 7B → 14.6% result used the 80-instance filtered set. This is genuinely small-data RL — much of the win is the framework, not the data scale.

2. **Disaggregating environment from training is presented as the *first* contribution, before anything algorithmic.** This is a refreshing inversion of the usual ML-paper rhetoric where systems work is buried in an appendix. They're explicitly making the case that for long-horizon-agent RL, **infrastructure is the bottleneck, not the loss function.**

3. **The producer-consumer pipeline is ordinary distributed-systems craft.** None of `asyncio.Queue`, bounded back-pressure, or three-stage pipelining is novel as a pattern. The contribution is *recognizing that RL rollout for agents has the right shape for it* and being willing to build it.

4. **They open-sourced the K8s configs.** Most "we ran on a Berkeley cluster" papers leave reproduction as an exercise. Shipping the deployment YAML lowers the bar from "have a Berkeley cluster" to "have a K8s account."

5. **The 8B Qwen3 model jumps 3.6% → 9.4%, the largest *relative* gain in the set (2.6×).** Worth understanding why the non-thinking variant gains more than the thinking-enabled 14B. Possibly because the non-thinking model has more headroom — it's leaving capability on the table that the agent loop can recover.

## Limitations and honest gaps

The post itself flags some, others are visible by what's missing:

- **Single environment family.** v0 only supports SWE-Bench. WebArena, WebDev, and other long-horizon benchmarks are listed as future work. Their "generic environment execution" claim is partly aspirational until proven on a second domain.
- **Algorithm work is explicitly deferred.** They use stock GRPO/PPO from VeRL. No claims about a long-horizon-specialized algorithm. If outcome-only sparse reward turns out not to scale beyond SWE-Bench, this becomes the bottleneck.
- **No ablation of the 4–5× speedup claim** in the post itself — we don't see "async only" vs "async + pipeline" vs "naive" decomposed numbers.
- **No analysis of what the trained agents *learn***. The post is a system+results post, not an interpretability post. Did the 7B learn to use the editor better? To loop less? Both? Unknown from this writeup.
- **Reward hacking risk is unaddressed.** Outcome-based reward on `tests pass` invites the agent to discover degenerate fixes (skipping the failing test, hardcoding outputs). The blog mentions tracking loop/finish failures but doesn't discuss test-gaming.
- **Sample inefficiency vs. SFT comparison missing.** Would 300 SFT examples on the same data move the score this much? The post doesn't include this baseline.

## Connections to existing vault knowledge

- [[reinforcement-learning-implementation-complexity-barrier]] — SkyRL is a direct, infrastructure-shaped attack on the implementation complexity that gates this kind of work. Worth updating that note with SkyRL's "decouple env from training" pattern as a concrete remediation strategy.
- [[reinforcement-learning-in-language-models]] — extends the multi-turn-RL strand with explicit long-horizon support.
- [[reinforcement-learning-from-human-feedback-rlhf]] — different reward structure (outcome-based, programmatic) but same on-policy update mechanics.
- [[asynchronous-execution-and-fallback-strategy-in-agent-systems]] — SkyRL's 3-stage queue pipeline is a concrete instance of the async pattern in this note; the back-pressure-via-bounded-queue technique is worth atomizing into its own zettel.
- [[asymmetric-learning-rates-for-actor-critic]] — orthogonal but related: SkyRL inherits PPO/GRPO from VeRL without modification, so the actor/critic LR asymmetry question is downstream of their stack.

## What I'd want to extract as zettels

Candidate atomic concepts surfaced by this source (pre-ingest list):

1. **Disaggregating RL environments from training compute** — the architectural pattern, when to reach for it, what it costs.
2. **Three-stage producer-consumer pipelining for agent rollouts** — init/run/eval as separable stages with bounded queues.
3. **Data filtering by partial-success rate** — train on tasks the model can solve sometimes, not always or never.
4. **Async per-trajectory generation unlocks continuous batching** — why sync-step rollouts defeat modern serving engines.
5. **Outcome-based sparse reward for long-horizon code agents** — when it works, when it collapses (zero-success regime).
6. **CodeAct as a tool-use abstraction** — three actions cover most software work.
7. **`crun` over `runc` for high-density container deployment** — concrete optimization with measurable yield (80–100 containers / 16-CPU node).


