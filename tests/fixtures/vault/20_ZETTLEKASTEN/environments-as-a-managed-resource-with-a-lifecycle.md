Created: 20-08-2026 00:20
Status: #baby
Tags: [[systems-engineering]] [[distributed-systems]] [[reinforcement-learning-in-language-models]]

# Environments as a Managed Resource with a Lifecycle

## What This Is
The default instinct when an agent needs a sandbox is to spawn one inline: `docker run`, do the work, tear it down. At one rollout that is fine. At 512 concurrent rollouts per training iteration it collapses, and it collapses for reasons that have nothing to do with the agent.

Three mechanisms, all variations on treating an environment as a pooled resource rather than something you create on demand:

**Remote execution services** (rStar2-Agent, SkyRL). A central scheduler dispatches tasks to a large pool of remote workers. The point is not parallelism per se — it is that the trainer's thread never sits inside a `docker run`. Environment interaction stops blocking the training loop, and the worker pool scales on cheap CPU nodes independently of the expensive GPU nodes.

**Efficient sandbox infrastructure** (Kimi-Researcher, GLM-4.5, Daytona, SkyRL). Once you churn hundreds of short-lived containers, the fixed costs dominate. Two fixes that sound like plumbing and are actually first-order: cache container images on storage-optimized nodes, because at 512 rollouts/iteration image *pull* time beats execution time; and swap the container runtime from `runc` (Go) to `crun` (C), which has markedly lower per-container startup latency and memory overhead. When container lifetime is seconds-to-minutes, startup overhead is a leading term.

**Centralized environment pools** (AgentFly). For *stateful* tools — a filesystem, a browser session — each task needs its own isolated instance, but cold-starting one per task is waste. Keep warm pools: allocate on task start, return on completion. This is connection pooling applied to sandboxes, and it attacks setup latency, which for a browser or a dependency-heavy image can dwarf the actual agent work.

## Why It Matters
The through-line is that agentic RL drags container-startup latency into the same optimization budget as attention kernels. That is a genuinely unusual claim about where engineering effort should go, and it is what makes a 512-way rollout affordable on a handful of CPU nodes instead of a fleet.

The density number is the payoff: SkyRL reports **80–100 containers per replica on a 16-CPU node** — roughly 5–6 containers per core. That is the single quantitative claim in the source, and it is what converts "512 concurrent environments" from an infrastructure project into a config file.

## Key Details
- Orchestration in SkyRL's stack is Kubernetes; Daytona sits one level up, abstracting container lifecycle behind provisioning APIs so the RL team never writes YAML.
- Fault tolerance, state checkpointing, and reproducibility are *named* as difficulties and then dropped — the source's solutions cover scale and isolation only. Nothing on a container dying mid-rollout, on making a stateful multi-step rollout deterministic for debugging, or on a flaky external API poisoning the reward signal.
- Security is one sentence. Running untrained model output as code inside containers at 512-way parallelism is a live adversarial surface, and sandboxing is discussed purely as *resource isolation*, not as a security boundary.
- Sept-2025 snapshot: SkyRL v0, rLLM, Agent-Lightning, AgentFly, Agentarium were all young. Treat configuration specifics as illustrative.

## Tensions
This is the "somewhere else" that [[colocated-environments-starve-the-inference-engine]] demands, and the bottom tier of [[agent-layer-inverts-who-owns-the-generation-loop]]. The scheduling it enables is refined by [[kv-cache-aware-scheduling-for-long-horizon-rollouts]], which points out that once environments are cheap, the next scarce resource is serving memory. The warm-pool idea is the same amortization logic as [[paged-kv-cache-management]] — both replace worst-case per-request allocation with a shared, reused pool. What it does *not* solve is credit assignment or sparse reward; the source concedes systems fixes can't touch those, which is [[credit-assignment-problem]] and [[agentic-rl-as-a-pomdp]] territory. Compare [[sandbox-environment-with-procedural-events-and-resources]] for environments designed for simulation rather than throughput.

# References
1. [[Jiachen Liu - When LLMs Grow Hands and Feet, How to Design our Agentic RL Systems]]
