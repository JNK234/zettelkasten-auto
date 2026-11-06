Created: 20-08-2026 00:20
Status: #baby
Tags: [[reinforcement-learning-in-language-models]] [[distributed-systems]] [[systems-engineering]]

# Agent Layer Inverts Who Owns the Generation Loop

## What This Is
In a chat-RL framework the trainer owns the generation loop. It holds the prompts, calls the inference engine, pulls tokens out, hands the string to a reward model, and computes a gradient. The rollout lives *inside* the training process.

That ownership stops working the moment the rollout stops being a forward pass. An agentic rollout opens a Docker container, hits the network, may occupy a GPU for twenty minutes, and may never terminate. Putting that inside the trainer means the training process now owns container lifecycles, network flakiness, and unbounded wall-clock.

The Agent Layer is the fix, and the fix is an inversion: **the environment calls the LLM, not the other way around.** A three-tier decomposition —

- **RL Framework** — training engine (weight updates) plus inference engine, which exposes an OpenAI-style HTTP endpoint.
- **Agent Layer** — a scheduler that dispatches rollout tasks, hands each one an inference endpoint, and captures returned trajectories into a replay buffer.
- **Agent Execution Environments** — distributed containers, sandboxes, browsers, GPUs, on their own machines.

The agent runs its own loop on a remote worker and treats the trainer's inference engine as just another API endpoint — the same one it would hit in production. The RL system's entire interface narrows to two obligations: serve an endpoint, accept a trajectory.

## Why It Matters
That narrow interface is what buys everything downstream. Because the agent talks HTTP, an agent written in LangGraph, a bespoke ReAct loop, or an OpenAI-Agents-SDK app all train unchanged — no rewriting into a framework DSL. Because environments live on their own machines, their resource appetite stops competing with training GPUs. Because trajectories arrive asynchronously into a buffer, slow rollouts stop blocking fast ones.

The deeper point is about where the bottleneck went. For chat RL the bottleneck was inference throughput plus reward scoring — both GPU-bound, both a kernel problem. For agentic RL the bottleneck moves off the GPU entirely and becomes orchestration and scheduling of thousands of heterogeneous environments. Your RL system's performance is now a distributed-systems problem.

## Key Details
- Reference implementation the source points at: Agentarium (`github.com/Agentarium-AI/agentarium`).
- The endpoint being OpenAI-shaped is load-bearing, not cosmetic — it is the reason arbitrary third-party agent code works without modification.
- Interfaces flow in both directions across the middle tier: trajectories go up to the trainer, an LLM API endpoint comes down to the environment, dispatch goes down and results come back up.
- The source is a Sept-2025 landscape survey with essentially no numbers; treat the architecture as the durable claim and the named systems as illustrative.

## Tensions
This is the agentic-scale version of the placement question in [[colocated-versus-disaggregated-rl-training-and-inference]] — but where that tradeoff was about GPU idle time versus memory contention, here disaggregation is not a tuning choice, it is structurally forced by the environment's weight. The middle tier is a control plane in the sense of [[control-plane-vs-data-plane-in-distributed-training]]: it moves dispatch decisions and trajectory handles, not activations. It only works if the trajectory format is standardized, which is [[trajectory-capture-by-instrumenting-the-llm-api-boundary]], and it only pays off if the environments underneath are pooled and cheap, which is [[environments-as-a-managed-resource-with-a-lifecycle]]. The direct motivation is [[colocated-environments-starve-the-inference-engine]]. Compare [[agents-environment-interface-types]] for what "environment" can mean, and [[agentic-rl-as-a-pomdp]] for why the rollout became stateful in the first place.

# References
1. [[Jiachen Liu - When LLMs Grow Hands and Feet, How to Design our Agentic RL Systems]]
