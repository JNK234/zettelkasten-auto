Created: 20-08-2026 00:20
Status: #baby
Tags: [[reinforcement-learning-in-language-models]] [[systems-engineering]] [[distributed-systems]]

# Colocated Environments Starve the Inference Engine

## What This Is
The intuitive reason not to run agent environments on your training node is "they're heavy, they'll eat resources." That's true but it's the weak half of the argument. The strong half is that the environment constraint and the throughput constraint are *coupled*, and colocation makes you lose both at once.

Walk the chain. Agentic environments are heavy — a pinned Docker container, a browser session, an allocated GPU. If you host them on the training node, the number you can run concurrently collapses to something small; the source's example is a rollout batch of 8. That's the first loss: the algorithm wanted 128 prompts × 4 generations = 512 concurrent rollouts, and you can't have them.

The second loss is the non-obvious one. A batch of 8 **starves the LLM inference engine**. vLLM and SGLang are throughput machines that want hundreds of concurrent sequences to fill the GPU and amortize the KV cache. Feed them 8 and the serving GPUs sit largely idle. So you have simultaneously failed to get the parallelism the algorithm needs *and* wasted the expensive hardware you bought to serve it.

## Why It Matters
This is the load-bearing justification for decoupling execution from training. If the only cost of colocation were "fewer environments," you could argue about whether 8 is enough. Because the costs compound — small batch also means idle serving GPUs — there is no batch size at which colocation is a good deal. You cannot tune your way out of it; you have to move the environments.

Generalizing: when two subsystems have opposite scaling preferences (environments want few-and-heavy, inference wants many-and-concurrent) and you place them on shared hardware, the binding constraint of one becomes an efficiency tax on the other. Physical separation is the only fix.

## Key Details
- Source's concrete numbers: local hosting forces batches around 8; the algorithmic target is 128 prompts × 4 generations = 512 live environments per iteration.
- Environment cost is not just heavy but **heterogeneous** — a web-search rollout and a code-execution rollout in the same batch share nothing resource-wise, so you can't even size a uniform slot for them.
- Once decoupled, environments run on cheap CPU nodes that scale independently of the GPU fleet — a different unit-economics regime entirely.
- Uncosted by the source: decoupling means paying for a separate CPU/container fleet alongside the GPUs, and the crossover point is never quantified.

## Tensions
This is [[training-efficiency-vs-inference-speed-tradeoff]] surfacing somewhere people don't expect it — not inside the model, but between the environment fleet and the serving fleet. It is the direct argument for [[agent-layer-inverts-who-owns-the-generation-loop]]. Note the difference from [[colocated-versus-disaggregated-rl-training-and-inference]]: there, colocation genuinely wins on some axes (no idle GPUs during backward, cheap weight sync); here it loses on every axis, because the third party competing for the node is not a trainer but an unbounded container workload. The starvation mechanism is the same one described in [[rollout-generation-as-the-online-rl-bottleneck]] and [[kv-cache-pool-size-versus-declared-max-sequence-length]] — concurrency, not FLOPs, is what fills a decode-bound GPU. Compare [[zero-advantage-groups-shrink-the-effective-batch]] for the algorithmic version of "your effective batch is smaller than you think."

# References
1. [[Jiachen Liu - When LLMs Grow Hands and Feet, How to Design our Agentic RL Systems]]
