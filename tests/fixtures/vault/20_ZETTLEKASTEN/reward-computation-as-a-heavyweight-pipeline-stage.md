Created: 20-08-2026 00:20
Status: #baby
Tags: [[reward]] [[systems-engineering]] [[reinforcement-learning-in-language-models]]

# Reward Computation as a Heavyweight Pipeline Stage

## What This Is
In chat RL, computing a reward is a forward pass through a reward model. It is GPU work, it is fast, it is deterministic, and it never fails. Every RL framework's design quietly assumes all four.

Agentic RL breaks all four. The reward for "did the agent fix this bug" is *run the test suite in a container*. The reward for "did the agent find this fact" is *hit the live web and check*. That means reward computation is now: heavyweight (minutes, not milliseconds), CPU/container-bound rather than GPU-bound, non-deterministic (flaky tests, rate-limited APIs, network timeouts), and capable of failing outright.

The systems consequence is that reward stops being an appendix to the rollout and becomes its own scheduled stage. SkyRL makes this explicit with a three-stage fine-grained pipeline — `initialize → rollout → reward calculation` — each a separate producer-consumer segment. While task A is being scored, task B is mid-rollout and task C's container is warming. Pipelining converts sequential per-task latency into overlapped throughput, and the win scales with how uneven those stages are.

## Why It Matters
Two distinct costs, usually conflated.

The **throughput cost**: if reward takes minutes and you compute it inline at the end of each rollout, you have added a long serial tail to every trajectory. Treating it as a pipeline stage is what recovers the overlap.

The **signal cost**, which is worse and less discussed: an external tool that flakes produces a *wrong reward*, not an error. A test that fails for network reasons tells the policy its correct solution was wrong. External-tool flakiness is a genuine source of reward noise, and it is noise that no amount of algorithmic care downstream can distinguish from real failure. Your reward infrastructure's reliability becomes an upper bound on your learning signal's quality.

## Key Details
- The source treats this as a gap in its own coverage: reward computation is acknowledged as a pipeline stage but never as a first-class design challenge. Arguably it deserves to sit alongside agent integration, environment management, and stragglers.
- Nothing is offered on how to make a stateful multi-step rollout reproducible for debugging, or how to detect that a reward was corrupted by infrastructure rather than by the policy.
- Open question the source poses: how do you get reproducible rollouts when the environment includes a live web?
- This is the reward analogue of the general lesson elsewhere in the piece — a component that was a forward pass in chat RL becomes a distributed-systems problem in agentic RL.

## Tensions
This is the infrastructure underneath [[verifiable-rule-based-rewards-prevent-reward-hacking]] and [[rlvr-reinforcement-learning-with-verifiable-rewards]]: "verifiable" is an algorithmic virtue that cashes out as an operational liability, because verification means *execution*. It sits inside the pipelining logic of [[partial-rollouts-produce-policy-heterogeneous-trajectories]] and depends on [[environments-as-a-managed-resource-with-a-lifecycle]] for the containers it runs in. Flaky-reward noise is a different failure than [[reward-hacking-as-specification-gaming]] — hacking is the policy exploiting a correct-but-gameable signal, this is the signal itself being wrong at random, and the two need different diagnostics. Compare [[execution-based-tool-evaluation]] for the eval-time version, and [[four-layer-scoring-stack-for-rl-environments]] for how scoring gets decomposed. Where reward is terminal over 30 tool calls, cheap per-step shaping from tool outcomes is tempting but invites gaming — see [[credit-assignment-problem]] and [[process-reward-model-vs-outcome-reward-model]].

# References
1. [[Jiachen Liu - When LLMs Grow Hands and Feet, How to Design our Agentic RL Systems]]
