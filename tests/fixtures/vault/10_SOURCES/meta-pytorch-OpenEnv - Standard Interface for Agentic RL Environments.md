Created: 24-09-2026 16:19

Tags:

Status: #baby

## References

1. [OpenEnv: Agentic Execution Environments (meta-pytorch/OpenEnv, now huggingface/OpenEnv)](https://github.com/meta-pytorch/OpenEnv)

# Notes

Decoded from a full clone of the repo on 24-09-2026: README, all RFCs (000, 001, 002, 003, 004, 005, 008, 010, 011, 012), the `.claude/docs` design files (PRINCIPLES, INVARIANTS), the scaling tutorial, and the core source (`src/openenv/core`). The GitHub URL `meta-pytorch/OpenEnv` now redirects to **`huggingface/OpenEnv`**. The repo was created 2025-10-01, has about 2.6K stars, and was pushed the same day as this note. It is very active.

---

## 1. The problem: "environment" means five different things

By late 2025 every RL-for-LLMs stack (TRL, veRL, SkyRL, Prime Intellect's verifiers, torchforge, ART, Unsloth) had its own idea of an environment. Each one mixed some of these together:

- a **Gym-style simulator** (`reset`/`step`) that holds state,
- a **sandbox** (Docker, E2B, Modal, Daytona) that holds files and runs code,
- a **tool surface** (usually MCP) that the model calls,
- a **reward function** (verifier, unit tests, LLM judge),
- a **task dataset** (prompts plus ground truth).

So an environment built for one trainer could not run on another without an adapter, and the adapter usually broke something: rewards computed in a different place, tools exposed a different way, reset semantics that leaked to the agent. RFC 001 says it directly: "The term 'Environment' is an overloaded term in a field currently lacking established abstractions and terminology."

OpenEnv's bet is that **the environment is a network service with a fixed contract**. You package it once (Docker image plus typed Pydantic models) and put it on the Hugging Face Hub as a Space. After that, any trainer can run it at scale over a WebSocket. The ambition is to be the POSIX for RL environments: a boring shared interface so environment builders reach every trainer and trainers reach every environment. The README credits Farama's Gymnasium as the main inspiration. See [[Epoch AI (Denain & Barber) - An FAQ on Reinforcement Learning Environments]] for why environments became the bottleneck, and [[environment-as-installable-package-not-config]] for the same packaging idea in another stack.

---

## 2. Who is behind it (the company angle)

This part matters for reading the design as a signal of what the companies care about.

**Origin.** It started as a Meta-PyTorch project. The four authors on RFCs 000–004 are @Darktex, @pankit-eng, @jspisak and @zkwentz. v0.1 launched on 22 Oct 2025. It was built as the environment layer for **torchforge**, PyTorch's agentic RL framework. The featured example is GRPO BlackJack on torchforge. RFC 002 even sketches an "internal RL training stack" built on top of OpenEnv: 100% CodeAct, TorchForge for async RL, "TimeWalk for tree search", "tool-aware checkpointing". That reads like Meta describing its own internal distribution of OpenEnv. The analogy they give: OpenEnv is Linux, and in-house stacks are Ubuntu or Red Hat.

**It now lives under Hugging Face.** The canonical repo is `huggingface/OpenEnv`, the docs are on `huggingface.co/docs/openenv`, and every environment ships as an HF Space (`pip install git+https://huggingface.co/spaces/openenv/echo_env`). HF owns the distribution channel (the Hub) and the main trainer (TRL). RFC 008 is careful to keep "HF-as-hub" (an operator like any other, with no special code in the repo) separate from "HF-as-sandbox-provider" (a local primitive). That tells you there is some tension about HF becoming the default gatekeeper.

**Technical committee** (README, current): Meta-PyTorch, Reflection, Unsloth, Modal, Prime Intellect, Nvidia, **Mercor**, **Fleet AI**, Microsoft, Hugging Face, RadixArk, Nebius.

**Supporters:** Scaler AI Labs, Patronus AI, Surge AI, LastMile AI, vLLM, SkyRL (Berkeley), Lightning AI, Axolotl, Stanford Scaling Intelligence Lab, Mithril, OpenMined, Halluminate, Turing, Scale AI, Scorecard, Snorkel AI, SGLang, Miles.

What each group does in the repo, from evidence in the code and docs:

| Company / group | Concrete footprint in the repo |
|---|---|
| **Meta-PyTorch** | Core abstractions, RFCs 000–005, torchforge integration, "Principles" and "Invariants" docs. @Darktex is the named reviewer for alignment flags. |
| **Hugging Face** | Hosting (Spaces), TRL GRPO integration, docs, `openenv push`, and the HF Sandbox validation provider. @adithya-s-k wrote the Harbor capture RFC 012. |
| **Microsoft** | `thinkingbox_env` (microsoft/ThinkingBox). RFC 010 ECHO is based on `microsoft/echo-rl`. There is an Azure Container Apps provider (`aca_provider.py`) and a "Foundry Fine-Tuning" backend in the ECHO example. RFCs 002-amendment, 010 and 011 are by @thegovind. |
| **Prime Intellect** | On the committee. Its "True Agents Model the World" post is cited in RFC 010. `sophistry_bench_sprint_env` says "reward numbers are identical to the Prime Intellect Hub env". Parity with PI's Environments Hub is an explicit goal (see [[Prime Intellect - Environments Hub]]). |
| **Snorkel AI** | `finqa_env`: 290 SEC 10-K questions from `snorkel-ai/FinQABenchmark`. Data is `snorkelai/finqa-data`, "evaluation only". |
| **Turing** | `calendar_env` (a "Calendar Gym" MCP tool set). Its notebook is named `OpenEnv_and_mcp_Single_Gym_Client_Meta_Turing.ipynb`, so it looks like a Meta × Turing collaboration. |
| **RadixArk / Miles** | Terminal-Bench-2 GRPO example in Miles. `tbench2_env` is in the repo. |
| **Modal, Nebius, Daytona, Novita, E2B** | Container / sandbox providers (`modal_provider.py`, `daytona_provider.py`, `novita_provider.py`). E2B backs `coding_tools_env`, `opencode_env`, `terminus_env`. |
| **Unsloth, SkyRL, ART (OpenPipe), Oumi, Lightning** | Trainer integrations listed in the README (Unsloth 2048 with gpt-oss-20B, SkyRL example, and so on). |
| **Mercor, Fleet AI, Scale, Surge, Patronus, Halluminate, Scaler, Turing** | These are the **environment/data vendors**. They sell environments and expert tasks to frontier labs. They get the most from a shared format: build once, sell to every lab's trainer. |

**Reading the incentives** (my inference, not stated in the repo): the committee is almost exactly the two sides of the environment market. On one side are *buyers/trainers* (Meta, Reflection, Nvidia, Microsoft, Prime Intellect, Unsloth, RadixArk). On the other are *sellers/builders* (Mercor, Fleet, Turing, Scale, Surge, Snorkel, Patronus, Halluminate) plus *compute* (Modal, Nebius). RFC 008's phrase "An environment is only useful for improving models if a **frontier-lab intake** would accept it" gives it away. The six-property quality bar in issue #778 is basically **the acceptance spec a lab uses when buying an environment from a vendor**. If you want to understand what Mercor, Fleet, Turing or Scale are paid for, RFC 008's 44 tests are the most concrete public statement of it. (See also [[Mercor Post-Training Assessment - GRPO Rollout Scheduling Under a Fixed Budget]], [[Mercor benchmarking Take Home]], [[Patronus AI - RL Environments Tutorial and Examples]], [[HUD - Verifier and Reward Design for RL Environments]].)

---

## 3. The four design principles (RFC 000)

1. **Minimize lifecycle deltas.** Training, evals and production should use *identical* tool interfaces.
2. **Minimize human–agent divergence.** Tools that work for humans should work for agents.
3. **Be hands-on.** Ship working implementations, not just specs.
4. **Design for LLMs.** Optimize for context efficiency, in-distribution behavior and token cost.

Principle 1 drives almost every non-obvious decision below. The worry is a **train/prod gap**: a model trained against a mocked tool with one signature, then deployed against a real API with another, is out of distribution the moment it ships.

Roadmap (RFC 000): Phase 1 (to v0.3) covers only the narrow environment, meaning sandboxing, versioning, and MCP tools. Phase 2 (to v0.6) adds rewards. Phase 3 (to v0.9) adds evals, from scripted to LLM-judged to fully agentic. Breaking changes are allowed until 1.0, "can be fixed by LLM coders". In practice the phases overlapped. The rubrics (RFC 004) and a harness/eval layer already exist.

---

## 4. The core abstraction, simple version

Three calls, as in Gym:

```python
obs = env.reset(seed=..., episode_id=...)   # new episode
obs = env.step(action)                      # act, get observation (+ reward, done)
st  = env.state                             # episode metadata (episode_id, step_count, ...)
```

The environment runs as a **FastAPI server in a Docker container**. Your training code holds a typed **client** (`EnvClient[ActT, ObsT, StateT]`) that talks to it over a **WebSocket** (`/ws`). Async is the default. `.sync()` gives a blocking wrapper.

```
Trainer process                         Docker container (HF Space / Modal / local)
┌───────────────────────┐   WebSocket   ┌──────────────────────────────────┐
│ EchoEnv(EnvClient)    │──reset/step──▶│ FastAPI HTTPEnvServer            │
│ Action/Obs Pydantic   │◀─observation──│  └─ EchoEnvironment(Environment) │
└───────────────────────┘               │      ├─ rubric (reward)          │
                                        │      ├─ MCP tools                │
                                        │      └─ sandbox state            │
                                        └──────────────────────────────────┘
```

Scaffold: `openenv init my_env` creates `models.py` (Action/Observation/State), `client.py`, `server/your_environment.py`, `server/app.py`, a `Dockerfile` and an `openenv.yaml` manifest. `openenv push` deploys to a Space. `openenv import` wraps third-party formats, including **ORS/OpenReward and Verifiers** (Prime Intellect's format). That import path is the interoperability wedge.

---

## 5. The technical layers

### 5.1 Wire types: strict Pydantic

From `src/openenv/core/env_server/types.py`:

```python
class Action(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    metadata: Dict[str, Any] = {}

class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", ...)
    done: bool = False
    reward: bool | int | float | None = None
    metadata: Dict[str, Any] = {}

class State(BaseModel):
    model_config = ConfigDict(extra="allow", ...)   # note: State is permissive
    episode_id: Optional[str]; step_count: int = 0
```

Things worth noticing:

- **`extra="forbid"` on Action and Observation.** A malformed action from the policy (a hallucinated field) is rejected at the boundary with a `VALIDATION_ERROR` WebSocket code. It is not silently ignored. For RL this matters: a silently ignored field is a free no-op the policy can learn to exploit.
- **Reward lives on the Observation**, not in a separate return tuple. That is a change from Gym's `(obs, reward, terminated, truncated, info)`. Everything the server computes comes back in one typed object.
- **State is `extra="allow"`.** It is internal metadata, not an agent-facing contract.
- `ServerMode` is an enum `{SIMULATION, PRODUCTION}`. That enum is how one codebase supports the train/prod split (5.3).
- The WebSocket error codes include `CAPACITY_REACHED`: sessions per container are capped (5.6).

The server base class (`interfaces.py`):

```python
class Environment(ABC, Generic[ActT, ObsT, StateT]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = False
    REQUIRES_SINGLE_THREAD_EXECUTOR: bool = False
    def __init__(self, transform=None, rubric=None): ...
    @abstractmethod
    def reset(self, seed=None, episode_id=None, **kwargs) -> ObsT
    @abstractmethod
    def step(self, action: ActT, timeout_s=None, **kwargs) -> ObsT
    @property
    @abstractmethod
    def state(self) -> StateT
    async def reset_async / step_async   # default: call sync version
    def _apply_rubric(action, obs) -> float   # 0.0 if no rubric
```

Two knobs are easy to miss. `SUPPORTS_CONCURRENT_SESSIONS` defaults to **False**: an environment has to opt in to sharing a process across WebSocket sessions, because shared working dirs and global state are the usual bugs. `REQUIRES_SINGLE_THREAD_EXECUTOR` handles libraries that are not thread-safe (game engines, simulators).

There is also an optional **`TaskProvider`** protocol (`list_splits`, `num_tasks`, `get_task`, `get_task_range`). It is structural, so you just define the methods. The server then exposes them as HTTP routes, or returns `501` if they are missing. Task selection is done by passing `split` and `index` to `reset()`. So the dataset belongs to the environment, but it is discoverable from outside without starting an episode.

### 5.2 Two interfaces, two audiences (RFC 001), the central idea

This is the one idea to remember from OpenEnv.

| Boundary | Protocol | Who calls it | Operations |
|---|---|---|---|
| **Agent** | **MCP** (Model Context Protocol) | the policy | `tools/list`, `tools/call` (`search()`, `execute_sql()`, …) |
| **Infrastructure** | Gym-like over WebSocket/HTTP | the training orchestrator | `reset()`, `step()`, `state()`, health, metrics |

**"The Gym-like API is NOT accessible to the agent being trained."** The rationale, quoted from INVARIANTS.md: *"If an agent could reset after crashing a car, it would learn that consequences are reversible — which breaks the training paradigm."*

RFC 001 works an example. There is a database-maintenance environment with an SQLite snapshot of employee records. The agent calls `execute_sql("DELETE FROM employees")` and gets a penalty. The *training loop* calls `env.reset()` to restore the snapshot. The agent learns not to delete, *because from its point of view the deletion was permanent.* If `reset` were a tool, the policy could learn "delete, then reset" as a valid recovery. That is fine in simulation and a disaster in production.

This is enforced in code. `MCPEnvironment._validate_tool_names` rejects tools named with `RESERVED_TOOL_NAMES` (`reset`, `step`, `state`, `close`). RFC 005 goes further for harnesses and requires that the harness subprocess cannot even reach the orchestration port over the network.

This is a safety-by-design argument as much as an engineering one. **The information boundary of the action space shapes what the policy can learn.** It ties to [[agent-layer-inverts-who-owns-the-generation-loop]] and to control-evaluation work like [[2504.10374 - Ctrl-Z - Controlling AI Agents via Resampling]], where the question is also which affordances the agent can reach.

### 5.3 The time problem: simulation vs production

RFC 001 names a duality that most RL-environment frameworks ignore:

- **Simulation time**: time advances only on `step()`. The agent can think for an hour and the world waits. You can reset forever. Code blocks run atomically.
- **Real time**: events arrive on their own schedule ("people get hired *now*, not when the agent is ready"). There is bounded latency, no reset, and deletes are one-way doors.

> "You can simulate production (via event queues), but you can't 'productionize' simulation (can't pause reality)."

Their answer is an **event queue** as a first-class constructor argument. An empty queue means a static environment (chess, coding puzzles). A populated queue means a dynamic environment where external events change state independently of the agent. This is the right abstraction for customer-service or ops agents, where turn-based RL environments are badly out of distribution.

**Graceful degradation to production** (RFC 002). The *same image* runs in `MODE=sim` or `MODE=prod`. In production the HTTP layer loses `reset`/`step` and keeps only health, metrics and logs. The MCP layer is byte-identical. Tool duality: a SendGrid MCP server logs to a file in sim and sends real email in prod, with the same `send_email(to, subject, body)` signature. The roadmap expects tool providers to eventually ship official sim/prod pairs through an HF Hub registry ("`search: algolia/search-mcp` pulls both sim and prod servers").

### 5.4 MCP as the only agent action space (RFC 003), plus CodeAct

**Every agent action is an MCP call**, including actions that are not tools in the usual sense. The argument: "the general need to list all possible actions and to execute one overlaps exactly with MCP."

Two action styles, and you build the environment once:

| | Tool-calling | CodeAct |
|---|---|---|
| Discover | `env.step(ListToolsAction())` | `CodeAction("return [t for t in list_tools() if 'browser' in t]")` |
| Act | `env.step(CallToolAction(tool_name="read_text_file", arguments={...}))` | `CodeAction("text = read_text_file(...); return [l for l in text.split('\\n') if 'needle' in l]")` |
| Tool calls per step | 1 | many |

**The double-marshaling problem.** A non-obvious performance point. A local Python function decorated with `@mcp.tool` gets serialized Python → JSON-RPC by the MCP server. In CodeAct you then want to call it from a Python interpreter, which means JSON-RPC → Python again. That is a full round trip back to where you started. RFC 003 notes that Anthropic's and Cloudflare's "code mode" write-ups don't address this, because they mostly use remote MCP servers. OpenEnv's fix is introspection. `MCPEnvironment.get_callables()` pulls the underlying Python functions out of the local MCP server and injects them straight into the `exec` namespace. MCP still handles *discovery and prompting* (schemas for the model). Execution goes around it. Remote servers still use the normal protocol.

Since many OpenEnv environments are ported Gym environments that become *local* MCP servers, this matters a lot for rollout throughput.

`MCPEnvironment` (654 lines) wraps a FastMCP server. It handles `ListToolsAction`/`CallToolAction` inside `step()`, has a `tool(mode=...)` decorator so tools can differ by sim/prod mode, and `supports_code_mode`. A separate `/mcp` JSON-RPC endpoint lets an existing MCP client skip `step()` entirely, which is the production path.

### 5.5 Rewards: the Rubric system (RFC 004)

**Rewards are computed inside the environment** (decision in RFC 002). The reasoning: domain knowledge lives there, results are reproducible across clients, and the reward can use hidden state the client never sees. The RFC 002 wording is that this is "opaque to the agent."

The API is copied from `torch.nn.Module` on purpose:

```python
class Rubric:
    def forward(self, action, observation) -> float: ...   # return 0.0-1.0, sync OR async
    # __call__ runs pre-hooks → forward → post-hooks; auto-awaits coroutines
    # children auto-register when assigned as attributes
    # named_rubrics(), get_rubric("code.syntax"), state_dict(), load_state_dict(), reset()
```

Leaf example: `Compiles.forward` returns `1.0` if `ast.parse(action.content)` succeeds, else `0.0`.

Composite example, with each term explained:

```python
class CodeRubric(Rubric):
    def __init__(self, test_weight=0.7):
        self.compiles = Compiles(); self.tests = PassesTests(); self.style = LLMStyleJudge()
    def forward(self, a, o):
        if self.compiles(a, o) < 1.0: return 0.0         # hard gate
        t = self.tests(a, o); s = self.style(a, o)
        return t * self.test_weight + s * (1 - self.test_weight)
```

Written as math: $R = \mathbb{1}[\text{compiles}] \cdot \big(w\,T + (1-w)\,S\big)$.
- $\mathbb{1}[\cdot]$ is the gate. A non-compiling submission scores 0 no matter how stylish it is. Without it, a policy can collect the $(1-w)S$ share from an LLM style judge on code that doesn't run. That is a textbook reward hack.
- $T \in [0,1]$ is the fraction of tests passed. $w = 0.7$ makes correctness dominate.
- $S \in [0,1]$ is the LLM judge's style score.

**Container rubrics** (like `nn.Sequential` / `ModuleList` / `ModuleDict`):

| Container | Semantics |
|---|---|
| `Sequential(*r)` | Runs in order and returns 0 at the first child that returns 0 (fail-fast, RLTF-style hierarchical gating). Awaits children one by one. |
| `Gate(r, threshold=1.0)` | Returns the child's score if it is ≥ threshold, else 0. |
| `WeightedSum(rs, weights)` | $\sum_i w_i r_i$. Async children run **in parallel via `asyncio.gather`**. |
| `RubricList` / `RubricDict` | Containers with no aggregation. `RubricDict` routes multi-task envs (`self.games[obs.game_id]`). |
| `LLMJudge(prompt_template, client)` | Formats `{action}`/`{observation}`, calls the LLM, regex-extracts the first number `(\d+\.?\d*)`, clamps to [0,1], and falls back to `default_score=0.0` if parsing fails. |

**Hooks for observability.** `register_forward_hook` on every `named_rubrics()` entry gives a per-component score dict (`{"compiles":1.0,"tests":0.8,"style":0.7}`) you can log to W&B without the rubric author doing anything. This is what RFC 008's "rubric introspectability" check reads. Reward attribution is a validation property, not a nice-to-have.

**The appendix maps each container to a paper**: RLTF (hierarchical gating), AdvancedIF (all-or-nothing), Rubicon (multi-dimensional, "defense rubrics" such as a `NoSycophancy` gate), OpenRubrics (hard rules then soft principles), and Scale's "Rubrics as Rewards" (holistic LLMJudge beats point-summing sometimes). See [[instance-specific-rubric-as-reward-function]] and [[evolving-reward-criteria-against-a-moving-policy]].

**Rejected alternatives**, which are informative:
- *Criterion vs Rubric as separate classes.* Rejected because "nn.Module doesn't distinguish between layer and model."
- *Declarative YAML.* Rejected because interesting rubrics involve LLM calls, sandboxes and branching. "The ecosystem (TRL, OpenRLHF, veRL) has converged on code-based rewards."
- *Magic lazy proxy objects for auto-parallelism* (like CUDA tensors). Prototyped, then dropped as "fragile" because overriding `__add__` and `__lt__` caused confusing edge cases.

**Delayed rewards: `TrajectoryRubric`.** Because one environment instance equals one trajectory (5.6), the rubric can just *accumulate the trajectory itself*. `__call__` appends `(action, obs)` and returns `intermediate_reward` (default 0) until `obs.done`, then returns `score_trajectory(traj)`. Credit assignment goes in `compute_step_rewards()`. The provided `ExponentialDiscountingTrajectoryRubric` does:

$$r_t = \gamma^{\,T-1-t}\, R_{\text{final}}, \quad t = 0,\dots,T-1$$

- $T$ is the trajectory length and $R_{\text{final}}$ is the terminal score (chess win 1, draw 0.5, loss 0).
- The exponent $T-1-t$ is 0 for the last move and $T-1$ for the first. So with $\gamma<1$ **later moves get more credit** (they are "closer to the outcome"). With $\gamma = 1$ every move gets the same credit.
- Note that this is the *reverse* of the usual return discounting. Here the discount is applied to a known final outcome and spread backwards, not summed forwards over a reward stream.

Custom credit assignment example (Cursor "plan mode"): **all** credit goes to the plan step and zero to execution steps, since "execution just follows the plan". Constraint: trajectories are kept **CPU-only**, so environments must move GPU tensors off-device before returning. Compare [[decoupling-clipping-unit-from-credit-assignment-unit]] and [[grpo-group-relative-policy-optimization]].

**Code vs spec.** RFC 004 says the `Environment` base class *requires* `self.rubric` and "the framework validates this at environment startup." The actual `interfaces.py` has `rubric: Optional[Rubric] = None`, and `_apply_rubric` returns `0.0` when it is missing. So an environment with no rubric silently gives reward 0. That is a small but real way to get a zero-signal environment. RFC 008's validator exists partly to catch this kind of problem.

### 5.6 Scaling: one env = one trajectory, many sessions per container

**No multiplexing** (a key decision). One `Environment` instance serves exactly one trajectory. To get a batch, you *stack* instances (the planned `EnvPool.step_batch`). This keeps rubric authors writing plain synchronous `forward()` with no batching logic.

But one *container* can host many instances. Each WebSocket connection gets its own `Environment` object server-side ("the connection itself is the session"). Cleanup happens automatically on disconnect. From the scaling tutorial:

| Approach | Containers | Memory | Startup |
|---|---|---|---|
| HTTP, 1 env = 1 container | N | N × ~100 MB | N × ~5 s |
| WebSocket, N sessions = 1 container | 1 | ~200 MB | ~5 s |

Per-step overhead is quoted as ~10–50 ms TCP handshake for HTTP vs ~0.1 ms per WebSocket frame. Knobs: `WORKERS` (uvicorn processes, default 4) and `MAX_CONCURRENT_ENVS` (sessions per worker, default 100). They report that simple text-game envs reach **~2000 concurrent sessions with 8 workers**. Beyond one box: Docker Swarm, Envoy round-robin across containers (with WebSocket upgrade enabled), or cloud providers.

**History.** RFC 002 originally chose HTTP/REST ("easy to debug with curl"). INVARIANTS.md now says HTTP is being deprecated for WebSocket-only (PR #252). The cloud-provider amendment makes **WebSocket upgrade the conformance test**: "A `200` on `/health` does **not** prove conformance". HTTP/2-only buffering proxies break it.

**Container providers** (`ContainerProvider`: `start_container → base_url`, `stop_container`, `wait_for_ready`). Implementations: LocalDocker, DockerSwarm, UV (no Docker), Daytona, Modal, Novita, Azure Container Apps, K8s (planned). The June-2026 amendment adds **security invariants for hosted sandboxes**, treating the workload as hostile: wss/https only, ingress URLs treated as bearer capabilities that are never logged, **default-deny egress plus blocking cloud metadata `169.254.169.254`** (so agent code can't steal the sandbox's managed-identity token), control-plane credentials never inside the sandbox, no command interpolation from agent-controlled input, and CPU/mem/disk/idle limits. This is a good threat-model checklist for anyone running agent code. See [[sandbox-fidelity-is-part-of-the-reward-function]] and [[environments-as-a-managed-resource-with-a-lifecycle]].

### 5.7 Agentic harnesses: when the agent lives inside the env (RFC 005, Feb 2026)

The newer problem: people want to RL-train *through* harnesses like Claude Code, OpenClaw, Gemini CLI, Goose, opencode and codex. These harnesses own the ReAct loop, history and built-in tools. So a harness overlaps with an environment (tools, filesystem) but also has "brains". Who owns what?

The answer is to **wrap** it. The OpenEnv container provides the sandbox. The harness runs as a subprocess inside it. Environment MCP tools are **injected** into the harness config before startup (for OpenClaw, by writing `.openclaw/mcp.json` that points at an `openenv-mcp-bridge`). The key semantic change:

| Concept | Traditional env | Harness env |
|---|---|---|
| one `step()` | one atomic action / tool call | **one conversational turn** (many LLM calls + tool calls inside) |
| trajectory | `(Action, Observation)` pairs | stream of `HarnessEvent`s (LLM_REQUEST, LLM_CHUNK, TOOL_CALL, TOOL_RESULT, …, TURN_COMPLETE) |
| step count | actions | turns |

They rejected "one step = whole episode" because it stops the orchestrator from giving intermediate feedback ("the tests are still failing, check line 42"). Tool-name collisions get an `env_` prefix, and identical names with different schemas raise an error instead of silently overriding. The security boundary is stated explicitly: rubrics run *after* the turn and outside the harness loop. If you want in-loop checks like "does it compile?", expose them as tools that return *facts*, not reward signals. That is a quiet but important anti-reward-hacking rule. See [[agent-layer-inverts-who-owns-the-generation-loop]].

**Harbor (RFC 012 + `harbor_env`).** This pushes the harness idea to production scale: "Train one policy against many coding agents". There are **16 harnesses across 4 wire dialects** (OpenAI chat/responses, Anthropic, Google) and **23 sandbox backends**, all chosen per rollout. A capture proxy sits between the harness and the inference server and records, per model call, `prompt_token_ids`, `completion_token_ids` and `per_token_logps`. The reason is that harnesses re-template prompts. If you re-tokenize the text later, you get tokens the model never sampled, and importance ratios break. See [[training-inference-logprob-mismatch-forces-recomputation]] and [[trajectory-capture-by-instrumenting-the-llm-api-boundary]].

RFC 012's hard rules:
- `purpose ∈ {eval, train, auto}`. Hosted providers (no token access) are **eval-only**. "Endpoint reachability is not training certification."
- Training sampling must be full-vocabulary with positive finite temperature (`top_k=-1, top_p=1.0`). Otherwise the logged logprobs are not the behavior policy's.
- "Missing sampled probabilities are never filled with zeros." A zero verifier reward is kept distinct from missing or failed grading.
- The trainer, not the env, owns advantages, weighting, staleness and weight sync.

These are exactly the bugs that silently wreck off-policy corrections in real agentic-RL systems. Compare [[Hugging Face TRL Team - Keep the Tokens Flowing, Lessons from 16 Open-Source RL Libraries]] and [[Cao et al - SkyRL-v0 Train Real-World Long-Horizon Agents via Reinforcement Learning]].

### 5.8 Validation: the "frontier-lab intake" bar (RFC 008, Aug 2026)

The motivating numbers: the hub hosts **4,300+ environments**, manual review can't keep up, and "one submitted verifier returned reward 1.0 for the empty string." Issue #778 defines six properties that break into **44 acceptance tests**:

1. Scales on infra. 2. **Learnable** (a model can hill-climb within O(100s) of steps). 3. Secure. 4. **Not prone to reward hacking**. 5. Observable and self-describing. 6. Reproducible.

Four levels with three cost budgets:

| Level | What | Who runs it |
|---|---|---|
| 1 Static | manifest schema, pinned deps, reproducible build, SBOM | author laptop/CI |
| 2 Runtime | starts; seed/episode determinism; declared tools and tasks match reality; reward well-formed; **network policy enforced; host containment; cross-episode isolation** | author |
| 3 Semantic | **oracle replay hits max; do-nothing floor sits a margin below max; no solution leakage in observations; canary trajectories stay at floor** | author |
| 4 Statistical | pass@k band, difficulty separation across model tiers, headroom, reward variance, adversarial panel | **operators only** (hub/lab) |

`openenv validate` covers 33 of the 44 locally.

**The oracle/floor gap contract**, the core idea:

$$R(\text{oracle}) \ge R_{\max} - \tau, \qquad R(\text{reset}) = R_{\text{floor}}, \qquad R_{\max} - R_{\text{floor}} \ge m$$

- $R(\text{oracle})$ is the reward from a known-correct solution, supplied either as *injected state* (preferred, most deterministic) or as an *executable oracle script* (Harbor's `solution/solve.sh` precedent). **No oracle means FAIL**, not skip: "an environment is only good if verifiable."
- $\tau$ is the declared tolerance (non-zero for LLM-judged envs, which must pin judge model, version and params in the manifest).
- $R_{\text{floor}}$ is **measured** from doing nothing after reset, not declared. An agent that deletes the repo can legally score below the floor.
- $m$ is the required margin. If doing nothing scores close to max, the verifier is vacuous (the empty-string bug).
- Determinism is a precondition. A nondeterministic env has its semantic checks SKIPped with the dependency named, not passed or failed misleadingly.

Architecture: *signature selects parser → parser writes one normalized manifest → capabilities select contract graders, type tags select domain graders.* **"Graders read the manifest, never the signature."** So served OpenEnv (`openenv.yaml`), Harbor (`task.toml`) and PostTrain/BenchFlow (`task.md`) all share one grader registry. Third parties add graders through an entry-point group.

The political decision: **"ship local validation and the contracts that let any team build a hub; do not ship a hub."** A reference hub in the shared repo "would quietly make one operator's answers the standard." That is committee governance written into architecture: HF, Prime Intellect and private lab intake pipelines can all run their own gates on comparable reports. And: "Validation says an environment *can* be trained on, not that it *should* be."

The canary idea shows up concretely in `sophistry_bench_sprint_env`. Its reward proxy peaks at exactly 8 `<claim>` tags (a deliberate "claim_count_cliff"). Four **weight-0 canaries** (alternation, starts-with, length-band, template-echo) detect format hacking without feeding reward. `SPRINT_EXPOSE_CORRECTNESS` is off by default "so a harness can't accidentally leak [ground truth] to the policy." That is a working example of reward-hacking instrumentation. See [[a-bounty-list-is-a-map-of-what-cannot-be-verified]] and [[HUD - Verifier and Reward Design for RL Environments]].

### 5.9 ECHO: the environment as a free loss function (RFC 010, draft)

In normal agentic RL, **environment tokens are masked out** of the loss. You only train on action tokens ([[tool-output-masking-in-agentic-rl]]). ECHO ("Terminal Agents Learn World Models for Free", arXiv 2605.24517, `microsoft/echo-rl` on SkyRL; plus Prime Intellect's "True Agents Model the World") says: also train the policy to *predict* the observation tokens.

$$\mathcal{L}_{\text{ECHO}} = \mathcal{L}_{\text{GRPO}}(\text{action tokens}) + \lambda \cdot \mathcal{L}_{\text{env}}(\text{observation tokens})$$

From the reference `echo_loss.py`, symbol by symbol. With logits $[B,T,V]$ and next-token convention (a token's role is taken from the *target* position $t{+}1$):

- $\ell_t = \log \pi_\theta(x_{t+1}\mid x_{\le t})$, the per-token log-prob gathered from `log_softmax`.
- $a_t, o_t \in \{0,1\}$ are the action mask and observation mask (disjoint). A third `warning_mask` removes harness boilerplate (echoed commands, `[stdout]` tags) from $o$.
- $A_b$ is the per-sequence GRPO group-relative advantage.
- $\mathcal{L}_{\text{GRPO}} = -\dfrac{\sum_{b,t} \ell_t\, a_t\, A_b}{\sum a_t}$: REINFORCE on action tokens, normalized by action-token count.
- $\mathcal{L}_{\text{env}} = -\dfrac{\sum_{b,t} \ell_t\, o_t}{\sum o_t}$: plain cross-entropy (SFT) on the environment's tokens, normalized separately.
- $\lambda$ is `world_model_coeff`. $\lambda=0$ gives vanilla GRPO exactly. `use_rl=False` gives **verifier-free** world modeling, meaning training with no reward at all.

Why it's "free": the policy **already computed logits over the observation tokens** in the same forward pass, because it conditions on them. You only change the mask. Neat implementation trick: SFT = RL with a constant positive advantage. So set per-token advantage $=A_b$ on action tokens and $=\lambda$ on env tokens, and reuse the existing policy-gradient code path.

Reported results (quoted by the RFC from the paper): **~2.3× faster RL**, TerminalBench-2.0 pass@1 roughly doubles, recovers 50–104% of expert-SFT gain with no teacher, and verifier-free training still improves held-out tasks. Why it helps: when a small model's pass rate is near 0, or GRPO group variance collapses, $A_b \approx 0$ and the RL term gives nothing. The env term turns *every turn, including failures,* into dense supervision. "The terminal is the teacher."

Caveats they flag honestly: **keep λ small**. Prime Intellect saw collapse at 0.05 on GLM-4.5-Air (stable at 0.005), while echo-rl's Qwen3-8B config uses 0.05, so it is model-dependent and needs a sweep. It works best when tool outputs are "predictable-without-memorization and complex". Pure retrieval outputs (memorizing a database) don't generalize.

What OpenEnv has to add is plumbing: per-token role masks in the trajectory schema (today `EpisodeRecord` is message-level) and a λ knob on the optimizer seam. This is a good example of an environment *standard* having to change because of a *training-objective* result. See [[world-state-serialization-for-llm-agents]].

### 5.10 Discovery (RFC 011, draft)

With thousands of environments, the problem is *finding* one. The proposal is an "Environment Card" carried over **Agentic Resource Discovery (ARD)**, with a strict rule: discovery never installs, imports, pulls images, wakes deployments, resets or steps anything. The card separates *declarations*, *verified evidence* and *unknowns*, so a search score is never read as quality. A deterministic lexical baseline counts as a valid outcome, and semantic ranking has to beat it.

---

## 6. The environment catalog (what's in `envs/`)

About 40 environments in the repo, and the HF catalog has thousands more. Grouped by what they teach:

- **Infra tests:** `echo_env` (the hello-world), `grid_world_env`, `maze_env`.
- **Classic RL ported to the LLM interface:** `atari_env` (ALE), `openspiel_env`, `dm_control_env`, `connect4_env`, `chess_env` (moonfish engine), `snake_env` (multi-agent), `textarena_env`, `unity_env`, `sumo_rl_env` (traffic signals), `wildfire_env`, `carla_env` (driving), `finrl_env`.
- **Code and terminal:** `coding_env` (smolagents Python sandbox), `coding_tools_env` (SETA-style, E2B), `jupyter_env`, `julia_env`, `repl_env` (Recursive Language Model runner with `llm_query`/`rlm_query`), `git_env` (Gitea, fast reset), `tbench2_env`, `terminus_env`, `kernrl` (write CUDA/Triton kernels, rewarded on correctness plus measured speedup, optional Nsight profiling).
- **Harness wrappers:** `opencode_env`, `pi_env`, `harbor_env`, all with per-token logprob capture.
- **Tool-use / enterprise:** `agent_world_model_env` (**AgentWorldModel-1K: 1,000 synthetic MCP-server environments, 10,000 tasks**, with SQL- and code-based verifiers), `calendar_env` (Turing), `finqa_env` (Snorkel), `websearch_env`, `browsergym_env`, `openapp_env`.
- **Reasoning / math:** `reasoning_gym_env`, `qed_math_env` (CMU QED-Nano proofs with an LLM rubric on a 0–7 scale, process-isolated `math_verify`, backpressure and worker restart for concurrent rollouts), `pelican_svg_env` (blind SVG drawing, scored).
- **Safety:** `dipg_safety_env` (high-stakes medical-domain safety), `sophistry_bench_sprint_env` (reward-hacking sprint with canaries).

RFC 002 states a quality bar for contributions: favor "environments that reflect real-world use cases", avoid "toy environments with no production analog" and "made-up APIs that don't match real services". The catalog only partly meets that. Many entries are ported games, which fits the "reproducible research" audience more than the production-parity principle.

---

## 7. What's interesting or non-obvious

1. **The reset boundary is a safety claim.** Most frameworks treat `reset` as plumbing. OpenEnv treats "the agent can reach reset" as a way to train in the wrong belief that actions are reversible. It enforces this with reserved tool names, network isolation for harnesses, and "orchestration-only" lifecycle controls in cloud providers.
2. **MCP as the action space, not just the tool protocol.** Making even non-tool actions MCP calls means production inference (any MCP client) and training share one schema. This is the concrete mechanism for "minimize lifecycle deltas".
3. **PyTorch idioms applied to rewards.** `nn.Module` → `Rubric`, `Sequential` → fail-fast gating, forward hooks → per-component reward logging. Researchers already know the mental model.
4. **Validation as governance.** "Do not ship a hub" is architecture chosen to keep a multi-company committee from turning one member's gate into the standard.
5. **The spec follows training research.** ECHO (RFC 010) shows a change in the training objective forcing a change in the environment's trajectory format. The line between environment and trainer keeps moving.
6. **Token-exact capture is a standard, not an afterthought.** RFC 012's "never fill missing logprobs with zeros" and "hosted = eval-only" reflect the hard-won lesson that re-tokenization quietly corrupts off-policy RL.

---

## 8. Genuine limitations and open questions

- **Experimental and moving fast.** The README warns: "expect bugs, incomplete features, and APIs that may change." Most RFCs are still "In Review" or "Draft". HTTP→WebSocket is mid-migration, with both live.
- **Spec vs code drift.** The rubric is "required" in RFC 004 but `Optional` in code (silent 0.0 reward). RFC 002's design section still shows `HTTPEnvClient`. RFC 001's `Environment(mode, mcp_servers, event_queue)` constructor is not the shipped signature. The event queue for dynamic environments isn't visible in the core base class. Read the RFCs as intent and the code as truth.
- **Evals are still undefined.** RFC 001 says "Specific interfaces TBD." There is a `core/evals/` with an `inspect_harness` (UK AISI's Inspect), but no settled eval abstraction. Phase 3 is not done.
- **One env = one trajectory costs something.** Stacking containers or instances is simple but wasteful for cheap text environments. Multiplexing is "left to future work". `EnvPool` is still marked "(Future)" in RFC 004.
- **Docker overhead is an accepted cost** ("performance for isolation"). For very high-throughput, cheap environments (math verification), a network hop per step can dominate. That is presumably why the `UVProvider` (no Docker) and in-process `LocalREPLEnv` exist.
- **Reward-inside-env limits reward research.** You can't easily swap reward models from the trainer side. The escape hatch is server-side `Transform` pipelines, and calling reward models from inside the env needs RPC out of the sandbox, which RFC 000 lists as an open design problem.
- **LLMJudge parsing is naive.** It takes the first number in the response and falls back to 0. A judge that writes "On a scale of 1 to 10, I'd give 7" returns 1, which clamps to 1.0. A careful user should pass a strict `score_pattern`.
- **Statistical learnability checks (level 4) are reserved, not built.** The "adversarial panel" procedure "is not yet specified anywhere." So the hardest part of reward-hacking detection is still left to operators.

---

## 9. Connections

- Same problem, other stacks: [[Prime Intellect - Environments Hub]] (verifiers format, which `openenv import` can wrap), [[Patronus AI - RL Environments Tutorial and Examples]], [[HUD - Verifier and Reward Design for RL Environments]], [[Epoch AI (Denain & Barber) - An FAQ on Reinforcement Learning Environments]].
- Trainers that consume it: [[TRL and PEFT Training Libraries]], [[huggingface-trl - vLLM Integration for Online RL Generation]], [[Cao et al - SkyRL-v0 Train Real-World Long-Horizon Agents via Reinforcement Learning]], [[Hugging Face TRL Team - Keep the Tokens Flowing, Lessons from 16 Open-Source RL Libraries]], [[Prime Intellect - INTELLECT-3 A 100B+ MoE trained with large-scale RL]].
- Concepts: [[environment-as-installable-package-not-config]], [[environments-as-a-managed-resource-with-a-lifecycle]], [[sandbox-fidelity-is-part-of-the-reward-function]], [[agent-layer-inverts-who-owns-the-generation-loop]], [[trajectory-capture-by-instrumenting-the-llm-api-boundary]], [[training-inference-logprob-mismatch-forces-recomputation]], [[tool-output-masking-in-agentic-rl]] (which ECHO partially reverses), [[grpo-group-relative-policy-optimization]], [[decoupling-clipping-unit-from-credit-assignment-unit]], [[the-critic-deletion-tension-in-agentic-rl]], [[instance-specific-rubric-as-reward-function]], [[evolving-reward-criteria-against-a-moving-policy]], [[a-bounty-list-is-a-map-of-what-cannot-be-verified]], [[world-state-serialization-for-llm-agents]].
- Company context: [[Mercor Post-Training Assessment - GRPO Rollout Scheduling Under a Fixed Budget]], [[Mercor benchmarking Take Home]]. Mercor sits on the OpenEnv technical committee, and RFC 008's intake bar describes the product that environment vendors sell.
- Safety: [[2504.10374 - Ctrl-Z - Controlling AI Agents via Resampling]]. Both treat the set of affordances the agent can reach as the safety lever.
