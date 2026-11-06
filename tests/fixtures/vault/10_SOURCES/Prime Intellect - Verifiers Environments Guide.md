Created: 24-09-2026 16:20

Tags:

Status: #baby

## References

1. [Environments — Verifiers (Prime Intellect docs)](https://docs.primeintellect.ai/verifiers/environments)

# Notes

## First, what this page is now

The URL above no longer resolves. As of Sept 2026 the page has moved to `docs.primeintellect.ai/verifiers/legacy/environments` and is labelled **v0, deprecated, to be removed**. The decode below is of that page (~5K words, all code and API prose, no experiments), plus the short v1 pages (Overview, Architecture, Env, Tasksets, Harnesses) because you can't understand Prime Intellect's current approach without seeing what v0 turned into.

So read this in two layers:

1. **v0 (`import verifiers as vf`)**: an environment is `dataset + rubric + rollout loop`. That is the mental model that shipped with the [[Prime Intellect - Environments Hub]] launch and trained [[Prime Intellect - INTELLECT-3 A 100B+ MoE trained with large-scale RL]].
2. **v1 (`verifiers.v1`)**: the same idea split into `Taskset × Harness × Runtime → Trace`, with model traffic going through an interception proxy. The v0 page already hints at this in its last section ("`vf.Env` / `vf.Taskset` / `vf.Harness` — preferred pattern").

If you're reading to understand how the company thinks, the move from 1 to 2 is the most useful thing on the page.

---

## The problem an "environment" solves

RL on language models needs three things per training example: a prompt, a way to let the model act (one reply, or many turns with tools), and a number saying how good the result was. Before frameworks like this, each lab wrote all three as one-off glue inside its training loop. So the math task's reward was tied to one trainer, the coding sandbox was tied to one inference server, and nothing could be reused or shared.

Prime Intellect's bet (the thesis of the Hub post) is that **environments are now the scarce input**, not models or algorithms. For that to be true, an environment has to be a separate, installable artifact: a Python package that any trainer (their `prime-rl`) or evaluator (`prime eval run`) can load, run concurrently thousands of times, and score. This page is the spec for that artifact.

The contract is one function:

```python
def load_environment(**kwargs) -> vf.Environment
```

Everything else on the page is about what goes inside the returned object.

---

## Layer 1: The simplest environment

```python
dataset = Dataset.from_list([{"prompt": [...], "answer": "4"}, ...])
async def correct_answer(completion, answer) -> float:
    return 1.0 if answer in completion[-1]["content"] else 0.0
rubric = vf.Rubric(funcs=[correct_answer])
return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
```

Each dataset row becomes a **rollout**: send the prompt, get the completion, score it. That's the whole idea.

Look at the reward function in the official "first environment": `answer in response`. That is a substring check. For `answer="4"`, the replies "14", "40", or "it's 3 or 4 or 5" all score 1.0. It's fine for a toy, but it is exactly the [[ostensible-positives-from-low-entropy-answer-spaces]] failure: when the answer space is small (single digits), a lenient matcher gives reward to wrong answers, and RL will find and push on that. The docs don't warn about it. The built-in `MathRubric` (parses `\boxed{}` and checks symbolic equality with `math-verify`) is the grown-up version.

### Dataset columns and prompt building

A row is a Hugging Face `datasets` row with:

- `prompt`: a list of chat messages, **or** `question`: a string wrapped into one user message (ignored if `prompt` exists).
- `answer`: a ground-truth string (optional).
- `info`: structured metadata, a dict **or a JSON string**. The advice to use JSON strings when rows have different schemas is a real Arrow detail: HF datasets need one column type, so mixing `{"type":"math","difficulty":3}` and `{"type":"code","language":"python"}` as dicts would force a merged struct with lots of nulls or fail outright. A string sidesteps the schema and gets parsed back to a dict at rollout time.

`system_prompt=` puts a system message at the front, but only if the prompt doesn't already start with one.

`eval_dataset` is separate from `dataset`; `prime eval run` uses the eval split by default and **silently falls back to the train split** if none is given. Watch out for this: an env published without an eval split gets evaluated on its training data.

`DatasetBuilder` is just a no-argument function that returns a `Dataset`, run on first access. The reason is operational, not about ML: when you run many environment replicas (one per worker process), each one downloading and shuffling a large HF dataset at startup is wasted work.

---

## Layer 2: Rubrics — how reward is built

A `Rubric` is a list of functions with weights. The scalar reward for rollout $i$ is:

$$R_i = \sum_{k=1}^{K} w_k \, f_k(\text{rollout}_i)$$

- $f_k$: the $k$-th reward function, usually returning a value in $[0,1]$.
- $w_k$: its weight. Weight $0$ means "log it as a metric but don't let it change the reward" (`add_metric` is shorthand for this).
- The sum is **not normalized**. With weights `[1.0, 0.1]` the reward range is $[0, 1.1]$. That's fine for group-relative advantages (a constant shift cancels out), but it matters if something downstream assumes rewards live in $[0,1]$ (pass thresholds, `BestOfNEnv`'s success threshold).

### Argument injection by name

Reward functions get data by **naming parameters**: `completion`, `prompt`, `answer`, `info`, `state`, plus anything registered with `rubric.add_class_object("name", obj)` (e.g. a parser, a judge client). The framework looks at the function signature and passes in what it asks for. This is the key ergonomic choice: a reward is just a plain async function with no base class, so it can be tested on its own.

### Ordered execution + shared mutable state

Functions run **in the order they were added**, and `state` is one mutable dict shared by all of them. The docs use this to compute something expensive once:

```python
similarity_score  -> writes state["similarity"], weight 0.0 (logged only)
similarity_threshold -> reads it, returns 1.0 if > 0.8, weight 1.0
```

So you train on the thresholded binary signal while still watching the continuous one. That split (log the dense signal, optimize the sharp one) is a good pattern. The cost is that the order is now a hidden dependency: move one line and `state["similarity"]` gets a KeyError.

### Group-level reward functions — the RL part

This is the section that shows the framework was built for GRPO-style training ([[grpo-group-relative-policy-optimization]]).

During training, each example is rolled out $G$ times, which makes a **group**. 100 rows × 4 rollouts = 100 groups of 4. The trainer then computes advantages *within* a group:

$$A_i = R_i - \frac{1}{G}\sum_{j=1}^{G} R_j \quad(\text{optionally} \div \text{std})$$

If you use **plural** parameter names (`completions`, `prompts`, `answers`, `infos`), a reward function is called once per group and must return a list of $G$ scores. The example:

```python
async def diversity_bonus(completions) -> list[float]:
    responses = [c[-1]["content"] for c in completions]
    return [0.2 if responses.count(r) == 1 else 0.0 for r in responses]
```

Why this matters: a group reward can only be computed by looking at the siblings, and it feeds straight into the advantage calculation above. It's a direct handle on the [[zero-gradient-prompt-filtering]] problem. If all 4 rollouts are correct and identical, every advantage is 0 and the group teaches nothing. A diversity term brings back some variance.

But the example as written is easy to game: uniqueness is exact string equality, so a trailing space or different whitespace makes a response "unique." Under RL pressure, expect meaningless surface variation, not real diversity. A real version would compare normalized text, embeddings, or the extracted answers.

Also, when you add a diversity bonus you're mixing *what* gets rewarded (correctness) with *how the group is spread out*. If you also use std-normalized advantages, see [[std-normalization-difficulty-bias-in-group-rl]]: a small 0.2 bonus in an otherwise tied group gets scaled up a lot.

### Judges

`JudgeRubric` stores an LLM client and gives reward functions a `judge(prompt, completion, answer)` callable that returns the judge's text verdict. The example turns it into a reward with `"yes" in verdict.lower()`. Same issue as the substring matcher: "No, this is not correct, yes the format is fine" scores 1.0. It's a weak parse used as the default example. Custom `judge_prompt` templates and direct access to `judge_client` / `judge_model` / `judge_sampling_args` let you do structured scoring properly.

### RubricGroup and monitor rubrics

- `RubricGroup([r1, r2, ...])` acts like one rubric. Its members **run in parallel** and **their rewards add up**. Note the contrast: functions *inside* one rubric run in order and share state; rubrics *inside* a group run concurrently. So state dependencies must stay inside one rubric.
- **Monitor rubrics** are weight-0 metric bundles that each environment class adds on its own and that stack up the class hierarchy: `MultiTurnEnv` → `num_turns`; `ToolEnv` → `total_tool_calls` + per-tool counts; `SandboxEnv` → sandbox wait/exec time; `PythonEnv` gets all of them. You get observability without changing the reward.

This maps onto the [[four-layer-scoring-stack-for-rl-environments]] only partly. v0 gives you "reward functions" and "metrics," but no first-class way to say "hard pass/fail check that zeroes the reward." You'd have to build it yourself (e.g. a multiplicative gate in a later function that reads `state`). The weighted sum is additive, so a safety violation worth −1 can be outweighed by success elsewhere unless you design it so it can't.

---

## Layer 3: Multi-turn and tools

### Everything is a MultiTurnEnv

The class hierarchy is simpler than it looks: **every** v0 environment is a `MultiTurnEnv`. `SingleTurnEnv` is `MultiTurnEnv(max_turns=1)` with a placeholder `env_response`. `ToolEnv` adds tool execution. `MCPEnv` adds MCP server connections. `StatefulToolEnv` adds per-rollout resources. `SandboxEnv` / `PythonEnv` are built-in stateful ones on Prime Sandboxes.

### The rollout loop

```
setup_state(state)
loop:
    messages = get_prompt_messages(state)     # prompt or history + env reply
    response = model(messages)
    if any @vf.stop(state) -> break
    env_msgs = env_response(messages, state)  # abstract; you implement
render_completion(state)  -> state["completion"]
all @vf.cleanup(state)
```

`env_response` is the one abstract method: it takes the full history and returns the **new** messages to add (tool results, game feedback, a simulated user turn). Things you can optionally override: `setup_state`, `get_prompt_messages` (for non-linear conversations, e.g. trimming or branching context), `render_completion`, and `add_trajectory_step`. The last one is the hook for **per-turn rewards/advantages**, which is where process-reward or per-step credit assignment would go ([[trajectory-as-the-atomic-unit-of-policy-optimization]]).

### Tools from signatures

A tool is a Python function. The name becomes the tool name, type hints become the parameter schema, and the Google-style docstring `Args:` block becomes the parameter descriptions (OpenAI-compatible tool calling). The rollout continues until the model replies without calling a tool or hits `max_turns`.

The docs' own `calculate` example calls `eval(expression)` on text the model wrote, running in the environment process, not a sandbox. In an RL loop the policy is actively searching for any reward-increasing behavior, and `eval` on its output is arbitrary code execution on your worker. Don't copy this. Anything that runs model-written code belongs in `SandboxEnv` / `PythonEnv`.

### Stateful tools and hidden arguments

`ToolEnv` assumes tools are stateless. For per-rollout resources (a sandbox, a DB connection, a session), `StatefulToolEnv` adds two things:

1. `args_to_skip=["session_id"]`: that parameter is **removed from the schema the model sees**.
2. `update_tool_args(tool_name, tool_args, messages, state)`: fills it in from `state` before every call.

So the model sees `run_code(code: str)` and the environment calls `run_code(code, session_id=state["session_id"])`. This is good design: the model can't make up or change session IDs, so it can't reach another rollout's sandbox. It's a simple isolation boundary.

### Stop conditions, errors, termination

- `@vf.stop` methods return bool and are checked after every model reply. `priority=` sets the order (higher first), so cheap checks (look for "FINAL ANSWER:") run before expensive ones (an LLM validator). Built-in stops: error, prompt too long, `max_turns`, `max_total_completion_tokens`. Wall-clock timeout is set at eval time with `--timeout`.
- Errors form a typed hierarchy under `vf.Error`: `ModelError` (e.g. empty response), `OverlongPromptError`, `ToolError` (`ToolParseError`, `ToolCallError`), `InfraError` (`SandboxError`, `TunnelError`). A raised `vf.Error` is **caught and stored in `state["error"]`**, and the built-in `has_error` stop ends the rollout cleanly. At 2,000 concurrent rollouts, one bad sandbox shouldn't take down the batch.
- `stop_errors=[vf.ToolParseError]` picks which tool errors end the rollout. The rest get **returned to the model as tool messages** so it can recover. This is an RL design choice hidden in an API: returning errors teaches the policy error recovery; stopping on them doesn't. Also note the typing separates `InfraError` from `ModelError`, which lets a trainer tell "the environment broke" apart from "the policy did badly." You don't want to give negative reward for your own infrastructure failures.
- `state["final_env_response"] = msgs` inside `env_response` ends the rollout **with the environment having the last word**. You need this when the final feedback (game over + score, test results) is what the reward reads.
- `@vf.cleanup` runs per rollout and `@vf.teardown` once at shutdown. Cleanup **must be idempotent**, because cancelled or interrupted rollouts may run it more than once or with resources half-built.

---

## Layer 4: Packaging and composition

- An environment is a pip package: `environments/my_env/{my_env.py, pyproject.toml, README.md}`, built with hatchling, scaffolded by `prime env init`. The `[tool.verifiers.eval]` table stores default eval settings (`num_examples=20`, `rollouts_per_example=5`). So the environment ships with its own eval recipe, which is what makes Hub results comparable.
- `vf.ensure_keys([...])` checks for needed API keys at load time and fails fast with `MissingKeyError`. Secrets come from the Hub's Secrets tab, `env_file` for hosted training, or the local shell.
- `EnvGroup(envs=[math, code, reasoning], env_names=[...])` merges datasets and adds `info["env_id"]` for routing, so each row goes to its own sub-env's rollout logic and rubric. This is the multi-task training entry point. Each sub-env keeps its own reward scale, and `EnvGroup` doesn't normalize across them. If the math rubric gives rewards in [0,1] and code gives [0,1.1], the mixture is already weighted in ways nobody chose. Group-relative advantages per example help here (comparisons stay within a group, so within one env), but reward-scale differences still show up in any cross-env logging or thresholds.

---

## Layer 5: Performance — the part that shows how they actually run it

All rollouts share **one asyncio event loop**. Any synchronous call blocks every other rollout. The page's math:

$$10\,\text{ms sync call} \times 2{,}000\ \text{concurrent rollouts} = 20\,\text{s of serialized wall-clock blocking}$$

That's why every reward function in the docs is `async def`. The listed offenders are `time.sleep`, `requests`/sync `OpenAI` clients, `deepcopy`, `json.dumps` on large payloads, and file I/O. Fixes, in order of weight:

1. Async-native library (`asyncio.sleep`, `httpx.AsyncClient`, `AsyncOpenAI`, `aiofiles`).
2. `asyncio.to_thread(...)`: frees the event loop but **still holds the GIL**, so it's only good for I/O-bound or GIL-releasing work.
3. `ProcessPoolExecutor` for real CPU-bound work (> ~50 ms): compiling, heavy parsing, symbolic math verification.

`env.set_concurrency(256)` resizes the default thread executor plus any executors you registered with `register_executor`. `prime eval run` and `prime-rl` set this automatically from the concurrency level.

The non-obvious point: the whole environment design (async functions, typed errors saved to state, idempotent cleanup, lazy datasets) comes from one limit. They run thousands of rollouts at once in a single process for async RL, and one slow reward function stalls the whole GPU pipeline waiting on it. That matches [[scoring-as-a-pluggable-interface-in-rl-infrastructure]]: scoring is on the critical path of the async generate → score → train pipeline, so it has to be fast.

---

## Integrations and experimental classes

- Wrappers: `TextArenaEnv` (text games), `ReasoningGymEnv` (procedural reasoning data), `BrowserEnv` (Browserbase, DOM or computer-use mode), `OpenEnvEnv` (OpenEnv gym/MCP contracts on Prime Sandboxes), `GymEnv` (any Gymnasium API).
- `CliAgentEnv`: runs **an entire agent program** (not just a model) in a remote sandbox and intercepts its API calls through the rollout loop. It takes CPU/memory/disk/GPU sizing; GPU > 0 switches to VMs. `OpenCodeEnv` / `OpenCodeRLMEnv` are specific versions of this for the OpenCode CLI, with sub-LLM calls routed through the proxy.
- `HarborEnv` (Harbor-format agent benchmarks), `SandboxDebugEnv` (no agent: create the sandbox, apply a gold patch or command, run tests. Useful for checking a SWE-style task is solvable and its tests are correct before any RL).

`CliAgentEnv` is where v0 breaks and v1 starts.

---

## The shift to v1 — Prime Intellect's current approach

v0 assumes **the environment owns the rollout loop**: it calls the model, parses tool calls, runs tools, builds messages. That works for chat-style tasks. It stops working once the thing you want to train is an existing agent program like Claude Code, Codex, or mini-swe-agent, which has its own loop, tool format, and context management that you can't and shouldn't rewrite as `env_response`.

v1 turns this around. The pieces:

- **Taskset**: loads `Task`s. Each task pairs an immutable, serializable `TaskData` (prompt, files, reference answer, resource needs, timeouts) with a `Task` class carrying the behavior (`@vf.reward` methods that read a `Trace`, stop conditions, setup, tools). Configs (`TasksetConfig`, `TaskConfig`) hold only the settings users can change.
- **Harness**: the program the model runs inside (built-ins: Claude Code, Codex, `bash`, `browser_use`, `null`). A custom harness implements `setup(runtime)` and `launch(ctx, trace, runtime, endpoint, secret, mcp_urls, data)`, and declares what it supports (`SUPPORTS_MCP`, `SUPPORTS_RESUME`, `APPENDS_SYSTEM_PROMPT`).
- **Toolset**: task-defined tools, served as **MCP servers** and installed into harnesses that support MCP. (v0: tools are Python functions the env calls. v1: tools are services the harness connects to.)
- **Runtime**: `subprocess` (debug only; rollouts can interfere with each other, e.g. one overwriting a harness config file), `docker` (local), sandboxes like `prime` or `modal` (production).
- **Agent** = harness × model × runtime policy → a **Trace** (message graph, rewards, metrics, errors, one `ModelCall` record per provider exchange; for training, also tokens and logprobs built with their `renderers` library).
- **Env** = the control flow **between agents**: `Env.run(task, agents)`. Built-ins: `SingleAgentEnv`, `AgenticJudgeEnv` (solver then judge, either `Shared` runtime so the judge can inspect what the solver left behind, or `Isolated`), `UserSimEnv` (the user is also an agent), `BestOfNEnv` (n attempts; marks best and pass@n, for rejection sampling).

### The interception server is the key idea

The harness **never calls the model provider directly**. It's given an `endpoint` that points to Prime's interception server (local, or over Prime Tunnel), which speaks whatever API the harness expects (OpenAI Responses for Codex, Anthropic Messages for Claude Code) and forwards the request to the actual model. Because every model call passes through this one point, they get:

1. **Live trace building**: trajectories (and for training, token IDs and logprobs) are recorded from the wire, without the harness cooperating.
2. **Sampling control** for harnesses that don't expose temperature etc.
3. **Reward-hack blocking**: rewriting tool responses or server-side web-search results before the model sees them (e.g. removing search results that contain the benchmark answer).

This is the answer to "how do you RL-train a model inside a third-party agent you don't control?" You don't change the agent. You own the network path between it and the model. For RL specifically this is necessary: on-policy training needs the exact tokens and logprobs the policy produced, and a closed-source CLI will never give them to you.

The third point also matters for alignment work. It moves part of the reward-hack defense out of the reward function and into the **observation channel**. Rather than trying to detect a hack after the fact ([[verifier-failure-vs-rubric-design-failure]]), you stop the model from ever seeing the leaked information.

### What changed conceptually, v0 → v1

| | v0 | v1 |
|---|---|---|
| Unit of work | dataset row | `TaskData` + `Task` behavior |
| Who runs the loop | the environment (`env_response`) | the harness (any agent program) |
| Tools | Python functions called by env | MCP servers installed into harness |
| Record | `state` dict + `completion` | `Trace` with per-call `ModelCall` records |
| Reward input | named args (`completion`, `answer`, `state`) | the `Trace` |
| Multi-agent | not first-class | `Env.run(task, agents)` |
| Model access | env calls the client | interception proxy |

The one line summary: **v0 treats an environment as a scored dataset with a chat loop; v1 treats it as a task spec that any agent program can be dropped into, with scoring done on a trace recorded from the wire.** That's a bet that the future of RL post-training is training models *inside production agent harnesses*, not in toy loops written for training.

---

## What's genuinely worth taking from this

- **Signature-based injection** for rewards and tools keeps reward code as plain functions you can test on their own. Easy to copy in any home-built RL setup.
- **Weight-0 metrics next to weighted rewards**: always log the continuous signal even when you train on a thresholded one.
- **Group reward functions** as a first-class hook: the only clean place to put anything that depends on siblings (diversity, ranking, pairwise preference) in a GRPO pipeline.
- **Hidden tool arguments** as isolation: resources that belong to the environment shouldn't show up in the model's action space.
- **Typed errors kept separate from reward**: tell infra failure apart from policy failure before it becomes a gradient.
- **Async-everything as a hard rule**, with the 10 ms × 2,000 = 20 s calculation as the reason.

## Genuine limitations visible on the page

- The example reward functions (`answer in response`, `"yes" in verdict.lower()`, exact-string diversity) are all hackable under RL pressure, and the docs present them without caveats. People copy examples.
- The `eval()` tool example runs model-generated code in the host process.
- Additive, unnormalized reward composition (within `Rubric`, across `RubricGroup`, across `EnvGroup`) has no built-in hard-gate layer. A safety violation is just another term in the sum unless you build a gate yourself.
- Silent eval → train dataset fallback.
- Hidden order dependency among reward functions sharing `state`.
- The whole v0 API is deprecated, so anything built on it now needs a migration. v1 is the target, and its docs are much thinner than this page. The detailed guidance on rubrics, stop conditions, and performance only exists for the version being removed.

## Connections

- Strategic context: [[Prime Intellect - Environments Hub]]; proof at scale: [[Prime Intellect - INTELLECT-3 A 100B+ MoE trained with large-scale RL]].
- Reward design: [[HUD - Verifier and Reward Design for RL Environments]], [[four-layer-scoring-stack-for-rl-environments]], [[generic-rubrics-fail-instance-specificity-is-the-mechanism]], [[ostensible-positives-from-low-entropy-answer-spaces]].
- Group RL mechanics the group-reward hook feeds: [[grpo-group-relative-policy-optimization]], [[std-normalization-difficulty-bias-in-group-rl]], [[zero-gradient-prompt-filtering]].
- Infra: [[scoring-as-a-pluggable-interface-in-rl-infrastructure]], [[trajectory-as-the-atomic-unit-of-policy-optimization]].
- Background: [[Epoch AI (Denain & Barber) - An FAQ on Reinforcement Learning Environments]].
