Created: 24-09-2026 16:18

Tags:

Status: #baby

## References

1. [StonyBrookNLP/appworld — A Controllable World of Apps and People for Benchmarking Function Calling & Interactive Coding Agents](https://github.com/StonyBrookNLP/appworld)
2. [AppWorld paper (arXiv 2407.18901), ACL 2024 Best Resource Paper](https://arxiv.org/abs/2407.18901)
3. [Parallelizing Worlds guide (RL rollouts)](https://github.com/StonyBrookNLP/appworld/blob/main/guides/parallelizing_worlds.md)
4. [Evaluating Terminal Agents via AppWorld MCP](https://github.com/StonyBrookNLP/appworld/blob/main/guides/evaluating_terminal_agents.md)

# Notes

## The problem: agent benchmarks were too shallow to test real assistants

Picture the task: "My roommates and I split utilities. Approve every Venmo payment request from my roommates for this month, and decline the rest." A person would open their phone contacts or a notes app to find who the roommates are, open Venmo, log in, list received requests, filter by sender and date, approve some, decline others. That is 5 to 15 API calls across 2 or 3 apps, with loops, conditionals, and data carried between steps.

Before AppWorld (2024), tool-use benchmarks mostly tested something far simpler: pick one API from a list, fill its arguments, maybe chain 2 or 3 calls in a straight line. Examples are ToolBench, API-Bank, and the Berkeley function-calling set. Those test *API selection and argument filling*. They do not test whether an agent can:

- operate over **many apps at once** that share one simulated person's life,
- write **real code** (loops, branches, exception handling, date math) instead of one call at a time,
- **explore interactively**, reading docs and inspecting outputs before deciding the next step,
- finish the goal **without breaking anything else** along the way.

The last point matters most, and most earlier benchmarks could not measure it. If you grade by comparing the agent's call sequence to a reference sequence, you punish valid alternative solutions and you miss side effects. If you grade by LLM judge, you get noise. AppWorld's central move is to **grade by the state of the world**, not by the path the agent took. That idea, plus the engineering to make it cheap and deterministic, is why this repo matters. It is also why it has turned into a standard RL environment and not only a leaderboard. See [[benchmarks-repurposed-as-training-environments]].

## Two layers: the Engine and the Benchmark

The repo ships two separable things.

**AppWorld Engine** is a sandbox that simulates a digital life. It has 9 day-to-day apps (Amazon, Spotify, Venmo, Gmail, phone, file system, notes, todo list, splitwise-style bill splitting) served by a local backend of **457 APIs** with 1,470 arguments in total (about 50 APIs per app). Behind them sit **101 database tables, 726 columns, and ~370K rows**. These are populated with the activities of ~106 fictitious people (ages 19–60) who are related as roommates, coworkers, spouses, and siblings, with 2+ years of account history.

Two helper apps complete the interface:
- `ApiDocs`: the agent can look up documentation *from inside the environment*, on demand.
- `Supervisor`: returns the person-in-charge's passwords, addresses, and payment cards. It also has the special `complete_task(answer=...)` API that ends an episode.

**AppWorld Benchmark** is 750 tasks built on this world. Each task has three parts:
1. `Supervisor`: whose behalf the agent is acting on.
2. `Instruction`: the natural-language request.
3. `Initial State`: a database snapshot plus a frozen "current datetime".

### Why build a whole engine instead of wrapping real APIs?

Real APIs cannot be reset, have rate limits and side effects (you cannot let an agent send real money on Venmo 10,000 times during RL), and change over time. So results cannot be reproduced. A simulated engine gives you:
- **Resettability**: every task starts from an exact DB state.
- **No consequences**: sending money or deleting a playlist harms no one.
- **Full observability**: you can diff the entire DB after the episode, which enables state-based grading.
- **Determinism**: time is mocked per task using `freezegun`, so "last month" always means the same thing.

The cost is fidelity. The authors put real work into making the engine feel real. It has **1,780 unit tests with 98% code coverage** on the API implementations, and the API docs are *generated from code*, so docs and behavior cannot drift apart. The APIs are consistent with each other too. Placing an Amazon order clears the cart *and* sends a confirmation email to Gmail. So the generated history is internally consistent across apps.

### How the population was generated (non-obvious)

They did not ask an LLM to write rows of data. The history is built by **calling the tested APIs themselves, in time order**, with time mocked forward. So the ~370K rows exist *because the app logic produced them*, and every foreign-key and cross-app side effect is valid. ChatGPT was used only for semantic free text (for example note contents or product descriptions), with humans reviewing it. This "populate the world through its own APIs" pattern is the right way to build a synthetic environment. It turns data consistency into a property of the code, not something you have to check afterwards.

## Task generation: scenarios, variants, distractors, hurdles

Tasks are not written one by one. The authors write **scenarios**: a task template with placeholders ("repurchase my last {color} {apparel}"). Each scenario has three programs:

- **Setup program**: changes the base DB so that this task variant is solvable and interesting.
- **Solution program**: a reference solution, used only to *validate* the task is solvable. It is not used for grading.
- **Evaluation program**: the state-based unit tests.

250 scenarios × 3 variants = **750 tasks**. Splits: Train 105, Dev 60, **Test-Normal 168**, **Test-Challenge 417**. Test-Challenge requires at least one API from **Amazon or Gmail**, which are held out ("unseen apps"), so it tests generalization to apps the agent has never seen demonstrated.

The setup program guarantees four properties. Each one exists to block a specific shortcut:

1. **Well-defined**: the preconditions actually hold (the supervisor really has past T-shirt orders in the requested color and size).
2. **Distractors**: plausible decoys exist (several T-shirt orders in *other* colors). A lazy agent that grabs the first result fails.
3. **Hurdles**: realistic obstacles (the default payment card has expired). The agent has to notice and adapt, not just follow a script.
4. **Contrast sets**: the 3 variants of a scenario differ in ways that change the correct action. That is what makes the SGC metric meaningful (below).

Task complexity (from the validation solutions):

| Metric | Test-N | Test-C |
|---|---|---|
| Avg apps per task | 1.5 | 2.0 (max 6) |
| Avg unique APIs | 8.2 | 10.5 (max 26) |
| Avg solution code lines | 41.3 | 56.9 (max 134) |
| Avg evaluation tests | 5.9 | 8.0 (max 24) |

Solutions use loops, conditionals, try/except, dicts and sets, regex over free text, and a lot of datetime arithmetic. This is **programming**, not function calling.

## The key idea: state-based evaluation

### Plain version

Do not grade *how* the agent did it. Grade *what the world looks like afterwards*. Did the right things change? Did anything else change that should not have?

### Technical version

Let $D^s$ be the DB state at task start and $D^f$ the state when the agent calls `complete_task`. The engine computes a diff:

$$\Delta D = \text{diff}(D^s, D^f)$$

This is the set of row-level inserts, updates, and deletes across all app databases. It is computed with hashing so it is fast (~0.6 s per task). A task **passes** iff:

1. **Expected changes present**: every change the task requires shows up in $\Delta D$ with the right values. For example, `music_player.is_playing` is true and the set of queued song IDs equals the private ground-truth set.
2. **No collateral damage**: $\Delta D$ contains *nothing else*, except changes on an **allow-list** of harmless side effects (a login-token table updating, for example).
3. For answer-seeking tasks, the `answer` passed to `complete_task` matches the ground truth.

Each check is a Python assertion with a natural-language `requirement` string. The README shows a real failure report:

```
assert model changes match spotify.MusicPlayer.
    test.case(models.changed_model_names(), "==", {"spotify.MusicPlayer"})
AssertionError:  set() == {'spotify.MusicPlayer'}
```

The line `changed_model_names() == {...}` is the collateral-damage check in its simplest form. The agent must have changed *exactly* this set of tables.

### The `no_op_pass` / `no_op_fail` labels (subtle and important)

Each test is labeled by what happens when the agent **does nothing**:
- `no_op_fail`: this test fails if the agent does nothing. It measures actual accomplishment.
- `no_op_pass`: this test passes when the agent does nothing. It guards against damage ("don't decline requests from roommates").

Why label them? A partial-credit score that counts `no_op_pass` tests would reward a do-nothing agent. The labels let you tell "the agent achieved something" apart from "the agent avoided breaking things". This matters when you turn test pass-rates into a **dense RL reward**. Naive per-test reward gives a sizeable free baseline to the empty policy, and a policy can learn to be passive. Anyone using `world.evaluate()` as a reward should weight or filter by this label. See [[four-layer-scoring-stack-for-rl-environments]].

### Why state-based beats trajectory matching

- **Many valid paths**: you can find the roommates via phone contacts or via Venmo friends. Both are fine. Trajectory matching would reject one.
- **Language-agnostic**: because only the DB is graded, the agent can use Python, curl, JS, or MCP tool calls. The README states this directly: "evaluation is state-based (database) and not process-based (code/APIs)".
- **Catches harm**: a trajectory can look plausible while quietly deleting an unrelated file. The DB diff shows it.

This is a concrete case of [[static-versus-dynamic-agent-evaluation]]. The environment is dynamic, but the grader is a deterministic program over final state. No LLM judge is needed.

## Metrics: TGC and SGC

**Task Goal Completion (TGC)** = % of tasks where *all* tests pass. It is all-or-nothing per task.

**Scenario Goal Completion (SGC)** = % of scenarios where the agent passes **all 3 variants**.

$$\text{SGC} = \frac{1}{|S|}\sum_{s \in S} \prod_{t \in \text{variants}(s)} \mathbb{1}[\text{task } t \text{ passes}]$$

SGC is a **robustness/consistency** metric. If an agent solves each variant independently with probability $p$, SGC ≈ $p^3$. A TGC of 50% would then give SGC ≈ 12.5%. Real SGC is higher than $p^3$ because variants of a scenario are correlated, but still far below TGC. That gap measures how much success is **luck versus skill**. Contrast-set variants are built so the same memorized procedure fails on at least one of them.

## Results from the paper (2024 baselines)

Agent scaffolds tested:
- **ReAct**: think, write a code block, see output, repeat. It explores the docs interactively.
- **PlanExec**: write a high-level plan, then run each step ReAct-style.
- **FullCodeRefl**: write the whole solution program at once, run it, reflect on errors, retry (up to 5 times).
- **IPFunCall**: native (parallel) function calling, one API per tool call, no free code.
- **ToolLLaMA / CodeAct**: fine-tuned open models.

TGC | SGC:

| Model | Method | Test-N | Test-C |
|---|---|---|---|
| GPT-4o | ReAct | **48.8 \| 32.1** | **30.2 \| 13.0** |
| GPT-4o | PlanExec | 44.6 \| 23.2 | 19.7 \| 7.9 |
| GPT-4o | FullCodeRefl | 33.9 \| 26.8 | 19.2 \| 12.2 |
| GPT-4o | IPFunCall | 32.1 \| 16.1 | 18.0 \| 10.1 |
| GPT-4 Turbo | ReAct | 26.8 \| 12.5 | 17.5 \| 5.8 |
| Llama 3 70B | FullCodeRefl | 24.4 \| 17.9 | 7.0 \| 4.3 |
| DeepSeek-Coder 33B | FullCodeRefl | 13.1 \| 8.9 | 5.8 \| 2.9 |
| CodeAct / ToolLLaMA | — | 0.0 | 0.0 |

What stands out:

- **Interactive code beats function calling by ~17 points** (ReAct 48.8 vs IPFunCall 32.1 on Test-N, same model). Letting the model write loops and keep variables across turns (the IPython shell is stateful, like Jupyter) is a big structural advantage. One code block can do what takes 20 tool calls. This is an early data point behind "code as the action space" ([[tool-use-and-environmental-interaction-in-llm-agents]], [[code-generation-with-llms-in-simulations]]).
- **Planning up front hurts on hard tasks** (PlanExec 19.7 vs ReAct 30.2 on Test-C). When the environment is full of distractors and hurdles, a plan written before seeing any data is wrong more often than right. Looking before you plan wins.
- **Fine-tuned tool models score exactly 0.** ToolLLaMA and CodeAct, trained on simpler tool benchmarks, do not transfer at all. That is strong evidence that the earlier benchmarks were measuring a different, easier skill.
- **Difficulty collapse**: GPT-4o ReAct scores 58.3% on level-1 tasks, ~40% on level-2, and 21.0% on level-3. Tasks needing 60+ lines of solution code are under 20% for every method.
- **API retrieval is not the bottleneck.** Giving GPT-4o the oracle set of relevant APIs adds only +7.8 (Test-N) and +9.8 (Test-C). The failure is in *reasoning and coding over the environment*, not in finding the right tool. This goes against the common assumption that tool-use agents mostly fail at tool selection.
- **More demos help Normal but not Challenge.** Going from 3 to 10 demonstrations, with similarity-based selection, gives up to +6.6 on Test-N and ~0 on Test-C. Prompt engineering does not generalize to unseen apps.
- **Cost**: $0.02–$1.33 per task depending on method. The full paper suite cost ~$10K.

### Failure taxonomy (manual GPT-4o analysis)

1. **Interaction avoidance / hallucination**: the model makes up a roommate's email instead of looking it up. This is the most important one. The model acts as if it knows the world instead of querying it.
2. **API misunderstanding**: reads fields that do not exist in the response schema, or calls `sent_payment_requests` when it needed `received_payment_requests`.
3. **Partial instruction following**: approves *all* requests when told to approve only roommates' requests. The distractors are built to catch exactly this.
4. **Commonsense confusion**: mixes up a song's *added-to-library* date with its *release* date.
5. **State-tracking failure**: loses track of what it already did, repeats work, and runs out of the call budget.

Most of these are not about model knowledge. They come from grounding and discipline. That points toward training (RL on the environment) rather than prompting as the fix.

## How the repo is engineered (their approach, useful if you talk to this team)

The design choices show a clear philosophy: **one environment, many ways in, one grader.**

**Serverless by default.** The APIs are real HTTP endpoints built on FastAPI. By default, though, calls go through FastAPI's `TestClient`, which simulates HTTP inside the same process without a network. The first task load takes 4–5 s (loading the apps). After that, each task loads in under 0.5 s, even though each one has a different DB. The DBs are SQLite, and each task stores only its **diff from the base DB** to save disk.

**Three API calling forms, all grading the same:**
- Functional: `apis.spotify.like_song(song_id=123, access_token=...)`. This is what the paper experiments use, because the agent has fewer decisions to make.
- REST: `requester.post("/spotify/songs/123/like", data={...})`.
- MCP: `world.mcp.call_tool("amazon__show_product", {"product_id": 1})`. The newer addition.

The functional form wraps the REST form, which in turn wraps `requests` or `TestClient`. There is a deterministic mapping between them. That is the reason the authors chose the simplest one for experiments.

**Guardrails built into the loop:**
- A failed API call (e.g. 422) becomes a **Python exception with a traceback**, so the agent sees its error as normal code feedback (`raise_on_failure`).
- A request cap of **2,000 calls per task** catches infinite loops. Execution timeouts exist too.
- `raise_on_unsafe_syntax` does AST-level static analysis to block dangerous modules. `raise_on_unsafe_execution` monkey-patches destructive stdlib functions (and also disruptive ones like `time.sleep` and `exit`). The authors say this is not foolproof, so they ship a Docker mode (`appworld serve environment --docker`) that only mounts `data/` and `experiments/outputs/`.

**State checkpointing.** `world.save_state()` / `world.load_state(id)` can roll back the DB. The README warns that agents using it **must not be compared** with agents that don't, since real life has no undo on a Venmo payment. That is a sensible fairness rule. For tree search or RL with branching rollouts, this is the feature you would use.

**Decoupled servers.** There are three servers: `environment` (HTTP wrapper around the whole `AppWorld` class), `apis`, and `mcp`. Each can run in-process, as a separate process, or in Docker. The browser playground (`appworld play`) drives the Python environment remotely through the environment server. Speed order: in-process > API server > MCP server (each hop adds latency). HTTP MCP transport is faster than stdio.

**Parallel rollouts for RL** (Parallelizing Worlds guide). Two hard rules:
1. Each world needs a unique `experiment_name + task_id`, or the output directories overwrite each other. For several rollouts of the same task, add a path suffix: `experiment_name + "/roll_out_1"`.
2. **A server process can hold only one world at a time.** The reason is that `freezegun` mocks time *process-wide*, and many DB connections at once would leak memory. So parallelism means one process per world: `multiprocessing.Pool` in unified mode, or N server processes in decoupled mode.

This is the practical limit to know for RL. You scale AppWorld by processes, not by threads or async inside one process. The guide recommends checking any parallel setup by running the released ground-truth solutions and confirming every task passes (`appworld verify tasks`). The same check runs in their CI. See [[multi-agent-environment-interface-design]] for the general version of this interface problem.

**Terminal agents via MCP.** The guide shows how to plug Claude Code, Codex, and Gemini CLI into AppWorld as MCP clients. It also warns you to put `APPWORLD_ROOT` *outside* the agent's working directory. A terminal agent with file access could otherwise read the task ground truth and cheat. That is a real reward-hacking vector, and they call it out plainly.

**Agent library (`appworld-agents`).** Scaffolds were reimplemented as "simplified" agents with no framework. The base `Agent` class is ~100 lines: a loop of generate code → execute → feed back output until `complete_task` or the max step count. Subclasses are `FunctionCallingAgent`, `ReActCodeAgent`, and `FullCodeAgent`. There are also smolagents and openai-agents versions. Experiments are named `{agent}/{base_llm_creator}/{base_llm_name}/{dataset}` and defined in jsonnet configs generated from Jinja templates. The model list covers Claude Sonnet 4.5 / Opus 4.1, GPT-5, Gemini 2.5, Grok 4, Qwen3, DeepSeek v3.2, Kimi K2, GLM-4.6, and more. `appworld run auto --agent-name X --model-name Y --dataset-name Z` builds the config on the fly.

## Contamination defenses (an unusual amount of care)

- App implementations, tests, task-generation code, and task data ship in **encrypted `.bundle` files** (in git LFS and in the pip package). `appworld install` unpacks them locally. The goal is that crawlers never see plaintext, so it cannot end up in pretraining data.
- A **canary string** (`appworld:d17ac3f:26b5c67b-...`) is embedded so training pipelines can filter it out.
- Users are asked not to post extracted code or data publicly.
- **Test sets ship only evaluation programs**. There are no setup or solution programs for Test-Normal/Test-Challenge, so you can score yourself but cannot see how the initial state was built.
- The split rules are written down clearly. Train is for demos, RL, and error analysis. Dev is for hyperparameter tuning. Test is for aggregate scores only: no looking at per-task reports, no tuning prompts against test failures. Generic hints learned from train/dev are allowed ("log in before calling authenticated APIs"). **Hardcoding API calls into the agent is not allowed** (for example, logging in to every app in a fixed first step), though prompting the agent to do it itself is fine.

The weak point is obvious. Evaluation code for the test sets is local, so the rules run on the honor system. The authors say a fully hidden test set with new apps is future work.

## Genuine limitations

- **API-only.** There is no GUI. Real assistants often have to deal with screens, and AppWorld does not test visual or UI grounding.
- **Single agent.** There is no coordination between two assistants (such as two people's agents scheduling a meeting). See [[single-agent-myopia-in-llm-evaluation]].
- **Small for training.** 105 train tasks (plus dev) is small for supervised fine-tuning. The paper suggests augmentation or self-instruct. In practice, RL on the environment with the state-based reward is how people use it now. Scenario templates make it possible to generate more variants, and the repo includes guides for writing new apps and task generators.
- **Cultural and geographic bias.** The apps and habits reflect North American and European digital life.
- **Simulation fidelity ceiling.** The world only contains what 9 apps and ~100 people encode. Agents that overfit to AppWorld's API conventions (consistent pagination, tokens from `Supervisor`) may not transfer to messy real APIs.
- **One world per process** (freezegun) makes large-scale RL rollouts heavier in memory than a thread-safe design would be.
- **Scores are aging.** The 48.8 / 30.2 GPT-4o numbers are 2024 baselines. The live leaderboard with newer models is the current reference, and the gap between Normal and Challenge (seen vs unseen apps) is still the number to watch.

## Why this matters / connections

- **A benchmark that became a training environment.** Deterministic resets, a programmatic dense-ish reward (`world.evaluate()` works mid-episode, and the README names "RL rewards" as a use), parallel rollouts, and a train split make it an RL gym for agentic coding. This is the pattern in [[benchmarks-repurposed-as-training-environments]]. The reward is a heavy program (DB diff + assertions), which ties to [[reward-computation-as-a-heavyweight-pipeline-stage]].
- **The collateral-damage check is a safety measurement.** "Did the agent do exactly what was asked and nothing more" is a small, concrete version of side-effect avoidance. The `no_op_pass` label makes "do no harm" tests separate from "achieve the goal" tests, so the two can be reported apart. This is useful framing for AI-safety-oriented agent evals.
- **Code as action space.** ReAct-with-code beating native function calling by a wide margin, and fine-tuned tool models scoring 0, support the view that agents should act by writing programs in a stateful interpreter. See [[tool-use-and-environmental-interaction-in-llm-agents]] and [[agentic-rag-approach]] for the retrieval side, which AppWorld shows is *not* the main bottleneck.
- **Consistency metric design.** SGC (all variants of a scenario must pass) is a cheap, reusable way to separate reliable skill from lucky success. It could be applied to any benchmark that has templated variants. See [[static-versus-dynamic-agent-evaluation]].
- **Meta-agent work.** Automated scaffold search ([[meta-agent-search]]) needs exactly this kind of cheap, deterministic, programmatic grader to score candidate agents.
