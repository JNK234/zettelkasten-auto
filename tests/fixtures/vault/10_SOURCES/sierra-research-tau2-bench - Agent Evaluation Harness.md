Created: 24-09-2026 16:20

Tags:

Status: #baby

## References

1. [sierra-research/tau2-bench (τ-bench / τ²-bench / τ³-bench code)](https://github.com/sierra-research/tau2-bench)

# Notes

Read at commit `b7ea907` (17 Sep 2026), release line 1.0.1. This note is about the **code**: how the harness is built and how it scores. The ideas from the papers (POMDP framing, pass^k math, the τ² dual-control results, simulator error rates) are already in [[2406.12045 - Tau-Bench (Tool-Agent-User Interaction)]]. Read that first, then this. For why Sierra's approach matters to an application, see [[Sierra SE Agent Architecture - Application]].

## The problem the repo solves

Sierra sells customer-service agents. Before it can ship one, it has to answer a plain question: "if this agent talks to 10,000 real customers, how often does it do the right thing, and does it do it *every* time?" Chat-quality judges cannot answer that. A judge reads the transcript and says "looks helpful". It cannot tell you whether the refund actually went to the right card.

So the repo's whole design follows one idea: **build a small fake business (a database plus tools plus a written policy), let an agent and a simulated customer talk, then check what the database looks like at the end.** If the database matches what a correct agent would have produced, the agent passed. Nothing else about the conversation matters by default.

The repo has grown from that seed into four things at once:

1. A **text benchmark** in five domains: `mock`, `airline` (50 tasks), `retail` (114), `telecom` (2,285 generated, 114 used), `banking_knowledge` (97 tasks over 698 documents).
2. A **voice benchmark** (τ-voice, arXiv 2603.13686): full-duplex audio, where both sides can talk at once.
3. A **retrieval benchmark** (τ-knowledge, arXiv 2603.04370): the policy is no longer in the prompt. The agent has to search a document store for it.
4. An **RL environment**: a Gymnasium wrapper with train/test splits, so the same domains can train agents, not just score them.

The package is still called `tau2` and the repo `tau2-bench`, but the README calls the current release **τ³-bench**.

## Layer 1: the pieces of one simulation

Four actors. Think of a phone call to an airline.

- **Environment.** A Python object that holds the database (`db.json` or `db.toml`) and exposes tools. Each tool is a method with a decorator such as `@is_tool(ToolType.READ)` or `@is_tool(ToolType.WRITE)`. The READ/WRITE tag matters later for scoring.
- **Agent.** By default `LLMAgent`. Its whole system prompt is:

  ```
  <instructions>
  You are a customer service agent that helps the user according to the <policy> provided below.
  In each turn you can either:
  - Send a message to the user.
  - Make a tool call.
  You cannot do both at the same time.
  Try to be helpful and always follow the policy. ...
  </instructions>
  <policy>{domain_policy}</policy>
  ```

  That is the entire scaffold. No planner, no reflection. This is on purpose. The leaderboard's "standard" category requires this exact scaffold, so the number measures the model and not someone's prompt engineering.
- **User simulator.** Another LLM that plays the customer, with a task-specific scenario ("you are Bob Nakamoto, a crypto trader, income $95,000, credit score 540..."). Its guidelines tell it to disclose information gradually, never invent facts, and end the call with one of three control tokens: `###STOP###` (goal done), `###TRANSFER###` (handed to a human), `###OUT-OF-SCOPE###` (the scenario doesn't say what to do next).
- **Orchestrator.** The loop that passes messages. `Orchestrator` is half-duplex: agent → user → agent → tool → agent, one party at a time. `FullDuplexOrchestrator` is for voice (below).

Defaults from `config.py`: `max_steps=200`, `max_errors=10`, seed 300, agent and user both `gpt-4.1-2025-04-14` at temperature 0. One detail stands out: the LLM used to *review the user simulator* (auth classification, hallucination checks) defaults to `claude-opus-4-5`. The simulator is audited by a stronger model than the one playing it.

There are three agent classes besides the default:
- `LLMGTAgent` gets the task's reference actions injected as `<resolution_steps>` in its prompt. This is an **oracle ceiling**. If it fails, the task or the simulator is broken, not the agent. That is a cheap way to find bad tasks.
- `LLMSoloAgent` gets no user at all, only a ticket. In telecom this removes the coordination problem, which is how τ² separated "can't reason" from "can't coordinate".
- Gym agents, for RL.

## Layer 2: how a run is scored

This is the core of the repo and the part most people misread. The repo now has a dedicated doc (`docs/evaluation.md`) because people kept getting it wrong (issues #224, #129).

### The task schema

Each task has `evaluation_criteria` with five fields:

| Field | What it is |
|---|---|
| `actions` | **One** reference sequence of tool calls that solves the task |
| `env_assertions` | Python predicates run on the final environment |
| `communicate_info` | Strings the agent must say to the user |
| `nl_assertions` | Plain-English checks graded by an LLM (experimental) |
| `reward_basis` | Which of the above actually count |

### The reward is a product

$$
R \;=\; \mathbb{1}[\text{terminated normally}] \;\cdot \prod_{c \,\in\, \text{reward\_basis}} r_c, \qquad r_c \in \{0, 1\}
$$

Symbol by symbol:
- $\mathbb{1}[\text{terminated normally}]$ is 1 only if the run ended with `AGENT_STOP` or `USER_STOP`. Hitting `max_steps`, too many tool errors, or a crash gives reward 0 before any evaluator runs.
- $c$ ranges over the component types listed in the task's `reward_basis`: `DB`, `ENV_ASSERTION`, `COMMUNICATE`, `NL_ASSERTION`, `ACTION`.
- $r_c$ is each component's score. Each is all-or-nothing.
- A component **not** in `reward_basis` can still be computed (for diagnostics) but does not enter the product.
- A missing criterion scores 1.0, not 0. "Not checked" is not "failed".

Because it's a product, one zero kills the task. There is no partial credit in the official score.

The default for airline, retail and telecom is `reward_basis = [DB, COMMUNICATE]`, which matches the original 2024 paper.

### How the DB check actually works (the clever part)

`EnvironmentEvaluator.calculate_reward` builds **two fresh environments** from the same constructor:

1. **Predicted environment.** Load the task's initial data, then **replay every tool call from the recorded trajectory** into it (`set_state(message_history=full_trajectory)`). Note it does not read the live environment from the run. It rebuilds it from the transcript. That makes any saved trajectory re-scorable later, on any machine.
2. **Gold environment.** Load the same initial data, then replay the task's reference `actions`.

Then it hashes both databases and compares:

$$
r_{\text{DB}} = \mathbb{1}\big[\,h(\text{DB}^{\text{agent}}_{\text{pred}}) = h(\text{DB}^{\text{agent}}_{\text{gold}})\,\big] \cdot \mathbb{1}\big[\,h(\text{DB}^{\text{user}}_{\text{pred}}) = h(\text{DB}^{\text{user}}_{\text{gold}})\,\big]
$$

There are two databases because of τ²'s dual control. In telecom the **user** has a phone with state (airplane mode, SIM status, roaming, data saver). The customer's device is a database too, and it has to end up right.

What follows from this design:
- `actions` is **not a checklist**. It is just a convenient way to say what the final database should look like ("play these steps on a clean copy"). Any path that reaches the same end state passes.
- Read-only calls never change the hash. An agent can skip every lookup in the reference trajectory and still score 1.0.
- The docs' worked example (airline task 1): the reference actions are two reads, `get_user_details` and `get_reservation_details`. The correct behavior is to *refuse* a cancellation. So the target hash equals the *initial* hash, and **an agent that makes zero tool calls and just politely refuses gets full reward.** The task also carries an `nl_assertion`, "Agent should not approve the cancellation", but `NL_ASSERTION` is not in `reward_basis`, so that line is diagnostic only.

This is outcome-based checking in its purest form. It is very hard to game by writing a nice transcript. It is blind to *how* the outcome was reached. That's the trade-off discussed in [[reward-hacking-as-eval-false-positive]].

### The other components

- **COMMUNICATE**: every string in `communicate_info` must appear as a **substring** in the agent's messages. Cheap, deterministic, and brittle. A real example is stored in the repo (`data/tau2/domains/retail/task_issues/task_4_issue_*.json`). It's a voice run of retail task 4. The LLM reviewer's summary says the agent authenticated the user and did the item modifications correctly, and the DB check passed (`db_match: true`). The final reward is still 0, because the required string `'10'` was "not communicated". The repo keeps it as a known issue. Most likely the number was said in a form the substring match didn't catch. Substring matching on spoken output is fragile.
- **ENV_ASSERTION**: arbitrary predicates on the final state. Telecom uses these. For example, `assert_mobile_data_status(expected_status=True)` and `assert_internet_speed(expected_speed=200)` run against the *user's* phone environment. This checks "is the customer's problem actually fixed?" instead of "did the agent follow the script?"
- **ACTION**: the agent's tool calls must include every reference action. Only 9 of the 97 `banking_knowledge` tasks use it (the other 88 use `[DB]` alone). Airline, retail and telecom never use it. The docs warn that turning it on assumes the reference path is the *only* valid path.
- **NL_ASSERTION**: LLM-as-judge. Marked WIP. Never in the official basis.

Across the whole repo, **the default official score never depends on an LLM judge.** LLM judges exist (reviewers, auth classifier, hallucination checker), but only as diagnostics or to decide when to re-run a simulation. That's a deliberate stance against [[llm-as-judge-evaluation-pattern]] for the headline number.

### Diagnostics that don't count

`tau2 view` and `agent_metrics.py` report things like `partial_action_reward` (m/n reference actions matched, split by READ vs WRITE), DB match counts, termination reasons, auth outcome (succeeded / failed / not needed, from an LLM classifier), and LLM-judge error counts by severity for agent vs user, including which side made the *first* critical error. The READ/WRITE split exists to catch one specific failure: "DB passes but the agent never attempted the write". That happens when the correct end state equals the start state.

### pass^k in code

```python
def pass_hat_k(num_trials, success_count, k):
    return math.comb(success_count, k) / math.comb(num_trials, k)
```

For each task, $\hat{p}^k = \binom{c}{k} / \binom{n}{k}$, averaged over tasks. $n$ is the number of trials of that task, $c$ how many had reward exactly 1 (within 1e-6), and $k$ the number of independent tries that must *all* succeed. The derivation and why it's unbiased are in the τ-bench note and [[pass-hat-k-reliability-metric]]. The leaderboard "strongly prefers" at least 4 trials, so pass^1 to pass^4 are all estimable.

## Layer 3: the telecom domain and generated tasks

Telecom is where τ² put its main idea into code.

**The user has 30 tools** on a simulated phone (`user_tools.py`): `check_status_bar`, `toggle_airplane_mode`, `reseat_sim_card`, `toggle_roaming`, `turn_data_saver_mode_off`, `run_speed_test`, `set_network_mode_preference`, and so on. The agent cannot touch the phone. It can only tell the user what to do, then read back what the user reports. That is what tech support actually is.

**Tasks are generated, not hand-written.** Each intent (`mobile_data_issue`, `service_issue`, `mms_issue`) has a `TaskManager` with:
- `selection_sets`: groups of atomic faults, each paired with a breaking function and a fixing function (`data_roaming_off` / `fix_data_roaming_off`, `set_vpn_slow_speed` / `fix_set_vpn_slow_speed`, `set_data_usage_exceeded` / `fix_data_usage_exceeded`...)
- `compose_tasks`: `itertools.product` over each selection set plus `None`. A task is any combination of 0-or-1 fault from each set, so multi-fault tasks come for free.
- `get_env_assertions`: the success predicates (data on, speed 200, rated "excellent")
- `is_fixed` and a `task_validator` that throws out impossible combinations
- persona variants encoded into the task id, e.g. `[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]`

That gives 2,285 tasks (`tasks_full.json`), a `small` set of single-fault tasks, and a balanced 114 (`tasks.json`) for the leaderboard. The troubleshooting policy is also shipped as flowcharts (`workflows/tech_support_path{1,2,3}_*.dot`). The policy is literally a decision graph, and tasks are walks through combinations of faults on it.

Why this matters: it breaks the hand-annotation bottleneck that capped τ-bench at 50/115 tasks. It also gives a clean difficulty axis (number of combined faults).

## Layer 4: banking_knowledge, the policy you have to find

In airline and retail the whole policy sits in the system prompt. Real banks have hundreds of documents. τ-knowledge asks: **what happens when the agent has to retrieve the policy before it can follow it?**

The corpus is 698 JSON documents (credit cards, bank accounts, disputes, ATM fees...) for a fictional "Rho-Bank". Tasks list their `required_documents`. For example, task_006 needs six credit-card documents to figure out that the only card that fits a 540-credit-score customer's fee constraints is `EcoCard`.

### Retrieval is the experiment variable

The `--retrieval-config` flag swaps how the agent gets at the corpus, with everything else held fixed:

| Config | What the agent gets |
|---|---|
| `no_knowledge` | nothing (floor) |
| `full_kb` | all documents pasted into the prompt |
| `golden_retrieval` | exactly the task's required docs in the prompt (retrieval ceiling) |
| `bm25`, `openai_embeddings`, `qwen_embeddings` | a `KB_search` tool; add `_reranker` / `_grep` suffixes |
| `grep_only` | a grep tool |
| `terminal_use` / `terminal_use_write` | a sandboxed shell over the document files |
| `alltools` (**default**) | `KB_search_bm25` + `KB_search_dense` (text-embedding-3-large) + read-only shell |

This is a proper ablation grid. `golden_retrieval` minus `bm25` isolates the retrieval gap. `full_kb` minus `golden_retrieval` isolates the cost of irrelevant context. `terminal_use` tests the Claude-Code-style "just give the agent a shell" approach against classic RAG. The shell runs in Anthropic's `sandbox-runtime` (pinned `@anthropic-ai/sandbox-runtime@0.0.23`), which is interesting on its own: a competitor's sandbox used as benchmark infrastructure. The retrieval pieces (embedders, BM25/cosine/grep retrievers, BGE/Qwen/pointwise-LLM rerankers) are modular in `src/tau2/knowledge/`. Related: [[retrieval-augmented-generation]], [[vector-databases-in-rag]].

### Discoverable tools

This is the most unusual design in the repo. Many of the bank's write tools are **hidden**. They have names like `file_credit_card_transaction_dispute_4829`, `order_replacement_credit_card_7291`, `get_atm_deposit_images_8473`. The agent's visible toolset only has meta-tools:

- `unlock_discoverable_agent_tool(name)`: works only if you know the exact name, which you can only get by reading the right document
- `call_discoverable_agent_tool(name, json_args)`: fails with "has not been unlocked" if you skipped the unlock
- `give_discoverable_user_tool(name, args)`: hands the *customer* a tool (e.g. "apply for the card yourself"). That's dual control again, this time through documents.

The random numeric suffixes stop the model from guessing a tool name from pretraining. Retrieval is required just to get the capability, not only to learn the rule. That makes "did it read the right document" something you can check on the DB state, not something you judge.

### The grading bugs, and what they teach

CHANGELOG 1.0.1 (2026-07-15) is worth reading as a case study in how outcome-based grading fails:

- **Prudent reads zeroed the reward (#329).** Every discoverable call, including reads, was logged to an `agent_discoverable_tools` table, and that table is part of the DB hash. So an agent that double-checked its work (e.g. calling `get_pending_replacement_orders_5765` after ordering a card) changed the hash and got 0. **Careful behavior was being punished.** The fix: log writes always, and log reads only if they're on a per-task `read_log_allowlist` (reads the gold trajectory needs). Re-grading only moved scores up, by up to ~9 points pass^1 (GPT-5.5 xhigh 37.37 → 46.39; GPT-5.4 xhigh 30.67 → 39.43).
- **`25` vs `25.0` (#397).** Numeric args changed deterministic record IDs and so the hash. Now all JSON numbers are parsed as floats.
- **Answer leakage in fixtures (#374).** Fee rows in the DB carried notes like "(SHOULD BE FREE - 1ST OF 2)". They told the agent which fees were wrong instead of making it work that out from the policy. Separately, the gold refund for task_074 was itself wrong ($8.00 vs the policy-faithful $14.50).
- **Tool/doc mismatch (#403).** A document said transactions come back newest-first. The tool returned insertion order, so a tie-break rule couldn't be applied.
- **Hallucinated tool calls during replay** used to raise, which triggered retries and finally an `INFRASTRUCTURE_ERROR`. Those are *excluded* from pass^k, so a failing agent could silently drop out of the denominator. Now they're treated as no-ops, the same as in the live environment.

The lesson: a hash-equality verifier is only as good as your control over **what goes into the hash**. Any side effect you didn't mean to grade (logs, number formatting, ordering) becomes a hidden requirement. After 1.0.1, banking_knowledge scores from before and after are **not comparable**. `tau2 evaluate-trajs --fresh-tasks` re-grades old trajectories against the current task definitions. It's possible only because scoring rebuilds state from the transcript. The same theme appears in [[label-noise-dominates-saturated-benchmarks]]: at these score levels, task bugs are a big share of the measured error.

The README credits a separate effort for the 75+ task fixes across airline/retail/banking: **SABER** (Cuadron et al., ICLR 2026 workshop). The older `sierra-research/tau-bench` repo now warns that its original tasks are outdated and buggy.

## Layer 5: voice (τ-voice)

### Full-duplex orchestration

Text is turn-based. Real phone calls aren't: people interrupt, say "mm-hmm", cough, talk to their kids. `FullDuplexOrchestrator` moves time forward in **ticks** (200 ms by default). Each tick, both sides may emit an audio chunk, tool calls and results. The trajectory becomes `list[Tick]` instead of `list[Message]`.

- Agent side: `DiscreteTimeAudioNativeAgent` with a provider adapter. OpenAI Realtime, Gemini Live and xAI Grok Voice are fully supported. Nova Sonic, Qwen, Deepgram and LiveKit are experimental.
- User side: `VoiceStreamingUserSimulator`, an LLM that decides to yield, interrupt, backchannel or wait. Its speech comes from ElevenLabs (`eleven_v3`) and is transcribed with Deepgram `nova-3`. The audio goes through 8 kHz μ-law telephony compression, background and burst noise, and frame drops.
- Personas: 7 synthetic voices (2 "control", 5 "regular"). Leaderboard voice runs must use `regular`. Sierra's own voice IDs are internal. External users generate equivalent ones with a seeded script, and Sierra re-runs final evaluations on its own voices for parity.

Every evaluator has a full-duplex twin that flattens ticks back into messages before scoring. The env evaluator orders them user-tool-calls-first within each tick, matching how the orchestrator executes them. So **the task-success definition is identical across text and voice.** Only the channel changes. That's what makes text-vs-voice gaps interpretable.

### Interaction metrics, computed without judges

Voice adds a second panel scored from tick-level speech flags:

| Metric | Meaning |
|---|---|
| L_R, R_R | response latency / fraction of user turns answered before the user had to speak again |
| L_Y, R_Y | yield latency / fraction of real interruptions where the agent stopped within 2.0 s |
| I_A | agent-interrupts-user events per user turn (can exceed 1) |
| S_BC, S_VT, S_ND | selectivity: correctly talking through backchannels, ignoring vocal tics, ignoring speech not directed at the agent |

An error on a backchannel means yielding within 1.0 s. An error on a tic or undirected speech during silence means responding within 2.0 s. The priority when categories overlap is backchannel > tic > non-directed > real interruption. The end-of-call `###STOP###` creates a fake one-tick user segment, which is removed so it doesn't count as a no-yield. The leaderboard hides any rate backed by fewer than 10 events. The design shows a real tension: yielding fast (good R_Y) and ignoring noise (good S_BC/S_VT) pull in opposite directions, and scoring both keeps an agent from maxing one by giving up the other. Related: [[llm-latency-mismatch-in-real-time-simulations]], [[turn-based-synchronous-agent-execution-model]].

### Hallucination retry

In voice mode, an LLM fact-checker (`hallucination_reviewer.py`) reads the *user simulator's* lines and flags any fact not in its instructions, not said by the agent, and not returned by a user tool. Examples: an invented zip code, or choosing a preference the scenario didn't give. If it finds one, the simulation is re-run with feedback (`--hallucination-retries`). This directly answers τ²'s finding that 16–47% of simulator trajectories had errors: the environment's own failure rate is now measured and partly filtered out, not just reported.

## Layer 6: RL and interoperability

- **Gym** (`--extra gym`): `AgentGymEnv` (you play the agent against the simulated user) and `UserGymEnv` (you play the user against an automated agent). Standard `reset()` / `step(action)`. An action is either a message or a tool call like `"check_network_status(user_id='user_123')"`. `solo_mode` works here too. Splits: airline 30/20, retail 74/40, telecom 74/40 train/test, with `base` = everything for the leaderboard. The repo is built to double as a **training environment with a verifiable reward**: the same DB-hash check that grades the leaderboard is a clean binary RL reward. Connects to [[reinforcement-learning-in-language-models]] and [[tool-use-and-environmental-interaction-in-llm-agents]].
- **Agentified τ-bench** (`src/experiments/agentify_tau_bench`): wraps the benchmark as an **A2A** (Agent-to-Agent protocol) service for Berkeley RDI's AgentX-AgentBeats competition. Any A2A agent can be scored with no adapter code. The benchmark becomes a peer agent, not a library you import.

## The leaderboard rules as a statement of values

The standard vs custom line tells you what Sierra thinks the number means:

- **Text standard** = a general LLM behind the default scaffold, standard tools and prompts, no training on τ-bench. Adding a planner, router, extra tool or changed control flow makes it **custom**.
- **Voice standard** is looser. The submitted *system* can have any internal architecture (its own ASR/TTS, multiple models, internal tools and routing) as long as it sits behind the standard τ-voice interface and needs no benchmark-side changes.

That asymmetry is deliberate. For text, the object of evaluation is the **model**. For voice, it's the **product**. That matches how Sierra actually buys and ships: voice agents are full systems, and nobody ships a raw realtime model.

Other rules: same agent and user models across all domains, all tasks (no `--num-tasks` filtering), 4+ trials, and results submitted by PR with full trajectories so maintainers can recompute everything (interaction metrics included) from raw data.

## What's interesting / non-obvious

1. **Re-scoring by replay.** Scoring never trusts the live run's state. It rebuilds from the transcript. That one choice makes re-grading, bug-fixing and cross-version comparison possible, and it's why the 1.0.1 fixes could be applied backward.
2. **Doing nothing can be optimal, and the benchmark rewards it.** Refusal tasks have target = initial state. An agent that's biased toward acting fails them. An agent that's biased toward inaction passes them without tool calls. The READ/WRITE diagnostic split exists because this is easy to misread.
3. **Being careful was penalized** until July 2026. A strict hash verifier punished extra verification reads, the opposite of what you want in a bank. Strict outcome checks need an explicit rule for which side effects count.
4. **Hidden tools with random suffixes** turn "did it retrieve the right document?" into a DB-state question. That's a neat way to avoid an LLM judge for retrieval quality.
5. **The oracle agent (`LLMGTAgent`)** is a built-in task-validity probe. Running it is a cheap check for whether a task is solvable at all.
6. **LLM judges are everywhere except the score.** Reviewer, auth classifier, hallucination checker, NL assertions: all diagnostics or re-run gates. The official number stays deterministic.

## Genuine limitations

- **Substring COMMUNICATE is brittle**, especially over voice (the retail task 4 issue file above: DB correct, review says correct, reward 0).
- **Outcome-only DB checks can't see process violations** that leave the state unchanged: skipping authentication, leaking another customer's data in chat, saying something off-policy. `nl_assertions` would catch some of these, but they're WIP and outside the official basis.
- **The user simulator is still an LLM** (gpt-4.1 at temperature 0 by default). The hallucination retry exists only in full-duplex mode. Text runs have no automated simulator-error filter, so text scores still mix agent failure with simulator failure.
- **Scores aren't stable across releases.** banking_knowledge changed by up to ~9 points from grading fixes alone. Any number quoted without a commit/version is ambiguous.
- **Voice parity depends on Sierra's private voices.** External runs are "functionally equivalent" but not identical, so official voice scores come from Sierra re-running them.
- **Small official test sets** (50 airline, 114 retail/telecom, 97 banking) with ~4 trials give wide confidence intervals. A few points of difference between models is often noise. See [[evaluation-specific-pitfalls-in-agent-simulation-environments]].

## How this connects

- Paper-level theory, pass^k derivation, τ² results: [[2406.12045 - Tau-Bench (Tool-Agent-User Interaction)]], [[pass-hat-k-reliability-metric]]
- Dual control and environment interfaces: [[multi-agent-environment-interface-design]], [[agents-environment-interface-types]]
- Outcome verifiers and their false positives: [[reward-hacking-as-eval-false-positive]], [[reward-hacking-as-specification-gaming]]
- Simulated tool execution as a proxy for intent: [[simulated-tool-execution-as-intent-proxy]]
- Customer-service agent deployment context: [[agentic-ai-usage-in-customer-service]]
- Benchmark noise: [[label-noise-dominates-saturated-benchmarks]]

## One-line takeaway

Sierra's evaluation philosophy in code: **grade the world, not the words.** Build a fake business, replay the transcript into it, hash the end state, and keep every LLM judge out of the official number. The repo's own changelog then shows the cost: a hash-based grader is exactly as correct as your choice of what goes into the hash.
