Created: 27-09-2026 23:30
Updated: 27-09-2026 23:30
Type: topic-overview
Sources: 6
Zettels: 23
Tags: [[reinforcement-learning-in-language-models]] [[agentic-ai-systems]] [[memory-management]] [[memory-mechanism]]

# Context Management in Long-Horizon Agent RL

The core question: when an agent's rollout is longer than the model can read at once, what should it forget, who decides, and how do you still train on the result?

## Why does the window run out at all?

A chat answer is a few thousand tokens. An agent run is every file it read, every command output, and every block of reasoning, stacked in one transcript. That transcript grows with every step, and the behaviors that make agents better are the ones that grow it fastest. Cursor reports that on its internal CursorBench, performance on hard coding tasks tracks more thinking and more codebase exploration; in their words, "agent trajectories are expanding faster than the context length of models" ([[summary-chained-rollouts-train-past-the-context-window]]). The numbers from other sources agree. On BrowseComp, more than 20% of DeepSeek-V3.2's trajectories pass its 128K window ([[discard-all-context-reset-matches-parallel-sampling]]). Surge's long-horizon training runs take 80K-100K tokens, and anything above 50K already forces heavy context-parallel sharding ([[offload-large-tool-outputs-and-compact-retry-loops]]).

Running out is not the only failure. Long before the hard limit, "context rot" sets in: stale reasoning and redundant tool output dilute attention, and performance drops on long-range reasoning even when the relevant tokens are present ([[context-management-as-a-learned-policy-not-a-scaffold]]). That is why a bigger window is not the whole fix, and why the agent-side symptoms (forgetting which branch it is on, redoing finished steps) show up in [[context-loss-in-long-horizon-agent-tasks]] well inside the limit. The hard ceiling itself is [[context-window-ceiling-limitation]].

So some form of forgetting is required. The rest of this page is about which form, and what it costs in training.

## What are the cheap rules, and how far do they get?

The first answers are fixed rules written into the harness (the code that runs the loop around the model). They cost nothing to design and no training.

**Keep reasoning only until the user speaks again.** DeepSeek-R1 deleted the previous turn's hidden reasoning whenever a new message arrived. In a tool loop, a new message arrives after every tool call, so the model had to rebuild its plan every step. DeepSeek-V3.2 moves the trigger: reasoning is dropped only on a new *user* message; tool results do not trigger it, and tool calls and results always stay ([[reasoning-retained-across-tool-calls-until-next-user-turn]]). The catch is that behavior now depends on which role the harness puts tool results in. Harnesses that send tool results as user messages (Roo Code, Terminus) trigger the drop every step. V3.2's Terminal Bench 2.0 score is 46.4 through Claude Code but 39.3 through Terminus in non-thinking mode. The same weights score differently depending on the harness.

**Keep big payloads out, and collapse failed retries.** Surge fixed two leaks in the harness: huge tool outputs (a whole spreadsheet, a full paper) go to files the agent can grep instead of into context, and once a malformed tool call is corrected, the retry sequence is collapsed and only the latest reasoning trace kept ([[offload-large-tool-outputs-and-compact-retry-loops]]). In RL this matters beyond efficiency: a rollout that dies from overflow gets low reward for a reason unrelated to the skill being trained, so harness waste turns into reward noise.

**Throw everything away.** DeepSeek compared three strategies triggered at 80% of 128K on BrowseComp: summarize and restart, drop the oldest 75% of tool history, or drop all tool history ([[discard-all-context-reset-matches-parallel-sampling]]). The crudest one won on efficiency. Discard-all reached 67.6, comparable to running many independent trajectories in parallel with far fewer steps; with no context management the score was 51.4. Summarization stretched trajectories to an average of 364 steps and topped out at 60.2. The reading: for search, the useful state is what the agent is looking for and what it has ruled out, which lives in its current reasoning; old raw tool outputs are mostly bulk. Whether more compute helps an agent at all is its own question ([[agent-test-time-scaling-limited-by-verification-gap]]).

These rules work where the environment itself holds the state. verl-agent keeps only 2 steps of history for ALFWorld and WebShop because the current observation is nearly sufficient ([[non-concatenating-multi-turn-rollouts-with-bounded-memory]]). They should fail where state is long and sequential, like a multi-hour coding task, and that is where the next question comes in.

## Who should write the compressed context: the harness or the model?

Three answers exist, and they differ in who owns the memory.

1. **The harness writes it, the model lives with it.** A summarizer prompt or a scheduled pruning pass decides what survives ([[memory-pruning-and-summarization]]). Its problem is that it was never told whether its choices helped the task finish. A prompted summarizer guesses what matters; a sliding window does not guess at all.
2. **Memory moves out of the model and into files.** Anthropic found model-written compaction "doesn't always pass perfectly clear instructions to the next agent," so its long-running coding harness carries state in a progress note, the git log, a feature list with pass/fail flags, and an `init.sh` ([[environment-held-handoff-state-for-multi-session-agents]]). This makes the environment legible rather than making the model remember. A related finding: when the source is one tool call away, agents ignore explicit memory tools entirely. In MemTrack, GPT-5 scored 0.601 without memory, 0.610 with Mem0 and 0.601 with Zep, all inside a run-to-run spread of 0.034-0.061 ([[explicit-memory-tools-go-unused-when-source-is-re-readable]]).
3. **The policy writes it, and RL trains it.** Prime Intellect states this as future work: give the model tools to cut its own context and learn the policy end to end ([[context-management-as-a-learned-policy-not-a-scaffold]]). Cursor's Composer demonstrates a version of it, covered below.

## Why can't you train without compaction and switch it on at deployment?

Because the model would then meet context shapes it never trained on: abrupt summaries, missing middle turns, rewritten history. MiniMax calls the resulting distribution shift "severe" and runs context management inside its Forge RL loop, treating "the context was compacted" as one possible transition in the environment ([[context-compaction-inside-rl-rollouts-avoids-train-inference-shift]]). The general rule: any transformation the harness applies to the context at inference is part of the environment dynamics and belongs in training. Forge handles both a white-box case (compaction modeled as an agent action) and a black-box case (the agent rewrites history however it likes, and Forge trains only on the LLM calls that crossed its gateway; see [[trajectory-capture-by-instrumenting-the-llm-api-boundary]]). This fits the view in [[harness-environment-step-is-a-conversational-turn]] that the real harness should run inside training. Note that MiniMax shows no ablation of in-training vs deploy-only compaction; the size of the shift is argued, not measured.

## What does compaction break in the trainer?

Most RL pipelines assume, without saying so, that an episode is one growing token sequence: every prompt the model saw at turn $t$ is a prefix of the final sequence the trainer scores. Policy gradient relies on this, because it recomputes each token's probability under the context the token was sampled in, and compares it to the sampling-time probability through the ratio $\pi_\theta(y_t \mid c_t) / \pi_{\theta_\text{old}}(y_t \mid c_t)$.

Compaction breaks the assumption. At turn 5 the model saw a summary instead of turns 1-3, but the concatenated final sequence contains either the full history (which the model never saw at turn 5) or nothing matching the summary-shaped prefix. NeMo Gym calls this **non-monotonic history** and says plainly that handling it in training "remains an open research question" ([[context-compaction-makes-agent-trajectories-off-policy]]). Chat templates that strip earlier reasoning cause the same problem, which is why Gym's practical advice is to turn that stripping off. Gym's usual repair for text-round-trip drift, splicing the original tokens back in, assumes a strict prefix and so cannot help here ([[retokenization-and-retemplating-drift-in-http-rl-rollouts]]).

The workable answer is to stop treating the episode as one sequence. Each segment, or each LLM call, becomes its own training sample scored under its own actual prompt. That creates a new cost: many samples now share long, identical beginnings. MiniMax's fix is prefix-tree packing, which stores the shared context once, uses an attention mask so each token sees only its own ancestors (equal to ordinary causal attention along each path), and claims a 40x training speedup on its agent data mix ([[prefix-tree-packing-for-shared-context-training]]).

On the inference side, the summary itself can be cheap. Cursor asks for it as a new user turn appended to the existing conversation, so the policy already has the whole history in its KV cache (the stored per-token computations that let later tokens skip re-reading earlier ones). A separate summarizer call would pay full prefill on the 80K-token history just to produce the summary ([[summary-as-appended-turn-reuses-the-kv-cache]]). The cache benefit ends at the summary boundary: the next segment starts from a different prefix.

## How does a summary learn what to keep?

Composer's design: when generation hits a fixed trigger (80K or 40K tokens in their experiments), the harness inserts a user turn asking the model to summarize. The next segment starts from that summary plus harness-kept state (plan, remaining tasks, a count of past summaries). At the end of the chain the task is graded once, and that single reward is applied to every token in the chain, summary tokens included ([[self-summarization-trained-by-shared-terminal-reward]]). The gradient has the ordinary shape

$$\nabla_\theta J = \mathbb{E}\left[\sum_{k=1}^{K}\sum_{t\in g_k} A\,\nabla_\theta \log \pi_\theta(a_t \mid c^{(k)}_t)\right]$$

where the only change from single-segment RL is that $c^{(k)}_t$, the context for segment $k$, is the previous summary plus new tokens rather than the raw history. The mechanism is blunt. A summary that dropped a critical fact tends to sit in failing chains, so over many rollouts it becomes less likely. No one writes a rule for what a good summary contains.

Two things follow. First, only each segment has to fit in the window, so the trainable task length is no longer capped by context length ([[summary-chained-rollouts-train-past-the-context-window]]). An early Composer checkpoint solved Terminal-Bench 2.0's make-doom-for-mips in 170 turns, compressing more than 100,000 tokens to about 1,000 at summary points; the attempt count is not stated. Second, the model trains on context it wrote itself, so there is no deploy-time shift to worry about.

Against a tuned prompted compaction baseline, Cursor reports that self-summarization "consistently reduces the error from compaction by 50%" at one-fifth the summary tokens. "Error from compaction" most plausibly means the score drop relative to a no-compaction reference, so this is a claim about the gap, not absolute accuracy, on an internal benchmark with no variance reported.

## What does a trained summary look like, and what can go wrong with it?

It is short and it points. The prompted baseline runs thousands of tokens with nearly a dozen sections and produces summaries over 5,000 tokens; the trained model gets roughly "Please summarize the conversation" and writes about 1,000 ([[outcome-trained-summaries-keep-pointers-not-checklists]]). In a coding environment, files and test output can be re-read with one tool call, so the summary keeps where to look and what was learned, not copies. The published make-doom-for-mips summary shows this: the non-obvious fact that the VM uses its own syscall numbers (0=read, 1=write) instead of MIPS Linux's 4003+, one-line cause-and-fix pairs so the next segment does not retry dead ends, a warning about its own hack, a live hypothesis about the crash at `0x400151`, and exact file paths. This is the same pattern as MemTrack's result: when the source is cheap to re-read, good memory is an index.

The risk is that nothing checks the summary against the history it replaced ([[outcome-rewarded-summaries-are-not-checked-for-truth]]). Reward selects for summaries that are useful, which in coding mostly means true, because a wrong fact usually fails a test. The same published summary contains a slip: it says "PIC was required" where the fix was `-fno-pic` and the next sentence says non-PIC fixed it. Once the summary replaces the history, a wrong fact is worse than a gap, because a gap prompts re-reading and a confident error does not. Keeping file paths, keeping structured state in the harness, or storing state in files as in [[environment-held-handoff-state-for-multi-session-agents]] are ways to keep claims checkable.

Credit is the other weak point. A summary decides what the entire next segment can see, yet gets the same advantage as any other token, with no critic and no separate summary reward. The cost of dropping the wrong thing shows up many turns later, the [[credit-assignment-problem]] at its longest. Cursor shows no ablation of why this works.

For how this fits the wider picture of multi-turn training, see [[agentic-and-multi-turn-rl]]; for the systems side (packing, rollout capture, off-policy correction), see [[rl-training-infrastructure]]; for where the harness ends and the environment begins, see [[rl-environment-architecture]]; for the agent failures that motivate all of this, see [[where-llm-agents-fail]].

## Open Questions

- How should a trainer handle non-monotonic history in general? NeMo Gym calls it open. Per-segment samples plus prefix-tree packing is one practical answer, but it gives up any credit link between a segment and the summary that shaped it beyond the shared terminal reward.
- When to compact is still a fixed trigger (80K, 40K, or 80% of 128K). Would a model that chooses its own compaction points do better, and can RL learn that decision with a single end-of-task reward?
- Does the "summary should be pointers" result survive environments where the source is expensive, rate-limited or gone after one read? The MemTrack and Composer evidence both come from settings where re-reading is cheap.
- Should summaries get a faithfulness term in the reward, and would it cost task success? No source measures this.
- How large is the train/inference shift that Forge describes? No ablation exists in the sources.
- Would discard-all still beat summarization for search if the summary were trained, as in Composer, rather than prompted?

## Contradictions

- **Summarize vs discard.** DeepSeek finds the crudest reset (discard-all, 67.6) beats summarization (60.2) on BrowseComp. Cursor finds a trained summary halves compaction error on coding. These are not directly opposed: the tasks differ (search state lives in current reasoning; coding state is long and sequential), and DeepSeek's summary was not trained on outcome.
- **Train the summary vs don't trust summaries.** Anthropic concluded model-written compaction is too lossy and moved memory into files. Cursor concluded the fix is to train the summary with reward, while still keeping some state (plan, task list) in the harness. Both move structured state out of prose; they disagree on whether prose memory can be made reliable.
- **Harness-owned vs policy-owned.** Forge keeps the harness's compaction rule and trains the model to survive it; Composer and Prime Intellect make the model write or control it. Forge's black-box mode also accepts any harness, including ones whose compaction makes trajectories off-policy in exactly the way NeMo Gym warns about.
- **Retain vs drop reasoning.** DeepSeek-V3.2 keeps reasoning across tool calls to avoid re-deriving plans, and then reports redundant self-verification with trajectories past 128K on MCP-Mark tasks. Surge keeps only the latest reasoning trace. Neither rule is learned.

## Sources

1. [[Cursor - Training Composer for longer horizons]]
2. [[2603.24477 - Composer 2 Technical Report]]
3. [[MiniMax - Forge Scalable Agent RL Framework and Algorithm]]
4. [[NVIDIA-NeMo-Gym - Environments for Training and Evaluating Agents]]
5. [[2512.02556 - DeepSeek-V3.2]]
6. [[Surge AI - Training on Long-Horizon Agent Tasks]]
