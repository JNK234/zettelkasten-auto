Created: 24-09-2026 16:15

Tags:

Status: #baby

## References

1. [RL Environments: Tutorial & Examples (Patronus AI Guide, Chapter 1)](https://www.patronus.ai/guide-to-rl-environments)

# Notes

## What this is, and how to read it

This is Chapter 1 of an 8-chapter Patronus AI guide series (the others cover LLM post-training, RL applications, RL training, online RL, RL evaluation, model-based RL, and RLVR). It is a vendor explainer: two-thirds basics plus a toy code example, one-third product pitch for Patronus RL Environments. Read it for two things: (1) the way Patronus frames the "environment" as an evaluation pipeline with a verifier attached, which tells you a lot about their business; (2) the concrete toy example, which is useful mainly because it has real bugs that show how easy it is to get reward code wrong.

Source quality is vendor-blog level. The basics are correct. One historical claim is muddled (see "The DeepSeek moment" below) and the code does not do what the prose says it does (see "The reward function is broken").

## The problem it claims to solve

The argument runs in three steps.

1. **SFT copies answers.** Supervised fine-tuning trains on labeled examples. It works when there is one right answer (spam detection, sentiment). It struggles on open-ended or multi-step tasks, where there is no single target sequence to copy.
2. **RLHF scales badly.** RLHF optimizes against human preference judgments, which fixed a lot of chatbot behavior (helpful, honest, safe). But humans are slow, expensive, and inconsistent, and the policy can only get as good as the raters. For long-horizon agent tasks, a human in the loop on every rollout is the bottleneck.
3. **Agents fail in ways a language modeling loss can't see.** An agent that calls the wrong tool, loses track of the task midway, or misuses an API gets no direct penalty from next-token prediction. Supervised data rarely covers the failures that only show up when the agent runs on its own.

Their answer: replace the human rater with a **programmatic check** that runs inside a simulated world. That world plus the check is the RL environment. This is the same bet as [[2411.15124 - Tulu 3 (RLVR Post-Training)]] and [[2501.12948 - DeepSeek-R1 (RL for Reasoning)]]. If you can write code that checks success, you can generate unlimited training signal.

## "The DeepSeek moment" (read with care)

The article says DeepSeek "trained an LLM-based research agent using a fully automated pipeline with no human-labeled data," using "a copy of Wikipedia as a sandbox" and "internal metrics to reward good outputs." That doesn't match what [[2501.12948 - DeepSeek-R1 (RL for Reasoning)]] actually did. R1-Zero used GRPO with **rule-based rewards** (answer correctness on math, test execution on code, plus a format reward) and no Wikipedia sandbox. See [[DeepSeek R1 - Paper Notes]] and [[2503.20783 - Understanding R1-Zero-Like Training]]. The "Wikipedia sandbox for a research agent" description sounds like later search-agent RL work being folded into the DeepSeek story. The broad point still holds: R1 showed that cheap, automatic, verifiable rewards could drive large capability gains, and labs then poured money into environments. Just don't cite this article for the details.

## When to use an RL environment (and when not)

The article gives a decision rule, which is the most useful practical part.

- **Don't** use one for one-shot classification-like tasks with good labeled data. SFT is cheaper and enough.
- **Do** use one when both of these hold:
  1. The task is a **sequence of decisions** that is too variable to hard-code, so the agent has to explore.
  2. Success can be **checked by code**. Their example is text-to-SQL: run the query and compare the result set to the expected answer. It either runs and matches, or it doesn't.

The second condition is the real gate. The environment is only as good as its verifier. This is the whole theme of [[HUD - Verifier and Reward Design for RL Environments]] and [[four-layer-scoring-stack-for-rl-environments]].

The same environment works for **evaluation** as well as training: run a trained agent through edge-case scenarios and score it. That dual use is Patronus's core pitch (see the company section below).

## The MDP, mapped onto LLM agents

The article describes an environment as a Markov Decision Process. Formally, an MDP is a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- $\mathcal{S}$ is the state space: what the agent can observe.
- $\mathcal{A}$ is the action space: what it can do.
- $P(s' \mid s, a)$ is the transition dynamics: the probability of ending up in state $s'$ after taking action $a$ in state $s$.
- $R(s, a)$ is the reward: a scalar score for that step.
- $\gamma \in [0,1]$ is the discount factor: how much future reward counts compared to immediate reward. The article skips it, since LLM episodes are short and usually scored at the end, so $\gamma \approx 1$ in practice.

The agent's goal is a policy $\pi(a \mid s)$ that maximizes expected return $J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t)\right]$, where $\tau = (s_0, a_0, s_1, a_1, \dots)$ is one episode (trajectory) and $T$ is the step at which the termination condition fires.

Here is how each part maps to LLM work, following the article:

| Component | Classic RL | LLM-agent version | Design tension the article points out |
|---|---|---|---|
| State | pixels, joint angles | conversation history; task spec plus code so far | Give **enough** context, not everything. A support bot needs the query and account details, not the whole DB schema |
| Action | move left/right, torque | token sequences, or discrete tool calls with arguments | Realism vs. tractability. Free-form SQL text is a huge space. Structured JSON actions (table, columns) shrink it and make checking easy |
| Reward | game score | unit tests passed; exact-match answer; valid JSON parse | "Use verifiable signals whenever possible." |
| Transition | physics | simulated user reply, tool output, DB state change | The environment must handle bad actions (nonsense SQL, repeated questions) by returning an "invalid action" state plus a penalty, not by crashing |
| Termination | goal reached / fell over | answer submitted, max steps hit, user says goodbye | Episodes give clean resets and a clean credit-assignment boundary |

Two design points worth keeping:

- **Constraining the action space is reward engineering in disguise.** Forcing JSON-structured SQL actions doesn't just speed up exploration. It also removes whole classes of ungradeable outputs, so the verifier becomes simpler and more trustworthy.
- **Robust transitions are part of the reward function.** If a malformed action crashes the environment, you get either a missing rollout or an accidental zero. That ties to [[mask-infrastructure-failures-never-reward-them-zero]] and [[sandbox-fidelity-is-part-of-the-reward-function]]. The article only goes as far as "don't crash, return invalid-action plus penalty." It doesn't separate *agent* errors (which should be penalized) from *infrastructure* errors (which should be masked).

A small mistake in the text: it says that to check "valid JSON" you should "parse the output directly using an LLM or any automatically verifiable heuristic." Using an LLM to parse JSON isn't deterministic verification. `json.loads` is. The next sentence ("deterministic feedback makes environments robust") argues against the LLM option.

## The "learning mechanism": a sixth component, and it isn't RL in the usual sense

The article adds a sixth component to the MDP: the **learning mechanism**, meaning how the agent actually improves from reward. Classic RL updates weights with policy gradients, PPO, or Q-learning (see [[PPO vs GRPO - Policy Optimization for Alignment]]). The article then claims that for LLM environments "full gradient-based training is often too costly or impractical," and lists three weight-free alternatives:

1. **Reward-conditioned prompt rewriting.** After each episode, an LLM rewrites the system prompt using the reward score.
2. **Self-reflection loops.** The model critiques its own output and rewrites its instructions (Reflexion-style).
3. **Evolutionary prompt search.** Generate prompt variants, score them with the reward function, keep the best.

This is the key reframe for understanding Patronus's view, and it's also the most debatable claim in the article. What they describe is **prompt optimization with a reward-based fitness function**, not reinforcement learning of a policy. The policy's parameters $\theta$ never change. Only the conditioning context changes. In MDP terms you are searching over a discrete, context-level "policy parameter" (the prompt) with a black-box optimizer, using the environment's return as the objective. That approach is legitimate and cheap, and it connects to [[llm-feedback-as-gradient-substitute]]: an LLM's text critique plays the role a gradient would. But it has a much lower ceiling than weight updates, and it's a different claim from the "DeepSeek moment" story used to motivate the article, which was all about gradient RL on weights. The frontier labs buying environments are running GRPO/PPO against them, not rewriting prompts.

Option 3 is basically evolutionary search with an LLM mutation operator and the environment's return as fitness. That's close in spirit to LLM-driven evolutionary methods like LEAR.

## The worked example: banking chatbot

**Setup.** A LangGraph ReAct agent (GPT-4o via `langchain_openai`) with four stub tools: `login_user`, `verify_2fa`, `get_balance`, `logout_user`. The stubs return fixed strings ("Your account balance is $10,000."). The intended behavior is: log in, then verify 2FA and fetch the balance, then log out **only when the user asks**. Logging out right after showing the balance is supposed to cost a penalty.

**Components in code.**

- *State*: `AgentState` = list of messages plus a `remaining_steps` counter (only there for LangGraph's recursion limit).
- *Actions*: the four tool names.
- *Transition*: `transition(state, msg)` appends the message to history. That's all. The "world" has no state beyond the transcript. The tools don't change anything (no logged-in flag, no session), so calling `get_balance` before `login_user` succeeds anyway.
- *Termination*: a keyword match on the last message ("thank you", "bye", "goodbye", "logged out", "session ended"). The article says that in practice you'd use an LLM to judge termination.
- *Reward*: `compute_reward(called_tools)`. The intended stepped reward is 1 after login, 3 after verify plus balance, 4 after a proper logout, and −1 for an immediate logout.
- *Learning*: `update_prompt(reward, prompt, actions)` asks GPT-4o to rewrite the system prompt, with stronger changes when the reward is lower.

**Training loop.** The outer loop runs one episode per iteration: build the agent from the current system prompt, then an inner loop alternates the agent's turn and a human typing into `input()`. After each agent turn it collects all tool calls, dedups them with `dict.fromkeys` (keeping first-occurrence order), and recomputes the reward on the running list. If the reward is −1, the loop recomputes it without logout and subtracts 1 (so 3 → 2). At episode end, if reward < 4, the prompt gets rewritten.

Note that **the user in this environment is a live human typing at a Colab prompt.** So despite the "no human-in-the-loop" pitch, the example has no simulated user. A real environment would need a user simulator, like the LLM user in [[2406.12045 - Tau-Bench (Tool-Agent-User Interaction)]]. Tau-bench is the rigorous version of exactly this toy: tool-agent-user interaction in customer-service domains, graded by final database state.

## The reward function is broken (and that's the lesson)

The docstring says the reward checks the **order** of calls and penalizes an immediate logout. The code does neither. I ran the function on all 24 orderings of the four tools. **Every ordering scores 4**, including `logout_user, get_balance, verify_2fa, login_user`.

Why:

1. **Order is never checked** for login/verify/balance. Each check is `any("login" in t for t in called)`, which is a set-membership test, so position doesn't matter.
2. **The −1 penalty can't be reached.** It needs `called.index("logout_user") > called.index("get_balance")` **and** `len(called) - called.index("get_balance") == 1`. The second condition means `get_balance` is the **last** element. But then nothing can come after it, so logout's index can't be larger. The two conditions contradict each other, so the penalty branch is dead code. The `-1` handling in the training loop is dead too.
3. **Dedup erases repeat behavior.** `dict.fromkeys` collapses repeated calls, so an agent that calls `login_user` five times in a loop looks the same as one that calls it once.
4. **The scale doesn't match.** The reward runs 0 to 4, but `update_prompt` tells the rewriter "you receive a reward score out of 3" and "achieve the maximum reward of 3/3", with rules for "high (3)". So a policy scoring 3/4 is told it's already at max and to "make minimal or no changes."
5. **Substring matching.** `"login" in t` would also match a hypothetical `relogin_user` or `login_failed_handler`. That's harmless with four tools, but it's a real risk in bigger tool sets.

The article's reported run (reward 2 after a premature-logout penalty) can't come from the published `compute_reward`. Either the notebook differs from the article, or the output was edited.

Why this matters beyond nitpicking: it's a clean small example of **the verifier silently disagreeing with its spec**. A policy trained against this reward learns "call all four tools in any order, as early as possible," which is exactly the behavior the designer wanted to penalize. The initial system prompt even says "Call all the tools for one user query," and the reward never pushes back. This is reward hacking with no adversary involved, just a bad check. See [[goodhart-at-the-contribution-layer]], [[METR Reward Hacking and Eval Integrity]], [[2605.12474 - Reward Hacking in Rubric-Based Reinforcement Learning]]. The fix is to **unit-test the reward function** against hand-built good and bad trajectories before any training. Here that would be the 24 permutations plus "logout in the same turn as balance," with expected scores. That habit is cheap and catches exactly this. It's also why [[reward-computation-as-a-heavyweight-pipeline-stage]] treats reward code as production code, not a helper function.

What a correct version would check: positions (`idx(login) < idx(verify) < idx(balance)`), *turn* boundaries (logout must come in a later agent turn than balance, which means tracking calls per turn instead of in one flat deduped list), and preferably **state** instead of call names. For example, the tools could set `session.logged_in`, `session.verified`, and a balance-shown flag, and the reward could check the end-of-episode state the way tau-bench checks final DB state. Rewarding outcome/state is harder to game than rewarding a list of tool names.

## Patronus's approach (company section)

This part matters if you're trying to understand how Patronus positions itself.

**Core identity.** Patronus calls itself a platform for **agent evaluation, debugging, and reinforcement-based improvement**. The tech stack they list is fine-grained tracing, root-cause analysis, and scoring across multi-step workflows. The key sentence: these capabilities, "combined with automated verifiable feedback, allow Patronus to effectively turn evaluation pipelines into practical RL environments." Put plainly, **Patronus is an eval company that noticed a good eval harness plus a verifier is already an RL environment.** Their product sequence is evals (Core Platform), then agent debugging (Percival, their trace-analysis agent), then RL Envs. The site nav also shows a research arm and a benchmark release (SpeedrunBench).

**The three claimed differentiators.**

1. **Real-world task alignment.** Environments built around real domains (coding, SQL, customer support, finance Q&A, trading) instead of synthetic benchmarks, on the theory that gains transfer to production systems. This is a bet on domain realism over scale. Compare [[Prime Intellect - Environments Hub]], which bets on an open, community-contributed volume of environments.
2. **Automated verifiable rewards.** Deterministic verifiers: unit tests, SQL result validation, **database state-change checks**, schema constraints. Pitched as reproducible, scalable, auditable, with lower variance and cost than human judgment. The "database state changes" item suggests their real environments do outcome-state checking, which is the right answer to the toy example's bugs above.
3. **Generalization via "reward scaffolding across tasks."** They claim that consistent evaluation logic across domains (formatting, operation ordering, error handling) lets learned behaviors transfer and avoids "overfitting to a prompt." This is asserted with no evidence: no experiments, numbers, or definition of "reward scaffolding." Treat it as a hypothesis, not a result.

**Supported environment types.** Coding agents (generate and correct program outputs), SQL/DB Q&A, customer service, finance Q&A over financial data, trading agents (markets or simulated UI), and web-navigation / user-journey agents for review and e-commerce tasks. Each comes with structured tasks and built-in checks for **correctness plus procedural order, error conditions, tool usage, and consistency**. That's a multi-criterion verifier, which is the more mature version of the one-dimensional tool-list reward in the tutorial. It lines up with the layered scoring idea in [[four-layer-scoring-stack-for-rl-environments]].

**Reading between the lines.**

- Their competitive edge is **the verifier and the domain task library**, not the RL algorithm. The article barely touches training algorithms and falls back on prompt optimization in its own demo. That fits a company whose customers may range from enterprises tuning prompts to labs running GRPO. The environment is the product, and the learning mechanism is left to the customer.
- Their heritage in LLM-as-judge and evaluation shows in the "use an LLM to parse / judge termination" suggestions. Their real edge probably includes calibrated judge models for the parts that can't be verified deterministically, though this article doesn't say so.
- They lean on evaluation-plus-training dual use: the same environment scores a production agent and trains the next one. That's the sales pitch, and it's technically sound. The main risk is that an environment used for both training and eval stops being a clean eval once you optimize against it.

## What's genuinely useful to keep

- The **decision rule**: use environments when the task is sequential *and* success is checkable by code. Otherwise SFT.
- **Constrain the action space** so the verifier stays simple.
- **The environment must absorb bad actions** gracefully (invalid-action state plus penalty), never crash.
- **Unit-test the reward function.** The article's own code is the cautionary example: a stepped, order-aware reward that is actually an order-blind 4-for-everything reward, with an unreachable penalty.
- **Reward state, not call lists**, wherever the environment can hold real state.

## Genuine limitations of the source

- It's a marketing funnel. The generalization claim has no evidence, and there are no benchmarks or numbers.
- The history in "The DeepSeek moment" is garbled.
- It motivates with gradient RL on weights but demos prompt rewriting, and never says these are different regimes with different ceilings.
- The toy environment has no simulated user (a human types), no world state (stub tools), and a reward function whose key logic is dead code.
- No mention of reward hacking, verifier false positives/negatives, or infra-error masking, which are the hard parts of building environments in practice.
