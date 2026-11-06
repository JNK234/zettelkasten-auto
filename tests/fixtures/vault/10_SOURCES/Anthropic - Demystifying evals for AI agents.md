Created: 27-09-2026 11:53

Tags:

Status: #baby

## References

1. [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

# Notes

Anthropic Engineering post, published 9 January 2026, written by Mikaela Grace, Jeremy Hadfield, Rodrigo Olivares and Jiri De Jonghe. It is a practitioner's field guide, not a paper: there are no experiments of its own, but it packs in a lot of hard-won numbers and war stories from Anthropic's internal evals and from customers (Descript, Bolt, Qodo, METR, the Terminal-Bench team). Depth chosen: comprehensive. The article is long (about 5,000 words) and dense with operational advice and worked failure stories, though its only math is the pass@k / pass^k definitions.

## The problem: agents are hard to test, and teams without tests go blind

A chatbot eval is simple. You send a prompt, you get a response, and some grading logic decides whether the response is good. An agent is different in three ways that break that simple picture.

First, an agent acts over many turns. It calls tools, reads what comes back, and changes the state of the world (files, databases, a browser). A small mistake early can compound into a large failure late, so the thing you grade is no longer one output but a whole run.

Second, agents are non-deterministic. The same task can pass on one run and fail on the next, so a single run tells you little.

Third, strong models find solutions the eval designer did not anticipate. The post's opening example: Opus 4.5 solved a τ2-bench flight-booking task by finding a loophole in the airline policy. By the benchmark's rules it "failed", yet it produced a better outcome for the user. A static eval can therefore punish the very capability you want.

The practical consequence the post keeps returning to is what happens to teams that skip evals. Early on, manual testing and dogfooding (using your own product internally) carry you surprisingly far. Then the agent scales, users report that it "feels worse" after a change, and the team has no way to tell a real regression from noise. Debugging becomes reactive: wait for complaints, reproduce by hand, patch, and hope nothing else broke. The authors' framing is that eval costs are visible upfront while the benefits compound later, which is why teams under-invest.

## The vocabulary: what an agent eval is made of

The post defines a set of terms that are worth keeping precise, because most confusion in agent evaluation comes from mixing them up.

A **task** is one test case with defined inputs and success criteria. A **trial** is one attempt at a task; because outputs vary, you run several trials per task. A **grader** is a piece of logic that scores one aspect of performance, and a task can have several graders, each containing several **assertions** (individual checks).

The most useful distinction is between the **transcript** and the **outcome**. The transcript (also called trace or trajectory) is the full record of a trial: every model output, tool call, reasoning step and intermediate result. For the Anthropic API it is literally the final messages array. The outcome is the final state of the environment. The post's example makes the gap concrete: a flight agent can say "Your flight has been booked" in the transcript, but the outcome is whether a reservation row actually exists in the SQL database. Grading the transcript tells you what the agent claimed; grading the outcome tells you what happened. This is the same principle as the verifier layer in [[four-layer-scoring-stack-for-rl-environments]], where success is checked against final environment state, never against the agent's description of its own work.

Two kinds of "harness" are also separated. The **evaluation harness** is the infrastructure that runs evals: it supplies instructions and tools, runs tasks in parallel, records everything, grades and aggregates. The **agent harness** (or scaffold) is the system that turns a model into an agent by orchestrating its tool calls. The non-obvious point is that when you evaluate "an agent" you are evaluating model plus scaffold together. A score change can come from either, and as the CORE-Bench story below shows, a constraining scaffold can hide most of a model's ability.

Finally, an **evaluation suite** is a collection of tasks sharing a broad goal, such as a customer-support suite covering refunds, cancellations and escalations.

## Three families of graders and what each is good for

Every grader looks at some part of the transcript or the outcome. The post sorts them into three families.

**Code-based graders** are deterministic programs: string or regex matches, unit tests (fail-to-pass, meaning a test that failed before the fix must pass after it, and pass-to-pass, meaning tests that already passed must still pass), static analysis such as linters and type checkers, direct checks of environment state, checks of which tools were called with which parameters, and transcript statistics like turn count and token usage. They are fast, cheap, reproducible and easy to debug. Their weakness is brittleness: a valid answer phrased differently from the expected pattern fails.

**Model-based graders** use an LLM as the judge. Methods include rubric scoring, natural-language assertions ("the agent showed empathy"), pairwise comparison between two outputs, comparison against a reference answer, and consensus across several judges. They handle nuance and open-ended output, but they are non-deterministic, cost more, and above all "require calibration with human graders for accuracy". The vault has a lot on why this calibration is subtle: agreement should be measured against the human-human agreement ceiling rather than 100% ([[judge-agreement-against-human-ceiling]]), judges have position bias ([[llm-judge-position-bias]]), and a judge can match humans on overall scores while failing to locate the actual error ([[holistic-judge-scores-mask-localization-failure]]). See also [[2306.05685 - Judging LLM-as-a-Judge (MT-Bench)]], [[2501.10970 - The Alternative Annotator Test for LLM-as-a-Judge]] and [[2511.21140 - How to Correctly Report LLM-as-a-Judge Evaluations]].

**Human graders** include subject-matter-expert review, crowdsourcing, spot-check sampling, A/B tests and inter-annotator agreement studies. They are the gold standard and the thing you calibrate model graders against, but they are slow and expensive.

Per task, grader scores can be combined in three ways: weighted (the combined score must pass a threshold), binary (every grader must pass), or a hybrid. This choice matters more than it looks. A binary AND over five graders is very strict and will make tasks look harder than they are if any one grader is flaky; a weighted sum lets a strong result on one axis hide a failure on another.

## Capability evals versus regression evals

The post draws a distinction about what an eval is for, and it changes how you read the score.

A **capability eval** asks "what can this agent do well?" It should start at a low pass rate. Its job is to give the team a hill to climb, so if it starts near 100% it is useless.

A **regression eval** asks "does the agent still do everything it used to?" It should sit near 100%, and any drop signals breakage.

The two are linked by a lifecycle: once a capability eval is saturated and the feature is shipped, its tasks "graduate" into the regression suite, and the question they answer shifts from "can we do this at all?" to "can we still do this reliably?". This is a clean way to reuse eval investment instead of discarding saturated suites.

## How the advice changes by agent type

The post walks through four agent types. The common thread is that the more verifiable the outcome, the more you lean on code; the more subjective, the more you lean on calibrated LLM judges.

**Coding agents** are the easy case because software has a natural verifier: does it run, do the tests pass. SWE-bench Verified hands an agent a real GitHub issue and passes it only if the failing tests now pass and no existing tests break. Terminal-Bench tests end-to-end technical jobs such as building a Linux kernel or training a model. The post adds that once outcome tests exist, it is worth also grading the transcript, for code quality or for how the agent interacts with the user. Its illustrative YAML for an auth-bypass fix stacks five graders (required unit tests, an LLM code-quality rubric, `ruff`/`mypy`/`bandit` static analysis, a state check that a `security_logs` event of type `auth_blocked` exists, and a required tool-call sequence) plus tracked metrics for turns, tokens and latency. The authors are explicit that this is a showcase: in practice coding evals "typically rely on unit tests for correctness verification and an LLM rubric for assessing overall code quality."

**Conversational agents** (support, sales, coaching) are harder because the quality of the conversation itself is part of what you are grading, and because you need someone to talk to. The standard answer is a second LLM simulating the user, the design used by [[2406.12045 - Tau-Bench (Tool-Agent-User Interaction)]] and τ²-bench ([[sierra-research-tau2-bench - Agent Evaluation Harness]]); Anthropic also uses this for its alignment auditing agents. Success is multidimensional: was the ticket resolved (state check), was it done in fewer than 10 turns (transcript constraint), was the tone right (LLM rubric)? The simulator is itself a source of error that needs auditing ([[llm-user-simulator-design-and-auditing]], [[dual-control-environments]]), a point this post does not dwell on.

**Research agents** have no unit tests. Whether a report is "comprehensive" or even "correct" depends on context: a market scan and an acquisition due-diligence memo have different standards. Experts disagree, ground truth shifts as the web changes, and long outputs leave more room for error. The suggested decomposition is three checks: groundedness (is each claim supported by a retrieved source?), coverage (does the answer contain the key facts a good answer must have?), and source quality (are the sources authoritative, not just the first hit?). Exact match still works for questions with one right answer. Because the grading is subjective, the LLM rubrics need frequent recalibration against expert humans.

**Computer-use agents** act through screenshots, clicks and keystrokes. They must be evaluated in a real or sandboxed environment by checking the end state. WebArena checks URLs and page state, plus backend state for tasks that modify data, so it confirms that an order was actually placed rather than that a confirmation page appeared. OSWorld inspects file system state, app configs, database contents and UI properties. A non-obvious product detail: browser agents face a trade-off between DOM extraction (fast, token-heavy) and screenshots (slower, token-light). Summarizing Wikipedia favors DOM; shopping on Amazon favors screenshots. Claude for Chrome has evals that check whether the agent picks the right mode for the context, which is an example of evaluating a decision, not just an end result.

## Non-determinism: pass@k and pass^k

Because each task has its own per-trial success probability, call it $p$, one run is a coin flip. The post introduces two metrics that summarize $k$ independent trials of a task in opposite directions.

**pass@k** is the probability of at least one success in $k$ trials. If trials are independent with success probability $p$, then

$$\text{pass@}k = 1 - (1-p)^k$$

Here $(1-p)$ is the chance one trial fails, $(1-p)^k$ is the chance all $k$ fail, and one minus that is the chance at least one succeeds. It rises toward 1 as $k$ grows. It fits settings where you can generate many candidates and keep one that works, for example when a test suite can pick the winner. For coding the post says the usual target is pass@1, getting it right first time.

**pass^k** (pass-hat-k, from τ-bench) is the probability that all $k$ trials succeed:

$$\text{pass}^k = p^k$$

It falls as $k$ grows. It fits customer-facing agents, where every user interaction must work. The post's worked number: $p = 0.75$, $k = 3$ gives $0.75^3 \approx 42\%$.

At $k = 1$ both equal $p$. As $k$ grows they separate. Some values computed from the formulas above:

- $p = 0.5$: pass@3 = 0.875, pass^3 = 0.125; pass@10 ≈ 1.0, pass^10 ≈ 0.001.
- $p = 0.75$: pass@3 ≈ 0.984, pass^3 ≈ 0.422; pass^10 ≈ 0.056.
- $p = 0.9$: pass^10 ≈ 0.349.
- $p = 0.99$: pass^10 ≈ 0.904.

This makes one of the post's claims more precise. It says that "by k=10" pass@k approaches 100% while pass^k "falls to 0%". That holds for moderate $p$, but for a reliable agent ($p = 0.99$) pass^10 is still about 90%. The real lesson is that pass^k is extremely sensitive to small gaps from perfect reliability, which is exactly why it is the right metric for customer-facing agents: going from 90% to 99% per-trial success moves pass^10 from 35% to 90%.

Two practical notes the post leaves implicit. The formulas above assume trials are independent, which is why Step 4 below (isolating trials) matters: correlated infrastructure failures break the math. And in practice $p$ is estimated from $n$ trials, so small $n$ gives noisy estimates; [[2411.00640 - Adding Error Bars to Evals]] covers how to put error bars on these numbers.

## The roadmap: from zero evals to evals you trust

The core of the post is an eight-step roadmap. Each step is ordinary advice on its face, but each comes with a specific failure story that shows why it matters.

**Step 0: start early, and small.** Teams delay because they think they need hundreds of tasks. The post says 20 to 50 tasks drawn from real failures is enough at the start. The reasoning is statistical: early in development each change has a large effect, and large effects are detectable with small samples. Only mature agents, where improvements are small, need big suites. Waiting also has a cost: early, product requirements translate directly into test cases; later, you are reverse-engineering success criteria from a live system.

**Step 1: start from what you already check by hand.** The manual checks before each release, the bug tracker, and the support queue are ready-made task sources, and ranking them by user impact tells you where to invest.

**Step 2: write unambiguous tasks with reference solutions.** The test for a good task is that two domain experts would independently reach the same pass/fail verdict. Ambiguity in the spec shows up as noise in the metric. The Terminal-Bench audit example is concrete: a task asked for a script without specifying a path, but the tests assumed one, so an agent could fail "through no fault of its own." The most striking rule here: with frontier models, "a 0% pass rate across many trials (i.e. 0% pass@100) is most often a signal of a broken task, not an incapable agent." The fix is a reference solution for every task, a known output that passes all graders, which proves both that the task is solvable and that the graders are wired correctly. This is the same guard [[generated-task-validity-vs-difficulty]] argues generated environments lack.

**Step 3: build balanced problem sets.** Test both when a behavior should happen and when it should not. If you only test "does the agent search when it should?", you optimize toward an agent that searches for everything. Anthropic hit this building web-search evals for Claude.ai: the model should search for the weather but answer "who founded Apple?" from memory. Balancing undertriggering against overtriggering "took many rounds of refinements to both the prompts and the eval." One-sided evals produce one-sided optimization.

**Step 4: a stable, isolated harness.** The eval agent should behave like the production agent, and each trial should start from a clean environment. Shared state causes two kinds of error. It can deflate scores through correlated failures (leftover files, cached data, resource exhaustion) and it can inflate them: in some internal evals Claude gained "an unfair advantage on some tasks by examining the git history from previous trials." If several trials fail because of the same environment limit, such as low CPU memory, they are not independent samples, and the pass@k math above no longer applies.

**Step 5: design graders carefully.** The ordering of preference is deterministic graders where possible, LLM graders where needed, humans for validation. Four sub-points carry most of the value:

1. *Grade the outcome, not the path.* The instinct to require a specific sequence of tool calls produces brittle tests, because "agents regularly find valid approaches that eval designers didn't anticipate." This is in tension with the post's own YAML examples, which include `tool_calls` graders with required sequences; the reconciliation is that those examples are labeled as showcasing every grader type, and the prose recommendation is the one to follow.
2. *Give partial credit.* A support agent that identifies the problem and verifies the customer but fails the refund is better than one that fails immediately, and the metric should reflect that.
3. *Engineer the LLM judge.* Calibrate against human experts. Give the judge an escape hatch, an instruction to return "Unknown" when it lacks information, to reduce hallucinated verdicts. Grade each rubric dimension with a separate, isolated judge rather than one judge scoring everything. This isolation idea parallels the separation of verifier, rule check and rubric in [[four-layer-scoring-stack-for-rl-environments]].
4. *Audit the grader itself.* This is where the post's most valuable numbers are. Opus 4.5 first scored 42% on CORE-Bench. An Anthropic researcher found rigid grading that rejected "96.12" when the key said "96.124991…", ambiguous specs, and stochastic tasks that could not be reproduced exactly. After fixing those and using a less constrained scaffold, the score went to 95%. More than half of the "failure" was the instrument. METR found a related bug in its time-horizon benchmark ([[2503.14499 - Measuring AI Ability to Complete Long Tasks (METR Time Horizon)]]): tasks said "reach score X", but grading required exceeding X, which "penalized models like Claude for following the instructions, while models that ignored the stated goal received better scores." A grading bug can therefore invert a ranking, not just add noise. The vault's [[grader-failure-routing]] makes the same argument from the other side: charging grader failures to the model measures the grader, not the model.

The post also asks for graders resistant to hacks, where passing "genuinely requires solving the problem rather than exploiting unintended loopholes." It does not go further than this one sentence. The vault covers the rest: [[reward-hacking-as-eval-false-positive]], [[METR Reward Hacking and Eval Integrity]], [[transcript-spotting-heuristics-for-reward-hacking]] and [[2601.20103 - TRACE Reward Hack Detection via Contrastive Analysis]]. There is an interesting unresolved tension here with the opening τ2-bench story: a model finding a policy loophole is called a better solution there, while exploiting loopholes is called cheating here. Where the line falls (serving the user's real goal versus satisfying the grader's letter) is left to judgment.

**Step 6: read the transcripts.** You cannot know whether graders work without reading transcripts and grades from many trials. Anthropic built tooling for viewing transcripts and reads them regularly. The criterion: "Failures should seem fair", meaning it is clear what the agent did wrong. The post states a firm rule: "we do not take eval scores at face value until someone digs into the details of the eval and reads some transcripts."

**Step 7: watch for saturation.** A suite at 100% tracks regressions but gives no signal for improvement. Near saturation, progress looks slow because only the hardest tasks remain, so "large capability improvements appear as small increases in scores." Qodo, a code-review startup, was initially unimpressed by Opus 4.5 because its one-shot coding evals did not capture gains on longer tasks, and it had to build an agentic eval to see the difference. There is a small inconsistency in the post's SWE-bench Verified numbers: one section says models went "from 40% to >80%" in a year, another that scores "started at 30% this year." Both agree on the >80% endpoint.

**Step 8: maintain the suite as a living artifact.** What worked at Anthropic was a dedicated evals team owning the core infrastructure, with domain experts and product teams contributing most tasks and running evals themselves. The post pushes "eval-driven development": write capability evals for features before the agent can do them, treating them as bets on future model capabilities, then run the suite when a new model drops to see which bets paid off. It also suggests letting product managers, customer success and sales contribute tasks as pull requests via Claude Code.

## Evals are one layer, not the whole picture

The last section places automated evals among six methods: automated evals, production monitoring, A/B testing, user feedback, manual transcript review, and systematic human studies. Each has a distinct blind spot. Automated evals can create false confidence if they do not match real usage. Production monitoring is reactive, since users hit problems first. A/B tests take days or weeks and explain little about why. User feedback is sparse and skews toward severe issues. Transcript review does not scale. Human studies are slow and expensive.

The mapping to development stages is the practical part: automated evals in CI on every change and model upgrade; production monitoring after launch to catch distribution drift; A/B tests for large changes once traffic allows; feedback triage and a weekly transcript sample as ongoing habits; human studies reserved for calibrating LLM graders and for subjective outputs. The post uses the Swiss Cheese Model from safety engineering: each layer has holes, but stacked layers rarely have holes in the same place.

## What is interesting, and what is missing

The most useful non-obvious idea is that eval failures are often instrument failures. The CORE-Bench jump from 42% to 95%, the METR threshold bug and the Terminal-Bench filepath issue all say the same thing: before trusting a low score, check whether the task is solvable and the grader fair. The "0% pass@100 means a broken task" heuristic turns that into a concrete trigger.

The second is the reframing of evals as specification. Two engineers reading the same spec will handle edge cases differently, and an eval suite resolves the ambiguity by making expected behavior executable. This is why the post says evals are useful from day one, before any regression exists to catch.

Genuine limitations. The post is written from Anthropic's vantage and its customer anecdotes are uncontrolled; the Descript, Bolt and Qodo stories illustrate practices but are not evidence that those practices beat alternatives. It gives no guidance on how many trials per task are needed or how to report uncertainty, despite introducing pass@k; [[2411.00640 - Adding Error Bars to Evals]] fills that gap. It says little about the external validity of clean tasks versus messy real work, which [[task-messiness-as-external-validity-gap]] shows can be large. Its treatment of reward hacking is one sentence. And the user simulator in conversational evals is treated as a solved component, when it is itself a measurable error source.

For broader context on grading frameworks and eval practice in the vault, see [[2507.21504 - LLM Agent Evaluation Objectives (COMPREHENSIVE + NETLOGO)]], [[Hamel Husain - AI Evals FAQ (Everything You Need to Know)]], [[AISI - Evaluation Validity Checklist]], [[2507.06893 - Inspect Evals (Eval Orchestration Infra)]] and [[evaluation-unit-scales-from-answer-to-trace-to-world]]. The appendix lists frameworks (Harbor, which ships Terminal-Bench 2.0; Braintrust; LangSmith; Langfuse; Arize Phoenix) with the advice to pick one quickly and put effort into tasks and graders, since frameworks are "only as good as the eval tasks you run through them."

## Key claims

1. **A 0% pass rate over many trials usually means the task is broken, not that the agent is incapable.** No number beyond the 0% pass@100 threshold. Holds for frontier models on well-intended tasks.
> With frontier models, a 0% pass rate across many trials (i.e. 0% pass@100) is most often a signal of a broken task, not an incapable agent, and a sign to double-check your task specification and graders.
(Roadmap, Step 2)

2. **Grading and scaffold bugs hid most of Opus 4.5's CORE-Bench ability: 42% rose to 95% after fixes.** Setup: Opus 4.5 on CORE-Bench; fixes to rigid numeric matching, ambiguous specs, irreproducible stochastic tasks, plus a less constrained scaffold.
> For example, Opus 4.5 initially scored 42% on CORE-Bench, until an Anthropic researcher found multiple issues: rigid grading that penalized “96.12” when expecting “96.124991…”, ambiguous task specs, and stochastic tasks that were impossible to reproduce exactly. After fixing bugs and using a less constrained scaffold, Opus 4.5’s score jumped to 95%.
(Roadmap, Step 5)

3. **A grading bug can invert rankings by punishing instruction-following models.** Setup: METR time-horizon tasks where the prompt stated a score threshold but grading required exceeding it.
> This penalized models like Claude for following the instructions, while models that ignored the stated goal received better scores.
(Roadmap, Step 5)

4. **20 to 50 tasks from real failures are enough to start, because early changes have large effects.** Holds in early agent development; mature agents need larger suites.
> In reality, 20-50 simple tasks drawn from real failures is a great start. After all, in early agent development, each change to the system often has a clear, noticeable impact, and this large effect size means small sample sizes suffice.
(Roadmap, Step 0)

5. **pass^k collapses quickly with k: 75% per-trial success gives about 42% for three consecutive passes.** Assumes independent trials with fixed per-trial success probability.
> If your agent has a 75% per-trial success rate and you run 3 trials, the probability of passing all three is (0.75)³ ≈ 42%. This metric especially matters for customer-facing agents where users expect reliable behavior every time.
(Non-determinism section)

6. **Shared state between trials can inflate scores, not only add noise.** Setup: internal Anthropic evals without clean per-trial environments.
> For example, in some internal evals we observed Claude gaining an unfair advantage on some tasks by examining the git history from previous trials.
(Roadmap, Step 4)

7. **Grade the outcome rather than the exact tool-call path, because agents find valid unanticipated routes.** A design recommendation, no number.
> We’ve found this approach too rigid and results in overly brittle tests, as agents regularly find valid approaches that eval designers didn’t anticipate. So as not to unnecessarily punish creativity, it’s often better to grade what the agent produced, not the path it took.
(Roadmap, Step 5)

8. **LLM judges work better with an "Unknown" escape hatch and one isolated judge per rubric dimension.** Design recommendation, no measured effect reported.
> To avoid hallucinations, give the LLM a way out, like providing an instruction to return “Unknown” when it doesn’t have enough information. It can also help to create clear, structured rubrics to grade each dimension of a task, and then grade each dimension with an isolated LLM-as-judge rather than using one to grade all dimensions.
(Roadmap, Step 5)

9. **One-sided evals cause one-sided optimization; Anthropic needed evals in both directions for Claude.ai web search.** Setup: Claude.ai search triggering, balancing under- and overtriggering.
> Test both the cases where a behavior should occur and where it shouldn't. One-sided evals create one-sided optimization.
(Roadmap, Step 3)

10. **Near saturation, large capability gains show up as small score increases.** SWE-bench Verified cited at >80% for frontier models.
> As evals approach saturation, progress will also slow, as only the most difficult tasks remain. This can make results deceptive, as large capability improvements appear as small increases in scores.
(Roadmap, Step 7)

11. **Static evals can mark a better-than-intended solution as a failure.** Setup: Opus 4.5 on a τ2-bench flight booking task.
> For instance, Opus 4.5 solved a 𝜏2-bench problem about booking a flight by discovering a loophole in the policy. It “failed” the evaluation as written, but actually came up with a better solution for the user.
(The structure of an evaluation)

12. **Anthropic does not trust a score until someone has read transcripts.** Stated as an internal rule.
> As a rule, we do not take eval scores at face value until someone digs into the details of the eval and reads some transcripts.
(Roadmap, Step 7)
