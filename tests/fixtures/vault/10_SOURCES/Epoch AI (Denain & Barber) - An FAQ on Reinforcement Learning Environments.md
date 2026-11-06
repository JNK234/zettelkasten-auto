Created: 24-09-2026 16:18

Tags:

Status: #baby

## References

1. [An FAQ on Reinforcement Learning Environments](https://epoch.ai/gradient-updates/state-of-rl-envs)

# Notes

Authors: Jean-Stanislas Denain (Epoch AI, leads the Insights team) and Chris Barber (guest author, angel investor). Published 12 Jan 2026 in Epoch's *Gradient Updates* newsletter, which Epoch labels as "opinionated or informal takes" and not Epoch's house view. Method: 18 interviews across RL-environment startups, neolabs, and frontier labs (9 calls, 9 text/email replies, plus 4 more people who sanity-checked without giving input). Every quote is anonymous, so treat the numbers as practitioner hearsay that points in a direction, not measured data.

## The problem: compute is scaling faster than the things to spend it on

The story starts with OpenAI's o1: RL on math and code problems whose answers can be checked automatically. Since then labs have done two things together: widened the kinds of tasks they train on, and put much more compute into RL. Karpathy's 2025 review gives the intuition: train on enough different verifiable tasks and the model "spontaneously develops strategies that look like 'reasoning'."

The catch sits in one sentence of the post: **without diverse, high-quality environments and tasks, extra RL compute is wasted.** RL gets no signal from a task the model always fails, always passes, or can cheat on. So the bottleneck has moved from GPUs to *task supply*, and a mostly private market has grown up to fill it. The post is a field report on that market: what gets built, who builds it, what it costs, and what makes a good one.

This is the market view of ideas already in the vault: [[benchmarks-repurposed-as-training-environments]] and [[environment-and-eval-are-the-same-object]] explain *why* an environment can be sold as both a training input and an eval. This post adds *who pays whom and how much*.

## Plain version first: what is an environment, what is a task?

Take the standard LLM-RL loop:

1. The model gets a **task** (a prompt with an objective).
2. It takes **actions**: runs code, thinks out loud, clicks buttons, searches documents.
3. A **grader** scores the attempt. Usually this is automated: a unit test, an exact-match check, or an LLM judging against a rubric.
4. The scored attempts update the weights, so strategies that scored well get reinforced.

The **environment** is everything that fixes steps 2 and 3 *except* the prompt: which actions exist, and the context that decides what those actions do (file system, environment variables, the state of a simulated app). In practice it usually ships as a **Docker container**, but not always. Prime Intellect's Environments Hub uses their lighter `verifiers` library instead, and most math needs no environment at all because the model uses no tools (see [[Prime Intellect - Environments Hub]], [[environment-as-installable-package-not-config]]).

A **task** is prompt + grader. The post admits the words are loose. In classic RL terms (footnote 3), the "environment" holds the transition dynamics, the reward function, and the distribution over start states, and a "task" is just one start state. In industry use, "environment" can mean one container holding one task, or a collection of containers holding many tasks.

### Four worked examples (each is environment -> task -> grader)

| Environment | Example task | What the agent sees and does | Grader |
|---|---|---|---|
| Git repo at a fixed commit (SWE-bench Verified style) | Fix the bug so the failing tests pass | OS and shell tools on the repo | Run the test suite |
| Airbnb clone | Cheapest 2-bedroom listing in city X on dates Y | DOM / accessibility tree; outputs click or type actions | Check the final answer |
| Bloomberg terminal clone | 5-year CAGR for a list of companies | Simulated terminal UI and data | Compare figures with the true values |
| Excel clone | Pivot table of revenue by region | A spreadsheet app with realistic features | Compare with a reference solution |

**A structural point that is easy to miss:** how many tasks share one environment depends on the domain. A computer-use clone (Excel) is expensive to build and cheap to reuse, so one environment can host *hundreds* of tasks. A coding environment usually holds *one* task, because setting up a repo at a commit is cheap. That changes the cost model: in computer use you pay mostly for the environment, in coding you pay mostly per task.

## How labs use them: RL first, then SFT, then benchmarks

Three uses:

- **RL** (the main one). One startup employee put RL demand at roughly **10-20x** benchmarking demand.
- **Benchmarking.** Benchmarks are usually single-turn. RL buyers increasingly want *multi-turn* agent-user interaction, so the two products are drifting apart even though they are the same object underneath.
- **SFT on successful trajectories** ("gold trajectories" = human-verified successful runs), used in midtraining. It is growing especially for interleaved thinking and tool calling.

**Why SFT vs RL is a real engineering choice.** RL needs *several* rollouts per task with *different* outcomes so the algorithm can compare them. SFT needs only *one* good trajectory. So SFT is the practical choice when good trajectories are easy to produce but a reliable grader, or enough spread between attempts, is hard to get. One founder added that **trailing labs lean toward SFT**: "It's quite hard to get your really large scale RL pipelines and it's easy to throw stuff into your SFT mix." Put simply, whether a lab uses RL or SFT tells you how mature its infrastructure is.

The vendors often **cannot see** how their product is used. "To what extent are they using it to generate gold trajectories versus doing RL? I think that would be pretty hard to answer." One interviewee expects environment designers to start building *for* synthetic data generation on purpose.

## Who builds them: four kinds of company

This is the part to read for company approaches. The post names four supplier types, and each has a different edge.

### 1. Specialized RL-environment startups
They build environments across software engineering, computer use, math, and finance. Chris Barber keeps a list of them at **Pavlov's List** (https://pavlovslist.com/). It is the directory to scan for company names. Their edge is focus and engineering quality on the environment itself. Their weakness, stated by one founder: the cap on their revenue is "simply the difficulty of scaling up task creation at the required quality level." Their limit is operations, not demand.

### 2. Traditional data providers: Mercor, Surge, Handshake, Turing
These companies started with human-labeled data (RLHF preferences, SFT demos) and now sell environments too. Buyers pay partly for QA processes and supervision infrastructure, but the main thing, as a founder put it, is **"they have the guys"**: an existing contractor network, so they can staff a large task-creation project faster than a lab could hire. Their edge is labor supply and management at scale. This fits the post's main bottleneck claim: if management and QC are the hard part, the firms that already run large expert-contractor workforces are well placed. (Context from the vault: [[Mercor benchmarking Take Home]], [[Mercor Post-Training Assessment - GRPO Rollout Scheduling Under a Fixed Budget]] show what Mercor tests for on the builder side.)

### 3. In-house teams at model developers
Both frontier labs and **neolabs** do this. The post cites xAI and Anthropic job postings for environment-building roles, and names **Cursor** as a neolab that can turn *its own user data* into training tasks. That data is an edge no vendor can sell. One founder saw "substantially more in-housing" recently. The three reasons given:
- avoid paying vendor margins;
- keep training priorities secret (buying a Salesforce environment tells the vendor what you plan to train);
- some domains need expertise the lab has and vendors lack. The example given is **AI research itself**.

### 4. Product companies partnering with labs
If you are Salesforce or Slack, you know your own interface and edge cases better than anyone. Named partnerships:
- **Benchling + Anthropic**: biology workflows.
- **OpenAI + Shopify and Stripe**: shopping.
- **OpenAI's health integrations.**
The authors say plainly that they *don't know* the details, only that the product company is "likely at least collaborating" on environment or task creation. The counter-example is **Amazon**, which is hostile to agent traffic: it sued Perplexity over its agentic shopping tool and blocks most AI agents. So access to the real product is contested, and that is part of why clones exist.

**Reading across the four:** each type owns a different scarce input. Startups own environment engineering. Data providers own labor at scale. Labs own secrecy and proprietary data. Product companies own ground-truth fidelity. The market splits along whichever input is scarcest for a given domain.

## What it costs, with the economics worked through

| Item | Figure | Source in the post |
|---|---|---|
| Contract size | 6-7 figures **per quarter**; one founder says often 7+; one neolab researcher saw $300k-$500k | Interviews |
| Website replica ("UI gym") | ~$20k each | SemiAnalysis |
| High-fidelity complex-product clone (e.g. Slack) | ~$300k | One interviewee |
| Per task | **$200-$2,000** typical; $20k rare (complex SWE) | Several interviewees agreed |
| Exclusivity premium | **~4-5x** non-exclusive price | Two founders, independently |
| RL compute spent per task | ~**$2,400** | Mechanize estimate |
| Anthropic RL-env spend discussed | >$1B over the next year | The Information, Sep 2025 |
| OpenAI 2026 R&D compute (projected) | ~$19B, about 2x 2025 | The Information |

**The non-obvious point is in the task-vs-compute ratio.** A typical task costs $200-$2,000, and the compute spent training on it is about $2,400. So **the task is the cheaper input.** If a cheap task is flawed (hackable, infeasible, trivially easy), you lose more than its price: you waste the larger compute bill spent on it. That is Mechanize's argument ("cheap RL tasks will waste compute") and it is the economic reason quality beats quantity here. Put simply, paying 2x for a sound task is cheap insurance on the compute.

**The exclusivity premium says something about what is being bought.** 4-5x for exclusivity means labs value *not sharing a capability edge* at roughly 3-4x the task's direct training value. The buyer is paying to keep a capability from rivals, not only paying for data.

**Scale check.** Even $1B/yr is small next to compute ($19B R&D compute at OpenAI in 2026). Environments are a big market for vendors but a small line item for labs. So labs have room to raise prices for quality, and quality should win on price.

## Domains: where demand is moving

**Math: first, but maybe shrinking.** Cheap to make (no environment, just a verifiable answer). But "math might be shrinking" and, per one founder, math tasks "don't transfer as well to other capabilities." That goes against the early-2025 assumption that math RL spills over into general reasoning. Worth comparing with [[2501.12948 - DeepSeek-R1 (RL for Reasoning)]], where math/code RL was the flagship.

**Coding: still large, and becoming "productionized."** Moving from SWE-bench-style `FAIL_TO_PASS` / `PASS_TO_PASS` checks (tests that should start passing / tests that must keep passing) toward realistic workplaces: GitHub + Linear + an IDE, i.e. "how does a SWE actually work in an environment?" The open problem is **grading beyond unit tests**: "There's probably many solutions that pass the unit tests. But there's some nicer solutions that have better trade-offs." Unit tests check correctness, not design quality, so a tests-only reward cannot teach good engineering. This is the gap in [[weak-verification-problem-in-automated-testing]] and [[validation-test-overfitting-in-agent-search]].

**Enterprise workflows: the growth area called for 2026.** Expense reports, pivot tables, slides from a brief, CRM updates. The founder's reason is about incentives: "labs index very heavily on what's valuable and what is quantifiable, and enterprise workflows are perfect for that." Three ways to build them:
- **MCP-style tool integrations** (API calls, no GUI);
- **Playwright-style browser control** (DOM actions);
- **Screenshot-based computer use** (pixels in, clicks out), needed for things like an ERP "that has no backend" you can reach.

Most use **clones** of apps like Slack or SAP. The sharpest warning in the piece comes from a lab researcher: "what everyone does is vibe code a buggy website which isn't useful. There's a large amount of useless bad environments out there." A buggy clone teaches the agent the clone's bugs, not the real app. That explains the 15x price gap between a $20k UI gym and a $300k Slack clone: fidelity is what you are paying for. See [[enterprise-agent-evaluation-challenges]] and [[state-verification-vs-output-verification]]. For enterprise tasks the grader often has to inspect the *final app state* (was the record updated?), not just the text answer.

**Prime Intellect's environment bounties** are the one public view into this space: environments built from well-known benchmarks, MCP-server interaction, AI-research tasks like nanoGPT speedrunning, and textbook-sourced questions.

**Direction:** coding + enterprise, both going **longer-horizon**: "full end-to-end tasks that involve navigating through multiple tabs, browsers, and then submitting something that involves multi-hop steps." Minor directions: multi-turn user interaction, multi-objective environments, and tools that let researchers *inspect and edit trajectories* as feedback. Compare [[human-time-horizon-as-grounded-difficulty-metric]] for how "horizon" gets measured.

## Quality criteria, with the mechanics

### 1. Robustness to reward hacking (named most important)
"High reward must mean the task was actually solved, not hacked." Concrete ways models hack:
- **looking up the solution** online;
- **checking out future commits** in a badly scripted repo, where the fix already exists in git history (a real SWE-bench Verified case the post links).

Graders "take many many iterations" to harden. **Surprise:** it is still the top priority *even though* models got better at not reward hacking over the past year (the post cites the Claude Opus 4.5 system card). Two readings: the better behavior may itself come from sturdier environments, or labs cannot count on the model's restraint and still have to fix the grader. The vault has the detection side: [[transcript-spotting-heuristics-for-reward-hacking]], [[reward-hacking-detection-via-at-scale-agent-qa]], [[model-awareness-of-intent-violation-while-cheating]], [[METR Reward Hacking and Eval Integrity]], [[2605.12474 - Reward Hacking in Rubric-Based Reinforcement Learning]], [[HUD - Verifier and Reward Design for RL Environments]], plus the layered-grader design in [[four-layer-scoring-stack-for-rl-environments]].

### 2. Difficulty calibration: why 0% and 100% are useless
Footnote 10 gives the mechanism, which is GRPO ([[grpo-group-relative-policy-optimization]]). For one task, sample a group of $G$ rollouts with rewards $r_1..r_G$. Each rollout's advantage is

$$A_i = \frac{r_i - \bar r}{\sigma_r + \epsilon}, \qquad \bar r = \tfrac{1}{G}\sum_j r_j$$

- $r_i$: reward of rollout $i$ (0/1 for pass/fail);
- $\bar r$: group mean, the baseline;
- $\sigma_r$: group standard deviation, the normalizer;
- $\epsilon$: stops division by zero.

If every rollout fails (or every rollout passes), then $r_i = \bar r$ for all $i$, so $A_i = 0$ and **the policy gradient from that task is exactly zero.** You paid for $G$ rollouts and learned nothing. That is the math behind "if the pass rate is 0% or 100%, the model won't learn," and why [[zero-gradient-prompt-filtering]] exists.

Now the thresholds the interviewees gave. With true pass rate $p$, the chance a group has *mixed* outcomes (nonzero signal) is $1-(1-p)^G - p^G$:

| $p$ | $G=8$ | $G=16$ | $G=64$ | $G=128$ |
|---|---|---|---|---|
| 0.78% (1/128) | 0.06 | 0.12 | 0.39 | 0.63 |
| 1.56% (1/64) | 0.12 | 0.22 | 0.63 | 0.87 |
| 2% | 0.15 | 0.28 | 0.73 | 0.93 |
| 3% | 0.22 | 0.39 | 0.86 | 0.98 |
| 30% or 70% | 0.94 | ~1.0 | 1.0 | 1.0 |

What this shows:
- The "2-3% floor" and the "1 success in 64-128 rollouts" floor are **the same rule stated two ways**. It is the pass rate at which a large group reliably contains at least one success. With small groups ($G=8$), a 2% task gives signal only 15% of the time, so the floor depends on the lab's group size. The post doesn't say this, but it follows from the numbers. Related: [[task-diversity-beats-group-size-under-fixed-rollout-budget]].
- Mixed-outcome probability is **symmetric** in $p$ and $1-p$ (30% and 70% both give 0.94 at $G=8$). So retiring a task at ~70% is **not** because its signal is gone. It is closer to diminishing returns: the model has mostly learned it, and harder tasks have more to teach. Also, std-normalization gives near-saturated groups oversized advantages ([[std-normalization-difficulty-bias-in-group-rl]]). A reason to retire them early.
- A 0% task is **ambiguous**: too hard, or *impossible* (broken task)? Hence the founder's rule: keep a 0% task only if "another annotator do[es] a blind solve and they succeed." That is a feasibility check by a human.

### 3. A smooth difficulty distribution (curriculum across the set)
"A very important feature of the RL environment is a smooth gradient: diversity in the difficulty of the tasks." A suggested mix: some at 0%, some at 5%, some at 30%. The 0% tasks are **not wasted**: as the policy improves they cross into the learnable range, so they are *future* signal. The set works as a built-in curriculum: tasks enter from 0%, get learned, and leave around 70%.

### 4. Compositionality
"The skills should not be disparate tasks, they should be similar to each other and leverage common skills." Shared sub-skills mean gradients from different tasks add up instead of pulling apart. Probably the same reason math "transfers less": its skills don't overlap much with agentic work.

### 5. Scaling without losing quality (the number one operational bottleneck)
"Finding the experts isn't that hard, but managing them and doing quality control is hard." The details are about people: motivating contractors, and checking they are "not just using LLMs" to write tasks. An LLM-written task is likely to sit at a difficulty the model already handles, which defeats the purpose. The post cites Kevin Lu's argument that scaling environments is a key constraint on AI progress.

### Who is good at this
- "Domain knowledge and expert level prompting is more important than ML skills for creating tasks."
- Product sense: "know how people are actually using these tools."
- Heavy frontier-model use: "a very heavy Claude Code user, a prompt whisperer like Riley Goodside, can be better at figuring out what the frontier is than an AI researcher."
- "The people who are probably best at this are the people who create benchmarks that actually get used."

Building the *environment* is mostly software engineering. Writing the *task* means knowing where the frontier model breaks. These are two skill sets and often two teams. The last quote links this role directly to benchmark design, which is again [[environment-and-eval-are-the-same-object]].

## Limitations of the source (real ones)

- **Small, anonymous, self-interested sample.** 18 people, about half by text. Founders gain from saying demand is large and quality is hard (it justifies prices). No figures are audited.
- **Borrowed numbers.** UI-gym price (SemiAnalysis), compute per task (Mechanize), the Anthropic $1B (The Information, "discussed", not confirmed) are second-hand. Mechanize's $2,400/task depends on assumptions the post doesn't unpack.
- **Partnership details are admitted guesses** ("we don't know the exact details").
- **Vendors can't see how products are used**, so the "RL is 10-20x benchmarking" figure is about what gets *requested*, not measured use.
- **Short shelf life by the authors' own account:** "we expect this picture to look quite different in a year." Published Jan 2026, so by late 2026 treat the domain mix and prices as a snapshot.
- No comparison of how different graders (unit test vs rubric LLM-judge vs state check) behave. See [[scoring-as-a-pluggable-interface-in-rl-infrastructure]] for that design question.

## Links out worth following from the post
- Pavlov's List (RL-env startup directory): https://pavlovslist.com/
- Mechanize, "Cheap RL tasks will waste compute": https://www.mechanize.work/blog/cheap-rl-tasks-will-waste-compute/
- SemiAnalysis, "RL environments and RL for science": https://newsletter.semianalysis.com/p/rl-environments-and-rl-for-science
- Prime Intellect scaling-environments program / bounties: https://www.primeintellect.ai/blog/scaling-environments-program
- Prime Intellect `verifiers`: https://github.com/PrimeIntellect-ai/verifiers
- Kevin Lu, "The only important technology is the internet": https://kevinlu.ai/the-only-important-technology-is-the-internet
- Epoch, "What skills does SWE-bench Verified evaluate?": https://epoch.ai/blog/what-skills-does-swe-bench-verified-evaluate
