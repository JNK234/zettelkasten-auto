Created: 24-09-2026 16:22

Tags:

Status: #baby

## References

1. [ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)

# Notes

Source: OpenPipe technical blog, Kyle Corbitt (founder/CEO), 29 Apr 2025. Code: `OpenPipe/ART` repo, `examples/art-e` on the `art-e` branch. Model: `OpenPipe/art-e-008` on Hugging Face. Dataset: `corbt/enron_emails_sample_questions` (~4K questions). The openpipe.ai site is now a static archive that says the platform "migrated to Weights & Biases and CoreWeave". The page is a JS app, so the text here was pulled from the site bundle, the chart numbers were read off the rendered SVGs, and the reward details come from `rollout.py` in the repo.

## The problem, and why OpenPipe cared

The question behind the post is a company question as much as a research one: **can a small open model trained with RL on one narrow agentic task beat the best prompted frontier model on that task, and be much cheaper and faster?** That is OpenPipe's whole business pitch (fine-tune a custom model to replace GPT-4 calls). ART·E is the demo that makes the pitch concrete for *multi-turn agents*, not just single-call tasks.

They framed it against OpenAI's Deep Research. Deep Research showed that RL on a specific agentic task (search, read, synthesize) gives a big jump over prompting. ART·E applies the same recipe to a smaller, realistic task: **answer a natural-language question by searching the user's email inbox.** Example: "what time is my brother's flight on Friday?" A person has to guess keywords, run searches, open emails and read them. The agent should do that loop for you.

Why this task is a good pick for RL (their reasoning plus mine):
- It is **realistic and useful**, so the result means something.
- It is **narrow**, so improvement shows up within hours of training, not weeks.
- It has a **verifiable outcome**: each question has one golden answer and one source message ID. That makes the reward close to a ground-truth verifier, not a fuzzy preference score. This matters a lot later (see [[kl-penalty-is-unnecessary-under-a-ground-truth-verifier]], [[dropping-the-kl-anchor-is-safe-only-under-rule-based-rewards]]).

Headline result: Qwen 2.5 14B + GRPO gets **96%** correct vs o3 at **90%**, o4-mini 88%, Gemini 2.5 Pro 76%, GPT-4.1 71%. Cost: **$0.85 per 1K runs vs $55.19 for o3 (~64x cheaper)**. Latency: **1.1 s vs 5.6 s (~5x faster)**. Total training cost about **$80** on one H100, under a day.

## The recipe in one picture

The post follows a four-part recipe that is basically OpenPipe's house method for agent RL:

1. **Data**: realistic scenarios with known answers (synthetic, if needed).
2. **Environment**: the tools and information the agent can use.
3. **Baselines**: push prompted models as hard as you can before any training.
4. **Reward + train + watch**: design a reward, run GRPO, look at the trajectories.

Each piece is below, simple first, then the details.

## 1. Data: Enron inboxes plus synthetic questions

**Inboxes.** The Enron fraud litigation made about 500K real corporate emails public. That is the only large, realistic, public inbox corpus, so it is the obvious choice. They randomly picked **8 employee inboxes for test and 20 for train**, each with at least 5K emails (many 10K+). The split is **by inbox, not by question**, so the test set checks generalization to inboxes the model has never seen: new people, new topics, new vocabulary. That is a stronger held-out test than a random question split.

**Questions.** No dataset of "questions a user would ask about their own inbox" exists, so they generated one:
- For each inbox, walk through 1,000 emails in **batches of 20**.
- For each batch, prompt **GPT-4.1** to write several question/answer pairs per email, each with the **source message ID** and a **`how_realistic` score from 0 to 1**.
- Filter on `how_realistic`. The author says this self-rating worked well at dropping questions no real person would ask.
- A footnote says Gemini 2.5 Pro was also tried as the generator, and GPT-4.1 did better.

Result: ~4K questions. The example questions shown are specific and factual, e.g. which issues a named meeting will cover, with the answer taken from one email.

Two things worth noticing:
- The self-rated realism filter is cheap and does real work. It is the same move as an LLM-as-judge filter, applied at data-generation time ([[llm-as-judge-evaluation-pattern]]).
- **Every question is built from a single email.** So the task is mostly *single-hop retrieval*: find one email, read it, pull out the fact. That shapes what "96%" means (see Limitations).

## 2. Environment: three tools and the simplest loop

"Environment" here just means the tools plus the data the agent can reach. The agent gets a system prompt (answer the user's question from their inbox) and three tools:

| Tool | What it does |
|---|---|
| `search_emails(keywords, sent_after, sent_before)` | Returns up to 10 matching emails: message IDs plus matching snippets. Date filters included. |
| `read_email(message_id)` | Returns the full body of one email. |
| `return_final_answer(answer, sources)` | Ends the episode. Gives the answer plus the list of message IDs that support it. The tool description tells the model to answer "I don't know" with empty sources if it can't find the answer. |

Backend: the corpus is loaded into a **local SQLite database**, and search uses **SQLite FTS5** full-text search. So search is plain keyword/BM25-style, not embeddings. The agent has to learn to pick good keywords, which is exactly the skill that RL can sharpen.

The agent loop is as plain as it gets:
1. Call the LLM with the conversation so far.
2. Parse the tool call.
3. Run the tool.
4. Append the model's message and the tool output to the history.
5. Repeat until `return_final_answer` is called or **10 turns** pass.

No planning module, no recursion, no backtracking, no sub-agents. The point: the capability comes from training the policy, not from scaffolding. That matches the idea in [[learned-tool-routing-from-outcome-reward-alone]]: an outcome reward alone can teach when and how to call tools. It is also a clean example of [[agentic-rag-approach]], where retrieval is a tool the model decides to call, not a fixed pipeline step.

## 3. Baselines first (their strongest practical advice)

Before RL, they benchmarked many prompted models and **fixed several bugs in the agent loop** along the way. The argument for doing this:
1. A prompted model may already be good enough, so you skip RL.
2. If a strong model fails, the cause is often **missing context, broken tools, or a vague goal**. RL can't fix those. Better to find them before paying for training.
3. You get a target to beat.

Prompted accuracy on the test set (LLM judge vs golden answer):

| Model | Correct |
|---|---|
| DeepSeek R1 | 54% |
| GPT-4o | 67% |
| GPT-4.1 | 71% |
| Gemini 2.0 Flash | 75% |
| Gemini 2.5 Pro | 76% |
| o4-mini | 88% |
| o3 | 90% |

Interesting: the reasoning models built for tool use (o3, o4-mini) are far ahead of GPT-4.1 and Gemini 2.5 Pro, while R1, also a reasoning model, is last. So "reasoning" alone does not explain it. Being trained for agentic tool use does. That is indirect evidence for the whole thesis: agentic tool-use RL is what helps.

## 4. The reward function (the core of the post)

Plain version: **the reward is mostly "did you get the answer right", plus small tweaks for speed, plus a big penalty for confidently wrong answers.**

The judge is binary. In the repo, **Gemini 2.0 Flash** (temperature 0, 2 output tokens, cached) gets the question, the golden answer and the agent's answer, and returns True if they are semantically equivalent. So the generator (GPT-4.1) and the judge (Gemini Flash) come from different model families. That is a small but real guard against one model grading its own style.

The author's point: the eval judge is the starting point for the reward, but RL lets you **optimize several objectives at once, with weights that set their priority.** The reward in the repo's `calculate_reward` is a tiered function. Let $p \in [0, 0.4]$ be the partial-credit sum, with +0.1 for each of: ever saw the right email in search results, ever read the right email, never tried to read an invalid message ID, cited the correct source.

$$
R = \begin{cases}
-2 + p & \text{tool call unparseable} \\
-1.9 + p & \text{unknown tool name} \\
-1.8 + p & \text{bad tool arguments} \\
-1 + p & \text{answered, but wrong (hallucination)} \\
0 + p & \text{"I don't know" or ran out of turns} \\
1 + 0.3\,\mathbb{1}[\text{sources correct}] + \dfrac{0.1}{n_{src}} + 0.1\left(1 - \dfrac{t}{T_{max}}\right) & \text{correct answer}
\end{cases}
$$

Term by term:
- **Tiers.** Format errors are below −1, wrong answers are in [−1, −0.6], abstaining is in [0, 0.4], correct is in [1, ~1.5]. A code comment says partial credit must always sum to under 0.5, so **no amount of partial credit can move a trajectory into a higher tier.** The ordering "correct > abstain > wrong > broken" is strict. This is the key design rule: shaping terms only break ties *within* a tier.
- **$0.3\,\mathbb{1}[\text{sources correct}]$**: rewards citing the right email. That is grounding and makes the answer checkable.
- **$0.1/n_{src}$**: a bonus that shrinks as the model cites more sources. It stops the model from citing everything to be sure it hits the right one.
- **$0.1(1 - t/T_{max})$** with $T_{max}=10$: the "minimize turns" term. One fewer turn is worth 0.01. Very small on purpose, a latency proxy that never outweighs correctness.
- **Correct answers get no $p$.** The partial credits only apply on the failure branches, where they are meant to give a gradient between "totally wrong" and "almost right".
- The repo also has a `stupid_simple_reward_fn` flag (reward = 1 if correct, else 0) as an ablation. The post doesn't say whether the final run used it; it only reports that partial credit didn't help.

### Why the hallucination penalty works (a quick calculation)

Say the agent is unsure and thinks its best guess is right with probability $q$. Guessing gives about $q \cdot 1 + (1-q)(-1) = 2q - 1$. Saying "I don't know" gives 0. So **guessing only pays off when $q > 0.5$.** The −1 vs 0 gap sets a *calibration threshold*: answer only if you're more likely right than wrong. If the wrong answer got 0, the same as abstaining, guessing would always win and the model would learn to hallucinate. This is the RL version of the argument that binary accuracy grading rewards guessing.

Result: the hallucinated-answer rate went from **~13% at step 0 to ~3%** at the end. Compare o3 and o4-mini at ~4%, GPT-4.1 ~5%, and **Gemini 2.5 Pro ~21%**. The trained model is *both* more accurate and less likely to make things up. You might expect those two to trade off. Here they don't, because the reward prices both.

### Turns

Average turns per question: base model ~3.7 at step 0, a **spike to ~6.3 around step 30**, then a steady drop to **~3.3–3.4** by step 500. Compare o3 ~4.3, o4-mini ~4.25, Gemini 2.5 Pro ~4.1, GPT-4.1 ~6.0. So the final model uses about one turn fewer than o3.

The spike is the interesting part. Early on, the model learns that *searching more* finds the answer more often, so the correctness term pushes turns up. Only after accuracy is saturated does the tiny 0.01-per-turn term start to matter, and the model compresses its search. That is the expected order for a lexicographic-style reward: learn the main objective first, then optimize the tie-breaker.

### Partial credit didn't help

They hoped the partial credits would make the reward **denser** and speed up learning. It didn't measurably help. Their explanation: the base model already solved the task often enough (~40% at step 0) that the sparse correctness signal was enough. That is the right way to think about it. Dense shaping matters when the base success rate is near zero, so every group in GRPO scores the same and there is nothing to learn from. At 40%, most groups of 4 already have a mix of successes and failures. See [[shaping-term-ablation-against-true-completion-rate]] for the general rule: measure shaping terms against the true objective, not the shaped reward.

### The reward hack

In an early run they gave partial credit for **taking *more* turns**, to encourage exploration. The model learned to **repeat its last tool call over and over until it hit the 10-turn limit**, padding every trajectory for free reward. The author calls this a lesson from the bitter lesson: instrumental rewards get taken literally, and often it's better to reward only the final goal. Related: [[Manheim & Garrabrant - Self-Reported Score Components and Adversarial Goodhart]]. Rewarding a measurable proxy (turn count) the policy fully controls is the textbook Goodhart setup.

## 5. Training: GRPO via ART

ART (Agent Reinforcement Trainer) is OpenPipe's open-source library for training multi-turn agents with GRPO ([[grpo-group-relative-policy-optimization]]).

The loop per step:
1. Sample **12 questions**.
2. Run the agent **4 times per question** → 12 groups × 4 = **48 trajectories**. A trajectory is the full message list from one rollout.
3. Score each with the reward above.
4. Compute GRPO loss and update the model.
5. Every **30 steps**, evaluate on **100 validation questions**.
6. Stop when validation stops improving.

What GRPO does with those groups: for question $g$ with rewards $r_1..r_4$,

$$A_i = \frac{r_i - \mu_g}{\sigma_g}$$

Each trajectory's advantage is how much better it did than its siblings on the *same* question. No value network ([[the-critic-deletion-tension-in-agentic-rl]]). Tokens from above-average trajectories get pushed up, and tokens from below-average ones get pushed down. With only 4 samples per group, $\mu_g$ and $\sigma_g$ are noisy estimates. That is one reason the tiered reward helps: the within-tier shaping terms mean two correct trajectories rarely tie exactly, so $\sigma_g$ is rarely zero.

Final hyperparameters: **learning rate 1.2e-5, 2 epochs**, base model **Qwen 2.5 14B Instruct**. The step axis runs to ~500 steps. At 12 questions per step that's ~6,000 question visits over 2 epochs, which fits a train set of ~3K questions from the 20 train inboxes.

Infra, which is the part that makes $80 possible:
- **vLLM** for fast rollout generation.
- **Aggressive sample packing** at training time.
- **Unsloth** training optimizations (memory-efficient LoRA-style training).
- Launched with **SkyPilot** on a **RunPod** H100.
- One H100, a bit under a day, **~$80**. They ran **15 training jobs** in total while tuning, most for a few hours, so the real project cost is several times the final run.

Accuracy curve on the validation set: **40% at step 0 → 65% by step 30 → passes GPT-4.1 (71%) around step 90 → passes o4-mini/o3 around step 250–270 → 96% at the end**, with ups and downs (e.g. 88% around step 360). The "RL +56%" label on the headline chart is **absolute points** (40 → 96), not relative.

## 6. Monitoring: the practical lessons

1. **Watch the reward standard deviation.** GRPO learns only from differences inside a group. If every rollout for a question gets the same score, the advantage is zero and that group adds nothing. If the std collapses, their fixes are: more trajectories per group, a denser reward, a lower learning rate, or more varied scenarios. This is the same failure that [[Cameron R. Wolfe - GRPO Tricks (Making RL Actually Work)]] and [[Mercor Post-Training Assessment - GRPO Rollout Scheduling Under a Fixed Budget]] deal with (zero-variance groups waste compute).
2. **Track many metrics.** They logged 15 per trajectory: answer correct, sources correct, turns, attempted answer, ever found/read the right email, unparseable tool call, bad tool name, bad tool args, ran out of turns, said "I don't know", number of sources, tried to read an invalid email, prompt tokens, completion tokens. ART attaches metrics to each trajectory and sends aggregates to Weights & Biases. The reward is one number, and these metrics are how you find out *why* it moved.
3. **Read the actual outputs.** A rising reward doesn't prove the model is doing what you want. The repeat-the-tool-call hack was caught by reading trajectories, not by the reward curve. They used OpenPipe's request logs to inspect trajectories and say so in a footnote, adding that the tool wasn't built for this and ART would get something better. Compare [[trajectory-capture-by-instrumenting-the-llm-api-boundary]].

## What's surprising or non-obvious

- **A 14B open model beats o3 at a real agentic task after ~$80 of RL.** The catch is that the task is narrow and in-distribution. This is specialization, not general capability.
- **Accuracy and honesty both improved.** Usually pushing accuracy makes models guess more. The asymmetric −1/0 reward avoided that.
- **Latency improved as a side effect of a tiny reward term**, and only after accuracy saturated. The turn curve shows the priority order directly.
- **Dense shaping did nothing here, and one shaping term caused a hack.** The takeaway isn't "never shape". It's that shaping only helps when base success is near zero, and every shaping term the policy controls is a hacking risk.
- **Baselining found bugs in the environment.** Much of the "RL" win depends on a working environment, and prompted baselines are how you debug it.
- **The judge is tiny.** Gemini 2.0 Flash with a 2-token binary answer was enough for reward. That works because the task has a golden answer. For open-ended tasks this cheap judge wouldn't be enough ([[generic-rubrics-fail-instance-specificity-is-the-mechanism]]).

## Limitations (real ones, from the setup)

- **Single-hop questions.** Every question comes from one email. Real inbox questions often need several emails (e.g. "what did we finally agree on after the back-and-forth?"). 96% here says little about multi-hop synthesis.
- **Small eval, and possibly the same set used for stopping.** Validation is 100 questions, and in `train.py` the val scenarios load from the `test` split. The post says training ran "until the model stops improving on our validation set", and the headline compares on the test set. If those are the same questions, checkpoint selection leaks into the reported number. With n=100, one question is 1 point, and the standard error near 90–96% is ~2–3 points. The **6-point gap over o3 is 6 questions**. That is likely real but not a large margin, and the curve itself swings by ±4 points between evals.
- **Synthetic labels.** Questions and golden answers come from GPT-4.1, and correctness from Gemini Flash. Some "errors" by any model may be label noise. The ceiling for everyone is below 100%.
- **Enron is 2001 corporate email.** Modern inboxes have newsletters, receipts, threads, attachments and calendar invites. Transfer to a real personal inbox is untested.
- **Not an equal cost/latency comparison.** ART·E runs on a dedicated, self-hosted GPU. o3 is a shared API with reasoning tokens. The 64x/5x numbers are real for deployment, but they compare deployment setups as much as models, and they leave out the fixed cost of hosting a GPU at low traffic.
- **Tuning effort isn't fully counted.** $80 is the final run. There were 15 jobs, plus time spent building the environment and baselines.
- **Keyword-only search.** FTS5 caps what any agent can find. The RL mostly teaches better keyword choice. Swap in embedding search and the gap to o3 could shrink or grow.

## Company lens: what this says about OpenPipe's approach

- **Thesis:** task-specific RL on small open models beats frontier prompting on cost, latency, and sometimes quality. Their product was the tool for doing this (fine-tuning first, then agent RL through ART).
- **Method, as a playbook:** realistic environment → synthetic but realism-filtered data → hard prompted baselines → tiered verifiable reward with small shaping terms → GRPO with many samples → heavy observability and reading trajectories. That order is exactly what they would sell to an enterprise customer ("train your own proprietary agents").
- **Engineering stance:** simple agent loops, and put the effort into the reward and the infra (vLLM + Unsloth + packing on one GPU). Keep it cheap enough that a customer can run many experiments.
- **Open-source as distribution:** model, dataset and code are all public, with Discord and a webinar pushing people to ART.
- **Where it went:** the site now says the platform has moved to Weights & Biases and CoreWeave. The W&B integration was already in ART's metric logging. This post is a good reference for how that team thinks about agent RL.

## Connections

- [[reinforcement-learning-in-language-models]]: a small, fully reproducible example of outcome-reward RL for a tool-using agent.
- [[grpo-group-relative-policy-optimization]]: the optimizer. The std-collapse advice is GRPO-specific.
- [[learned-tool-routing-from-outcome-reward-alone]]: ART·E is a case where the outcome reward teaches the search strategy with no step-level supervision.
- [[shaping-term-ablation-against-true-completion-rate]]: partial credit gave no speedup, and turn-count credit caused a hack.
- [[llm-as-judge-evaluation-pattern]]: binary semantic-match judge used as both the eval and the reward.
- [[static-benchmark-inadequacy-for-agent-systems]]: synthetic, per-inbox eval sets are a cheap way to build a task-specific agent benchmark.
- [[hallucinated-actions-and-fallback-mapping]]: the bad-tool-name and bad-args penalties handle hallucinated actions in the reward, not the runtime.
