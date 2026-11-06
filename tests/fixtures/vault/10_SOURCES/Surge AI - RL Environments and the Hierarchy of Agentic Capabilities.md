Created: 27-09-2026 11:53

Tags:

Status: #baby

## References

1. [RL Environments and the Hierarchy of Agentic Capabilities (Surge AI)](https://surgehq.ai/blog/rl-envs-real-world)

# Notes

## The problem: a pass rate tells you who wins, not what is broken

By late 2025 the question about LLM agents had moved from "can it answer?" to "can it do a job?". Answering a job-shaped question needs a job-shaped test: a multi-step task where the model calls tools, reads what they return, and decides what to do next. The industry name for that test bed is an **RL environment** — a small simulated world with data and tools, where a model can act, get feedback, and (if you are training) be rewarded. The same object serves as a benchmark when you only score it, which is why [[environment-and-eval-are-the-same-object]] is the right frame for reading this post.

Surge AI (a human-data company that builds these environments) put nine frontier models into one such world and gave each 150 tasks. The headline number is a single pass rate per model. The post's real contribution is the step after that: reading the failed trajectories (the full logs of tool calls and replies) and noticing that failures are not random. They cluster by *kind of competence*, and the kinds seem to stack — a model that fails at the lower kind rarely gets far enough to show the higher kind. Surge calls the stack the **Hierarchy of Agentic Capabilities**.

This is a blog post with case studies, not a paper. There is no statistical analysis of failure categories, no counts per category, and the pass rates are shown only as bar charts. Treat it as a well-illustrated qualitative taxonomy backed by one quantitative ranking.

## The environment: Corecraft, Inc.

### What an environment is made of

Surge says every RL environment needs three parts. A **world model** is the big-picture setting (here, a company). **Entities** are the objects that live in that world and how they relate (customers, orders, products, support tickets). The **tool system** is the only way the agent can touch the entities — functions like `searchCustomers`, `searchOrders`, `searchProducts`, `getProduct`, `searchTickets`, and `validateBuildCompatibility`, exposed through MCP (the Model Context Protocol, a standard way to describe tools to a model with a name, argument names, and argument types). Because tools are effectively reads and writes against a database of entities, the grading can check the final state or answer against ground truth; this is the pattern in [[tools-as-database-read-write-for-verifiable-environments]].

A design choice worth noticing: Surge does not hand-design the whole world top-down. They build a framework that enforces consistent relationships and properties, then domain experts ("Surgers") populate it with entities and tasks from their own work experience. The claim is that real complex systems grow rather than being designed, so the environment should grow the same way. The upside is realism and diversity of tasks. The unstated cost is that quality depends on the contributors, and nothing in the post says how tasks were validated or how ground-truth answers were checked.

### The world and the role

Corecraft is a fictional online retailer of PC parts and custom builds. The agent plays a customer-support worker. Tasks range from trivial lookups ("How many refunds were there in July 2025?") to multi-system reasoning — for example, diagnosing why a CPU, motherboard and DDR5 memory order is flagged as incompatible and proposing the cheapest fix.

Why customer support rather than something flashy like research? Two reasons given. First, much of the economic value of AI is likely in ordinary daily work. Second, the role spans many difficulty levels, so it isolates general agent skills rather than domain knowledge. The same logic sits behind [[2406.12045 - Tau-Bench (Tool-Agent-User Interaction)]], which also uses retail/airline service tasks; Corecraft is a larger, messier sibling of that idea. The system prompt also fixes a "current date" (September 15, 2025), which matters later.

## The results

The pass rate is the share of 150 tasks where the full trajectory was judged correct. Values below are read from the bar charts on the page (no table is given, so treat each as ±1 point):

November run: GPT-5 ≈ 57%, Claude Sonnet 4.5 ≈ 47%, Kimi K2 Turbo ≈ 24%, Qwen3-Max ≈ 21%, Gemini 2.5 Pro ≈ 19%, Gemini 2.5 Flash ≈ 16%, Mistral Medium ≈ 11%, GPT-4o ≈ 10%, Nova 1 Pro ≈ 4%.

December update, same tasks: GPT-5.2 ≈ 61%, Claude Opus 4.5 ≈ 54%, Gemini 3 Pro ≈ 40%, Nova 2 Pro ≈ 27%.

Two things stand out. The top two are "in a league of their own": there is a gap of more than 20 points between Claude Sonnet 4.5 and the third model. And even the best model fails about four tasks in ten. The December jump for Gemini (19 → 40) and Nova (4 → 27) is much larger than the jump at the top (57 → 61), which fits the post's own thesis: the lower layers of the hierarchy are easier to fix with training than the top one.

Things the post does not report: number of runs per task, variance, how a "pass" was graded (human, rubric, or exact answer), and task difficulty mix. With 150 tasks, one task is about 0.67 points, so gaps of 2–3 points (Qwen3-Max vs Gemini 2.5 Pro) are within noise; the gaps of 20+ points are not.

## The hierarchy, layer by layer

The hierarchy has four foundation layers and one top layer: tool use, planning and goal formation, adaptability, groundedness, and common-sense reasoning. Surge places models on it by where their failures cluster. They are explicit that it is not strict: layers overlap and develop in parallel, and even strong models still slip on basics sometimes. The point is diagnostic — where is progress solid, and where is foundational work still missing?

### Layer 1–2: tool use, planning, goal formation

To finish even a simple agentic task, a model has to do four things reliably: split the task into mini-objectives, pick the right tool for each and the right order, map the information it has onto the right tool arguments, and carry out the plan without dropping a step. Surge's line for models that cannot: they are "chatbots with access to tools." GPT-4o, Mistral Medium and Nova 1 Pro sit here.

The examples are small, specific failures:

- Task: find gold/platinum-tier customers with open high-priority tickets. Nova 1 Pro called `searchTickets` with `"customer_id": "gold"` — it put a loyalty tier into an ID field. GPT-4o found the right customers but passed "high" into the `status` argument even though a `priority` argument existed. Mistral Medium passed an array into `customer_id`, which the MCP schema declares as a string.
- Task: list customers who ordered a recalled SkyForge X670E Pro in August 2025 with status fulfilled, paid, or pending. The correct plan is two hops: `searchProducts` to get the product ID, then `searchOrders` with that ID. Nova and Mistral skipped hop one and put the product *name* into `product_id`. GPT-4o did both hops but only searched "fulfilled", silently dropping two of three required statuses.

The interesting diagnosis is the mechanism Surge proposes for the skip-a-hop failure: the model seems to pick the single tool that looks like it produces the final answer, then force whatever data it has into the most plausible argument. The correct behavior is the reverse — look at what information you hold, find the tools whose arguments accept it, and chain them. This is a planning failure disguised as a tool failure. It lines up with the argument-hallucination category in [[hallucinated-actions-and-fallback-mapping]] and with the tool/plan families in [[three-family-agent-error-taxonomy]].

### Layer 3: adaptability — when the plan meets reality

Once the model can plan, the world stops cooperating: tools are poorly documented, data is formatted oddly, results come back empty. Adaptability is noticing that a step misfired and changing course. Gemini 2.5 (Flash and Pro) and Qwen3-Max sit here.

The clean example: Penny Whitcomb asks whether two Vortex Labs GPUs are compatible with her previous order and what her price would be. The correct plan is `searchCustomers` (tier and ID) → `searchOrders` (past parts) → `searchProducts` (GPU IDs) → `validateBuildCompatibility`. All three models got the plan right. At step 3 they searched `brand: "Vortex Labs"`, but the database stores `"VortexLabs"` with no space, so the search returned nothing. Surge is fair here: no model could be expected to know the spelling. The failure is what happened next — all three accepted the empty result as truth and told the customer Corecraft does not carry those cards. Claude Sonnet 4.5 hit the same empty result and tried other search parameters until it found the products.

The lesson is subtle. An empty result is ambiguous: "nothing exists" or "I asked wrong." Weak agents treat tool output as ground truth; strong agents treat surprising output as evidence about their own query. This is the same pattern as [[feedback-starvation-problem]] seen from the other side: the feedback was there, the model just did not use it.

### Layer 4: groundedness — staying tied to the context

Groundedness means staying tethered to what the context actually says: no invented IDs, no drifting facts, no answers that contradict the tool results just retrieved. Kimi K2 Turbo planned and adapted better than Qwen and Gemini but failed here often.

- The system prompt's first line says the date is September 15, 2025. Asked for orders from August 25–31, Kimi searched `2024-08-25` to `2024-08-31`, then reported back using 2025 again.
- Claude Sonnet 4.5, needing a customer's details, searched using an obviously fabricated email address. When that failed it recovered — strong adaptability covering for weak grounding.
- Worse, a silent case: Claude retrieved tickets that were clearly marked "normal" priority, then listed two of them as "high priority" in its answer and *also* repeated them in the normal section while saying they were already listed as high. The answer contradicts both the data and itself.

The second and third examples matter most. A fabricated email that fails loudly is cheap, because the tool error triggers recovery. A mislabel that appears only in the final prose has no tool to catch it, so it reaches the user. Surge names groundedness as a main factor separating Claude Sonnet 4.5 from GPT-5 — the 10-point gap is attributed mostly to this layer, not to planning. Compare [[environment-grounding-in-multi-agent-systems]].

### Layer 5: common-sense reasoning — the top

Once a model reliably uses tools, plans, adapts, and stays grounded, what remains is sense-making in unfamiliar situations: the "general" in general intelligence. Surge says this layer was the main thing separating GPT-5 from human-level performance in this run. Three GPT-5 failures, none of which is a tool or planning error:

1. **Missed inference.** Reclassify "other" tickets that should be "returns". One ticket asks for a refund and mentions "the package showed up a few hours ago." A refund request could be a return or a cancellation; the delivered package makes it a return. GPT-5 retrieved the ticket and did not connect the clue.
2. **Coherent but senseless plan.** Find likely "gamers" in August orders; the task even suggests heuristics (GPUs, prebuilts with GPUs, products mentioning gaming). The sensible route is to find gaming products first, then orders containing them. GPT-5 instead searched every August order one day at a time (to stay under the result limit), then guessed from product names which to inspect. Claude did the same. The plan was executed perfectly and was a bad plan.
3. **Misread identity cue.** "My name under my account should be set to Sarah Kim." GPT-5 read this as a request to rename the account rather than as the customer telling you who they are, so it never looked up her loyalty tier and gave generic pricing. The sentence is ambiguous alone, but context resolves it: no other identifier was given, a customer named Sarah Kim exists, and renaming would be irrelevant to a GPU price question.

Case 2 is the most interesting finding in the post. Both top models converged on the same brute-force strategy, which suggests it is a trained habit (exhaustive enumeration feels safe) rather than a one-off lapse. It is also the kind of failure a pass/fail grader might not even catch if the enumeration happened to succeed — an efficiency and judgment problem, not a correctness one.

## The correction: the pyramid is really a pillar

Surge then redraws the figure. The four foundation layers become thin slabs with GPT-5 standing on top of them; common-sense reasoning is a very tall block with a human on top. The message: proficiency in the first four layers is the *entry ticket* to talking about common sense, not most of the way to human-level. 2025 is "the year of agents" only in the sense that agents became coherent enough for their common sense to be measured.

Surge leaves open whether common sense is a set of trainable subskills or something that emerges from large-scale real-world training. That open question is the business case for their product: if it is trainable, you need many realistic environments like Corecraft.

## What to take away, and what to doubt

The useful idea is using failure *type*, not pass rate, as the diagnostic. A model at 20% and a model at 55% are not the same model with more polish; they are failing at different layers, and the fix for each is different (schema-following and planning data for the first, recovery and grounding training for the second). This is how [[evaluation-driven-development-for-agents]] should read its logs.

Genuine limitations:

- The layer placement of each model comes from hand-picked trajectories. There is no count of how many failures fell in each layer, so "Kimi lives at groundedness" is an impression, not a measured distribution.
- One environment, one role, 150 tasks, no reported variance or repeated runs. Pass-rate gaps under about 5 points should not be trusted.
- The grading method is not described, so we cannot tell whether a correct-but-wasteful trajectory (the day-by-day search) counted as a pass.
- Some "common sense" failures are arguably task ambiguity. The Sarah Kim sentence genuinely can be read two ways; Surge's argument that context resolves it is reasonable but it is a judgment call, and a stricter environment would let the agent ask a clarifying question.
- The author sells environments and human data, so the conclusion "you need realistic, expert-grown environments" is also a sales pitch. The examples are still concrete and checkable.

## Key claims

1. **Even the best models fail a large share of realistic agent tasks.** Setup: 9 models, 150 Corecraft customer-support tasks, November 2025. GPT-5 ≈ 57% and Claude Sonnet 4.5 ≈ 47% pass (read from chart).
   > Even GPT-5 and Claude Sonnet 4.5 failed over 40% of agentic tasks in one of our RL environments.

   (Results section, chart caption)

2. **Two models are far ahead of the rest.** Setup: same run; the third model (Kimi K2 Turbo) is ≈ 24%, more than 20 points behind Sonnet 4.5.
   > GPT-5 and Claude Sonnet 4.5 are in a league of their own.

   (Results section)

3. **Newer models improve modestly at the top and a lot lower down.** Setup: December 2025 re-run on the same tasks; GPT-5.2 ≈ 61%, Claude Opus 4.5 ≈ 54%, Gemini 3 Pro ≈ 40% (from ≈ 19% for 2.5 Pro), Nova 2 Pro ≈ 27% (from ≈ 4%).
   > GPT-5.2 and Claude Opus 4.5 show modest improvements over GPT-5 and Claude Sonnet 4.5, while Gemini 3 Pro and Nova 2 Pro make substantial gains over previous models.

   (December Update sidebar, chart caption)

4. **Failures cluster by level of competence, forming a hierarchy.** Setup: qualitative analysis of trajectories from the November run; no per-category counts given.
   > Each model’s struggles tended to cluster around certain levels of competence, revealing a natural hierarchy of skills that agentic models need to become proficient in before they can operate coherently in open-ended environments.

   (The Hierarchy of Agentic Capabilities section)

5. **The weakest models fail at basic tool use and planning.** Setup: GPT-4o, Mistral Medium, Nova 1 Pro (≈ 4–11% pass), e.g. putting "gold" into `customer_id` or a product name into `product_id`.
   > We saw that GPT-4o, Mistral Medium, and Nova 1 Pro lived here.

   (The First Step section)

6. **Mid-tier models plan correctly but do not adapt when a step fails.** Setup: Gemini 2.5 Flash/Pro and Qwen3-Max on the Penny Whitcomb task, where a "Vortex Labs" vs "VortexLabs" mismatch returned empty results.
   > Rather than realizing that there was an issue and trying a different strategy, all three models took the empty results at face value and reported back that those graphics cards were not carried by Corecraft.

   (Adaptability section)

7. **Groundedness is what separates Claude Sonnet 4.5 from GPT-5.** Setup: examples include a fabricated email lookup and mislabelling "normal" tickets as "high" in the final answer.
   > And while Claude Sonnet 4.5 had an impressive performance overall, it still had some notable issues with groundedness, and this was a major factor separating it from GPT-5.

   (Groundedness section)

8. **Kimi K2 Turbo drifts from the stated date.** Setup: system prompt fixes the date as September 15, 2025; Kimi queried 2024 dates.
   > Despite this, Kimi frequently got the year wrong in tool calls.

   (Groundedness section)

9. **Common-sense reasoning is GPT-5's main gap to human level.** Setup: examples include a missed return inference, a day-by-day brute-force order search, and misreading an identity cue.
   > In this run, problems with common-sense reasoning were the main thing separating GPT-5’s performance from human level.

   (Common Sense Reasoning section)

10. **Top models can execute coherent but bad plans.** Setup: the "gamers" task; both GPT-5 and Claude searched all 31 August days one by one.
    > GPT-5 behaved coherently and executed a plan, but it wasn’t a very sensible plan.

    (Common Sense Reasoning section)

11. **Mastering the four foundation layers is not near human-level.** Setup: the authors' redrawn figure and conclusion.
    > In other words, proficiency in the first four areas doesn’t mean a model is at the stage of being a human-level agent that can competently act in the real world.

    (So, Does This Mean GPT-5 Is Near "Human-Level"? section)
