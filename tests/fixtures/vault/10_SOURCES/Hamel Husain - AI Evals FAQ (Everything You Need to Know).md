Created: 15-09-2026 00:59

Tags:

Status: #baby

## References

1. [AI Evals: Everything You Need to Know](https://hamel.dev/blog/posts/evals-faq/)

# Notes

## The problem this document exists to solve

Hamel Husain and Shreya Shankar taught an AI-evals course to 700+ engineers and PMs. This FAQ is the compressed residue of the questions those people actually asked — which means it is not a taxonomy of evaluation invented from first principles, it is a record of where practitioners get stuck. That provenance matters for how you read it. Almost every answer is a correction of a default instinct.

The default instincts it is correcting, roughly in order of how often they appear:

1. "Let me set up eval infrastructure first." → No. Look at data first.
2. "Let me use the helpfulness/coherence metric from the eval library." → No. Those measure nothing about your product.
3. "Let me rate on 1-5 so I can track gradual improvement." → No. Binary.
4. "Let me write evals before I build (TDD for LLMs)." → No. You cannot anticipate LLM failure modes.
5. "Let me outsource the labeling." → No. The labeling *is* the learning.
6. "Let me try a better model." → Not until error analysis says the model is the problem.
7. "Let me buy the eval vendor with the best features." → Features are commoditized; pick on support.

The through-line: **evaluation is a sensemaking activity performed by a human on real traces, and everything else — judges, metrics, CI, tooling — is downstream scaffolding for something a human already discovered.** The whole document is an argument against skipping the discovery step, which is exactly the step every tool and vendor nudges you to skip because it is the one step that cannot be sold to you.

The authors flag their own epistemic status up front: "These are sharp opinions about what works in most cases. They are not universal truths." Published 2025-05-28, last modified 2026-09-01 — it is a living document, and the 2026 revisions visibly added agent-in-the-loop workflow (Claude Code clustering traces, active learning, the evals plugin) that was not in the original.

---

## The two things "evals" means — and why the distinction is load-bearing

**Model benchmarks** compare general-purpose models on shared tasks: GPQA Diamond (graduate science reasoning), Terminal-Bench (agents in command-line environments), MMLU (broad knowledge/reasoning). These are what model providers publish on release day. Their only use to you is as a prior for *which model to start with*.

**Product evals** measure whether *your* system does what *you* want. They cover the whole stack — model, prompts, retrieval, tools, application code — not just the model.

The example that makes the gap concrete: an order-cancellation agent. Its product evals check whether it selected the correct order, and whether it waited for the cancellation tool to *actually succeed* before telling the user the order was cancelled. A high GPQA Diamond score tells you nothing about that, "because those benchmarks don't have access to your systems." That's the whole argument in one line — benchmark scores are measured on a distribution that structurally excludes your product's failure surface.

This is the same boundary [[static-versus-dynamic-agent-evaluation]] draws from a different direction, and it's why [[importance-of-evaluation-in-ai-systems]] treats eval design as a product decision rather than an ML one.

The implementation mechanisms for product evals are only four: **code assertions, human review, LLM judges, and online experiments.** The rest of the document is mostly about choosing among them and about the human work that must precede the choice.

## Trace: the unit of analysis

A trace is "the complete record of all actions, messages, tool calls, and data retrievals from a single initial user query through to the final response" — every step across all agents, tools, and components in a session. Multiple user messages, assistant responses, retrieved documents, intermediate tool calls.

Worth noting the caveat the authors add: **observability vendors do not agree on what "trace" vs "span" means.** Alex Strick van Linschoten catalogued the divergence (as of 2025-07-02). This is a real operational hazard — your "100 traces reviewed" number is not comparable to another team's if your platform slices sessions differently. The FAQ's own definition (session-scoped, from first user query to final response) is the one to hold.

---

## Error analysis: the core method

This is the part of the document everything else hangs off. "Error analysis is the most important activity in evals. Error analysis helps you decide what evals to write in the first place."

The method is lifted, explicitly, from **qualitative research methodology** — grounded theory, essentially. That borrowing is the non-obvious design choice of the whole framework: the authors did not reach for statistics or ML, they reached for the social-science protocol for making sense of unstructured text corpora. The four steps:

### 1. Create a dataset
Gather representative traces. No production traffic yet? Generate synthetic (see below).

### 2. Open coding
A human annotator — ideally a single "benevolent dictator" — reads traces and writes **open-ended free-text notes**, not category labels. The document calls this "journaling."

Two rules here that carry a lot of weight:
- **Annotate at least 30 traces yourself before letting an agent suggest anything.** Rationale is explicitly anti-anchoring: "If the agent starts suggesting problems too early, its guesses can bias your judgment. You may miss failures that depend on product context or your definition of a good user experience." This is a bias argument, not a capability argument — the agent might well be competent, but its suggestions collapse your hypothesis space before you've explored it.
- **Note the first failure observed in a trace**, because upstream errors cause downstream ones. Tagging all independent failures is allowed once you're experienced, but the default is first-failure-only. This avoids inflating counts with cascade artifacts.

A **domain expert** must do this step. Not a general engineer.

### 3. Axial coding
Group the open-ended notes into a **failure taxonomy** — distinct categories — then **count the failures in each category**. The document says flatly: "Axial coding is the most important step." An LLM can help here (unlike step 2).

The counting is what turns qualitative notes into a prioritization signal. Without counts you have anecdotes.

### 4. Iterative refinement
The 2026-era addition. The agent clusters the data and picks a diverse initial sample. After your first 30 annotations, it searches the remaining traces for likely instances of the failures *you described*. You accept or reject each suggestion. Iterate until **theoretical saturation** — the grounded-theory term for "new reviews stop revealing failure modes or changing existing ones."

Working pool: **~100 diverse traces.** Crucially, the agent means you no longer read all 100 sequentially — it focuses your attention on the informative ones. This is a human-agent loop, not automation.

Stopping rule restated elsewhere: "Continue until new traces stop revealing failure modes... We recommend reviewing at least 100 traces. Continue past 100 while you are still learning."

The reason not to skip: error analysis "ensures that the evaluation metrics you develop are supported by real application behaviors instead of counter-productive generic metrics (which most platforms nudge you to use)." Note the causal claim — generic metrics are not merely useless, they are *counter-productive*, because they create false confidence.

---

## The sample-size question, stage by stage

This is the most operationally specific part of the document, and the numbers are worth holding precisely.

| Stage | What you do | How many examples |
|---|---|---|
| 1. Error discovery | Read traces, write failure notes | Start with 100 diverse traces; **annotate ≥30 yourself** before agent help |
| 2. Create + validate evaluators | Code-based or LLM judge | Code-based: Pass + Fail per condition + edge cases. **LLM judge: 100-200 labeled per failure mode** |
| 3. Repeatable eval set | Examples covering workflows + confirmed failures | Often grows to **100+** |

### Why code-based vs LLM judge is a cost decision, not a quality one

Use a **code-based eval** when a deterministic rule identifies the failure: does the JSON parse, does the tool call use correct arguments. Sample-size need is driven by *coverage*, not statistics — at minimum one Pass and one Fail per condition, plus edge cases from error discovery. A single-rule check may need only a handful.

Use an **LLM judge** when the failure requires subjective or domain judgment. Now you need 100-200 labeled examples *per failure mode*, from a trusted domain expert, with enough of both classes.

### The train/dev/test split for judges

This is the detail most people miss and the FAQ is precise about it:

- **10-20%** → train (examples that may appear in the judge prompt as few-shot)
- **40-45%** → dev (used while iterating on the judge prompt)
- **40-45%** → test (reserved for **one final evaluation**)
- Target **30-50 Pass and 30-50 Fail** in both dev and test

The heavy allocation to test (40-45%, versus the ~10-20% typical in ML) is a deliberate non-default choice. It makes sense once you see that judge-prompt iteration is a very high-variance, very low-regularization optimization process — you will overfit the dev set hard, so the held-out estimate has to be large enough to survive that. Note the train share is small precisely because those examples get *consumed into the prompt*, not learned from.

Relevant to [[labelled-dataset-creation]] and the agreement question in [[judge-agreement-against-human-ceiling]].

---

## Judge design decisions

### Binary over Likert

"Why do you recommend binary (pass/fail) evaluations instead of 1-5 ratings?" The engineer's intuition is that a 5-point scale carries more information and lets you track gradual improvement. The counter-argument has three legs:

1. **The gap between adjacent points is undefined.** 3-vs-4 is subjective and inconsistent across annotators, and inconsistent *within* one annotator over time.
2. **Statistical cost.** Detecting differences on an ordinal scale requires larger samples than a proportion test.
3. **Central tendency bias.** Annotators default to middle values to avoid committing. Binary "forces people to make a decision rather than hiding uncertainty in middle values."

Plus a speed argument: "you don't waste time debating whether something is a 3 or 4."

The replacement for gradations is **decomposition**: instead of rating factual accuracy 1-5, track "4 out of 5 expected facts included" as five separate binary checks. You keep resolution and gain objectivity. This is the right move and it's under-appreciated — a Likert score is a lossy scalar projection of what is actually a vector of independent binary conditions, and the projection is where the inter-annotator noise enters.

"Start with binary labels to understand what 'bad' looks like. Numeric labels are advanced and usually not necessary."

### Same model for task and judge — usually fine

Self-preference bias is documented in the literature, but the authors' position is that it doesn't dominate here **because the judge is doing a different task from the pipeline**: a scoped binary classification, not open-ended generation. What actually matters is not theoretical bias but measured alignment with human labels.

The metrics to optimize: **True Positive Rate (TPR) and True Negative Rate (TNR) on a held-out labeled test set.** Both, separately — not accuracy, because failure classes are usually imbalanced and accuracy would let a degenerate all-Pass judge look good.

If alignment is bad, *then* try a different model. The reason not to default to a different model is organizational, not scientific: "onboarding new model providers may involve non-trivial effort in some organizations."

Start with the most capable model available to establish alignment; optimize cost afterward. This is the right order — you want to know whether the task is learnable at all before you find out whether it's learnable cheaply.

Connects to [[llm-as-judge-evaluation-pattern]], [[llm-as-automated-evaluator-with-validation]], and the failure taxonomy in [[llm-verifier-exploitation-taxonomy]]. The self-preference concern is the mirror of [[llm-judge-position-bias]] — both are systematic judge artifacts that only surface if you validate against humans.

### Judges let you correct your failure-rate estimate

Buried in the RAG section but general: "Once you know your judge's True Positive and True Negative rates, you can correct its estimates to determine the actual failure rate in your system."

This is the key quantitative payoff of judge validation and it deserves spelling out. If your judge flags a fraction $\hat{p}$ of production traces as failures, and the judge has true positive rate $\text{TPR}$ and false positive rate $\text{FPR} = 1 - \text{TNR}$, then the observed rate relates to the true failure rate $p$ by:

$$\hat{p} = p \cdot \text{TPR} + (1-p) \cdot \text{FPR}$$

Solving for the quantity you actually want:

$$p = \frac{\hat{p} - \text{FPR}}{\text{TPR} - \text{FPR}}$$

Reading the terms: $p \cdot \text{TPR}$ is the real failures the judge caught; $(1-p)\cdot\text{FPR}$ is the good traces the judge wrongly condemned. The denominator $\text{TPR} - \text{FPR}$ is **Youden's J**, the judge's discriminative power. As it approaches zero the judge is no better than a coin flip and the correction blows up — which is exactly the right behaviour, since a non-discriminative judge genuinely carries no information about $p$. Without validation you have no TPR/FPR and cannot perform this correction at all, so you are stuck reporting $\hat{p}$ as if it were $p$. That is the concrete cost of skipping validation: your headline quality number is biased by an unknown amount in an unknown direction.

### Generic metrics and similarity metrics

Flat no on ready-to-use metrics as *quality measures*: helpfulness, coherence, quality. "Generic evaluations waste time and create false confidence when you use them as quality measures."

BERTScore, ROUGE, cosine similarity — "not useful for evaluating LLM outputs in most AI applications." The real-estate CRM example: suggesting showings that aren't available (catchable by an assertion), confusing client personas (catchable by an LLM judge). Similarity and verbosity catch neither.

The quoted line from the course: *"The abuse of generic metrics is endemic. Many eval vendors promote off the shelf metrics, which ensnare engineers into superfluous tasks."*

But — and this is the nuance most summaries drop — **generic metrics are legitimate as exploration signals.** They can surface traces worth human review even though they don't measure quality. "Experienced practitioners may use generic metrics as exploration signals. Once you understand why they fail as quality measures, you can use them to find interesting traces."

And similarity metrics retain genuine utility in their native domain: search and recommendation, hence RAG retrieval debugging. Cosine similarity measures semantic closeness in retrieval; average pairwise similarity measures output diversity (lower similarity = higher diversity).

### Evaluating abstention / calibrated refusal

To test whether a model "knows what it doesn't know," you need a set with both:
- **Answerable questions** — a verifiable answer exists in context or general knowledge
- **Unanswerable questions** — designed to tempt hallucination: false premises, information explicitly missing from context, topics far outside the knowledge base

Roughly balanced, but "the diversity and difficulty of the questions are more important than the precise ratio."

The eval is binary and **two-sided**: Pass requires answering the answerable *and* refusing the unanswerable. This bidirectionality is the whole point — a model that refuses everything scores perfectly on the hallucination half and is useless. Failure = fabricating an answer to an unanswerable question.

Literature term: **"Abstention Ability."** Worth searching arXiv for.

---

## Sampling: how to find traces worth looking at

The sampling table, ordered **most exploratory → most targeted**:

| Method | What it does | Main limitation |
|---|---|---|
| Random | Equal probability | Small batch misses rare cases |
| Clustering | Group by similar content, sample each group | Depends on feature/clustering choices |
| Data analysis | Review extremes (latency, tool count) | An extreme value may be unrelated to quality |
| Classification | Evaluator/model flags likely failures | Favors problems the classifier already knows |
| Feedback | Negative user feedback | Misses problems users don't report |

Two rules on top of the table:

1. **Start exploratory, shift targeted as you learn.** Early on you don't know what you're looking for, so signals can only mislead you.
2. **Keep some random traces in every batch, always.** "This gives you a chance to find failure modes that your current signals do not describe." This is the exploration term that prevents the whole loop from collapsing into a self-confirming search for known failures — the central failure mode of any signal-driven sampling scheme.

For **rare failures**: targeted sampling on correlated signals — a specific tool sequence, unusually long traces, retries, a known input pattern.

**Active learning.** Borrowed explicitly from ML: the system asks a human to label the points most useful for its next update. In Shankar's walkthrough, Claude Code clusters traces and picks examples per cluster; a `monitor` command watches `annotations.json` for new labels; on each new label the agent updates the failure taxonomy and searches for similar cases or different failures. The authors note this generalizes — active learning applies anywhere you're annotating, not just error analysis. Related: [[convincing-wrong-answer-active-learning]].

Beyond user feedback, three complementary approaches: random sampling (escalating to **stress testing** — deliberately crafted queries that test your prompt constraints — if random turns up little); using existing evals as a screen; and outlier detection / metric sorting / stratified sampling.

**Cadence.** Re-run full error analysis on significant changes: new features, prompt updates, model switches, major bug fixes. Heuristic: **100+ fresh traces per review cycle, cycles typically 2-4 weeks.** Between full analyses, **10-20 traces weekly** focused on outliers — unusually long conversations, sessions with multiple retries, monitoring-flagged traces. New systems: weekly until patterns stabilize. Mature systems: monthly. Always after incidents, complaint spikes, or metric drift.

---

## Synthetic data: structured generation, and where it lies to you

### The dimension-tuple method

The failure mode is prompting "give me test queries" and getting generic repetitive output. The fix is structure.

**Define dimensions** — categories describing types of variation in user behavior.
- Recipe app: Dietary Restriction (vegan / gluten-free / none), Cuisine Type (Italian / Asian / comfort food), Query Complexity (simple / multi-step / edge case)
- Support bot: Issue Type (billing / technical / general), Customer Mood (frustrated / neutral / happy), Prior Context (new / follow-up / resolved)

Choose dimensions from **failure hypotheses**, not from a neutral description of your domain. If you lack intuition, use the product extensively or recruit friends first.

**Write 20 tuples by hand.** A tuple picks one value per dimension: (Vegan, Italian, Multi-step). The manual work is for *your* understanding of the problem space, not for the data.

**Two-step generation** is the key mechanical trick:
1. LLM generates more structured tuples — (Gluten-free, Asian, Simple)
2. A **separate prompt** converts each tuple to natural language

Separating these avoids repetitive phrasing. (Vegan, Italian, Multi-step) → *"I need a dairy-free lasagna recipe that I can prep the day before."* Generating queries directly in one shot collapses toward a template because the model conditions on its own previous phrasings; splitting the steps breaks that conditioning.

**Two ways to make tuples:**
- **Cross product then filter** — generate all combinations, filter invalid ones with an LLM. Guarantees coverage including edge cases. Use when most combinations are valid.
- **Direct LLM generation** — more realistic combinations, but tends generic and misses rare scenarios. Use when many combinations are invalid.

That trade-off is the interesting one: exhaustive enumeration buys you the rare corners precisely because it doesn't know which are plausible, while LLM generation buys realism by regressing to the modal case. You pick based on the validity density of your dimension space.

**Fix obvious problems first.** "Don't generate synthetic data for issues you can fix immediately. If your prompt doesn't mention dietary restrictions, fix the prompt rather than generating specialized test queries."

Then run the synthetic queries through the **actual system** to capture full traces — ~100 diverse traces, agent-assisted sampling, ≥30 annotated by you.

### Where synthetic data fails

Genuine limitations, named explicitly:

- **Complex domain-specific content** — legal filings, medical records, technical forms. LLMs miss structure, nuance, quirks; critical edge cases go missing.
- **Low-resource languages or dialects** — generated samples are often unrealistic, so evals built on them don't reflect actual performance.
- **When validation is impossible** — if you can't verify realism (domain complexity, no ground truth), you need real data.
- **High-stakes domains** — medicine, law, emergency response. Synthetic data "lacks subtlety and edge cases," errors are consequential, and manual validation is hard.
- **Underrepresented user groups** — the model may misrepresent context, values, challenges. "Synthetic data can reinforce biases in the training data of the LLM."

And one structural limitation that holds even when synthetic data is good: **it cannot tell you how common a failure is in production.** Frequency is a property of the real query distribution; you've replaced that distribution with one you designed. Synthetic data gives you existence proofs, never rates.

Compare with real data as soon as real data exists.

---

## The human process

### Benevolent dictator

For most small-to-medium companies: **one domain expert as the definitive voice on quality.** A psychologist for a mental-health chatbot; a lawyer for legal document analysis.

The argument is about decision latency and consistency, not correctness: a single expert "eliminates annotation conflicts and prevents the paralysis that comes from 'too many cooks in the kitchen'." They can take input from others but they drive.

The diagnostic buried in there is sharp: **"If you feel like you need five subject matter experts to judge a single interaction, it's a sign your product scope might be too broad."** Annotation difficulty is being read as a signal about product scope, not about staffing. That's a genuinely non-obvious inversion.

Multiple annotators are warranted in larger orgs or across genuinely different domains (multinational, different cultural contexts). Then you must measure agreement with a **chance-corrected metric — Cohen's Kappa**. See [[krippendorfs-alpha]] for the generalization to >2 annotators and arbitrary scales.

### Disagreement resolution protocol

1. Label the same examples **independently, before discussing** (prevents anchoring — same principle as not letting the agent suggest failures before your first 30)
2. Measure agreement; collect the disagreement cases
3. Alignment session: ask *which part of the rubric* caused the disagreement, and *what rule* would make the next decision clear
4. Update the rubric with a definition, rule, or example covering the disputed case
5. Relabel affected examples
6. Still disagreeing? A domain expert makes the final call **and records the reason**

The framing here is worth noting: disagreement is treated as **a defect in the rubric, not a defect in the annotators.** Every resolution step pushes the fix back into the written criteria. That's what makes the process converge instead of relitigating the same judgment forever.

### PM + engineer collaboration

At the outset, collaborate to build shared context. **Engineers** catch technical issues — retrieval problems, tool errors. **PMs** catch product failures — unmet expectations, confusing responses, missing features.

Over time, converge on a benevolent dictator: a domain expert or PM who understands user needs.

The critical instruction: **have domain experts evaluate outcomes, not implementation.** Ask *"Has an appointment been made?"* not *"Did the tool call succeed?"* Same distinction as [[outcome-driven-verification-in-agent-evaluation]]. The enabler is tooling — custom annotation tools that display system outcomes alongside traces: the confirmation, the generated email, the database update that proves goal completion. **All context on one screen** so non-technical reviewers stay focused on results.

### Don't outsource annotation

"Outsourcing error analysis is usually a big mistake." The reason is not quality control — it's that **"the core of evaluation is building the product intuition that only comes from systematically analyzing your system's failures."** Outsourcing breaks the feedback loop between observing a failure and knowing how to fix it. You'd be paying someone to acquire the understanding you needed.

Three concrete problems:
- **Superficial labeling** — external teams lack the nuanced judgment even well-defined metrics require. "A critical misstep in error analysis is excluding domain experts from the labeling process."
- **Loss of unspoken knowledge** — the expert's tacit knowledge can't be fully captured in a rubric. Involving them *uncovers preferences they couldn't articulate upfront.* (This is the same phenomenon as criteria drift, below.)
- **Annotation conflicts** — without shared context, external annotators create more disagreement than they resolve.

**Capacity strategies that aren't outsourcing:**
- **Smart sampling** — "more effective to analyze 100 diverse traces to find patterns than to superficially label thousands"
- **Think-aloud protocol** — borrowed from usability testing. The expert verbalizes their reasoning while reviewing a handful of traces. "This method can uncover deep insights in a single one-hour session." The point is that it captures the *reasoning*, which a label alone discards.
- **Lightweight custom tools** to raise throughput

**Legitimate exceptions:**
- Purely mechanical tasks — identifying a phone number, validating an email — *after* a rigorous internal process defined the rubric
- Tasks without product context — translation needs linguistic expertise, not product knowledge
- **Hiring external SMEs as your internal domain experts is not outsourcing.** Example: AnkiHub hired 4th-year medical students to evaluate their medical RAG system, rather than sending it to generic annotators.

### What LLMs can and cannot do in the eval loop

**Can (with oversight):**
- **First-pass axial coding** — after *you* open-code 30-50 traces, the LLM organizes your raw notes into proposed groupings. Always review and refine.
- **Mapping annotations to failure modes** — "Given this annotation: [open_annotation] and these failure modes: [list_of_failure_modes], which apply?"
- **Suggesting prompt improvements** for recurring problems (review before adopting)
- **Analyzing annotation data** — finding patterns like "reports of lag increase 3x during peak usage hours" or "slow response times mostly from mobile users"

**Cannot / must not:**
- **Initial open coding** — "Always read through the raw traces yourself at the start... Never skip this or delegate it."
- **Validating failure taxonomies** — the worked example: an LLM might group "app crashes after login" and "login takes too long" under one "login issues" category. One is a stability problem, one is a performance problem; they need different fixes. The LLM's clustering is driven by surface similarity, and surface similarity is not the same partition as *cause* or *remedy*. That is the structural reason this step can't be delegated, not a contingent capability gap.
- **Ground truth labeling** — anything used to test/validate an LLM judge must be hand-validated. Otherwise your benchmark inherits the errors of the thing it's benchmarking.
- **Root cause analysis** — only human review catches patterns like "bugs that happen only when users paste data from Excel."

The load-bearing illustration for why: if users keep complaining about "lag" but the LLM lumps them into generic "performance issues," you never create a latency category and never prioritize the fix. **The categories an LLM proposes are drawn from its priors about what categories exist, and your product's real failure modes are, by definition, not in those priors.**

Summary line: **"Use LLMs to scale what you've learned, not to avoid looking at data."**

### Don't auto-optimize prompts (early)

Two arguments, and the second is the deeper one.

**First:** writing a prompt forces you to clarify assumptions and externalize requirements. "Good writing is good thinking" (citing Paul Graham, *Writes and Write-Nots*). Delegate too early and you never understand your own requirements.

**Second, and more fundamental:** automated prompt optimization **hill-climbs a predefined metric.** It refines against known failures. It **cannot discover new ones** — discovery requires error analysis. And the metric it climbs is not stable: research shows **evaluation criteria shift after reviewing model outputs**, a phenomenon Shankar et al. named **"criteria drift"** (*Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences*).

Criteria drift is the theoretical keystone of the whole document. If your evaluation criteria change as a consequence of looking at outputs, then evaluation is not a fixed objective function — it is an iterative, human-driven sensemaking process, and any scheme that freezes the objective and hands it to an optimizer is optimizing a target that was never correct. That's also the formal reason eval-driven development can't work (below), why generic metrics fail, and why the human must do open coding. All four positions are the same claim.

Pragmatic middle: use LLMs to improve prompts **based on your open coding** — human in the loop, still looking at data. Once you have high-quality evals, prompt optimization is effective for the last mile.

---

## Design choices that follow

### Eval-driven development: generally no

Writing evaluators before implementing features "sounds appealing but creates more problems than it solves." The disanalogy with TDD: "Unlike traditional software where failure modes are predictable, LLMs have infinite surface area for potential failures. You can't anticipate what will break."

TDD works because in deterministic software the specification *is* the failure taxonomy — you know the ways the function can be wrong because you defined what right means. For an LLM system the failure taxonomy is an empirical object you discover from traces. Writing the test first means writing a test for an imagined failure, which is how you get metrics with no impact on actual quality.

Instead: error analysis first, write evaluators for errors you **discover**, not errors you **imagine**.

**Exception:** specific constraints where you know exactly what success looks like. "Never mention competitors" — write that evaluator early. The tell is that this is a *policy* constraint, externally specified, not a behavioral failure mode you'd have to discover.

Always cost-benefit before implementing any eval.

### Don't automate every failure mode

**Fix prompt gaps first.** "Many teams discover their LLM doesn't meet preferences they never actually specified — like wanting short responses, specific formatting, or step-by-step reasoning." Fix those before building evaluation infrastructure. You do not need an eval for a requirement you forgot to write down.

**The cost hierarchy:**
- Simple assertions + reference-based checks — cheap to build and maintain
- LLM-as-Judge — **100+ labeled examples, ongoing weekly maintenance, and coordination between developers, PMs, and domain experts**

That third cost is the one teams forget. The judge isn't a piece of code you ship; it's a standing organizational commitment.

So: **only build expensive evaluators for problems you'll iterate on repeatedly.** Save LLM judges for **persistent generalization failures** — not anything trivially fixable. Start with regex, structural validation, execution tests. Reserve judges for subjective qualities rules can't capture.

### Budget

"Evaluation is part of the development process rather than a distinct line item, similar to how debugging is part of software development." You don't budget for debugging separately.

The number: **"In the projects we've worked on, we've spent 60-80% of our development time on error analysis and evaluation."** Most of that going to *understanding failures* — looking at data — not to building automated checks.

And the counter-intuitive target: **"Be wary of optimizing for high eval pass rates. If you're passing 100% of your evals, you're likely not challenging your system enough. A 70% pass rate might indicate a more meaningful evaluation."** A pass rate near 1.0 has no dynamic range left — it cannot register a regression or an improvement. Same principle as [[metrics-need-dynamic-range-at-the-failure-end]].

### Model selection: not the first lever

"Many developers fixate on model selection as the primary way to improve their LLM applications." Hamel's line from office hours: *"I suggest not thinking of switching model as the main axes of how to improve your system off the bat without evidence. Does error analysis suggest that your model is the problem?"*

### Making the case internally

"Don't try to sell your team on 'evals'. Instead, show them what you find when you look at the data."

Do the error analysis yourself on 50-100 real conversations. Then present: top failure modes; metrics on how often high-impact errors happen; **surprising ways users are actually interacting with the product**; bugs found and fixed, framed as "prevented production issues."

Keep a running log — error caught, what you learned, the fix, impact avoided. Share weekly/monthly. **"We caught 47 issues before users saw them"** beats an abstract pitch about evals.

"Don't just show dashboards and metrics; tell the story of what you're finding in the data." When you fix something, show that specific error rate go down. "Let results instead of methods lead the conversation."

Analogy offered: classic ML projects, where outcomes are speculative and progress is bounded by experiment iteration — so sharing per-experiment learnings *is* the progress report.

### Will this be obsolete in 5-10 years?

"Yes" [it will still be relevant]. Even with perfect models you must verify they're solving the right problem. Prompt-engineering tricks may die; understanding failure modes won't. And: "a LLM cannot read your mind, and research shows that people need to observe the LLM's behavior in order to properly externalize their requirements" — criteria drift again, now as a claim about the permanence of the method.

Both sides linked: "The model is the product" vs "The model is NOT the product."

---

## Tooling

### Build the annotation tool

"Build a custom annotation tool. This is the single most impactful investment you can make for your AI evaluation workflow." With Cursor/Lovable you can build one in hours. Claim: **"teams with custom annotation tools iterate ~10x faster."**

Why custom wins:
- All context from multiple systems **in one place**
- Product-specific rendering — images, widgets, markdown, buttons
- Workflow-specific affordances — custom filters, sorting, progress bars

Off-the-shelf is justified mainly when coordinating dozens of distributed annotators with enterprise access controls — "even then, many teams find the configuration overhead and limitations aren't worth it."

Example: Isaac's Anki flashcard annotation app, handling **400+ results per query** with keyboard navigation and domain-specific criteria "that would be nearly impossible to configure in a generic tool."

### What a good review interface has

Explicitly a menu, not a checklist — "I rarely implement all these features in a single app."

1. **Render traces intuitively, not generically.** Emails should look like emails; code gets syntax highlighting. Full trace available (user input, tool calls, LLM reasoning) but less important details collapsed.
2. **Progress + keyboard navigation.** Progress indicators ("Trace 45 of 100") bound the session and encourage completion. Hotkeys (N for next, label keys, quick save). The stated goal is keeping reviewers **in flow** — this is a human-throughput optimization, and given that annotation is the bottleneck for the entire method, it's a rational place to spend engineering time.
3. **Clustering, filtering, search.** Metadata filters, keyword search, semantic search for conceptually similar problems, clustering (e.g. by user persona) to spot recurring issues.
4. **Prioritize suspect traces.** Surface guardrail-flagged, CI-failed, evaluator-flagged traces. Buttons to add to datasets, file bugs, re-run pipeline tests. Show pipeline version, eval scores, reviewer info inline to kill context switching.

**General principle: keep it minimal.** Only add what pays for its complexity.

### Gaps vendors don't fill

Table stakes are handled: trace logging, metric tracking, prompt playgrounds, annotation queues. Four gaps you'll fill yourself:

1. **Error analysis and pattern discovery** — automatic clustering of similar issues. If several traces show casual language for luxury clients, something must recognize the broader "persona-tone mismatch" pattern. Build AI-suggested groupings, taxonomy rewriting, semantic search for similar cases.
2. **AI assistance throughout the workflow** — categorizing open observations into coherent failure modes ("wrong tone for investor" + "too casual for luxury buyer" → one category); proposing fixes (analyze the 20 cases where pet policies were omitted, suggest prompt modifications; draft SQL-instruction refinements when WHERE clauses go missing); data analysis of annotations in AI-in-the-loop notebooks like Julius or Hex, yielding findings like "location ambiguity errors spike 3x when users mention neighborhood names."
3. **Custom evaluators over generic metrics** — "Be prepared to build most of your evaluators from scratch."
4. **APIs that support custom annotation apps** — real bulk export, and **APIs that write annotations back efficiently.** "You shouldn't have to paginate through thousands of requests or handle timeout-prone endpoints just to get your data."

"Watch for vendors addressing these gaps: it's a strong signal they understand practitioner needs."

### Vendors

Refuses to rank. "It would be futile to compare their features. If I tried to do such an analysis, it would be invalidated in a week!" Most-encountered: **Langsmith, Arize, Braintrust.** Selection criterion in client work: **support quality, not features** — "it's mainly the human factor that matters, and dare I say, vibes." He builds custom tools on top of all of them anyway.

### Prompt versioning

Tension: prompts near the code vs. an environment non-technical stakeholders can reach.

Preference: **store prompts in Git** — versioned, reviewed, deployed atomically with application code. The accessibility objection is answered empirically: at GitHub he worked with lawyers and accountants who used the GitHub web interface and Desktop app effectively.

Why vendor prompt-management tools fall short: AI products have many moving parts (tools, RAG, agents), and **prompt tools can't easily execute your application's code.** Even when they can there's indirection, making it hard to test a prompt against your system's real capabilities.

**Notebooks are the recommended experimentation surface** when you have Python entry points: iterate on prompts against your actual agents with full tool and RAG capabilities, build widgets and small UIs inline. Teresa Torres's walkthrough shows a PM running the entire eval and experimentation lifecycle in notebooks. If notebooks aren't feasible, use an integrated prompt environment — but version in Git either way.

### System vs user prompt

"Nothing beats experimentation. Test both approaches (ideally with evals)... Models handle system and user prompts differently, and these differences vary by provider and model version."

Guidelines:
- **System prompt = the model's constitution.** Static instructions, role definitions, identity, behavioral constraints, output format requirements, standing instructions. *"You are a medical assistant. Never provide diagnoses. Always recommend consulting a healthcare provider."*
- **User prompt** = the actual task, relevant context, few-shot examples, data to process. Documents for analysis, query-specific variation.
- **When unclear, prefer the user prompt** — "more portable across models and easier to debug."

---

## Production

### CI vs monitoring

Two different jobs.

**CI evals** protect against **known regressions before deployment.**
- Small, purpose-built datasets — "in many cases 100+ examples"
- Cover core features, regression tests for past bugs, known edge cases
- Run frequently, so per-test cost matters → curate carefully
- **Favor assertions and deterministic checks over LLM judges**

**Online monitoring** finds failures in production traffic and **estimates how often they occur.**
- Sample live traces, run evaluators **asynchronously**
- You usually lack reference outputs → lean on **reference-free evaluators like LLM-as-judge**
- **Track confidence intervals.** "If the lower bound crosses your threshold, investigate further." Production rates are estimates from samples; a point estimate crossing a threshold is meaningless without knowing whether the sample could produce that by chance.

**Connect them:** when production monitoring + error analysis reveals a new failure pattern, **add representative examples to the CI dataset.** That's the ratchet — every production discovery becomes a permanent regression test. Without it, CI only ever tests failures you found before launch.

### Guardrails vs evaluators

Genuinely different objects, not two settings of one thing.

**Guardrails** sit **inline in the request/response path**, validating inputs or outputs before anything reaches a user:
- Fast and deterministic — "typically a few milliseconds of latency budget"
- Simple and explainable — regexes, keyword block-lists, schema/type validators, lightweight classifiers
- Targeted at **clear-cut, high-impact** failures — PII leaks, profanity, disallowed instructions, SQL injection, malformed JSON, invalid code syntax

On trigger: redact, refuse, or regenerate. Because triggers are **user-visible, false positives are treated as production bugs.** Hence: version guardrail rules, log every trigger, monitor rates, keep them conservative.

**Evaluators** run **after** the response, measuring what rules can't — factual correctness, completeness. Verdicts feed dashboards, regression tests, improvement loops. **They do not block the answer.** Usually async or batch, which is what affords heavy computation like a judge. Inline judge use is possible only if latency and reliability budgets allow — plausibly in a **cascade that runs only on borderline cases.**

Caution: **"Do not use llm guardrails off the shelf blindly. Always look at the prompt."** Same objection as generic metrics — someone else's notion of unsafe is not yours. See [[jailbreak-judge-as-load-bearing-metric]] for what goes wrong when an unvalidated judge becomes the number a safety claim rests on.

### Can evaluators auto-fix in production?

"Yes, but only a specific subset." The question reduces to whether the evaluator can function as a guardrail. Two criteria:

1. **Latency & cost** — can it run in the critical path without degrading UX?
2. **Error-rate trade-off** — false positives (blocking good outputs, frustrating users) vs false negatives (bad outputs reach users, cause harm). **The asymmetry is domain-dependent**: in medical advice, false negatives are more costly; in creative applications, false positives that block legitimate creativity may be worse.

Most guardrails are fast with very low FPR, so "you would almost never use a slow or non-deterministic LLM-as-Judge as a synchronous guardrail." But the authors leave the door open: "these tradeoffs might be different for your use case."

---

## Domain-specific

### "Is RAG dead?"

The viral article argues against **naive vector-database retrieval for autonomous coding agents** — not against retrieval. That's the distinction being lost to marketing.

RAG means retrieval-augmented generation: using retrieval to get relevant context into the model. "The question isn't whether to use retrieval, but how to retrieve effectively."

For code specifically, naive vector similarity often fails because **code relationships are complex and contextual.** Modern coding assistants like Claude Code still retrieve — they use **agentic search** rather than relying solely on vector DBs, "similar to how human developers work."

The strategy space runs from keyword matching → embedding similarity → LLM-powered relevance filtering, with hybrids and multi-hop LLM-guided retrieval in production systems. Connects to [[agentic-rag-approach]], [[retrieval-augmented-generation]], [[self-reflective-retrieval]], and the framing in [[rag-versus-long-context-trade-offs]].

Meta-point: "'RAG' has become a buzzword with no shared definition." Some mean any retrieval; some restrict it to vector DBs. Focus on the goal — getting the LLM the context it needs — and treat the mechanism as a product/engineering decision. **The failure mode here is arguing about a word.**

### Evaluating RAG

Two components, two evaluation approaches.

**Retrieval is a search problem** → traditional IR metrics:
- **Recall@k** — of all relevant documents, how many appeared in the top k
- **Precision@k** — of the k retrieved, how many were relevant
- **MRR** — how high up was the first relevant document

To build the retrieval eval set **without manual annotation**, run the generation process backwards: take documents from your corpus → extract key facts → generate the questions those facts would answer. You now have query-document pairs with known ground truth for free. This works because the document→question direction is well-posed (a fact determines a question) while the question→document direction is the thing you're trying to measure.

**Generation** → error analysis, human labels, targeted judges, validate the judges.

Maps onto Jason Liu's "There Are Only 6 RAG Evals":
- **Tier 1** — traditional IR metrics for retrieval
- **Tiers 2 & 3** — relationships among Question, Context, Answer:
  - **C|Q** — is the context relevant to the question
  - **A|C** — is the answer faithful to the context
  - **A|Q** — does the answer address the question

Beyond the six, error analysis on **your** data reveals domain-specific failure modes: a medical RAG consistently failing to distinguish adult vs. pediatric drug dosages; a legal RAG confusing jurisdictional boundaries. "These patterns emerge only through systematic review of actual failures."

And do not take Tier 2/3 judge prompts off the shelf — run the full validation process (error analysis, prompt iteration, labeled examples, accuracy against human labels), then use TPR/TNR to correct the estimates.

Order of operations: **debug retrieval first with IR metrics, then generation quality with validated judges.** Retrieval failures are cheap to diagnose and they poison generation, so fixing them first prevents you from chasing generation bugs that are really retrieval bugs.

### Chunk size for document processing

Important framing: this is **not RAG.** In RAG, chunks are optimized for *retrieval*. In document processing, **the model will see every chunk** — the goal is splitting so the model can reason without being overwhelmed.

Key non-obvious point: **even if the document fits in the context window, breaking it up may still be better.** Long inputs degrade performance through attention bottlenecks, "especially in the middle of the context."

**Fixed-output tasks → large chunks.** Output length doesn't grow with input: extract a number, answer a specific question, classify a section. *"What's the penalty clause in this contract?"* / *"What was the CEO's salary in 2023?"* Use the largest chunk that likely contains the answer — fewer queries, no context fragmentation. Caveats: avoid irrelevant text (models are distractible, especially at length); middle content is under-attended; if cost/latency binds, pre-filter with keyword search or a lightweight retriever before feeding a huge chunk.

**Expansive-output tasks → small chunks.** Output grows with input: summarization, exhaustive extraction. *"Summarize each section"* / *"List all customer complaints."* Process each chunk independently then aggregate (map-reduce). Respect content boundaries — paragraphs, sections, chapters. Bonus: chunking also sidesteps **output token limits**, since each piece's output stays bounded.

The mechanism, stated plainly: "A larger chunk means the model has to reason over more information in one go — essentially, a heavier cognitive load... If too much is packed in, the model might prioritize certain parts (commonly the beginning or end) and overlook or 'forget' details in the middle." **You are trading off global context for local focus.**

"No rule of thumb can perfectly determine the best chunk size... I treat chunk size as a hyperparameter to tune."

### Debugging multi-turn conversations

**Start simple.** Pass/fail on whether the *whole conversation* met the user's goal. Look at the entire trace, focus on the **first upstream failure**. Read the **user-visible parts first**; only then dig into tool calls and intermediate steps. This ordering matters — starting in the tool traces means debugging things that may be irrelevant to what the user experienced.

**Multi-agent logging:** assign a session/trace ID per user request; log every message with its source (which agent or tool), trace ID, and **position in the sequence**. That's what lets you reconstruct the full path.

**Annotation:** first failure only, at first. Downstream failures cascade. With experience, annotate independent failure modes in the same trace to go faster.

**Simplify to reproduce.** The example: a shopping bot gives the wrong return policy on turn 4. Before grappling with multi-turn complexity, collapse it to one turn — *"What is the return window for product X1000?"* If it still fails, **you've proven the error isn't about conversation context** — it's a retrieval or knowledge issue, far easier to debug. This is bisection applied to conversational state, and it's the highest-leverage move in the section: "Not every multi-turn failure requires multi-turn analysis."

**Test case generation, two approaches:**
- **LLM user simulation** — another LLM plays the user to create realistic multi-turn conversations
- **N-1 testing** — supply the first N-1 turns of a **real** conversation and test what happens next

"The N-1 approach often works better since it uses actual conversation prefixes rather than fully synthetic interactions, but is less flexible." The trade-off is exactly the synthetic-data trade-off again: real prefixes are on-distribution but you only have the ones that happened; simulated users cover more but drift off-distribution.

### Human handoffs

**The trace does not end at the handoff.** It "continues until the user's need is resolved or the session ends, not when AI hands off to a human."

Log: the handoff decision, **why** it occurred, context transferred, wait time, human actions, final resolution, and **whether the human had sufficient context.**

"Many failures occur at handoff boundaries where AI hands off too early, too late, or without proper context." Evaluate handoffs as failure modes during error analysis: Was the handoff necessary? Did the AI provide adequate context?

Track **both handoff quality and handoff rate** — "Sometimes the best improvement reduces handoffs entirely rather than improving handoff execution." Optimizing handoff quality alone would miss that the handoff shouldn't have happened.

### Multi-step workflows

Log **from initial trigger to final business outcome** — LLM calls, tool usage, human approvals, database writes.

**Two metric families:**
- **Outcome metrics** — did the final result meet requirements? Complete? Accurate? Properly formatted?
- **Process metrics** — efficiency: step count, time taken, resource usage

"Process failures are often easier to debug since they're more deterministic, so tackle them first." Deterministic-first is the recurring heuristic across this whole document.

**Segment error analysis by workflow stage.** Early (understanding user input) ≠ middle (data processing) ≠ late (formatting output). **"Early stage improvements have more impact since errors cascade in LLM chains."**

### Agentic workflows — two phases

**Phase 1: End-to-end task success.** Treat the agent as a black box: did it meet the user's goal? Define a precise success rule per task, measure with human review or validated judges, record the **first upstream failure** during error analysis.

Only once error analysis says which workflows fail most do you move to:

**Phase 2: Step-level diagnostics.** Score individual components:
- **Tool choice** — did it select the appropriate tool
- **Parameter extraction** — were inputs complete and well-formed
- **Error handling** — how did it handle empty results or API failures
- **Context retention** — did it preserve earlier constraints
- **Efficiency** — steps, seconds, tokens
- **Goal checkpoints** — key milestones in long workflows

The ordering is the point: step-level metrics are cheap to produce and easy to over-collect, and most of them won't correlate with task success. End-to-end first tells you which steps are worth instrumenting.

**Testing tool calls — four separate checks:** tool **name**, **arguments**, **result**, and **resulting state.** Use code assertions where behavior is objective. The full example: verify the agent selected `cancel_order`, passed the correct order ID, received a successful response, **and changed the order status** before it told the user cancellation succeeded.

That last conjunct is the interesting one. The common bug isn't a wrong tool call — it's a **correct-looking call whose effect the agent never verified**, followed by a confident report to the user. Checking the resulting state is what catches it. Compare [[execution-based-tool-evaluation]] and [[silent-failure-detection-requirements]].

"Also test authorization and preconditions. A valid tool call can still be wrong if the user did not approve the action or the system skipped a required check."

Decomposition example: *"Find Berkeley homes under $1M and schedule viewings"* → parameters extracted correctly / relevant listings retrieved / availability checked / calendar invites sent. **Each checkpoint passes or fails independently, making debugging tractable.**

### Transition failure matrices

The most concrete analytical tool in the document, and it recurs in three sections (multi-turn, multi-step, agentic).

Construction: **rows = the last successful state; columns = where the first failure occurred.** Cell counts show where failures cluster.

The worked example is a text-to-SQL agent: **GenSQL → ExecSQL transitions cause 12 failures, while DecideTool → PlanCal causes only 2.** "The counts show where to investigate first."

Why this is the right representation: a flat per-step failure count conflates "this step is broken" with "this step is where everything that was already broken finally surfaced." Conditioning on the last *successful* state removes the cascade contamination — it tells you where the system was still healthy immediately before it broke, which is where the causal defect lives. It's the quantitative version of the "annotate only the first failure" rule.

Bryan Bischof ("Failure is A Funnel," Data Council 2025) shows variation in these matrices **across experiments** — which turns the matrix from a snapshot into a diff you can track as you make changes. How you organize the matrix depends on your application; Bischof's text-to-SQL agent has an inherently sequential workflow he exploits for further analytical leverage.

Test cases for agent failures follow the multi-turn principles: **reproduce with the simplest test that still fails; use a multi-turn test only when the failure depends on conversation context.**

---

## What this document does not answer

Honest gaps, not manufactured ones:

- **No statistical treatment of eval uncertainty.** Confidence intervals get one sentence in the monitoring section. There is no guidance on sample size for *detecting a change* (as opposed to discovering failures), no power analysis, no treatment of multiple comparisons across many evals. The 100-200 labeled examples per failure mode is a practitioner heuristic, not derived from a target margin of error. [[2411.00640 - Adding Error Bars to Evals]] is the complement here.
- **Benevolent dictator has an unexamined failure mode.** The document is honest that it optimizes for decision speed and consistency. It does not address what happens when the single expert is systematically wrong, or has blind spots for user segments unlike themselves. The "underrepresented user groups" caution appears only under synthetic data, never under single-annotator risk — yet it applies at least as strongly there.
- **~10x faster iteration with custom tools** is an unquantified field observation, offered as such.
- **Adversarial pressure is absent.** Everything here assumes failures are accidental. Nothing covers evaluating under deliberate misuse, prompt injection reaching tool calls, or judges being gamed by the system under test. Guardrails get a mention for SQL injection and disallowed instructions, but there is no eval methodology for adversarial settings.
- **Cost of the method itself.** 60-80% of development time is a large claim presented as an observed fact from the authors' engagements; there is no discussion of when a product is too low-stakes to justify it.

## The compressed version

If you keep four things:

1. **Error analysis first, always.** Open code ≥30 traces yourself, axial code into a counted taxonomy, iterate to saturation at ~100. Everything else is downstream of this and nothing substitutes for it.
2. **Binary, application-specific, validated.** Never generic metrics as quality measures. Never Likert. Decompose instead of grading. Validate judges against human labels on a large held-out set and report TPR/TNR.
3. **Criteria drift is why humans can't be removed.** Your evaluation criteria change as a result of looking at outputs. Any process that freezes the objective and hands it to an optimizer — eval-driven development, automated prompt optimization, LLM-run open coding, off-the-shelf metrics — is optimizing a target that was never correct.
4. **Deterministic before expensive.** Assertions before judges, CI before monitoring, process metrics before outcome metrics, single-turn repro before multi-turn analysis, first failure before cascades.
