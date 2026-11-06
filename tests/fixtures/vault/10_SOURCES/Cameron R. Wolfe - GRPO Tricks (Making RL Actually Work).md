Created: 19-08-2026 23:56

Tags: [[blog]] [[post-training]] [[reinforcement-learning]] [[grpo]] [[policy-gradient]] [[implementation]] [[research-survey]]

Status: #baby

## References

1. [Tricks for Making RL Actually Work (GRPO++)](https://cameronrwolfe.substack.com/p/grpo-tricks)

# Notes

## The problem this article exists to solve

DeepSeek-R1 [2] reported 47% on AIME 2024 for an RL-Zero run on a 32B base model. When the open community took the recipe as described — vanilla [[grpo-group-relative-policy-optimization]] on Qwen-2.5-32B — they got **30%**. That 17-point gap is the whole reason this article exists.

The gap is not a data gap and not a compute gap. It is a gap in *undocumented algorithmic detail*. GRPO as written in the DeepSeekMath paper [13] is four lines of math and looks like it should just work. In practice, at scale, it exhibits four failure modes:

1. **Entropy collapse** — the next-token distribution sharpens onto single tokens, exploration dies, and the group-relative advantage becomes meaningless because every rollout in the group is nearly identical.
2. **Noisy, non-monotone reward curves** — training reward doesn't climb steadily; it thrashes.
3. **Excessive response lengths, concentrated in *incorrect* answers** — the model learns to ramble specifically when it's wrong.
4. **Poor sample efficiency** — slow learning, wasted gradient steps.

Wolfe's framing is that these are not four unrelated bugs. Three of them (1, 3, 4) trace back to specific, identifiable terms in the GRPO objective, and the fourth (a system-level one, discussed under TIS) traces to the fact that modern RL frameworks quietly turn on-policy RL into off-policy RL. Each of the papers surveyed here fixes one or two of them. The Olmo 3 recipe at the end is the demonstration that they *compose*.

The mental model to carry through: **vanilla GRPO is not a correct algorithm that needs tuning. It is a slightly-wrong algorithm with biases baked into its normalization terms, and the "tricks" are corrections to those biases.**

---

## Layer 0: what RL for reasoning actually is

Two flavors of online RL for LLMs, differing only in where reward comes from:

- **RLHF** — reward from a learned reward model trained on human preferences. See [[reinforcement-learning-from-human-feedback-rlhf]].
- **RLVR** — reward from a rule-based / deterministic verifier. See [[rlvr-reinforcement-learning-with-verifiable-rewards]].

Everything else in the loop is identical: sample prompts → generate completions from the current policy → score them → compute a policy update. GRPO is one choice of optimizer for that last step; PPO or REINFORCE would slot into the same hole.

Reasoning models live almost entirely in RLVR territory, because you need a domain where correctness is checkable: math (string-match the final answer) or code (run it in a sandbox against test cases). Wolfe flags a practical wrinkle here — **string matching is often not enough**. The DeepSeek quote he pulls says they use an *LLM judge* against the ground-truth solution rather than a strict parser like Math-Verify, because the stricter parser throws away correct-but-differently-formatted answers, and that lower yield produces worse downstream models. This is a nice, concrete instance of the general point in [[verifiable-rule-based-rewards-prevent-reward-hacking]]: verifiable rewards are safer against hacking, but the *verifier's recall* is itself a hyperparameter that affects final model quality.

A reasoning model is architecturally an ordinary LLM. The only difference is that it emits a long free-text chain-of-thought (a "rationale" / "reasoning trajectory") before its final answer, and can spend more inference compute by making that trajectory longer.

The striking claim — and this connects directly to [[reasoning-as-emergent-behavior-from-pure-rl]] — is that the sophisticated behaviors inside those trajectories (decomposition, self-critique, exploring alternatives, backtracking) are **not injected**. Nobody wrote SFT data teaching the model to say "wait, let me reconsider." It falls out of large-scale RL. The observable signature is that average completion length *rises* over training: the model discovers on its own that spending more tokens buys more reward.

**RL-Zero setup**: skip SFT entirely, run RLVR directly on the pretrained base model. DeepSeek-R1-Zero popularized this and reported the famous "Aha moment" — an intermediate point in training where the model spontaneously starts re-evaluating its own reasoning. Hold onto this; paper [3] below partially demolishes it.

---

## Layer 1: PPO, because GRPO is PPO with the critic amputated

You cannot understand what's broken in GRPO without the PPO machinery it inherits. See [[ppo-as-trpo-simplification]] and [[PPO and RLHF - Deep Dive]].

PPO's iteration loop:
1. Sample a diverse batch of prompts.
2. Generate one completion per prompt from the policy.
3. Compute advantage estimates for each completion.
4. Perform **several** policy updates over that sampled data.

Step 4 is the source of everything interesting. Because you take multiple gradient steps on data sampled from the *pre-update* policy, the data goes stale — you are doing off-policy updates on on-policy data. That is what the importance ratio exists to correct.

### The importance (policy) ratio

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$$

Symbol by symbol: $\pi_\theta$ is the policy *right now*, mid-update. $\pi_{\text{old}}$ is the policy as it was when this batch of rollouts was generated. $a_t$ is the token actually emitted at position $t$; $s_t$ is everything before it (prompt + generated prefix). So $r_t$ answers: *how much more (or less) likely has this exact token become since we sampled it?*

$r_t = 1$ means nothing changed. $r_t = 3$ means the update tripled that token's probability — a large, potentially destabilizing move.

### The clipped surrogate objective

The unclipped objective is the TRPO-style $r_t(\theta) \hat{A}_t$. PPO's surrogate takes the **minimum** of the unclipped term and a clipped one where $r_t$ is forced into $[1-\epsilon, 1+\epsilon]$:

$$L^{\text{CLIP}} = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

The `min` is what makes this a **pessimistic lower bound**, and it makes clipping *asymmetric*:

- **Positive advantage** (good token, push it up): the clip caps the benefit at $1+\epsilon$. Pushing $r_t$ past $1+\epsilon$ buys you nothing, so the gradient dies. You are *disincentivized* from large upward moves.
- **Negative advantage** (bad token, push it down): the clip caps the benefit at $1-\epsilon$ in the other direction, similarly killing the gradient for large downward moves.
- But in the direction where moving $r_t$ *hurts* the objective, the `min` keeps the unclipped term, so the penalty is unbounded.

Schulman's own phrasing, quoted in the article: *"We only ignore the change in probability ratio when it would make the objective improve, and we include it when it makes the objective worse."* This asymmetry is the trust region — it's a soft version of the hard KL constraint in [[trust-region-policy-optimization-monotonic-improvement-guarantee]].

**This asymmetry is exactly what DAPO's "clip higher" attacks. Remember it.**

### KL divergence — two places to put it

You typically add a KL penalty between the current policy and a frozen reference (usually the model at the start of RL), approximated via log-prob differences per Schulman's k3 estimator [12]. Two insertion points:

- **Subtract it from the reward** (PPO's choice).
- **Add it to the loss as a penalty term** (GRPO's choice).

**Crucially: all of the work surveyed in this article drops the KL term entirely.** The reasoning is sharp — KL-to-reference exists to keep an aligned chat model from drifting off its SFT distribution. But a reasoning model *is supposed to drift*. Learning long CoT means becoming behaviorally very different from the base model. Anchoring to the reference actively fights the thing you're training for.

### Advantage estimation, and why GRPO drops the critic

PPO's advantage is $A(s,a) = Q(s,a) - V(s)$, where $V$ comes from a **learned critic** — a separate transformer with a regression head (or a value head sharing weights with the policy) predicting expected cumulative reward per token. The critic is trained *alongside* the policy with an added MSE loss, because $V$ is on-policy and must track the changing policy. Advantage is then computed with GAE. See [[ppo-gae-value-function-triangle]] and [[four-model-setup-in-ppo-rlhf]] — PPO needs policy + reference + reward model + critic in memory simultaneously.

**GRPO's central idea**: replace the learned value function with an *empirical* baseline computed from a group of sampled completions.

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

Sample $G$ completions for the same prompt, score them all, then each completion's advantage is its reward standardized against its peers. "How much better than my siblings was I?"

This kills two of the four models: no critic (the group mean *is* the baseline), and with verifiable rewards, no reward model either. Massive memory and compute savings.

Two consequences that matter later:

- **$G$ must be large.** Since the baseline is a sample estimate over the group, small $G$ gives a high-variance, unreliable baseline. PPO and REINFORCE typically sample one completion per prompt; GRPO needs many (RLOO had explored this earlier). This is why batch size is such a live issue in GRPO.
- **Every token in completion $i$ gets the *same* advantage $\hat{A}_i$.** The reward is outcome-level; there is no per-token credit assignment. So the only thing that differentiates token $t$ from token $t'$ within one completion is the per-token importance ratio $r_t(\theta)$. **That single fact is the seed of GSPO's critique.**

The GRPO surrogate loss is then essentially the PPO clipped objective, aggregated over the $G$ completions in a group, with the KL added as a loss term rather than a reward subtraction.

---

## Layer 2: how to tell if your RL run is sick

Before any algorithmic fix, Wolfe puts a diagnostics section — and it's the most practically useful part of the piece. The framing quote from DAPO [1]:

> "Reinforcement learning on large language models is… an intrinsically complex systems engineering challenge, characterized by the interdependence of its various subsystems. Modifications to any single subsystem can propagate through the system, leading to unforeseen consequences… Even seemingly minor changes… can amplify through iterative reinforcement learning processes."

RL runs are expensive and failures are silent, so you need *intermediate* signals, not just end-of-run eval. Four metrics (compare [[ppo-monitoring-diagnostics-beyond-returns]]):

**1. Response length.** Should *increase* — that's the policy learning to use its CoT. But: it doesn't have to be monotone (stagnation or dips are OK), and **excessive** length is itself a symptom of a broken setup. Length is a two-sided indicator, not "more is better."

**2. Training reward.** Should increase *stably*. A noisy or chaotic reward curve is a clear defect signal. Caveat Wolfe flags: training reward is not performance — RL overfits its training set.

**3. Policy entropy.** The entropy of the next-token distribution, a proxy for exploration. Wants to be in a band:
   - Too low → distribution sharpened onto one token → no exploration → group rollouts become identical → advantage computation degenerates. See [[entropy-collapse-in-policy-gradient-methods]].
   - Too high → the model is emitting gibberish.
   - The DAPO authors' stated preference is the non-obvious one: *"maintaining a slow upward trend in entropy is conducive to the improvement of model performance."* Not flat — slowly rising.

**4. Held-out evaluation.** Small validation set, tracked throughout, specifically to catch reward hacking that training reward would hide.

The prescribed workflow: (i) start from a setup known to work, (ii) apply one intervention, (iii) watch these four metrics. Every paper below follows exactly this loop.

### Data and batching — check this before touching the algorithm

Wolfe is emphatic that these are more common failure causes than any algorithmic subtlety:

- **Batch size too small is the single most common GRPO mistake.** Reference point: Olmo 3 [5] uses batch size 512 = 64 prompts × 8 rollouts per prompt. Diagnostic heuristic: if increasing batch size visibly stabilizes your reward curve, your original batch was too small.
- **Difficulty diversity within each batch.** Nathan Lambert, quoted: *"With large batch sizes, the model should have questions that are trivial, somewhat challenging, and nearly impossible in each batch."* Olmo 3 does offline difficulty filtering (drop too-easy and too-hard), often gauged by measuring Pass@K with another LLM [9].
- **Curricula are largely irrelevant here** — you see each datapoint many times during RL, so ordering washes out. Focus on composition, not sequence. (Wolfe notes in a footnote that curriculum learning for LLM RL is simply not widely used yet.)
- **Delete easily-guessable question formats.** Multiple choice is the canonical offender: a policy that random-guesses collects reward without reasoning, which is reward hacking by construction.

---

## Fix 1 — DAPO [1]: four modifications that close the R1 gap

*Decoupled Clip and Dynamic sAmpling Policy Optimization.* Qiying Yu et al., arXiv:2503.14476. Code (on verl) and data openly released. This is the paper that turns 30% into 50% on AIME 2024 with Qwen-2.5-32B — beating DeepSeek-R1-Zero's reported 47%, **in half the training steps**.

### (1) Clip Higher — the fix for entropy collapse

This is the sharpest insight in the whole article, because it locates entropy collapse in a specific *structural asymmetry of the clipping bound*, not in some vague hyperparameter.

Split tokens conceptually:
- **Exploitation token** — already high probability under the current policy.
- **Exploration token** — low probability under the current policy.

Clipping constrains $r_t \in [1-\epsilon, 1+\epsilon]$. But $r_t$ is a *ratio*, so the upper bound $1+\epsilon$ translates into a **multiplicative** cap on probability increase — and a multiplicative cap on a small number is a tiny absolute allowance. DAPO's own worked example, with $\epsilon = 0.2$ and positive advantage:

| $\pi_{\text{old}}(a_t \mid s_t)$ | max $\pi_\theta$ after update = $\pi_{\text{old}}(1+\epsilon)$ | headroom |
|---|---|---|
| 0.01 (exploration token) | 0.012 | +0.002 |
| 0.9 (exploitation token) | 1.08 → effectively unconstrained | can reach 0.999 |

The high-probability token's cap is above 1.0, i.e. **not a constraint at all** — it can go to 0.999 in a single update. The low-probability token can crawl from 0.01 to 0.012. So the *same* $\epsilon$ imposes a savage constraint on exploration and no constraint at all on exploitation. Run this thousands of steps and probability mass ratchets monotonically toward the already-likely tokens. That is entropy collapse, and it's a direct mechanical consequence of using a symmetric ratio clip.

**The fix**: decouple the bounds. Clip to $[1 - \epsilon_{\text{low}},\ 1 + \epsilon_{\text{high}}]$ with $\epsilon_{\text{low}} = 0.2$ (unchanged default) and $\epsilon_{\text{high}} = 0.28$. Raising only the upper bound gives exploration tokens room to grow.

Non-obvious corollary, and DAPO is explicit about it: **do not raise $\epsilon_{\text{low}}$.** A larger lower bound lets the update drive token probabilities toward zero, which permanently removes tokens from the sampling space — collapsing the support rather than the entropy. The asymmetry of the fix mirrors the asymmetry of the problem.

### (2) Dynamic Sampling — the fix for sample efficiency

As training improves, more and more prompts have *all* $G$ completions correct. DAPO plots this ratio rising over training.

Why that's fatal: if every reward in the group equals 1, then $r_i - \text{mean} = 0$ for every $i$, so $\hat{A}_i = 0$ for every $i$, so the policy gradient contribution of that entire prompt is **exactly zero**. The prompt occupies batch slots, burns rollout compute, and contributes nothing. Your *effective* batch size silently shrinks, the batch gradient gets noisier, and — per the batch-size discussion above — noisy gradients are precisely what destabilizes GRPO.

**The fix**: over-sample prompts per batch, filter out any prompt with perfect accuracy, keep sampling until you have a full batch of prompts that actually produce gradient. The per-batch sampling cost becomes variable (hence "dynamic"), but the article's point is that this cost is *more than offset* by convergence speed — the model converges much faster with filtering on.

Note the elegance: this is the same trick as Reinforce-Rej (rejection-sampling-inspired filtering) and Olmo 3's "Zero Gradient Filtering," and it generalizes to filtering *all*-incorrect groups too, which are zero-gradient for the identical reason.

### (3) Token-Level Loss — the fix for length bias

Vanilla GRPO aggregates at the **sample level**: average the token losses *within* each completion, then average those per-completion numbers across the batch. Every completion gets equal weight regardless of length.

Trace the arithmetic. A completion of length $T$ contributes $\frac{1}{T}\sum_t \ell_t$ to the batch loss. So each individual token's weight in the final loss is $\propto 1/T$. **A token inside a 8000-token completion has one-tenth the influence of a token inside an 800-token completion.**

Two damaging consequences:
- You cannot effectively *learn from* the good patterns in long, high-quality completions — their tokens are individually muted.
- You cannot effectively *punish* the bad patterns in long, low-quality completions — same muting. Empirical signature Wolfe cites: excessively long samples visibly accumulate repeated words and gibberish, because the loss can barely reach in and suppress them.

**The fix**: aggregate as a flat average over **all tokens in the batch** — every token weighted equally, regardless of which completion it lives in.

Reported effect: this yields the *smallest* raw accuracy gain of DAPO's four changes, but it visibly stabilizes training — you get a clean, monotone rise in both entropy and response length. This is a good illustration of why the health metrics section came first: the intervention that improves the *metric you actually care about least* can be the one that keeps the run from diverging.

### (4) Overlong Reward Shaping — the fix for truncation noise

You impose a max generation length for efficiency. The policy sometimes exceeds it and gets truncated. The default RL reflex is to assign a negative reward.

DAPO's objection is a credit-assignment argument: what if the reasoning was *entirely correct* and merely long? You've now punished valid reasoning for a length violation, injecting noise into the reward signal.

Two remedies, tested:

- **Overlong filtering** — *mask* truncated samples: they contribute nothing to the policy update rather than being negatively reinforced. Improves both performance and stability. This is the same "delete the misleading signal rather than guess at it" move as dynamic sampling.
- **Soft overlong punishment** — a graduated penalty. Define max length $L_{\max}$ and cache length $L_{\text{cache}}$, forming a punishment interval $[L_{\max} - L_{\text{cache}},\ L_{\max}]$:
  - length $< L_{\max} - L_{\text{cache}}$ → penalty 0
  - length in the interval → penalty scales linearly from 0 down to $-1$
  - length $> L_{\max}$ → penalty $-1$

  This penalty is **added directly to the verifiable reward**. DAPO's settings: $L_{\max} = $ 16K tokens, $L_{\text{cache}} = $ 4K tokens. So the model gets a smooth "you're approaching the limit" pressure gradient instead of a cliff.

### DAPO experimental setup

- **Data**: DAPO-Math-17K, 17K prompts, deliberately curated so all answers are **integers** — making parsing and verification trivial and removing verifier noise as a confound. (Footnote: the HuggingFace release has ~1.8M rows, mostly duplicates; the DAPO code dedupes down to ~17K.)
- **Model**: Qwen-2.5-32B base, chosen specifically to match DeepSeek-R1-Zero's setup for a like-for-like comparison.
- **Domain**: math only. Wolfe notes this is standard practice — math RL is used as a cheap testbed and findings are *assumed* to transfer. Flag this as an assumption, not a result.
- **Result**: AIME 2024 accuracy 0% → **50%**, vs. 47% for DeepSeek-R1-Zero-Qwen-32B and 30% for vanilla GRPO, using **half** the training steps of R1-Zero.
- Ablations show all four modifications contribute positively.

---

## Fix 2 — Dr. GRPO [3]: the biases were in the normalization all along

*"Understanding r1-zero-like training: A critical perspective."* Zichen Liu et al., arXiv:2503.20783. Two contributions: an audit of **base models**, and an audit of the **GRPO objective**. Both produce uncomfortable results.

### Part A: base models are not what you think

Two questions asked of Qwen-2.5 (the workhorse of open RL-Zero recipes) and DeepSeek-V3-Base [14], probed on 500 MATH questions.

**Q1: Does the prompt template matter?** Base models are pure next-token predictors with no instruction tuning, so how you frame the prompt is load-bearing. Three template styles tested; GPT-4o-mini judges whether output *format* is correct (format only, not answer correctness).

Result: template choice massively changes performance, and **the best template differs per model**. DeepSeek-V3-Base performs very poorly without its correct chat template. Qwen-2.5 performs best with **no template at all** — just concatenated question-answer text — worth a **60% performance improvement**.

That last finding is the loaded one. The authors' hypothesis, quoted:

> "Since Qwen2.5 uses chat model's data (question-answer pairs) during the pretraining stage, we hypothesize that they might pretrain on the concatenated text… If our hypothesis turns out true, we shall be more careful about using Qwen2.5 models to reproduce DeepSeek-R1-Zero, since the base models are already SFT-like without templates."

If true, a large fraction of published "RL-Zero" results are **not RL-Zero** — they are RL on a model that has already effectively been instruction-tuned during pretraining. The hypothesis cannot be validated because Qwen doesn't disclose training data. This is a genuine, unresolved threat to the validity of a whole literature.

**The second-order finding is worse for RL's reported gains.** Template choice matters enormously *before* RL, but after RL most templates converge to similar performance. So a badly-matched template **artificially deflates your baseline**, and the RL delta you report is partly just recovering from the bad template. Wolfe states this plainly: the performance benefits of RL may be more modest than typically reported.

A nuance worth keeping: whether RL can recover from a template mismatch depends on **data coverage**. If model and template are well matched, RL gives benefits even on narrow data (e.g. GSM-8K alone). If mismatched, you need a broad, diverse dataset for RL to recover.

**Q2: Do base models already reason and self-reflect?** Pass@8 across temperature settings, to check whether the base policy can produce *at least one* correct trajectory — which is a hard prerequisite:

> "If a base policy cannot even sample a single trajectory that leads to the correct final answer, it is impossible for reinforcement learning to improve the policy because there is no reward signal."

This is the fundamental exploration constraint of RLVR, stated cleanly: **RL can only amplify what the base model can already occasionally do.** All models tested had non-zero Pass@8; Qwen-2.5 and DeepSeek-V3 were already strong.

### The "Aha moment," deflated

Keyword-searching DeepSeek-V3-Base outputs on MATH for reflection markers ("Aha", "wait") reveals the base model **already self-reflects on a large number of queries, before any RL**. RL does *increase* the frequency of these behaviors — but:

> "Although self-reflection behaviors occur more frequently in R1-Zero, we observe that these behaviors are not positively correlated with higher accuracy."

Two claims stacked: (a) self-reflection is largely *elicited/amplified*, not learned from scratch, and (b) more self-reflection doesn't measurably buy accuracy. This directly complicates the emergence story in [[reasoning-as-emergent-behavior-from-pure-rl]] — worth holding both readings in tension rather than picking one.

### Part B: the two biases in the GRPO objective

Observation that starts the analysis: response length rises during GRPO training, and everyone attributes this to "developing long CoT." But [3] shows **response length keeps climbing after reward has plateaued**, and the growth is concentrated in *incorrect* responses. That's not reasoning development; that's an artifact.

**Bias 1 — response-level length bias.** GRPO normalizes each sequence's summed token loss by that sequence's token count $|o_i|$. Since lengths vary across the batch, the normalizer varies, so the *effective learning rate per sequence depends on its length*. Work through both signs of advantage:

- **Positive advantage** (correct answer): the $1/|o_i|$ factor makes short responses receive **larger** gradient updates than long ones. Correct answers are pushed up harder when short.
- **Negative advantage** (incorrect answer): the same $1/|o_i|$ factor makes long responses receive a **smaller** penalty. Wrong answers are punished *less* the longer they are.

Combine them and the gradient literally prefers **short-and-correct or long-and-incorrect**. Wolfe's summary line: *"GRPO biases the model towards overthinking by using more tokens for incorrect answers!"* The model is not learning to think longer — it's learning that length is a cheap shield against penalty. This is the same defect DAPO found from a different angle (sample-level aggregation), and the two papers arriving at it independently is good evidence it's real.

**Fix**: divide the summed sequence loss by a **fixed constant** (`MAX_TOKENS`) rather than by the sequence's own token count. The normalizer is now length-independent, so the bias vanishes.

**Bias 2 — question-level difficulty bias.** The $\text{std}(\{r_i\})$ in the advantage denominator. Consider a question where 7 of 8 rollouts are correct: rewards are mostly 1s, so std is small, so dividing by it makes $|\hat{A}_i|$ **very large**. Same for a question where 7 of 8 are wrong. So near-trivial and near-impossible questions get **disproportionately large gradient updates** purely because of low reward variance — not because they're informative.

Wolfe's footnote sharpens this and is the part worth remembering: advantage normalization/whitening is a standard, healthy RL practice (see [[advantage-normalization]]) — **but it's normally done over a batch**. Batch-level normalization applies one consistent scale to everything. GRPO's std is computed **per question**, so the scale varies with each question's difficulty, and that per-question variation is what introduces bias. *Same operation, different scope, opposite effect.* Lite PPO (listed later) makes exactly this correction: compute the std over the whole batch instead of the group.

**Fix**: delete the std term. Advantage becomes just $\hat{A}_i = r_i - \text{mean}(\{r_1,\ldots,r_G\})$.

### Dr. GRPO summary and results

Two changes total:
1. Normalize summed sequence loss by a fixed constant, not by sequence length.
2. Remove std from the advantage denominator.

Reported effect: better performance **per token** — higher accuracy while producing *fewer* tokens on average. Note the direction: everywhere else in this literature, longer = better. Here, removing a bias makes the model both better *and* shorter, which is strong evidence the length growth was artifactual.

**Experiments**: Oat framework (openly released). Train on MATH; evaluate on OlympiadBench, AIME 2024, AMC, Minerva Math, Math500. Binary correctness rewards via Math-Verify. Training Qwen-2.5-Math-7B with the Qwen-Math template gives **43.3% on AIME 2024** — SOTA at that scale — in **~27 hours on 8× A100**. Wolfe highlights the cheapness as the actual contribution for researchers: a setup this small lets you iterate on RL interventions at all.

**Bonus finding**: continued domain-specific pretraining helps RL. Continually pretraining Llama-3.2-3B on math data before RL-Zero **raises the performance ceiling** RL can reach — consistent with the Pass@8 constraint above (better base coverage → more reward signal to exploit).

---

## Fix 3 — TIS [4]: your RL framework is secretly off-policy

Feng Yao et al., published as a Notion post (Aug 2025). This one is different in kind: not a flaw in the *math*, but a flaw in the *systems architecture* that every modern RL framework shares.

### The setup that creates the bug

RL alternates rollouts (generate completions) and policy updates (compute gradients). Rollouts dominate cost — Olmo 3 profiling shows **5–14× more compute spent on inference than on policy updates**. (Footnote adds the long-tail wrinkle: most completions are short and finish fast, but a few very long ones dominate wall-clock in a synchronous setup.)

So every serious framework uses **two different engines**:
- **Learner**: FSDP or DeepSpeed — a training stack.
- **Sampler**: vLLM or SGLang — an inference stack, often at reduced precision (fp8/int8) for speed.

### The bug

Even with **identical model parameters**, the two engines produce **different token probabilities**. Different kernels, different fusion, different precision, different parallelism, different sampling code.

The consequence is not cosmetic. GRPO/PPO assume $\pi_{\text{old}}$ — the distribution the rollouts were drawn from — is the learner's distribution. It isn't; it's the sampler's. So:

> "We show the implementation gap implicitly turns the on-policy RL to be off-policy."

In the worst case the probabilities are contradictory — **the learner would never have generated the completion the sampler produced.** You are computing gradients for trajectories your policy wouldn't take.

### Why you can't just fix the engineering

The obvious response is to hunt down and eliminate the implementation differences. [4] tried: they identified problem areas and patched several. **The gap persisted.** And:

- Full elimination requires chasing an unbounded number of subtle precision and sampling deviations.
- The work must be **redone every time an engine is changed or updated**.
- LLM inference is fundamentally **non-deterministic**, so the gap can be minimized but never removed.

This is a good example of the theme in [[reinforcement-learning-implementation-complexity-barrier]] — the algorithm on paper and the algorithm you ran are different objects, and the difference has teeth.

### The algorithmic fix: importance sampling, again

Importance sampling estimates properties of a target $f(x)$ by sampling from a cheaper proposal $g(x)$ and reweighting each sample by $f(x)/g(x)$.

Map it directly:
- $f(x) = \pi_{\text{learner}}$ — the target, but slow to sample from.
- $g(x) = \pi_{\text{sampler}}$ — the proposal, cheap, and what you actually sampled from.

So the correction is to weight the policy gradient by $\pi_{\text{learner}}/\pi_{\text{sampler}}$. **Truncated Importance Sampling (TIS)** caps that ratio at a maximum value $\rho$ for stability:

$$w_t = \min\left(\frac{\pi_{\text{learner}}(a_t \mid s_t)}{\pi_{\text{sampler}}(a_t \mid s_t)},\ \rho\right)$$

Note this is a **second, distinct importance ratio**, orthogonal to PPO's $\pi_\theta/\pi_{\text{old}}$. PPO's ratio corrects for *staleness across update steps*; TIS's ratio corrects for *engine mismatch within a step*. They multiply together; they do not replace each other.

Applying TIS to any optimizer is mechanical: take your optimizer's policy gradient expression, scale it by the truncated ratio. Works for GRPO, PPO, whatever. The authors' own framing of why they used the simplest possible version:

> "While there has been extensive study on how to design a stable and effective importance sampling, in practice we find it usually sufficient to use a classical technique, truncated importance sampling."

### Does it work?

**Experiment A** — Qwen-2.5-32B + DAPO on DAPO-Math-17K, stopped at 250 iterations (early stopping as a cheap intervention test — worth noting as a methodology trick in its own right). TIS gives a clear performance boost. Critically, **the engineering-centric approach does not match it** — patching engine bugs is strictly worse than the algorithmic correction.

**Experiment B — deliberately amplifying the bug.** Use quantized rollouts to *widen* the gap on purpose: sample at int8 vs bf16 on GSM8K. Lowering precision raises the maximum token-probability difference from **~0.4 to ~1.0**, confirming the mechanism. Then:
- Plain PPO with int8 rollouts → clear performance deterioration, **entropy collapse**, and **abnormally long average responses**.
- Add TIS → recovers to bf16-level performance, entropy collapse reversed.

That int8 result is the most interesting thing here. Entropy collapse and length inflation were diagnosed in DAPO and Dr. GRPO as *objective-function* problems. Here they are **reproduced from a pure systems artifact**. Same symptoms, entirely different cause. Practical takeaway: **the health metrics tell you something is wrong, not what is wrong.** You can chase clipping bounds forever when your actual bug is your sampler's precision.

Also: TIS helps *even with bf16 rollouts*, improving entropy further. And when the gap is genuinely small, the ratio $\approx 1.0$ and the objective reduces to standard PPO/GRPO — so TIS is safe to leave on. Low cost, no downside.

### What widens the gap (measured over 512 DAPO-Math-17K prompts)

- **Mean mismatch is roughly constant across implementations; maximum mismatch is what moves.** The damage is from rare, large single-token divergences, not from a uniform drift. This is why averages hide the bug — you must monitor the max.
- **Different parallelism strategies between engines** (e.g. sequence parallelism in the learner, tensor parallelism in the sampler) → significantly worse.
- **Same strategy, different degree** (TP=2 vs TP=4) → much less problematic. Kind of parallelism matters more than amount.
- **Longer rollouts** → worse: *"Responses capped at 20K tokens exhibit a higher maximum mismatch than those capped at 4K… the mean mismatch remains similar across both settings… longer sequences provide more opportunities for a single, large probability divergence."* More tokens = more lottery tickets for a big divergence.
- **Sampler backend choice** (vLLM vs SGLang vs SGLang-deterministic) → **no measurable impact**. Genuinely surprising, and useful: don't waste time switching backends.

Un-analyzed but flagged: MoE vs dense models likely differ; post-trained models likely have larger mismatch than base models; data difficulty/domain may modulate it.

---

## Fix 4 — the variance family: GSPO, GMPO, CISPO

All three attack the same root object from different angles: **the token-level importance ratio is a high-variance quantity, and GRPO lets it dominate the gradient.**

### GSPO [6] — move the ratio to the sequence level

*Group Sequence Policy Optimization*, Zheng et al., arXiv:2507.18071. Used to train **Qwen 3**.

The argument is a clean unit-mismatch observation. Recall from Layer 1: with outcome rewards, GRPO assigns the *same* advantage to every token in a completion. So:

- Rewards and advantages: **sequence-level**.
- Importance ratios and optimization: **token-level**.

> "Unlike previous algorithms that adopt token-level importance ratios, GSPO defines the importance ratio based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization."

The misalignment injects variance: you're modulating each token's gradient by a noisy per-token ratio when the actual learning signal carried no per-token information at all.

**GSPO's fix**: compute the importance ratio over the **whole sequence's likelihood**, and clip *that*. One stable weight applied to all tokens in the completion, instead of $T$ noisy ones.

Important implementation detail: the sequence-level ratio is **normalized by completion length $T$** (i.e., a length-normalized geometric-mean-like quantity in log space). Without this, a product of $T$ token ratios would explode or vanish with $T$ — the ratio would swing wildly based on length alone. Advantage estimation is unchanged from GRPO, so the critic-free efficiency is preserved.

**The MoE result is the most interesting empirical claim in this section.** Training MoE models with RL suffers **expert-activation volatility**: the set of experts routed to for a given prompt changes drastically after a policy update. Since the token-level importance ratio compares $\pi_\theta$ and $\pi_{\text{old}}$ token-by-token, and a routing flip changes which subnetwork computed that token, the ratios become garbage and training fails to converge.

Qwen 3's original workaround was **routing replay**: cache the experts selected on the first pass and force-reuse them for subsequent policy updates. It works but it's a hack with memory and fidelity costs. **GSPO makes MoE RL converge without routing replay at all** — because sequence-level likelihood is far more robust to individual routing flips than per-token likelihood is. A variance-reduction change eliminating an entire systems workaround is a strong signal the diagnosis was right.

GSPO also reports better sample efficiency and final performance generally, not just for MoE.

### GMPO [7] — keep token ratios, change the mean

*Geometric-Mean Policy Optimization*, Zhao et al., arXiv:2507.20673. Same problem (outlier token ratios destabilize training), opposite design choice.

GSPO's reasoning: token-level ratios are the problem, so eliminate them. GMPO's reasoning: **the arithmetic mean is the problem, so change the aggregator.** Arithmetic means are dominated by outliers; geometric means are not. So aggregate token-level losses with a **geometric mean** over the sequence.

> "GMPO is plug-and-play—simply replacing GRPO's arithmetic mean with the geometric mean of token-level rewards, as the latter is inherently less sensitive to outliers."

Implementation details, all of which matter:

- **Sign problem**: geometric means require non-negative inputs (you're taking roots). GMPO computes the geometric mean over **absolute values** of token-level losses, then multiplies by $\text{sign}(\hat{A})$ (±1) to restore the correct update direction. Magnitude from the robust aggregator, direction from the advantage.
- **Wider clipping range needed**: $[\approx 0.7, \approx 1.5]$ instead of GRPO's default $[0.8, 1.2]$. Because the geometric mean already suppresses outliers, the clip no longer has to do that job, so you can loosen it and preserve more gradient signal. The two mechanisms trade off against each other.
- **Compute in log space**: ratios and the whole geometric mean via log probabilities, for numerical stability — a geometric mean is an arithmetic mean of logs, so this is natural rather than a hack. Wolfe notes this is standard in PPO implementations anyway. The $[0.7, 1.5]$ range corresponds to clipping the **log-ratio** to $[-0.4, 0.4]$.

**Ablation worth noting**: GMPO finds that **token-level clipping outperforms sequence-level clipping** — a direct empirical disagreement with GSPO's design. Both papers report gains; they haven't been cleanly reconciled. Treat "sequence-level vs token-level" as an open question, not a settled one.

Results: more stable importance ratios, more stable entropy, up to **+4% absolute Pass@1** in math, with the **largest gains on multimodal and MoE models** — which is at least directionally consistent with GSPO's MoE story.

### CISPO [8] — stop deleting the tokens that matter

*Clipped Importance Sampling weight Policy Optimization*, from the **MiniMax-M1** technical report (Chen et al., arXiv:2506.13585).

The observation is specific and vivid. **"Fork tokens"** — the tokens that initiate reflection, like "aha" or "wait" — are exactly the pivots of a reasoning trace. But under the base model they are **rare and low-probability**. Low probability means that when the update wants to increase them, the ratio $r_t$ is large. Large ratio means **clipped**. Clipped means **zero gradient**.

> "We found that tokens associated with reflective behaviors… were typically rare and assigned low probabilities by our base model. During policy updates, these tokens were likely to exhibit high [importance ratio] values. As a result, these tokens were clipped out after the first on-policy update, preventing them from contributing to subsequent off-policy gradient updates… These low-probability tokens are often crucial for stabilizing entropy and facilitating scalable RL."

So GRPO's clip systematically discards the gradient contribution of precisely the tokens responsible for reasoning behavior — and, per the DAPO analysis, for entropy maintenance.

**Why DAPO's clip-higher isn't sufficient here**: MiniMax-M1 performs **16 policy updates per batch**, where standard setups do ~2–4. Over 16 updates, a token's ratio will drift outside *any* fixed clip range, and once clipped it's ignored for **every remaining update in that batch**. Raising the ceiling delays the problem; it doesn't remove it. The desired property is that pivotal tokens contribute to *all* updates.

**CISPO's mechanism**: keep GRPO's advantage estimation, but restructure the objective to resemble **REINFORCE** rather than clipped-PPO. Token losses are scaled by a **clipped importance ratio wrapped in a stop-gradient** — the ratio is treated as a *constant multiplier* on each token's gradient contribution, not backpropagated through.

The distinction is the whole point:

| | clipped token's contribution |
|---|---|
| PPO / GRPO | **zero** — hard clip removes it from the update entirely |
| CISPO | **non-zero but capped** — weight saturates at a max value |

Hard clipping vs. soft, saturating clipping. Rare pivotal tokens keep pushing, just with bounded influence. Compared against GRPO and DAPO on Qwen2.5-32B-Base for math reasoning, CISPO improves both stability and sample efficiency.

### The rest of the field (pointers)

- **RSPO** (Router-Shift Policy Optimization) — MoE-focused; rescales router logits for stability.
- **SAPO** (Soft Adaptive Policy Optimization) — replaces hard clipping with a soft gating mechanism. Same family as CISPO's soft-clip idea.
- **Lopti** (Low-Probability Token Isolation) — *reduces* the influence of low-probability tokens to emphasize high-probability-driven updates. Note this is roughly **the opposite prescription to CISPO**, which protects low-probability tokens. Unreconciled tension in the literature; likely regime-dependent.
- **VAPO** — builds on DAPO but **reintroduces value models** for efficiency, cutting against the critic-free trend.
- **Lite PPO** — extensive empirical study; a critic-free algorithm built on the *vanilla PPO loss* that reportedly beats both GRPO and DAPO consistently. Two ideas: token-level loss aggregation, and **computing the advantage std over the whole batch instead of the group** — precisely Dr. GRPO's diagnosed bias, fixed by rescoping rather than deletion.
- **DCPO** (Dynamic Clipping Policy Optimization) — dynamic clipping for token ratios plus reward standardization across *consecutive training steps*, to avoid zero-gradient cases.
- **Reinforce-Rej** — rejection-sampling-flavored filtering of fully-correct and fully-incorrect groups. Same insight as dynamic sampling, simpler framing.

---

## Putting it together: the Olmo 3 recipe [5]

The payoff of the whole article: **these fixes are not competing alternatives, they compose.** Olmo 3 Think's GRPO configuration, with attribution:

| Modification | What it does | From |
|---|---|---|
| Zero Gradient Filtering | drop prompts where the whole group gets identical reward (all-correct *or* all-wrong) | DAPO [1] |
| Active Sampling | keep spare samples ready so batch size stays constant after filtering | DAPO [1] |
| Token-Level Loss | normalize by total tokens across the batch, not per-sequence | DAPO [1] |
| No KL Loss | drop the KL term for policy-update flexibility | community standard |
| Clipping Upper Bound | $\epsilon_{\text{high}} > \epsilon_{\text{low}}$ | DAPO [1] |
| Truncated Importance Sampling | reweight for learner/sampler log-prob mismatch | TIS [4] |
| No Standard Deviation | remove std from the advantage denominator | Dr. GRPO [3] |

Note that Zero Gradient Filtering is a *generalization* of DAPO's dynamic sampling — DAPO filtered all-correct groups; Olmo filters any zero-variance group, since all-wrong is equally gradient-free.

Structurally, the Olmo objective is still recognizably GRPO. Four deltas: (i) different normalization, (ii) slightly changed advantage, (iii) asymmetric clip bound, (iv) TIS weighting. No new machinery, no critic, no reward model. **The entire gap between 30% and state-of-the-art lives in normalization constants, clip bounds, filtering rules, and one systems correction.**

---

## What's genuinely interesting here

**The unifying diagnosis.** Nearly every fix is a correction to a *normalization or scoping choice*, not a new algorithm:
- Dividing by sequence length instead of a constant → length bias (Dr. GRPO, DAPO).
- Dividing by *group* std instead of *batch* std → difficulty bias (Dr. GRPO, Lite PPO).
- Symmetric clip bounds on an asymmetric quantity (a ratio) → entropy collapse (DAPO).
- Arithmetic mean instead of geometric → outlier sensitivity (GMPO).
- Token-level ratio instead of sequence-level → variance, MoE non-convergence (GSPO).
- Hard clip instead of saturating clip → pivotal tokens get exactly zero gradient (CISPO).

Every one is a denominator or an aggregation scope. None required new theory. That's a useful prior for reading the next wave of these papers.

**Symptom aliasing is the practical trap.** Entropy collapse and length inflation each have *at least three* distinct causes documented here: clip asymmetry (DAPO), loss normalization (Dr. GRPO), and sampler/learner precision mismatch (TIS). The health metrics detect illness; they do not diagnose it. Someone tuning $\epsilon_{\text{high}}$ when their real bug is int8 rollouts will tune forever.

**Length growth was partly fake.** The single most cited visual evidence of "reasoning emerging via RL" — the rising response-length curve — is contaminated. It keeps rising after reward plateaus, and it rises fastest on wrong answers, because the objective under-penalizes long failures. Dr. GRPO gets *better accuracy with fewer tokens*. Any argument that leans on length-as-reasoning needs to check whether the run had the bias fixed.

**The RL-Zero foundation is shaky.** If Qwen-2.5 base was pretrained on concatenated Q-A pairs, it is SFT-like, and "pure RL from a base model" results built on it don't mean what they claim. Unfalsifiable without training-data disclosure — which is itself the finding.

**The Aha moment may be elicitation, not emergence.** Base models already emit reflection tokens; RL raises their frequency; the frequency doesn't correlate with accuracy. Read alongside [[reasoning-as-emergent-behavior-from-pure-rl]] and hold the tension.

**The "verifiable rewards" story is more nuanced than usual.** DeepSeek preferring an LLM judge over strict Math-Verify — because strict parsing yields fewer usable samples and worse models — cuts against the clean framing in [[verifiable-rule-based-rewards-prevent-reward-hacking]]. Verifier recall is a real tuning knob, and tightening it isn't free.

**Backend choice doesn't matter; parallelism strategy does.** Counterintuitive and directly actionable: vLLM vs SGLang is a non-issue for sampler-learner gap; mismatched parallelism strategies are the thing to fix.

---

## Genuine limitations

- **Almost everything is validated on math only.** DAPO, Dr. GRPO, and CISPO all evaluate primarily in math. Wolfe states the field's justification openly — math RL is the cheap testbed and transfer is *assumed*. That assumption is untested for coding, agentic, or open-ended domains. Olmo 3 is the only multi-domain data point mentioned.
- **Heavy Qwen dependence, and Qwen is the model [3] casts doubt on.** DAPO, Dr. GRPO, and CISPO all use Qwen-2.5 variants. If the SFT-like-pretraining hypothesis holds, the RL-Zero framing of these results is compromised — and the paper raising the concern shares the dependency.
- **Direct contradictions between methods.** GSPO says move ratios to the sequence level; GMPO's ablations say token-level clipping wins. CISPO protects low-probability tokens; Lopti suppresses them. VAPO reintroduces critics while the field abandons them. These aren't reconciled, and the article doesn't attempt to.
- **The composed recipe isn't ablated as a whole.** Olmo 3 shows seven tricks *can* be stacked, but there's no ablation demonstrating each still contributes in the presence of the others. Several address overlapping symptoms (DAPO's token-level loss and Dr. GRPO's constant normalizer both fix length bias) — redundancy or synergy is unmeasured.
- **TIS can only be validated indirectly.** Since inference non-determinism means the gap can never reach zero, there is no ground-truth "correctly on-policy" baseline to compare against. Evidence is inferential (int8 amplification reproducing the symptoms) rather than a clean control.
- **Some results are stopped early.** TIS's DAPO experiment halts at 250 iterations for resource reasons. Reasonable practice, but it's evidence about early training dynamics, not final performance.
- **The compute asymmetry is unaddressed.** All the algorithmic tricks fight over the ~10% of compute in policy updates. The 5–14× spent on inference — and the long-tail completion problem Wolfe raises in a footnote — is where the actual cost is, and no algorithmic fix here touches it.
- **[4] is a Notion post, not peer-reviewed.** Worth noting given how central TIS is to the composed recipe.

---

## Connections

Foundations: [[grpo-group-relative-policy-optimization]] · [[PPO and RLHF - Deep Dive]] · [[ppo-as-trpo-simplification]] · [[trust-region-policy-optimization-monotonic-improvement-guarantee]] · [[ppo-gae-value-function-triangle]] · [[four-model-setup-in-ppo-rlhf]] · [[reinforcement-learning-in-language-models]]

Direct extensions of prior notes: [[entropy-collapse-in-policy-gradient-methods]] (now given a precise mechanism — asymmetric ratio clipping) · [[advantage-normalization]] (the group-vs-batch scoping distinction) · [[ppo-monitoring-diagnostics-beyond-returns]] (the four-metric health check) · [[reinforcement-learning-implementation-complexity-barrier]] (TIS is the sharpest instance)

Complicated or contested by this article: [[reasoning-as-emergent-behavior-from-pure-rl]] (self-reflection may be elicited, not emergent; length growth partly artifactual) · [[verifiable-rule-based-rewards-prevent-reward-hacking]] (verifier recall trades against model quality)

Related: [[rlvr-reinforcement-learning-with-verifiable-rewards]] · [[reinforcement-learning-from-human-feedback-rlhf]] · [[distillation-beats-from-scratch-rl-for-small-reasoning-models]] · [[value-function-init-from-reward-model-in-rlvr]] · [[2501.12948 - DeepSeek-R1 (RL for Reasoning)]] · [[PPO vs GRPO - Policy Optimization for Alignment]] · [[2411.15124 - Tulu 3 (RLVR Post-Training)]]

---

## Bibliography (as cited by the article)

[1] Yu, Qiying, et al. "DAPO: An open-source LLM reinforcement learning system at scale." arXiv:2503.14476 (2025).
[2] Guo, Daya, et al. "DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning." arXiv:2501.12948 (2025).
[3] Liu, Zichen, et al. "Understanding r1-zero-like training: A critical perspective." arXiv:2503.20783 (2025).
[4] Yao, F., Liu, L., Zhang, D., Dong, C., Shang, J., Gao, J. "Your efficient RL framework secretly brings you off-policy RL training." Aug 2025. https://fengyao.notion.site/off-policy-rl
[5] Olmo Team, et al. "Olmo 3." arXiv:2512.13961 (2025).
[6] Zheng, Chujie, et al. "Group sequence policy optimization." arXiv:2507.18071 (2025).
[7] Zhao, Yuzhong, et al. "Geometric-mean policy optimization." arXiv:2507.20673 (2025).
[8] Chen, Aili, et al. "MiniMax-M1: Scaling test-time compute efficiently with lightning attention." arXiv:2506.13585 (2025).
[9] Kimi Team, et al. "Kimi k1.5: Scaling reinforcement learning with LLMs." arXiv:2501.12599 (2025).
[10] Hu, Jingcheng, et al. "Open-Reasoner-Zero: An open source approach to scaling up reinforcement learning on the base model." arXiv:2503.24290 (2025).
[11] Schulman, John, et al. "Proximal policy optimization algorithms." arXiv:1707.06347 (2017).
[12] Schulman, John. "Approximating KL Divergence." http://joschu.net/blog/kl-approx.html (2020).
[13] Shao, Zhihong, et al. "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models." arXiv:2402.03300 (2024).
[14] Liu, Aixin, et al. "DeepSeek-V3 technical report." arXiv:2412.19437 (2024).

Author: Cameron R. Wolfe, Ph.D. — Senior Research Scientist at Netflix, *Deep (Learning) Focus*.
