Created: 01-10-2026 17:00
Updated: 01-10-2026 17:00
Type: topic-overview
Sources: 2
Zettels: 22
Tags: [[llm-as-a-judge-framework]] [[evaluation-methodology]] [[inference-optimization]] [[experimental-design]]

# LLM Judge Score Bias and Contrastive Decoding

The question this page answers: when an LLM judge writes "4", how much of that is about the summary and how much is about the model liking the numeral 4? And if the answer is "a lot", can you subtract the liking out at decode time without retraining anything?

The short version. Judges have a prior over score numerals. That prior is shared by every model in a family. So a tiny, broken sibling model is a good measuring stick for the big model's prior, and subtracting the sibling's first-token log-probs from the big model's fixes much of the damage. The same "subtract a distribution that carries the unwanted preference" trick has been tried with a counter-prompt instead of a second model, and that attempt shows how easily implementation and baseline mistakes can fake or hide a decoding gain.

## Does relabelling the scale really change the verdict?

It should not. A 5-point scale labelled 1-5, 0-4, 2-6 or 3-7 carries the same information, so a good judge ranks summaries the same way under all four, and its correlation with human scores stays flat. [[score-range-bias-in-llm-judges]] is the finding that it does not stay flat. On SummEval coherence (1,600 summaries, 100 articles x 16 systems), with greedy decoding and a G-Eval prompt that forces the score to be the first token:

- Qwen2.5-14B: Spearman .468 at 1-5, .313 at 2-6.
- Llama-3.1-8B: Spearman .426 at 3-7, .257 at 2-6.

That is about a third of the agreement lost by moving the labels. The practical point: the label range is a hidden hyperparameter. A judge validated on 1-5 can quietly degrade when someone rebases the rubric to 0-4 "for cleaner math". Re-running the same items under shifted labels is a cheap robustness check, in the same family as swapping candidate order to catch [[llm-judge-position-bias]]. Both belong in [[llm-as-judge-biases-catalog]]: an input feature that should be irrelevant changes the score.

A caveat before reading too much into tiny models: the paper maps unparseable outputs to the lowest score and clamps out-of-range ones to the max. Near-zero correlations for small models are mostly formatting failures, not range bias. The same default-on-failure trap is in [[judge-parse-failure-defaults-bias-test-set-selection]].

## Where does the bias live?

Because the prompt forces the score to be the first token, you can read the model's logits over the candidate numerals directly. The pattern there is [[llm-score-numeral-preference-is-family-shared]]:

- Qwen2.5 3B and 7B put their highest first-token logit on "2" in the 0-4 range, while the most common human score is "3". The tilt shrinks from 3B to 14B but is still there at 14B.
- Llama 3B and 8B pile their outputs onto "4" in the 2-6 range.

So each family has its own favourite numeral, and siblings share it. Note the preferred numeral is not always the middle one, which is why this is not the same thing as [[central-tendency-bias-in-llm-agents]]. Siblings share the shape of the preference but not its size: Qwen2.5's maximum logit on 0-4 is about 25 at 3B, 30 at 7B and 34 at 14B.

Two consequences point in opposite directions. The bad one: checking a judge with a second model from the same family proves little, because they share the bias. That is the [[monoculture-problem-in-multi-agent-systems]] again, and the reason [[cross-family-judge-panel-with-unanimity]] draws judges from different labs. The useful one: if the bias is shared, a small sibling is a cheap instrument for measuring the big model's bias.

## How do you subtract it?

Contrastive decoding (Li et al.) was built to stop big models producing bland, repetitive text. It runs two models on the same input and prefers tokens the big model likes much more than a small model does. Whatever both believe cancels. Whatever only the big model knows survives.

[[weighted-contrastive-decoding-for-judge-scores]] applies this to the first score token:

score(i) = log p_main(i) − λ · log p_asst(i), where p_asst(i) = e^{l_i/t} / Σ_j e^{l_j/t}

Read it in words. If the assistant strongly prefers numeral i, its log-prob is near zero, so little is subtracted; numerals the assistant finds unlikely have very negative log-probs, so subtracting them pushes those up. Net effect: the shared favourite loses ground relative to the others. Two knobs, each solving a specific problem:

- λ exists because sibling logits live on different scales (25 vs 34 above). Without it, the subtraction mixes magnitudes arbitrarily. Original contrastive decoding has no such weight.
- t (temperature on the assistant) controls how sharp the subtracted preference is.

Both are grid-searched per (main model, assistant, range, metric) on a 10% dev split, over λ ∈ {0.01, 0.1, 0.5, 1.0} and t ∈ {0.5, ..., 5}. The chosen values swing from λ=0.01, t=5.0 to λ=1.0, t=0.5. There is no global setting.

What it buys: Qwen 14B with a 3B assistant goes from .313 to .426 Spearman at 2-6, and its four-range average from .392 to .438 (the 11.7% headline). Llama 8B with a 1B assistant goes from .257 to .337 at 2-6 and from .344 to .375 on average. It does not win everywhere. At 0-4 and 3-7, Llama 8B alone slightly beats both contrastive pairs, and Qwen 7B-3B is worse than 7B alone at 0-4 and 1-5. On BigGen-Bench the gain is small for Qwen (.500 to .517) and about zero for Llama (.349 to .352). The honest reading: it repairs the ranges where the bias hurts most and is roughly neutral elsewhere.

How it compares with other fixes: [[sampled-mean-judge-score-over-greedy]] averages many samples to recover granularity, but a prior that tilts every sample toward one numeral survives averaging. [[binary-judge-scoring-over-likert]] removes most numerals altogether, which leaves fewer labels to be biased between. Contrastive decoding keeps the Likert scale and corrects the prior.

## Why does a worse assistant work better?

This is the result that explains the method. [[weak-assistant-subtracts-bias-not-competence]]:

- Llama 1B as assistant beats 3B (average Spearman .375 vs .349).
- Qwen 14B with a 7B assistant gets .282 at 1-5, against .470 with a 3B assistant.
- The best assistants are terrible judges on their own. Qwen2.5-3B produces about 98% unparseable outputs as a standalone judge. Llama 1B has correlation exactly .000 at 1-5 and 2-6, which looks like a constant output.

Why: subtraction removes everything the two models agree on, and that includes the right answer. A capable assistant agrees with the main model about which summary is good, so subtracting it deletes signal along with bias. You want a model that shares the family's numeral prior but not its judgement. And a broken judge still works because the method reads only the assistant's first-token distribution over score tokens, never its generated text, so its formatting failures do not matter.

That reframes the assistant. It is not a second opinion; it is a measurement of the family prior. The method is closer to prior correction (estimate the label prior, divide it out) than to "ask a weak model and subtract". It also suggests an untested cheaper version: estimate a fixed per-family prior over score tokens from content-free inputs and subtract that, with no second model at all.

Cost is two forward passes. The paper notes that if you already run [[speculative-decoding-with-draft-models]], the draft model's logits are free. But that argument pulls against the weak-assistant result: a draft model is chosen to agree with the target as often as possible, which is exactly what makes an assistant worse here.

## Can you skip the second model and use a counter-prompt?

[[same-model-counter-prompt-contrastive-decoding]] tries this for a different bias: models trained mostly on English and Chinese drift into English when asked for Japanese. Run the same model on two prompts, one asking for Japanese and one for English, and prefer tokens the Japanese prompt makes much more likely than the English one. The English-conditioned pass plays the assistant's role: it carries the unwanted preference. The blog's hypothesis is that Mixture-of-Experts models make this especially promising, since contrast might suppress English-specialised experts.

The rule is not the paper's rule. It keeps tokens where P(JA) − P(EN) > 0.01, a difference of probabilities with no λ, and falls back to P(JA) if nothing survives. [[probability-difference-contrast-favors-already-likely-tokens]] explains why that matters. A difference scales with the size of the probabilities: 0.4 vs 0.3 gives 0.1, while 0.002 vs 0.0001 gives about 0.002 even though the second token is 20x more Japanese-specific. A log-ratio ranks the second far higher. So the difference rule means "take likely Japanese-prompt tokens the English prompt does not also love", not "find the most Japanese tokens". That is probably safer, since log-ratios blow up on rare junk tokens (the reason Li et al. add a plausibility cutoff), but it is a different method. The 0.01 floor also gives it a plausibility constraint for free: any surviving token has at least 1% probability under the Japanese prompt.

Reported results: gpt-oss-20b (MoE) line pass rate 90% to 100% on 10 Language Confusion Benchmark prompts; Qwen3-8B (dense) 100% to 80%, from 2 of 10 outputs leaking English reasoning. On 20 JamC-QA questions, gpt-oss went 20% to 15% and Qwen3 stayed at 25%. The 90% to 100% is one prompt; the 100% to 80% is two. The author says so.

## What can make a decoding gain fake?

The code behind the counter-prompt experiment is a compact catalogue of ways a decoding comparison stops measuring what it claims. Two are general enough to check in any contrastive or paired-log-prob setup.

The implementation can corrupt one of the two distributions. [[right-padded-batch-without-mask-corrupts-last-token-logits]]: both prompts go through one batched forward pass, the shorter row is padded on the right, and logits are read at position −1 with no attention mask. For the shorter row, position −1 is a pad, so its "next-token distribution" is the model's guess about what follows a run of padding. The English prefix is 9 tokens shorter than the Japanese one (15 vs 24 on the gpt-oss tokenizer), so the English row carries 9 pads at every step, and the pad id falls back to 0, a real vocabulary token, when the tokenizer has none. The bug is silent: the code runs, it is faster, and the probabilities look plausible. The repo's batching test checks only that batching runs and is faster. The test that would have caught it is equivalence: batched probabilities must match per-row probabilities within tolerance. Fix with left padding plus a mask, or by reading each row at its true last index. The judge paper sidesteps the issue by running one model per pass on a fixed prompt and reading only the first token.

The baseline can change more than the decoder. [[baseline-that-drops-the-instruction-confounds-decoding-gains]]: the contrastive arm's Japanese prompt includes "Generate in Japanese"; the baseline prompt is the same system message without that line. So "baseline vs contrastive" changes both the decoding rule and whether the model is told to write Japanese, and an explicit instruction could produce a one-prompt gain on its own. The write-up describes the baseline as getting the instruction; the code does not send it. The clean control is greedy decoding on the instruction-bearing prompt with no subtraction. The audit move that catches this is to diff the exact strings each arm sends to the model, not the description of the arms.

The same code has further problems that make its numbers uninterpretable. Raw "System: ..." strings bypass gpt-oss's harmony chat template, so outputs are text continuations and some of the "language confusion" may be template-induced ([[template-induced-capability-collapse]]). The QA parser strips spaces and newlines before matching patterns that contain them, so 15-25% accuracy sits at or below 4-way chance. All runs use temperature 0, so nothing is sampled.

The judge paper is not clean either. Its contrastive arm gets λ and t tuned per range on held-out data while the greedy baseline gets no tuning, so some of the stability may come from extra degrees of freedom ([[straw-man-baseline-problem]]). And its §4.2 prose for Llama (6.7%, .330 to .352) does not match Table 1 in v2 (.344 to .375 Spearman); the Qwen 11.7% does match. Cite the tables.

The common thread is the question [[eval-integrity-as-measurement-validity]] asks of any number: what else, besides the method working, could have produced it?

## How the pieces connect

Both works use one structure: find a second distribution that carries the unwanted preference and little of the wanted signal, then subtract it. The paper finds it in a small sibling model, measures it carefully, and the "smaller is better" ablation explains why it works. The counter-prompt version finds it in the same model under an opposing prompt, but its evaluation cannot separate the decoding rule from the prompt change, and the batching bug corrupts one of the two distributions. The paper is the result to cite; the counter-prompt idea is an untested hypothesis.

The same "subtract a direction" logic appears in activation space in [[contrastive-activation-addition]] and across layers in [[logit-lens-intermediate-predictions]] and DoLa-style decoding. Contrast used to detect rather than decode is [[contrastive-group-reward-hack-detection]]. For how judges should be built around these biases in the first place, see [[judge-design-for-spec-conformance-eval]]; for the case where judge scores become training rewards, [[rubric-rewards-and-verifier-design]].

## Open Questions

- Does a fixed per-family prior over score tokens, estimated once from content-free inputs, match the two-model method? If yes, the second forward pass is unnecessary.
- Is the family-shared numeral prior a pretraining artifact or a post-training artifact? Neither source tests this.
- Is a typical speculative-decoding draft model weak enough to be a good assistant, given that drafts are chosen to agree with the target?
- Does the bias, and the fix, hold beyond English summarization, above 14B, and for judge-finetuned models such as Prometheus?
- Does same-model counter-prompt contrast help MoE models more than dense ones once padding is fixed, the baseline gets the instruction, the chat template is applied, and the sample is large enough? A sharper test proposed in the source: MoE with shared experts (DeepSeek-V3) vs without (gpt-oss).
- How much of the paper's gain survives giving the greedy baseline an equal tuning budget?

## Contradictions

- Same-family correlation is a hazard for verification ([[llm-score-numeral-preference-is-family-shared]], [[monoculture-problem-in-multi-agent-systems]]) and the resource that makes the fix work ([[weak-assistant-subtracts-bias-not-competence]]). Both hold: correlation of bias is bad when you want independent errors and good when you want to cancel them.
- The speculative-decoding "free assistant" argument in [[weighted-contrastive-decoding-for-judge-scores]] conflicts with the finding that more capable, more agreeing assistants hurt.
- Original contrastive decoding uses log-ratios; the counter-prompt implementation uses a probability difference. They rank tokens differently ([[probability-difference-contrast-favors-already-likely-tokens]]), so results from one do not transfer to the other.
- The paper's Llama numbers disagree between §4.2 prose and Table 1 (v2).
- The counter-prompt write-up describes a baseline that gets "Generate in Japanese"; the code's baseline does not.

## Sources

1. [[2510.18196 - Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge]] — Part 1 (score range bias, family-shared numeral prior, weighted contrastive decoding, weak-assistant ablation, things to check); Part 2 and code walkthrough (counter-prompt decoding, padding bug, baseline confound).
2. [[Yoshinari Fujinuma - Homepage and Research Notes]] — the multilingual contrastive decoding experiment, probability-difference rule, code-level issues.
