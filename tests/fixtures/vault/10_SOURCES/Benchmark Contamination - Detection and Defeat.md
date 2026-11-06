Created: 11-08-2026 18:45

Tags:

Status: #baby

## References

1. [Rethinking Benchmark and Contamination for Language Models with Rephrased Samples (arXiv 2311.04850)](https://arxiv.org/abs/2311.04850)
2. [Detecting Pretraining Data from Large Language Models — Min-K% Prob (arXiv 2310.16789)](https://arxiv.org/abs/2310.16789)
3. [A Careful Examination of Large Language Model Performance on Grade School Arithmetic — GSM1k (arXiv 2405.00332)](https://arxiv.org/abs/2405.00332)
4. [Deduplicating Training Data Makes Language Models Better (arXiv 2107.06499)](https://arxiv.org/abs/2107.06499)
5. [Time Travel in LLMs: Tracing Data Contamination in Large Language Models (arXiv 2308.08493)](https://arxiv.org/abs/2308.08493)
6. [BIG-bench: Beyond the Imitation Game (arXiv 2206.04615)](https://arxiv.org/abs/2206.04615)
7. [LLM Decontaminator (reference implementation)](https://github.com/lm-sys/llm-decontaminator)
8. [Deduplicate Text Datasets (reference implementation)](https://github.com/google-research/deduplicate-text-datasets)

# Notes

## Why contamination breaks a benchmark

A benchmark is an estimator. You show a model `n` held-out items, count the fraction it gets right, and treat that fraction as an estimate of "how good is this model at the underlying capability." The estimator is only valid if the items are genuinely held out — if the model has seen the item (or something that maps onto it) during training, the response is drawn from retrieval, not from the capability you meant to measure.

This is the measurement-validity failure catalogued in [[eval-integrity-as-measurement-validity]]: the number stays a number, it just stops referring to the thing on the label. Contamination is the false-positive twin of [[elicitation-gap-as-eval-false-negative]] — the model scores higher than its true capability rather than lower — and it belongs in the same family as [[reward-hacking-as-eval-false-positive]], where the score rises without the capability rising. See also [[benchmark-contamination-risk-in-public-datasets]] and [[data-leakage-in-static-llm-benchmarks]].

The practically important consequence: **a contaminated benchmark loses discriminative power exactly where it is needed.** If several models have absorbed a public test set to differing degrees, the resulting ranking is a ranking of *contamination exposure*, not of capability. A benchmark is supposed to answer a decision question — "is model A better than model B on this skill" — and a contaminated one answers a different question than the one asked.

The number that should anchor your intuition: **Llama-2 13B fine-tuned on *rephrased* GSM8K test items scores 95.3% on GSM8K, up from a 28.7% baseline** (2311.04850). Not the original strings. Paraphrases. And every standard decontamination filter passed them clean.

---

## The detection toolkit, in escalating order

Each rung costs more and catches more. The critical structural point is that **each rung is defeated by a transformation that is cheap for a contaminator and invisible to the rung below it.**

### Rung 1 — n-gram / substring overlap

The industry default. GPT-3, GPT-4, PaLM, and Llama all report decontamination based on some variant of "flag a test item if it shares an n-gram with training data" (typically n = 8 to 13).

**What it does:** exact string matching on a sliding window.

**Cost:** minutes. This is the cheapest useful check available — the first thing to run against any eval set of unknown provenance.

```python
# ABOUTME: n-gram overlap contamination check between a test set and a training corpus.
# ABOUTME: This is the industry-baseline decontamination filter (GPT-3/PaLM/Llama style).

from collections import defaultdict


def ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    """Word-level n-grams. Normalize first — casing/punctuation defeat naive matching."""
    toks = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text).split()
    return {tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}


def build_index(train_docs: list[str], n: int = 13) -> dict:
    """Invert the training corpus: n-gram -> doc ids that contain it."""
    index = defaultdict(set)
    for doc_id, doc in enumerate(train_docs):
        for g in ngrams(doc, n):
            index[g].add(doc_id)
    return index


def contamination_report(test_items, index, n=13, min_hits=1):
    """Flag test items sharing >= min_hits n-grams with the training corpus."""
    flagged = []
    for i, item in enumerate(test_items):
        grams = ngrams(item, n)
        hits = {g: index[g] for g in grams if g in index}
        if len(hits) >= min_hits:
            overlap_frac = len(hits) / max(1, len(grams))
            sources = set().union(*hits.values())
            flagged.append(
                {"idx": i, "n_shared": len(hits), "overlap_frac": overlap_frac, "train_docs": sources}
            )
    return flagged, len(flagged) / max(1, len(test_items))
```

**Why it fails:** n-gram matching is a test for *string identity*, but contamination is a property of *semantic content*. Change one word every `n` tokens and the overlap goes to zero while the information the model extracted stays intact. 2311.04850 measured this directly: against rephrased and translated test samples, **10-gram overlap detection scored F1 = 0** — it caught nothing, at all.

### Rung 2 — MinHash / near-duplicate detection

Fuzzy matching. Instead of exact n-grams, estimate Jaccard similarity between shingle sets using a hashed signature, so small edits don't destroy the match.

Lee et al. (2107.06499) is the canonical implementation and the source of the standard hyperparameters:

- **NearDup (MinHash):** 5-gram shingles, 9,000 hashes, banded LSH with b = 20, r = 450, Jaccard threshold 0.8, plus an edit-similarity ≥ 0.8 confirmation pass.
- **ExactSubstr (suffix array):** flags any shared substring of ≥ 50 BPE tokens — a linear-time exact-match complement to the fuzzy pass.

Their findings are the strongest single argument that dedup is not optional hygiene but a correctness requirement:

| Dataset | NearDup examples removed | Token-level duplication |
|---|---|---|
| C4 | 3.04% | 7.18% |
| RealNews | 13.63% | 19.4% |
| LM1B | 4.86% | 0.76% |
| Wiki-40B | 0.39% | 2.76% |

- **Memorized emission dropped ~10×**: 1.57% of unprompted-generation tokens copied verbatim from training data → 0.26% after NearDup.
- **Train-test contamination exceeded 4% of validation sets**; for C4 specifically, **4.6% of validation examples had an approximate duplicate in train.** That is a benchmark whose reported perplexity is meaningfully wrong before anyone does anything adversarial.
- Deduplicated models matched or beat baseline accuracy **with fewer training steps**.

```python
# ABOUTME: MinHash LSH sketch for near-duplicate detection between eval items and a corpus.
# ABOUTME: Rung 2 of the decontamination escalation - catches edits that defeat n-gram matching.

import hashlib
import random
from collections import defaultdict

_MASK = (1 << 61) - 1  # Mersenne prime modulus for the hash family


def shingles(text: str, k: int = 5) -> set[str]:
    toks = text.lower().split()
    return {" ".join(toks[i : i + k]) for i in range(max(0, len(toks) - k + 1))}


def _hash_family(num_hashes: int, seed: int = 0):
    """Universal family h_i(x) = (a_i * x + b_i) mod p, seeded for reproducibility."""
    rng = random.Random(seed)
    return [(rng.randrange(1, _MASK), rng.randrange(0, _MASK)) for _ in range(num_hashes)]


def minhash_signature(text: str, coeffs, k: int = 5) -> list[int]:
    """Signature[i] = min over shingles of h_i(shingle). Jaccard ~= fraction of matching slots."""
    sh = shingles(text, k)
    if not sh:
        return [_MASK] * len(coeffs)
    base = [int.from_bytes(hashlib.sha1(s.encode()).digest()[:8], "big") for s in sh]
    return [min(((a * x + b) & _MASK) for x in base) for a, b in coeffs]


def estimated_jaccard(sig_a, sig_b) -> float:
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)


def lsh_candidates(signatures: dict, bands: int = 20):
    """Band the signature; items colliding in ANY band are candidate near-duplicates.
    P(collision) ~= 1 - (1 - s^r)^b -- an S-curve with the knee near s = (1/b)^(1/r)."""
    rows = len(next(iter(signatures.values()))) // bands
    buckets = defaultdict(set)
    for key, sig in signatures.items():
        for band in range(bands):
            chunk = tuple(sig[band * rows : (band + 1) * rows])
            buckets[(band, hashlib.sha1(str(chunk).encode()).hexdigest())].add(key)
    pairs = set()
    for members in buckets.values():
        if len(members) > 1:
            members = sorted(members)
            pairs.update((a, b) for i, a in enumerate(members) for b in members[i + 1 :])
    return pairs


# Usage: coeffs = _hash_family(128); sigs = {i: minhash_signature(t, coeffs) for i, t in enumerate(docs)}
# for a, b in lsh_candidates(sigs): if estimated_jaccard(sigs[a], sigs[b]) >= 0.8: flag(a, b)
```

**Demonstrating the failure directly.** Take one training document, and three test items: (0) a verbatim copy, (1) a *paraphrase* preserving the full meaning, (2) an unrelated sentence. Running both detectors above:

| Test item | 8-gram overlap | MinHash est. Jaccard |
|---|---|---|
| 0 — verbatim copy | **flagged** | **1.00** → flagged |
| 1 — paraphrase (same meaning) | *missed* | **0.00** → missed |
| 2 — unrelated | not flagged (correct) | 0.00 (correct) |

The paraphrase scores **identically to the unrelated sentence** under both filters. A contaminated item and a clean item are indistinguishable to every surface-form method, which is the mechanical reason behind the F1 = 0 result — not a tuning failure, but the detectors measuring the wrong quantity. Everything from Rung 3 onward exists to close this exact gap.

**Why it fails:** MinHash still operates on surface tokens. A paraphrase that preserves meaning while replacing most content words drives Jaccard well below 0.8. Translation to another language drives it to ~0. MinHash raises the cost of evasion from "trivial" to "cheap," not to "hard."

### Rung 3 — embedding similarity

Encode test items and training items into a semantic vector space; flag close neighbors. This *should* survive paraphrase, since paraphrases are semantically close by construction.

**Why it fails:** two ways, and 2311.04850 hit both. Using Multi-QA BERT and `distiluse-base-multilingual-cased-v1` at a 0.5 threshold, embedding search (a) produced **false positives** — legitimately distinct test items in the same topic cluster sit close together in embedding space, so a threshold tight enough to catch paraphrases also nukes clean items; and (b) **failed on translated samples**, because cross-lingual alignment in these encoders is weaker than same-language paraphrase alignment. There is no threshold that separates "paraphrase of test item" from "different problem about the same topic" cleanly enough to run unsupervised at corpus scale.

### Rung 4 — membership inference (Min-K% Prob)

Change the question. Rungs 1–3 ask "is this test string in the training corpus?" — which requires access to the training corpus. Min-K% Prob (2310.16789) instead asks **"does this model behave like it has seen this text?"** using only next-token probabilities. That is black-box, corpus-free, and therefore the only rung that works against a closed frontier model.

**The intuition:** if a model has never seen a passage, then somewhere in that passage it will be surprised — a few tokens will have genuinely low probability. If it has memorized the passage, even the hard tokens are predictable. So don't average over all tokens (that's just perplexity, which is dominated by the easy tokens); **look only at the worst tokens.**

**The statistic.** For text `x = x_1 ... x_N`, compute each token's log-likelihood `log p(x_i | x_<i)`. Take the set `Min-K%(x)` of the `k%` tokens with the *lowest* probability, and average over just those:

$$\text{Min-K\% Prob}(x) = \frac{1}{|\text{Min-K\%}(x)|} \sum_{x_i \in \text{Min-K\%}(x)} \log p(x_i \mid x_1, \dots, x_{i-1})$$

Decision rule: if the value exceeds threshold ε, call it a **member** (seen in training); below ε, **non-member**. Tuned `k = 20` in the paper.

The contrast with perplexity is the whole point. Perplexity averages over all `N` tokens, so a long passage full of easy function words looks confident whether or not the model memorized it. Min-K% discards that mass and concentrates entirely on the tail where memorization actually shows up.

**Results.** On WikiMIA — 394 balanced pairs, members = Wikipedia articles created before 2016, non-members = event articles created after 1 Jan 2023, bucketed at 32/64/128/256 tokens so the temporal split gives gold labels:

| Method | AUC |
|---|---|
| **Min-K% Prob** | **0.72** |
| Perplexity (LOSS) | 0.67 |
| Zlib compression ratio | 0.65 |
| Neighborhood attack | 0.65 |
| Lowercase comparison | 0.61 |

A 7.4% improvement over the best baseline. Case studies: detecting copyrighted Books3 excerpts in GPT-3 (`text-davinci-003`) reached **AUC 0.88**, with ~90% of the 100 tested books showing >50% of excerpts flagged. On controlled downstream contamination (LLaMA-7B trained on RedPajama with injected eval examples): BoolQ 0.91, CommonsenseQA 0.80, IMDB 0.98, TruthfulQA 0.74 — average 0.86 vs 0.84 for the best baseline.

**Why it fails / where to be careful:** an AUC of 0.72 is a weak classifier. It supports population-level claims ("this corpus is probably contaminated") but *not* per-item adjudication — at 0.72 you cannot responsibly delete a specific eval item and claim it was leaked. Detection also degrades as contamination gets subtler: the paper found detection is *easier* with higher learning rates and more repetitions of the contaminant, which is exactly the opposite of the modern regime (huge corpora, one epoch, low LR). And it requires logprobs, which frontier APIs increasingly restrict. Subsequent work has argued that much of the apparent MIA signal on temporal-split benchmarks like WikiMIA comes from distribution shift between the pre- and post-cutoff text rather than membership per se — treat AUCs from temporally-split MIA benchmarks as an upper bound.

### Rung 5 — guided probes (Time Travel in LLMs)

Behavioral rather than statistical. Golchin & Surdeanu (2308.08493) exploit a simple asymmetry: **if a model has memorized a specific dataset partition, telling it the dataset name and split should help it reconstruct held-out instances.** A clean model gains nothing from that hint.

The method:
- **Guided instruction** — prompt includes the dataset name and partition ("complete the second sentence of instance X from the *train* split of dataset D").
- **General instruction** — same completion task, no dataset/partition identifiers.
- **Instance-level detection:** measure overlap between completion and the true reference via ROUGE-L or BLEURT; flag if guided ≫ general, or if the match is exact/near-exact.
- **Partition-level detection:** either (a) statistical significance test on the guided-vs-general score difference across the partition, or (b) a few-shot GPT-4 classifier judging exact/near-exact matches.

**Result: 92–100% accuracy in detecting whether an LLM is contaminated**, across seven datasets. GPT-4 was flagged as contaminated on AG News, WNLI, and XSum.

**Why it fails:** it needs the model to reproduce reference text, so it works on datasets with distinctive verbatim strings (news snippets, NLI sentences) and works poorly on short-answer benchmarks where the "instance" is a number. It detects *verbatim* memorization; a model that learned the answer distribution without memorizing the phrasing passes clean. It is also easily suppressed — any RLHF'd refusal to regurgitate datasets kills the signal without removing the contamination.

### Rung 6 — held-out re-creation (GSM1k)

The nuclear option: don't detect contamination, *outrun* it. Build a fresh test set that matches the original's distribution and has demonstrably never been published.

Scale AI's GSM1k (2405.00332) re-created 1,000 grade-school math problems mirroring GSM8k's style and difficulty, using human annotators, and held it private. If a model's GSM8k score is real capability, GSM1k should match; if it's memorization, GSM1k drops.

**The honest read — and this is the part that gets misquoted:**
- Accuracy drops of **up to 8%** were observed, and *several model families showed systematic overfitting across almost all model sizes*.
- **But many models, especially frontier ones, showed minimal signs of overfitting.**
- **All models retained the ability to solve novel problems** guaranteed absent from training data. Contamination inflated scores; it did not manufacture the capability from nothing.
- The memorization link is real but partial: **Spearman r² = 0.36** between a model's probability of generating GSM8k examples and its GSM8k→GSM1k gap.

The disciplined conclusion is narrower than the headline: contamination is a *real and measurable* inflation term concentrated in specific model families, not a blanket indictment of every reported score. The common paraphrase — "GSM1k proved the benchmarks are fake" — over-reads the paper considerably. Related: [[memorization-paradox-in-dynamic-benchmarks]], [[static-benchmark-inadequacy-for-agent-systems]].

**Why it fails:** cost, and a clock. Re-creation requires expert annotators and careful distribution matching — expensive, slow, and itself a source of error if the new items are subtly harder or easier. And a held-out set is only held out until you use it: every evaluation leaks a little, and once results are published the set begins decaying toward the public one.

---

## The special case: the BIG-bench canary, and why a social contract is not a defense

BIG-bench (2206.04615) tried the cheapest possible solution. Every file in the benchmark carries a **canary GUID** — a unique random string plus an instruction that any document containing it must be excluded from training corpora. The idea is elegant: it gives crawler operators a machine-detectable "do not train" marker, and it gives researchers a probe (ask a model to reproduce the canary GUID; if it can, it was trained on BIG-bench).

It did not hold, for reasons worth being precise about, because the same failure modes recur in every "mark the data" scheme:

1. **It is opt-in and unenforceable.** The canary only works if every party assembling a pretraining corpus implements the filter. Most large scrapes did not, and there is no penalty for skipping it.
2. **It travels with the file, not with the content.** The moment a BIG-bench task is quoted in a blog post, discussed in a paper, reformatted into a leaderboard, or reposted to a forum, the content propagates and the canary does not. The mark is on the container; contamination happens through the content.
3. **Derivative and synthetic data strips it.** A model asked to generate practice problems "like BIG-bench task X" emits contaminated content with no canary attached.
4. **The probe direction is weak.** Failure to reproduce the GUID is not evidence of cleanliness — models are bad at reciting random strings seen once, so a negative result is uninformative. Only the positive result carries signal.

Multiple models have been observed reproducing the canary string, confirming BIG-bench material reached training corpora despite the mechanism. **The lesson generalizes: any contamination defense that depends on the *good behavior of the party being evaluated* is not a defense.** It is the core argument for holding a test set privately rather than publishing it alongside a request not to train on it.

---

## Accidental contamination via synthetic data

This is the failure mode that has grown fastest, and the most insidious, because **no one has to do anything wrong for it to happen.**

The mechanism: labs generate synthetic training data from a strong teacher model. That teacher has itself seen the benchmarks. Asked to produce "diverse math word problems" or "coding exercises," it emits things that are near-copies of test items it absorbed. The synthetic corpus is fresh text that has never appeared on the web, so it passes every n-gram, MinHash, and canary check trivially — while carrying test-set content.

2311.04850 quantified this in real, widely-used datasets:

| Dataset | Contamination found |
|---|---|
| StarCoder-Data (2.4G subset) | **15.9%** overlap with HumanEval |
| CodeAlpaca (GPT-generated) | **12.8%** — 21 of 164 HumanEval samples |
| RedPajama-Data-1T (16G subset) | **8.5%** overlap with HumanEval |
| Evol-Instruct-Code | **7.9%** overlap with HumanEval |
| MATH training set | **1.6%** self-contamination with its own test set |
| FLAN CoT | **0.5%** MMLU contamination |

CodeAlpaca and Evol-Instruct-Code are GPT-generated instruction datasets — the contamination arrived through the teacher, not through a scrape. The MATH figure is its own quiet scandal: a benchmark's *official train split* overlapping its own test split, meaning legitimate fine-tuning on the sanctioned training data leaks test content.

The 2311.04850 headline numbers show what this buys a model that trains on rephrased test items:

| Benchmark | Model | Baseline | After training on rephrased test items |
|---|---|---|---|
| MMLU | Llama-2 7B | 45.3% | **88.5%** |
| MMLU | Llama-2 13B | 54.8% | **89.9%** |
| GSM-8K | Llama-2 7B | 14.6% | **86.7%** |
| GSM-8K | Llama-2 13B | 28.7% | **95.3%** |
| HumanEval | CodeLlama 7B | 32.9% | **67.7%** |
| HumanEval | CodeLlama 13B | 36.0% | **81.1%** |

A 13B model reaching GPT-4-comparable numbers, with every standard filter reporting clean. This is the single most persuasive table in the contamination literature.

**The proposed mitigation — the LLM decontaminator:** two-stage. (1) For each test item, retrieve top-k most similar training items by embedding search. (2) Pass each (test, train) pair to a strong LLM (GPT-4) and ask whether they are substantively the same problem. Stage 1 gives recall cheaply; stage 2 gives precision, and crucially handles paraphrase and translation because the judge reasons about content rather than strings — the same LLM-as-judge pattern documented in [[llm-as-judge-evaluation-pattern]] and [[llm-as-automated-evaluator-with-validation]], applied to decontamination instead of scoring.

Cost is O(k) LLM calls per test item. That is affordable for a benchmark-sized test set and utterly infeasible for a corpus-vs-corpus comparison — which makes this a *test-set-side* defense, the right shape for anyone who controls the benchmark but not the training corpus. In practice: cache embeddings, batch the judge calls, and report both the flag rate and a sample of adjudicated pairs so the judge's own precision can be spot-checked.

---

## What actually works

The escalation above has a depressing shape: every detection method is defeated by a transformation cheaper than the detection. That asymmetry does not close. So the real answer is architectural, not algorithmic — **stop relying on secrecy of a static artifact.**

1. **Held-out / private test splits.** Publish a public dev set for iteration; hold the scored set privately and serve it only through an evaluation API. This is the single highest-leverage design choice, for the blunt reason that *a benchmark that is never published cannot be trained on.* It is also why commercial benchmark providers can offer something a public leaderboard structurally cannot — the guarantee that the items stayed unseen is itself the deliverable. The pattern mirrors [[dev-unseen-eval-split-for-credible-recipe-iteration]].
2. **Live / continuously refreshed benchmarks.** Draw items from a stream that postdates any plausible training cutoff — recent GitHub issues, newly published papers, newly filed cases, freshly authored problems. Contamination has a *half-life*: a benchmark refreshed faster than models retrain is structurally uncontaminatable. The cost is that scores across refreshes are no longer strictly comparable, so you need item-level difficulty calibration to keep the scale stable — which is exactly what IRT provides (see [[Benchmark Efficiency and Saturation - IRT and Label Noise]]).
3. **Held-out re-creation as an audit, not a product.** GSM1k-style twins are too expensive to run continuously, but running one periodically against your own public benchmark gives you a direct measurement of the inflation term.
4. **Contamination as a reported quantity, not a binary.** Ship a contamination estimate alongside the score — n-gram flag rate, LLM-decontaminator flag rate, and where logprobs are available a Min-K% population statistic. Labs can then discount appropriately instead of arguing about whether the benchmark is "clean."
5. **Dedup the corpus anyway.** 2107.06499's result stands independently: dedup cuts memorized emission 10×, improves perplexity, and reaches target accuracy in fewer steps. It is the cheapest thing on this list and it is strictly positive-value even ignoring contamination.
6. **Task formats that resist memorization.** Execution-graded, environment-interactive, and multi-turn agentic tasks are far harder to contaminate than static Q&A, because the answer is a trajectory through a stateful environment rather than a string. See [[execution-based-tool-evaluation]], [[static-versus-dynamic-agent-evaluation]], and [[static-versus-interactive-benchmark-divergence]].

The composite property worth naming: a contamination-resistant benchmark is **expert-authored, never published, continuously refreshed, and shipped with a measured residual contamination estimate.** Each of those four maps onto a distinct failure mode above — synthetic-data leakage, canary non-enforcement, half-life decay, and detector unreliability respectively. Dropping any one of them re-opens the corresponding hole. Note that n-gram dedup, the near-universal industry answer, addresses none of the four; it is table stakes and provably defeated by paraphrase at F1 = 0.

---

## Open questions to hold

- Min-K% at AUC 0.72 is too weak for item-level adjudication. Is there a variant with enough precision to justify deleting individual eval items — or is population-level estimation the ceiling for black-box detection?
- The synthetic-data channel means contamination now propagates through model weights rather than through text. No corpus-side filter can catch that. Does the only remedy become held-out-by-construction?
- What is the actual half-life of a published benchmark under current retraining cadence? That number determines the required refresh frequency for any live benchmark, and nobody has measured it directly.
- GSM1k found frontier models mostly clean. Is that because frontier labs decontaminate well, or because their scale makes single-exposure memorization weaker — and does that hold as post-training gets more benchmark-targeted?
