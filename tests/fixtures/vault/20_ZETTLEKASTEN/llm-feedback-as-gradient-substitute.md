Tags: #prompt-engineering #ai-safety #feedback-loops
Status: #baby
## What This Is
When you want to optimize a discrete natural-language input against a black-box model, you cannot do calculus: the input is text, the model exposes no gradients, and the objective (a judge's verdict) isn't differentiable. PAIR's insight is to replace the gradient with **an LLM reading the feedback and reasoning, in English, about what to change**. The attacker model reads the target's refusal text and the judge's score, writes a few sentences of diagnosis (the `improvement` field), and emits a revised prompt. That English reasoning is the directional update signal — the thing a gradient provides in a white-box optimizer like GCG.

## Why It Matters
This is the conceptual substitution at the heart of black-box prompt attacks, and it generalizes far beyond jailbreaking. It turns an intractable optimization (differentiate through a discrete, sampled, API-only system) into a tractable **conversational search guided by feedback**. The cost is rigor — you lose GCG's guaranteed optimization and any clean mechanistic account of *why* a winning prompt won — but you gain accessibility (runs on a laptop against an API) and produce human-readable, transferable artifacts.

## Key Details
- The formal objective is a constraint-satisfaction problem, not a differentiable loss (per source): find `P` such that `JUDGE(P, R) = 1` where `R ∼ q_T(P)`.
- No gradient of `JUDGE` w.r.t. `P` exists; `P` is discrete text, `T` is a black box. So PAIR is a search algorithm over semantic prompts.
- The `improvement` chain-of-thought field is the engine: read transcript → reason about the refusal → emit revised prompt.
- Ablation (per source): removing the `improvement` instruction drops success 93% → 87% — reasoning helps, though less than in-context examples (which drop it 93% → 76%).
- The same "scalar reward/judge becomes an update signal" structure underlies preference-optimization — there run toward alignment, here run adversarially against safety.

## Tensions
This trick is what makes [[pair-prompt-automatic-iterative-refinement]] black-box and cheap. It rhymes with how policy-optimization methods convert a reward into a gradient — but those have a real differentiable path, whereas here the "gradient" is an LLM's prose. The loss of mechanistic legibility is a real downside versus gradient methods, and is one of PAIR's stated limitations.

## References
1. [[2310.08419 - PAIR (Jailbreak in Twenty Queries)]]
