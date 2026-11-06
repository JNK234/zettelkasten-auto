Created: 20-08-2026 00:20
Status: #baby
Tags: [[reward]] [[evaluation-methodology]] [[reinforcement-learning-in-language-models]] [[experimental-design]]

# A Bounty List Is a Map of What Cannot Be Verified

## What This Is
A reading technique before it's a concept. When an organization publishes a list of things it will **pay** for, that list is a more honest statement of its research frontier than anything in its prose. Prose describes what you've done; a bounty describes what you can't do and want badly enough to buy.

Prime Intellect's Environments Hub launch buries a "moonshot environments" list at the end. Read as a research agenda rather than as decoration, it's the densest signal in the post — a direct statement of what they think currently cannot be trained. And every single item has the same shape: **the verifier is the hard part.**

- **Robust code-quality evaluations for agentic software engineering.** Note *quality*, not correctness. Tests-pass is already solved and already gamed. This asks for a verifier over maintainability — where programmatic checking runs out entirely.
- **Evaluating filesystem and memory usage for long-running tasks.** Long-horizon: rewards get sparser and credit assignment gets harder as the horizon grows.
- **Adaptive coherent instruction-following in realistic multi-turn interactions.** Multi-turn credit assignment.
- **High-quality creative writing and style adherence.** The unverifiable domain. No programmatic checker exists for good prose.
- **Generative generalist reward models with process critiques.** A *learned* verifier that explains itself — step-level rather than outcome-level.
- **NanoGPT speedrun optimizations, terminal-friendly data viz, research-plan generation with recent papers as golden targets.** ML research itself as an RL environment.

They are not asking for more math datasets. They are explicitly funding the frontier of reward design.

## Why It Matters
The cynical prediction for a hub launch is "please contribute lots of environments" — volume, breadth, easy wins. Instead the money goes to the four or five places where verification is genuinely unsolved. That inverts the expected incentive and tells you the constraint they actually feel.

The creative-writing item carries a specific admission: **RLVR's clean guarantee has to be abandoned somewhere.** The whole appeal of verifiable rewards is that a ground-truth checker can't be sweet-talked. Style adherence has no such checker, so anyone funding it is committing to fall back on judges or rubrics and accept the exploitability that comes with them.

Corroboration that the list was a roadmap, not decoration: the nanoGPT speedrun item later became Prime Intellect's own "Measuring Autonomous AI Research" work (153 runs across 18 frontier models).

## Key Details
- The taxonomy the list implicitly walks: programmatic checker → learned outcome verifier → learned step-level verifier → judge/rubric → nothing. Each bounty sits at or past the point where programmatic checking fails.
- "Code quality" is the hardest case named because it is simultaneously high-value, high-frequency, and has a *known-gamed* cheap proxy (tests pass) already in wide use.
- Generalizable heuristic (inference): apply the same read to any lab's job postings, grant calls, or RFPs — the funding list is the negative image of the capability set.

## Tensions
Items 1 and 4 land where you must fall back on [[llm-as-a-judge-framework]], inheriting [[llm-as-judge-biases-catalog]] and [[llm-judge-position-bias]], or on rubric scoring — see [[instance-specific-rubric-as-reward-function]] and [[generic-rubrics-fail-instance-specificity-is-the-mechanism]]. Item 5 is exactly the [[process-reward-model-vs-outcome-reward-model]] and [[process-reward-model-as-step-level-verifier]] direction, whose difficulty is documented in [[process-reward-models-and-mcts-failed-as-rl-training-signals]]. Item 2 is [[credit-assignment-problem]] at long horizons. Abandoning verifiability re-opens [[reward-hacking-as-specification-gaming]], which [[verifiable-rule-based-rewards-prevent-reward-hacking]] had closed — and note that [[kl-penalty-is-unnecessary-under-a-ground-truth-verifier]] stops applying once the verifier is learned. The alternative to funding hard verifiers is [[reshape-the-task-to-fit-the-verifier]].

# References
1. [[Prime Intellect - Environments Hub]]
