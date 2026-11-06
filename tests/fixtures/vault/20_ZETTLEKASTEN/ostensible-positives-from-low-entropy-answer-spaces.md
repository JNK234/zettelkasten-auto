Created: 20-08-2026 00:20
Status: #baby
Tags: [[reward]] [[ai-alignment]] [[evaluation-methodology]] [[reinforcement-learning-in-language-models]]

# Ostensible Positives from Low-Entropy Answer Spaces

## What This Is
Verifiable rewards are supposed to be hack-proof: check the final answer against ground truth, no learned reward model to exploit. That holds only while the answer space is large enough that guessing is worthless.

It breaks when the answer is binary. If ground truth is "True" or "False", a coin flip earns reward 1 half the time. The model produces the **right answer from a wrong reasoning chain** and the rule-based verifier hands out full credit. The source calls these **ostensible positives**.

Now run RL on a dataset with an excessive proportion of such items. You are not training reasoning; you are paying a 50% base rate for guessing, and the gradient rewards whatever surface heuristic gets the model to a plausible binary faster. The correlation between reward and the behavior you want is only as strong as the answer space is wide.

## Why It Matters
The fix used is **dataset surgery, not a fancier verifier**: strip out the excess True/False items before training. Cheap, honest, and effective, but almost nobody reports doing it — which means an unknown fraction of published RLVR gains includes free reward from binary guessing.

The generalizable rule: before trusting an outcome-based verifier, ask how much reward a policy that reasons randomly would collect. Answer-space entropy is a property of your *data*, and it bounds how informative your reward can be no matter how correct your checker is.

## Key Details
- Arises from verifier *coarseness*, not verifier error. The checker is right every time; it just cannot see the chain that produced the answer.
- Same shape appears in multiple-choice evals — few options means a high floor from guessing (my inference; the source discusses only True/False).
- The source's data pipeline: three difficulty tiers of 5,000 examples each (SimpleRL-Zoo-Data-Easy; the 5,000 easiest DeepMath-103k; DeepMath-103k sampled with probability proportional to difficulty), with difficulty assessed independently by GPT-4o. The True/False pruning is applied on top.
- Fixed answer format for eval — "Please reason step by step, and put your final answer within \boxed{}" — which keeps extraction unambiguous but does nothing about the guessing floor.

## Tensions
A concrete instance of [[reward-hacking-as-specification-gaming]] inside [[rlvr-reinforcement-learning-with-verifiable-rewards]], and it complicates [[verifiable-rule-based-rewards-prevent-reward-hacking]] — rule-based rewards prevent the *learned-reward-model* failure of [[reward-model-overoptimization-hump]], but they open a data-shaped one instead. Different attack surface, not no attack surface.

It is the outcome-supervision blind spot that motivates [[process-reward-model-vs-outcome-reward-model]] and [[process-reward-model-as-step-level-verifier]]: a step-level checker would catch a wrong chain reaching a right answer. The source declines that route on cost grounds and prunes the data instead — consistent with [[process-reward-models-and-mcts-failed-as-rl-training-signals]].

Mirror image of [[overlong-filtering-as-false-negative-removal]]: both are verifier failures on the same run, one handing out undeserved reward and one handing out undeserved punishment, and both are fixed by removing samples rather than improving the verifier. Related to [[reward-hacking-as-eval-false-positive]] on the evaluation side.

# References
1. [[2508.08221 - Tricks or Traps - A Deep Dive into RL for LLM Reasoning]]
