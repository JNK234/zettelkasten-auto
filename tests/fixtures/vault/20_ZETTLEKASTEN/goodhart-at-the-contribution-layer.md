Created: 20-08-2026 00:20
Status: #baby
Tags: [[reward]] [[evaluation-methodology]] [[ai-alignment]] [[experimental-design]]

# Goodhart at the Contribution Layer

## What This Is
Reward hacking is normally discussed one level down: a policy exploits a flawed reward function. This is the same failure moved up a level, to the humans who *supply* the reward functions.

Pay a bounty per contributed RL environment and you have created a reward function whose optimizers are contributors. The measured quantity is "an environment was submitted." The intended quantity is "training on this environment improves the model." Nothing connects them. So the rational contributor builds what is cheap to build, not what is valuable to train on — many easy, gameable environments crowding out the few hard, well-verified ones.

The failure compounds because of an asymmetry: **a broken verifier is worse than no environment.** A missing environment costs you a capability you never had. A wrong verifier teaches a wrong thing, efficiently, with the full force of the optimizer behind it. So a contribution layer that admits low-quality artifacts doesn't just dilute the hub — it makes the hub actively harmful to train on.

## Why It Matters
The unaddressed gap in the Environments Hub launch is precisely this: **no mechanism validates that a contributed reward function measures what it claims.** With arbitrary Python you can't lint it; you'd have to run it against known-good and known-bad trajectories and check the ordering.

The fix has to be a downstream signal — measure "did training on this environment improve a held-out capability" and pay on *that* — but this is expensive (it requires actually running training) and slow, which is exactly why per-artifact payment is tempting. The general lesson: **any incentive scheme that pays for an artifact rather than for the artifact's downstream effect will be Goodharted at the payment boundary.**

Note also that reward hacking gets no mention anywhere in the post. In a launch entirely about reward functions contributed by strangers, that is a conspicuous silence.

## Key Details
- The distribution of contributed quality under open submission is heavy-tailed toward bad — this is a prior, not a measurement; the post reports no environment count, no quality distribution, no throughput. Its sole quantitative claim is "over 30 researchers and companies contributed environments."
- Structure of the fix (inference, not in the source): a validation harness that runs each environment against (a) a hand-written oracle trajectory, which must score maximum, and (b) a deliberately bad trajectory, which must score low. Cheap, catches inverted and constant reward functions, catches nothing subtle.
- The distinguishing test between the two Goodhart levels: at the policy level the reward function is fixed and the optimizer moves; at the contribution level the optimizer is a human and the *reward functions themselves* are the moving output.

## Tensions
This is [[reward-hacking-as-specification-gaming]] lifted one level of indirection, and the artifact-versus-effect gap is the same one in [[best-of-n-as-a-misleading-prm-metric]] — optimizing a proxy for the thing you want. It's the who-validates-the-validators problem applied to reward functions; compare [[verifier-failure-vs-rubric-design-failure]] and [[llm-verifier-exploitation-taxonomy]]. Made worse by [[environment-as-installable-package-not-config]], where arbitrary code defeats static review, and by [[environment-and-eval-are-the-same-object]], since a bad environment is simultaneously a bad training signal and a bad public benchmark. Validation via known-good/known-bad trajectories is the pattern in [[shaping-term-ablation-against-true-completion-rate]] and [[consensus-filtering-for-process-supervision-data]]. It is the cost side of the ledger whose benefit is argued in [[open-source-lag-versus-structural-ceiling]].

# References
1. [[Prime Intellect - Environments Hub]]
