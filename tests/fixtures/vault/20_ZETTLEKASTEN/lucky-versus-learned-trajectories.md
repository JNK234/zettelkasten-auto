Created: 20-08-2026 00:20
Status: #baby
Tags: [[reinforcement-learning]] [[experimental-design]] [[evaluation-methodology]]

# Lucky Versus Learned Trajectories

## What This Is

A passing trajectory is not automatically a trajectory worth training on. Three properties decide:

1. **Repeatability** — run the same task multiple times under the same policy. If outcomes swing wildly, the passes may be **lucky rather than learned**. A lucky pass looks identical to a competent one in the training data: same task, same score, same "success." But the behavior it reinforces is whatever noise happened to work that time.
2. **Generalizability** — the trajectory respects the constraints and would work again on a variant, rather than exploiting a loophole or hardcoding an answer that happens to be right.
3. **Correct scoring** — the score, assigned by a *stable* grader, actually matches what the agent did.

The insight is that all three are properties you have to *measure with repeated runs*, and none of them are visible in a single trajectory's score.

## Why It Matters

Policy gradient methods reinforce whatever produced high reward. They have no channel for "this succeeded but for the wrong reason." So a training set contaminated with lucky passes teaches the model to reproduce coincidence — and it will, because the optimizer is faithful.

This also reframes what unstable pass rates mean. The instinct is to treat variance as a nuisance to average over. Under this framing, variance is a *warning about the training data*: high variance means a larger fraction of your positive examples are noise-driven, so the same aggregate pass rate carries much less real signal at high variance than at low.

## Key Details

- Diagnostic procedure: same policy, same task, repeated runs, look at pass-rate stability.
- If unstable, the source instability lies in the **environment**, the **agent**, or the **grader** — and each demands a different fix. The source names this three-way split and gives neither an attribution procedure nor the three fixes. That is the most conspicuous hole in the piece, since it's the first question a practitioner hits.
- Property 3 depends on grader repeatability (same trajectory → same score), which is also the stated precondition for tuning reward weights at all.
- Property 2 is why deliberately-constructed adversarial exploit trajectories belong in the grader's test fixture set: a loophole-exploiting trajectory that scores high is a generalizability failure caught before training rather than after.

## Tensions

This is the data-quality side of the same coin as [[best-of-n-verification-with-reward-models]] — Best-of-N also selects high-scoring samples and inherits exactly this problem, since a lucky sample is selected as eagerly as a learned one. It supplies the practical reason [[critic-versus-critic-free-advantage-under-environment-stochasticity]] matters: environment stochasticity is precisely what makes passes lucky, and it corrupts the advantage estimate before it reaches the policy.

The environment/agent/grader attribution gap is partly answered by [[grader-failure-routing]] for the grader branch. Property 2's loophole clause is the trajectory-level statement of [[reward-hacking-as-specification-gaming]], and screening for it before training is cheaper than [[transcript-spotting-heuristics-for-reward-hacking]] after.

Open tension the source doesn't address: some legitimate exploration variance is *desirable* — a policy that is perfectly repeatable has stopped exploring (see [[exploration-and-exploitation]]). So "unstable pass rate" cannot be a pure red flag; the discrimination between healthy exploration variance and luck is left unspecified (my inference).

# References
1. [[HUD - Verifier and Reward Design for RL Environments]]
