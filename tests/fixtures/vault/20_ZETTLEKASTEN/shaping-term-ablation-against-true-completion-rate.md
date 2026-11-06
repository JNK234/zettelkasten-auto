Created: 20-08-2026 00:20
Status: #baby
Tags: [[experimental-design]] [[reward]] [[reinforcement-learning]] [[evaluation-methodology]]

# Shaping Term Ablation Against True Completion Rate

## What This Is

"Is my agent reward hacking?" is normally answered by vibes — someone reads a hundred transcripts and forms an impression. This converts it into a controlled experiment with a directional prediction.

The procedure: train **with and without** each shaping component, and compare the runs on **true task completion rate, not proxy reward**. That second clause is the whole trick — comparing on reward is circular, because a hacked reward goes up.

Two diagnostic readings:

1. Removing the shaping term doesn't hurt completion → **it wasn't helping.** Delete it. Every shaping term you keep is exploit surface carried for free.
2. Adding the term raises proxy reward but *lowers* completion → **it's actively harmful.**

Case 2 is the payoff. "Proxy up, ground truth down" is the definition of specification gaming, expressed as a measurable divergence between two curves rather than a judgment call about intent.

## Why It Matters

It gives you a falsifiable test where the field mostly has anecdotes. A reward hack that you can only detect by reading transcripts is one you'll detect late, inconsistently, and only if you happen to look at the right trajectories. A divergence between two logged scalars you're already tracking is detectable automatically.

It also imposes an unusual discipline: shaping terms must *earn* their place. The default becomes deletion, since case 1 tells you a term contributing nothing to completion is pure liability. Most reward functions accrete shaping terms that nobody ever tests, on the theory that more signal can't hurt. It can.

## Key Details

- The comparison metric must be an independent measure of task success — the verifier's state assertion — not the training reward.
- Ablate **per component**, not the shaping block as a whole, or you can't attribute the effect.
- Complements the two magnitude rules: keep shaping small relative to terminal reward, and re-run the ablation after model upgrades, since a term that passed the test at one capability level may fail it at the next.
- Cost is the obvious objection: this is N+1 training runs for N shaping terms. The source recommends the method without pricing it.

## Tensions

Methodologically this is [[ablation-study-design-for-isolating-causal-effects]] applied to the reward function instead of the architecture, and [[ablation-driven-failure-diagnosis]] is the same instinct pointed at failures. It is the *quantitative* complement to [[transcript-spotting-heuristics-for-reward-hacking]] and [[reward-hacking-detection-via-at-scale-agent-qa]] — those find the mechanism, this finds the fact.

The proxy-vs-ground-truth divergence it measures is structurally identical to the [[reward-model-overoptimization-hump]], where RM score keeps climbing while human-preferred quality falls; the difference is that here the divergence is induced deliberately by an ablation rather than observed by accident. The need for an *independent* completion metric connects to [[dev-unseen-eval-split-for-credible-recipe-iteration]] — if the completion check and the training reward share a verifier, you've re-introduced the circularity you were trying to break (my inference).

Honest limitation: the source *recommends* this methodology and never *demonstrates* it. No experiments, no reported divergence, no worked example.

# References
1. [[HUD - Verifier and Reward Design for RL Environments]]
