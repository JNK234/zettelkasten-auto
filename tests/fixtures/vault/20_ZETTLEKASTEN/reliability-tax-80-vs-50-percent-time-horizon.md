Tags: #evaluation-metrics #agent-evaluation #frontier-models-in-ai
Status: #baby

## What This Is
The gap between what an agent can do *sometimes* and what it can do *reliably*. The 50% time horizon is the task length the model passes half the time; the 80% time horizon is the length it passes 80% of the time. Because longer tasks always fail more, the 80% horizon is shorter than the 50% horizon — and empirically about **5× shorter** (per source). So a model with a 50-minute *50%* horizon only does ~10-minute tasks *reliably*. Demanding reliability costs you roughly an order of magnitude of task length.

## Why It Matters
Headlines quote the 50% horizon because it's the bigger, more impressive number — but nobody deploys an agent that fails half the time. The reliability tax is the correction you must apply to translate a benchmark figure into "what work could this actually be trusted with unattended." It also reframes the **reliability boundary**: frontier models are near-100% on tasks humans do in under ~4 minutes but under ~10% on tasks over ~4 hours (per source), so the *reliable* operating regime is far tighter than the 50% number suggests.

## Key Details
- 80% horizon ≈ 50% horizon / 5 (empirical, per source).
- Obtained from the same logistic fit by solving for the task length where `P(success)=0.8` instead of `0.5` — see [[logistic-fit-of-success-versus-task-length]].
- A steep slope `β` would shrink this gap (crisp cliff between can/can't); a shallow `β` widens it (gradual rolloff means reliability is hard to reach).
- The reliability boundary view: ~100% under ~4 min, <10% over ~4 hr — the "interesting" regime (usually-succeeds → usually-fails) is exactly where the horizon sits.

## Tensions
- Argues against headline-number optimism in [[task-completion-time-horizon-as-capability-metric]] — the governed quantity for safety/oversight should arguably be the 80% (or higher) horizon, not 50%.
- Reinforced by [[task-messiness-as-external-validity-gap]]: even the 50% number is on clean tasks, so reliable real-world horizons are doubly shorter.
- The rolloff's shape is the contested point in [[compounding-error-in-long-horizon-sequential-tasks]].

## References
1. [[2503.14499 - Measuring AI Ability to Complete Long Tasks (METR Time Horizon)]]
