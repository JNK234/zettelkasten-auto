Tags: #benchmark-design #evaluation-methodology #distribution-shift #dangerous-capability-evals
Status: #baby

## What This Is
The central caveat on the whole time-horizon result: the tasks being measured are *clean* — well-specified, with clear success criteria, reversible actions, and tight feedback loops — while real economically valuable work is *messy*. "Messiness" is scored on 16 factors that make real tasks harder for agents: you can make irreversible mistakes, you consume limited resources per attempt, you can't cleanly attribute outcomes to your own actions vs. external causes, you can't easily measure whether you succeeded, the spec drifts. The suite lives in the clean corner of task-space, so the trend may not transfer to the messy work the extrapolation is supposed to be about.

## Why It Matters
This is the load-bearing external-validity worry. The 7-month doubling and the projected "1-month-task agents in ~5 years" are only as meaningful as the resemblance between the measured tasks and the real tasks we care about. If clean-task horizons systematically overstate messy-task ability — and the data says they do — then the forecast is optimistic and the metric overstates how soon agents can sustain real autonomous work.

## Key Details
- Mean messiness of the suite is **3.2/16, none exceeds 8/16** (per source). For calibration, "write a good research paper" would score ~9–15/16.
- Splitting tasks into messiest-vs-cleanest halves: performance on the messy half collapses — **no model tops ~30% success on the messiest 50% of tasks** (per source).
- METR's own later follow-up: a large fraction of agent-generated SWE-bench PRs would be **rejected by real maintainers** — "task passed the automated check" overstates real usefulness.
- The 16 messiness factors are scored on HCAST/RE-Bench tasks ≥1 minute, in an appendix; the authors themselves flag this as the key limitation.

## Tensions
- Directly discounts [[exponential-capability-trend-and-seven-month-doubling]] and the legibility claim of [[task-completion-time-horizon-as-capability-metric]].
- "Passed the automated check ≠ useful" is a sibling of [[outcome-driven-verification-in-agent-evaluation]] and [[static-benchmark-inadequacy-for-agent-systems]] failure modes.
- Compounds the [[reliability-tax-80-vs-50-percent-time-horizon]]: clean *and* 50% means doubly optimistic.
- A concrete instance of [[distribution-shift]] between eval distribution and deployment distribution.

## References
1. [[2503.14499 - Measuring AI Ability to Complete Long Tasks (METR Time Horizon)]]
