Tags: #evaluation-methodology #benchmark-design #agent-evaluation #ai-safety
Status: #baby
## What This Is
An automatically-scored agent benchmark is only trustworthy if agents can't *game* the scorer — finding a shortcut the grading function wrongly accepts (reward hacking). HCAST's defense is to run a strong agent against *every* task many times, then have humans **manually review all the transcripts** to confirm each "success" is a legitimate solve and not a scorer exploit. It's QA that scales the agent attempts but keeps a human in the verification loop.

## Why It Matters
Eval papers live or die on whether the scoring can be gamed. A binary pass/fail scorer is a tempting attack surface: an agent that writes to the expected output file without solving the task, or hits a degenerate path the scorer didn't anticipate, inflates the benchmark and invalidates every downstream time-horizon number. Catching reward hacking *before* publishing the scores is what lets you believe the curve.

## Key Details
- Claude 3.5 Sonnet attempted every task **5×** → **945 transcripts**, all **manually reviewed** to confirm legitimate solves and catch reward hacking (per source).
- Layered with: blind human QA (testers not involved in authoring run tasks to surface bugs/ambiguities), gold reference solutions where feasible, and baseline QC (manual review of most successful baselines via Loom recordings, prioritizing anomalies).
- Contamination is controlled structurally too: 178/189 tasks withheld from public release; a **canary string** (`hcast:5a51:...`) ships with the paper asking labs to exclude it from training corpora; ~2/3 of the suite is private-solution + private-problem (hard to memorize), only 15.4% easy-to-memorize (per source).

## Tensions
- Manual review of 945 transcripts is expensive and doesn't scale to the rate at which new tasks/models appear — a recurring cost ceiling on the whole suite.
- Distinct from [[outcome-driven-verification-in-agent-evaluation]] (which checks the *environment end-state* to prevent agents faking success via text) — here the concern is the scorer *itself* being exploitable, caught by human transcript audit rather than by better state-checks.
- Reward hacking is the eval-side mirror of the alignment threat in [[scheming-as-threat-model]] and [[control-evaluations-red-team-blue-team]]; relates to [[evaluator-circularity-in-ai-benchmarks]].

## References
1. [[2503.17354 - HCAST (Human-Calibrated Autonomy Software Tasks)]]
