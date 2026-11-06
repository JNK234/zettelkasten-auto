Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #supported

# A verifier that is too rigid, such as exact numeric matching, exact IDs or path-dependent records, fails correct agents, and these false negatives are as damaging as the false positives of a hackable check.

## Relations
informs:: [[QUE 02 - Designing a verifier so high reward means solved]], [[QUE 03 - Grade end state, trajectory or text]]
supported-by:: [[EVD - Grading checks only the unique end state and accepts any valid path. Fixed auto-incr]], [[EVD - On Task 32, every model in one trial each reached the gold's final state with a sing]], [[EVD - Opus 4.5 scored 42% on CORE-Bench until rigid numeric matching, ambiguous specs and]]
refines:: [[CLM - Anchor reward in a deterministic check of the outcome]]

## Scope
Holds for end-state and output checks; the fix is to accept every correct end state and output form while still rejecting trivial matches.

## Related zettels
- [[fixed-id-sql-verifiers-as-implicit-side-effect-checks]]
- [[grader-strictness-bugs-can-invert-model-rankings]]
- [[outcome-driven-verification-in-agent-evaluation]]
