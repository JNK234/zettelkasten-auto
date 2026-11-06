Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #supported

# State graders are stronger when they diff the initial and final state, checking both the intended changes and that nothing else changed.

## Relations
informs:: [[QUE 03 - Grade end state, trajectory or text]]
supported-by:: [[EVD - The released check functions also receive the initial state, so they compare before]], [[EVD - ABC's state-modification checks require the ground truth to include every reachable]]
refines:: [[CLM - Grade the observed end state rather than the trajectory]]

## Scope
Applies to environments with an inspectable database. Collateral checks must not fail harmless extra writes, such as logs or reads the spec does not forbid.

## Related zettels
- [[state-grader-must-check-untouched-state-and-all-valid-end-states]]
- [[state-diff-grading-with-collateral-damage-allowlist]]
