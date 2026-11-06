Created: 24-09-2026 16:30

Status: #baby

Tags: [[benchmark-design]] [[experimental-design]] [[rl-environment-design]]

# Scenario Templates with Distractors, Hurdles, and Contrast Sets

## What This Is
Writing agent tasks one by one is slow, and hand-written tasks are easy to shortcut: grab the first search result, follow the happy path, memorize a procedure. AppWorld writes **scenarios** instead: a template with placeholders ("repurchase my last {color} {apparel}") plus three programs:
- **Setup program:** edits the base DB so this variant is solvable and interesting.
- **Solution program:** reference solution, used only to *validate* solvability, never for grading.
- **Evaluation program:** the state-based unit tests.

The setup program guarantees four properties, each blocking a specific shortcut.

## Why It Matters
This is a recipe for task generation where difficulty comes from the *world*, not from obscure instructions. Each property targets a failure mode seen in real agents, and the solution program keeps generated tasks honest.

## Key Details
1. **Well-defined:** preconditions hold (the supervisor really has past T-shirt orders in that color and size).
2. **Distractors:** plausible decoys (orders in *other* colors). A lazy agent taking the first result fails.
3. **Hurdles:** realistic obstacles (the default payment card has expired). The agent must notice and adapt.
4. **Contrast sets:** the 3 variants of a scenario differ in ways that change the correct action, so one memorized procedure fails at least one of them.
- Scale: 250 scenarios × 3 variants = **750 tasks**. Splits: Train 105, Dev 60, Test-Normal 168, Test-Challenge 417. Test-Challenge requires at least one API from held-out apps (Amazon or Gmail) to test generalization to unseen apps.
- Complexity (from validation solutions), Test-N / Test-C: apps per task 1.5 / 2.0 (max 6); unique APIs 8.2 / 10.5 (max 26); solution lines 41.3 / 56.9 (max 134).
- Difficulty tracks performance: GPT-4o ReAct scores 58.3% on level-1, ~40% level-2, 21.0% level-3. Tasks needing 60+ solution lines are under 20% for every method.

## Tensions
- Solution-program-as-validator is [[oracle-solutions-for-task-solvability]]. Compare τ²'s generator in [[fault-composition-task-generation]]: τ² composes faults for scale, AppWorld composes world edits for shortcut resistance.
- Contrast sets are what make [[scenario-goal-completion-variant-consistency-metric]] meaningful.
- Distractors directly target "partial instruction following" in [[interaction-avoidance-as-dominant-agent-failure]].
- Only 105 train tasks is small for SFT; templates allow generating more variants (per source). Risk of template-idiosyncrasy overfitting: [[distribution-match-in-generated-benchmarks]].

# References
1. [[StonyBrookNLP-appworld - Controllable World of Apps for Interactive Coding Agents]] (§"Task generation: scenarios, variants, distractors, hurdles", §"Results" difficulty collapse, §"Genuine limitations")
