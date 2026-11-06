Created: 24-09-2026 12:10
Status: #baby
Tags: [[evaluation-methodology]] [[llm-as-judge]] [[educational-measurement]]

# Rubric vs Checklist vs Rating Scale

## What This Is
People call almost any scoring sheet a "rubric": a 1-5 scale, a points table, a list of yes/no items. The educational-measurement literature uses a narrower definition. A rubric has two parts: **criteria**, which say what to look for in the work, and **performance level descriptions**, which describe what the work looks like at each quality level, from low to high (per source §What counts as a rubric).

The key move: every scoring tool has criteria. What separates the tools is the scale attached to each criterion.

| Tool | Scale per criterion |
|---|---|
| Checklist | dichotomous: has it / does not have it |
| Rating scale | levels that do not describe the work: numeric (1-5), evaluative (Excellent/Good/Fair/Poor), frequency (Always/Sometimes/Never) |
| Rubric | each level is a description of what the work looks like at that quality |

So a true rubric is a matrix. Rows are criteria, columns are levels, and each cell says what the work does, not how good it is.

## Why It Matters
Most "LLM rubrics" are rating scales: a list of dimensions with "score 1-10" and no description of what a 4 looks like versus a 7 (per source §Takeaways). Most "rubrics as rewards" in RL are checklists: binary criteria summed with weights. Calling all three by one name hides which design choice is being tested. In Brookhart's review, 15 of 51 tools studied as "rubrics" (30%) were rating scales, point schemes, or rubrics with judgment-word levels, and the literature pooled them all as "rubric effects" (per source §Findings 1).

For a judge builder, naming the tool correctly forces the right question: am I relying on the criteria, or on the level descriptions? Those are different levers with different costs.

## Key Details
- Two further axes cut across the three tools: **analytic vs holistic** (score each criterion separately and get a vector, or weigh all at once and get one scalar) and **general vs task-specific** (a family of tasks vs one task).
- Among 29 general analytic rubrics in the review, 22 used 4 or 5 levels. That is the modal design in education.
- Brookhart coded the printed artifact, not the author's label. That is how mislabeled tools were caught. The same audit applies to an LLM judge prompt: read what the scale actually says.

## Tensions
- [[per-instruction-checklist-reward]] and [[instance-specific-rubric-as-reward-function]] are checklists in this taxonomy, not rubrics. Their success is evidence about criteria, not about level descriptions.
- [[binary-judge-scoring-over-likert]] argues for collapsing to the checklist form. Brookhart's taxonomy frames the choice: you give up level descriptions, you keep criteria. See [[criteria-not-level-descriptions-drive-rubric-effects]] for why that may cost little.
- The analytic vs holistic axis maps onto [[implicit-vs-explicit-rubric-aggregation]]: explicit aggregation is analytic scoring, implicit is holistic.
- Level wording is its own design choice: [[descriptive-vs-evaluative-performance-level-descriptions]].

# References
1. [[Brookhart - Appropriate Criteria Key to Effective Rubrics]] (§What counts as a rubric, §Findings 1, §Takeaways)
