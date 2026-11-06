Created: 24-09-2026 12:10

Status: #baby

Tags: [[evaluation-methodology]] [[benchmark-design]] [[measurement-validity]]

# Rubric Agreement Does Not Establish Rubric Validity

## What This Is
Two raters can agree perfectly that an essay deserves a 4 because it is long, while length is not the skill you meant to assess. **Reliability** means the score does not depend on who grades. **Validity** means the score measures what you claim. A rubric can have the first without the second.

The review's verdict on validity was blunt: rubrics do not facilitate valid judgment *per se* (per source conclusion 2, §3.2). A rubric is an instrument, and its validity has to be argued with evidence like any other instrument.

To make "argue with evidence" concrete, the review uses **Messick's unified validity framework**. It splits validity into six questions:

| Aspect | Question | Studies (of 75) |
|---|---|---|
| Content | Do the criteria cover the skill and stay on it? | 10 |
| Substantive | Do raters actually use the intended reasoning? | 1 |
| Structural | Do the weights and dimensions match the skill's structure? | 7 |
| Generalizability | Do scores carry across tasks, occasions, raters? | 3 |
| External | Do scores correlate with other measures of the skill, and not with unrelated ones? | 15 |
| Consequential | What happens because of the scores: fairness, effects on teaching? | 2 |

Only 25 of 75 studies reported any validity evidence (per source Table 2).

## Why It Matters
The coverage pattern is the lesson. The field collected the **cheapest** evidence: experts saying the criteria look right (content) and one correlation with another test (external). It almost never checked **substantive** validity (is the rater using the right reasoning?) or **consequential** validity (what does optimizing against the score do?). Those two are the ones that catch a rubric rewarding the wrong process. Mabry's (1999) "Writing to the rubric" argued that standardized rubrics narrow student writing toward what the rubric rewards.

Rubric-scored LLM evals repeat the pattern. "Judge agrees with humans at κ = 0.7" is external validity, one aspect out of six. When the rubric becomes an RL reward, consequential validity becomes the main risk: the policy is the student writing to the rubric, and it optimizes much harder than any student.

## Key Details
- Mapping the six aspects to a rubric reward (my transfer): content = criteria review; substantive = read judge rationales, check it cites the right evidence; structural = check the criterion weights match what matters; generalizability = stability across prompts, judges, reruns; external = agreement with human labels; consequential = inspect policy outputs after RL for rubric gaming.
- The reliability levers (analytic, specific, exemplars, training) can raise agreement by steering raters toward easily countable surface features. So a reliability gain can come with a validity loss. The source frames this as a possibility supported by the reference to Mabry. It is not a measured effect.

## Tensions
- [[eval-precision-vs-eval-validity]] is the general version: precision and validity fail independently. This note adds a concrete checklist (Messick's six) and evidence that a whole field skipped the expensive items.
- [[construct-and-operationalization-gap]] names the gap that substantive and content validity try to close.
- [[reward-hacking-as-specification-gaming]] and [[verifier-failure-vs-rubric-design-failure]] are what consequential-validity failure looks like when the rubric is a training signal.
- Trade-off with [[rubric-reliability-comes-from-exemplars-and-training]]: the same moves that raise agreement can narrow what is measured.

# References
1. [[Jonsson and Svingby 2007 - The Use of Scoring Rubrics]] (§3.2, Table 2, conclusion 2)
