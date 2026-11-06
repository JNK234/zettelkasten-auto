Created: 24-09-2026 12:10
Status: #baby
Tags: [[llm-as-a-judge-framework]] [[evaluation-methodology]] [[reward]]

# Judge Reasoning Ceiling Bounds Gradable Skills

## What This Is
To spot a broken proof you must be able to follow the proof. A judge model that cannot do a piece of reasoning also cannot tell when a response got that reasoning wrong. So the judge's own ability sets an upper limit on which skills it can grade reliably, and the limit is tightest on reasoning skills.

A better rubric does not lift this limit. It helps a weak judge a little, but not enough to replace a stronger judge.

## Why It Matters
When a rubric becomes an RL reward, the policy is pushed toward whatever the judge rewards. If the judge cannot see logic errors, fluent but wrong reasoning scores well, and the policy learns that. Choosing the judge is a bigger decision than tuning the rubric wording.

## Key Details
- Judge strength beats protocol. Human-judge Spearman ρ goes up about 0.25 when moving from GPT-3.5 to GPT-4, but only about 0.04-0.08 when moving from one score to FLASK's per-skill scoring (per source, Table 1).
  - Skill-agnostic GPT-3.5 0.360, FLASK GPT-3.5 0.424; skill-agnostic GPT-4 0.641, FLASK GPT-4 0.680.
- When GPT-3.5, Claude and GPT-4 are used as three raters, they disagree most on **Logical Robustness** (α = 0.339), a skill where humans agree fairly well. The authors attribute this to the gap in reasoning ability between the judges (per source, Table 4).
- Claude-1 as judge gives *higher* Logical Thinking scores and makes Vicuna look close to proprietary models on logic, contradicting human labels. The authors' explanation: Claude-1 is weak at logic itself, so it misses the errors (per source, C.7).
- Related bar: FLASK reserves Metacognition for questions no text model could answer, and that bar is set by what GPT-4 can do. The judge's ability defines the rubric.

## Tensions
- Partly contradicts [[rubric-scaffolding-substitutes-for-judge-capability]]: there, a checklist moves the hard part offline for a small judge. Here the rubric helps at every judge strength but does not close the gap. Possible reconciliation: scaffolding helps with deciding *what matters*, not with *checking* reasoning. (Inference.)
- [[reference-guided-judging-beats-chain-of-thought]] gives one workaround: a reference answer lets a judge check against a solution instead of re-deriving it.
- High self-agreement does not reveal this failure. A judge that consistently misses logic errors scores well on [[sampled-mean-judge-score-over-greedy]]-style consistency checks.
- Using several judge families, as in [[cross-family-judge-panel-with-unanimity]], exposes the ceiling as disagreement on reasoning skills.

# References
1. [[2307.10928 - FLASK Fine-grained Skill Set Evaluation]] (§Result 1, §Result 2, §Cost, §Metadata)
