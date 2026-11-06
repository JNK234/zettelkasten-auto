
Created: 11-08-2026 19:15

Status: #baby

Tags: [[evaluation-methodology]] [[llm-as-judge]] [[benchmark-design]]

# Grader Failure Routing

## What This Is

Your grader tries to parse a verdict out of a model's response and fails — the output was malformed, the regex missed, the API errored. What score does that item get?

The instinctive answer is zero. That answer is wrong, and the rule stating why is the sharpest single line in evaluation practice: **charging a grader failure to the model as 0.0 measures the grader's competence, not the model's.**

The correct routing is a third outcome. Not pass, not fail, but **unscored** — an item excluded from the numerator and denominator, and counted separately as instrument breakage.

This is implemented and not merely advised: mature eval frameworks return an explicit unscored result with a reason code (`grade_parse_failure`) rather than a zero.

## Why It Matters

Silent conversion of grader failures into zeros produces a bias that is invisible and directional. Every parsing bug, every timeout, every unexpected output format depresses the measured score of the system under test — and depresses it *more* for models whose output formatting differs from what the grader expected. You end up ranking models by how well their formatting matches your regex.

The bias is also self-concealing. A model at 60% with 10% grader failures reads as 54% with no indication that anything broke. Nobody investigates a plausible number.

Tracking unscored items separately converts a hidden bias into a visible quality signal: a rising unscored rate means your instrument is degrading, which is information you want and would otherwise never receive.

## Key Details

The validity rules that travel with this one, each preventing a specific failure:

**Every sample must be possible to succeed at.** If an environment defect makes a task unachievable, failures read as model limitation when they are harness breakage — a false negative about capability.

**Every sample must be possible to fail at.** A constrained action space where any behaviour scores as success manufactures a 100% that means nothing — a false positive. Zero-variance items carry no information in either direction.

**Substring matching only when the matched string is itself the ground truth.** A CTF flag is definitionally identical to success — matching it is unhackable. A refusal keyword is a *symptom*, and matching symptoms is trivially optimizable: `"I cannot recommend... but here are the steps"` scores a harmful completion as safe. The line is whether the string *constitutes* the outcome or merely correlates with it.

**Cap invalid samples at roughly 10%.** Beyond that, the measurement is dominated by dataset defects.

**Name the eval for what it actually covers.** An eval over 200 Python function-completion problems named `coding` invites every reader to treat the score as a coding-ability estimate.

Two hardening details for LLM graders, both learned from real exploits. Bind the grade regex to the **last** `GRADE:` match, not the first — a greedy leading match with DOTALL. A first-match regex lets the model under test grade itself by echoing `GRADE: C` into its own answer. And when neutralizing structural delimiters in model-controlled fields, match a **literal space** rather than `\s`, since `\s` also matches non-breaking space and hands the model a way to pre-neutralize its own payload.

Finally: grader temperature commonly defaults to the provider default, near 1.0. An unconfigured grader is *sampling*. Set `temperature=0` with a fixed seed, and measure epoch-to-epoch spread as the residual noise floor, since seeding is provider-limited and reduces rather than abolishes variance.

## Tensions

The behaviour-versus-statements rule is the deepest of the set and belongs with [[construct-and-operationalization-gap]]: a self-report is a different random variable from the behaviour it describes, and the error is systematic in the reassuring direction — models describe themselves as safer than they act.

Possible-to-succeed guards the same territory as [[elicitation-gap-as-eval-false-negative]] (harness defect masquerading as incapability); possible-to-fail guards [[reward-hacking-as-eval-false-positive]] (a score that cannot go down is not a measurement).

Grader self-grading via prompt echo is [[llm-agent-attacks-and-prompt-injection]] pointed at the evaluation harness rather than at a deployed application — the eval pipeline is an attack surface, and one that is rarely threat-modelled.

Open question: the unscored category is honest but creates a reporting burden nobody has standardized. There is no convention for how many unscored items invalidate a comparison, or whether unscored rates should be reported per-model — and since they differ systematically by model, omitting them leaves a confound in every published table.

# References

1. [[AISI - Evaluation Validity Checklist]]
