Created: 24-09-2026 16:50

Status: #baby

Tags: [[evaluation-methodology]] [[rl-environment-design]] [[llm-as-a-judge-framework]]

# Evaluation Unit Scales from Answer to Trace to World

## What This Is
Automated grading has moved through three units of evaluation. First, judge a single answer (is this response faithful to the context?). Then, judge an agent trace (where in a long run did it go wrong?). Then, build the whole world the agent acts in and judge behavior there. Each step applies the same skill, reliable automated grading, to a bigger unit. Patronus's public history follows this arc, and it explains why an evaluation company ended up selling RL environments.

## Why It Matters
It shows why grading skill transfers into environment building: the verifier is the hard part of an environment, and trace-level error finding is exactly the tool needed to catch reward hacks. It also suggests the environment's scoring stack should include all three levels: answer-level checks, trace-level review, world-state checks.

## Key Details
- Patronus product arc, marked [context — verify before citing] in the source note: FinanceBench (2023, QA over filings), Lynx (2024, hallucination judge) with HaluBench, GLIDER (2024, small rubric judge), Percival / TRAIL (2025, error finding in long agent traces), then "digital world models" (2026).
- Why "world model" and not "sandbox" (the note's reading): the replica simulates how the system reacts, state changes, errors, other users, not a static mock. "The difference between a fake login screen and a working fake bank."
- In RL terms: state s_t, action a_t, transition s_{t+1} = T(s_t, a_t) (the world-model part), trajectory reward R(τ).
- Datadog as a strategic investor hints at the next step: production agent monitoring sits next to pre-production evaluation (note's inference).
- The article says nothing on whether replicas are hand-built, LLM-generated, or mixed.

## Tensions
- Answer-level stage: [[evidence-spans-before-score-in-judge-output]], [[randomized-field-names-force-criterion-reading]].
- World-level stage: [[generative-simulator-four-tuple]], [[environment-and-eval-are-the-same-object]].
- Static to dynamic: [[static-versus-dynamic-agent-evaluation]], [[multi-level-agent-evaluation-framework]].

# References
1. [[TechCrunch - Patronus AI Lands 50M to Build Digital Worlds that Stress-Test AI Agents]] (§What Patronus builds, §Background on Patronus, §The numbers)
