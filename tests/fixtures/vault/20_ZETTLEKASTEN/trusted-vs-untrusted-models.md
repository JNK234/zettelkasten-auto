
Created: 03-06-2026 01:56

Status: #baby

Tags: [[ai-safety]] [[capability-measurement]] [[frontier-models-in-ai]]

# Trusted vs Untrusted Models

## What This Is
The problem: you want to extract useful labor from a powerful frontier model, but you cannot rule out that it's scheming — and you can't ask the possibly-scheming model to certify itself, because that's circular. How do you get a trustworthy auditor without a trustworthy model?

The building block of every control protocol is a split by *capability*, not by inspection:
- **Untrusted model (U):** the powerful frontier model you actually want to use, capable enough that you cannot rule out it's scheming (e.g. GPT-4).
- **Trusted model (T):** a weaker model assumed safe *purely because of a capability gap* — too weak to plausibly scheme without detection (e.g. GPT-3.5).

The honest concession the whole framework rests on: "The only approach we know of for confidently ruling out scheming without fundamental breakthroughs is via lack of capabilities — some models (e.g., GPT-4) just aren't capable enough to plausibly be scheming."

## Why It Matters
This split is what dodges the circularity. T can supervise U *without* asking the suspect model to vouch for itself: before U's code is ever executed, "we first send a query to an inference server running our trusted model saying, 'Hey, was this code sketchy?'" You get a cheap, dumb, *trustworthy* auditor standing between a smart, untrustworthy worker and the real world.

A third resource completes the hierarchy: scarce, expensive, fully-trusted human labor H, with an audit budget of roughly 2% of problems — the trusted oracle you can barely afford to use.

## Key Details
- T's roles: monitoring (rate suspiciousness), editing (strip likely backdoors while keeping U's structure), and serving as a fallback solver.
- Trust in T is justified *only* by weakness, which is a moving target — as models improve, today's "trusted" model may cross the threshold into plausibly-scheming.
- The authors flag the assumption as under-verified: "We think it's pretty obvious that GPT-4 isn't scheming against us, but we would like to see more investigation of this, because it seems so crucial."

## Tensions
The whole hierarchy **collapses if T is actually scheming** — monitoring, editing, and deferral all inherit the very problem they were built to escape. Separately, as the capability gap widens, T may be too weak to *comprehend* U's subtle attacks (motivating multi-step and resampling extensions like [[resampling-for-agent-control]]). Underpins [[ai-control-vs-alignment]] and is the substrate the game in [[control-evaluations-red-team-blue-team]] operates on.

# References
1. [[2312.06942 - AI Control - Improving Safety Despite Intentional Subversion]]
2. [[Shlegeris Greenblatt - The Case for Ensuring Powerful AIs are Controlled]]
