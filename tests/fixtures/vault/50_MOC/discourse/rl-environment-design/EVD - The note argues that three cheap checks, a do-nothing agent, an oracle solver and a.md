Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: C

# The note argues that three cheap checks, a do-nothing agent, an oracle solver and a look at suspiciously high pass rates, would have caught the tau-bench, CVE-Bench outbound-server and likely SWE-Lancer flaws.

## Relations
from:: [[2507.02825 - Agentic Benchmark Checklist (ABC)]]
supports:: [[CLM - Check the oracle hits max and the empty policy hits the floor]]

## Quote
> A do-nothing agent (R.13), an oracle solver (T.9), and a look at suspiciously high pass rates (T.10) would have caught τ-bench, CVE-Bench's outbound server, and most likely SWE-Lancer.

## Setup
Retrospective reading across the paper's ten audited benchmarks plus CVE-Bench; the note author's inference, no measured detection rate.
