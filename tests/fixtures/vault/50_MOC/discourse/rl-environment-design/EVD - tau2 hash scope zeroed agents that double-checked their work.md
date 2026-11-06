Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: A

# tau2 hash scope zeroed agents that double-checked their work

## Relations
from:: [[sierra-research-tau2-bench - Agent Evaluation Harness]]
supports:: [[CLM - Deterministic graders fail silently when their scope drifts from the spec]], [[CLM - Grader and test weaknesses inflate benchmark scores]]

## Quote
> So an agent that double-checked its work (e.g. calling `get_pending_replacement_orders_5765` after ordering a card) changed the hash and got 0. **Careful behavior was being punished.**

## Setup
Documented bug #329, fixed in v1.0.1 (banking domain).
