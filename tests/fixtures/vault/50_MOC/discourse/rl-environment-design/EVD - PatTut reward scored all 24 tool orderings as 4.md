Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# PatTut reward scored all 24 tool orderings as 4

## Relations
from:: [[Patronus AI - RL Environments Tutorial and Examples]]
supports:: [[CLM - Deterministic graders fail silently when their scope drifts from the spec]], [[CLM - Check the oracle hits max and the empty policy hits the floor]]

## Quote
> I ran the function on all 24 orderings of the four tools. **Every ordering scores 4**, including `logout_user, get_balance, verify_2fa, login_user`.

## Setup
Direct test of the tutorial reward by the note author; the docstring promised order checks.
