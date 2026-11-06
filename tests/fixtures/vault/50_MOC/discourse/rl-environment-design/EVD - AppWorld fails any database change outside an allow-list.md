Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# AppWorld fails any database change outside an allow-list

## Relations
from:: [[StonyBrookNLP-appworld - Controllable World of Apps for Interactive Coding Agents]]
supports:: [[CLM - Grade the observed end state rather than the trajectory]]

## Quote
> **No collateral damage**: $\Delta D$ contains *nothing else*, except changes on an **allow-list** of harmless side effects (a login-token table updating, for example).

## Setup
AppWorld's grader design: DB diff plus allow-list.
