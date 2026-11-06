Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: B

# The generated check functions run with full Python builtins inside the trainer's own process despite a variable named safe_globals.

## Relations
from:: [[2601.05808 - EnvScaler]]
supports:: [[CLM - A fixed typed contract should separate environment from trainer]]

## Quote
> Second, despite the name `safe_globals`, the functions get the full Python builtins and run in the trainer's own process, so an LLM-written check could import modules or touch files.

## Setup
Code reading, env_util.py:102-127 @ 96ae8b0.
