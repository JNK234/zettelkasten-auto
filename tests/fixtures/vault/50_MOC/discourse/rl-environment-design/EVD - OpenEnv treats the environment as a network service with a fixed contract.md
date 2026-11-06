Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #evidence

Strength: C

# OpenEnv treats the environment as a network service with a fixed contract

## Relations
from:: [[meta-pytorch-OpenEnv - Standard Interface for Agentic RL Environments]]
supports:: [[CLM - A fixed typed contract should separate environment from trainer]]

## Quote
> OpenEnv's bet is that **the environment is a network service with a fixed contract**.

## Setup
Docker image + typed Pydantic models on the HF Hub.
