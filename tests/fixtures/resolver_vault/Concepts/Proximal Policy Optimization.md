---
aliases: [PPO, Proximal Policy Optimisation]
tags: [rl]
---
# Proximal Policy Optimization

## Clipped Objective

The ratio is clipped to [1 - ε, 1 + ε]. ^clip-eq

### Why clip?

Large policy steps destroy performance.

## KL Penalty: adaptive

An alternative to clipping.
