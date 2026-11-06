Created: 10-04-2026 10:45
Updated: 10-04-2026 10:45
Type: comparison
Tags: [[robot-memory]] [[memory-management]] [[memory-mechanism]]

# Proactive vs Reactive Memory Management

## Summary

Two philosophies for handling limited memory capacity in robot policies. Proactive management decides *what to remember* at write time, filtering experiences before they enter memory. Reactive management stores everything it can, then compresses when capacity runs out. Both exist because the naive baseline — [[fifo-memory-eviction-failure-modes]] — throws away old memories regardless of importance, which destroys the temporal context a robot needs to complete multi-step tasks.

## Proactive: Write-Time Gating

MemER uses [[surprise-driven-memory-gating]] to nominate keyframes at the moment they occur. The policy computes a surprise signal (prediction error between expected and observed states) and only writes frames that cross a threshold into the memory bank.

The key insight: most robot experience is redundant. A gripper hovering over an object for 50 frames contains roughly the same information as a single frame. By filtering at write time, MemER keeps memory lean and focused on state transitions that actually matter — grasps, placements, unexpected collisions.

The risk: surprise is a heuristic. A slowly developing situation (an object gradually sliding off a table) may never trigger a sharp prediction error, so the memory bank never captures it. You lose information you cannot recover.

## Reactive: Post-Hoc Consolidation

MemoryVLA takes the opposite approach. Store every frame up to capacity, then run [[memory-consolidation-via-adjacent-merging]] when the buffer fills. Adjacent merging compresses temporally neighboring frames that are visually similar, freeing slots for new observations.

The key insight: you defer the decision about what matters. Early in an episode, you cannot know which frames will prove important for a task 200 steps later. By keeping everything initially, you preserve optionality.

The risk: if the episode is long enough, you still lose information — just later and more gracefully than FIFO. And you waste compute storing and then compressing frames that were obviously redundant from the start.

## Comparison Table

| Dimension | Proactive (MemER) | Reactive (MemoryVLA) |
|---|---|---|
| When filtering happens | Write time | At capacity |
| Selection signal | Surprise / prediction error | Visual similarity of neighbors |
| Memory utilization | Sparse — many slots unused early | Dense — always near capacity |
| Information loss risk | Silent omission of gradual changes | Lossy compression of merged frames |
| Compute cost | Low (one check per frame) | Bursty (consolidation pass at capacity) |
| Best regime | Long episodes, few critical moments | Short dense episodes, uniform importance |

## When Each Wins

**Proactive** excels in long-horizon tasks where a robot must remember a handful of pivotal moments across hundreds of steps. Think kitchen cooking: the moment you placed the pot on the burner matters; the 300 frames of stirring do not. Surprise-driven gating naturally captures the transitions.

**Reactive** excels in short, dense episodes where nearly every frame carries unique information. Assembly tasks with rapid sequential steps benefit from keeping everything and compressing only when forced. The adjacent merging preserves the general trajectory even under compression.

Both approaches benefit from storing visual memories rather than text summaries — [[visual-memory-superiority-over-text-in-robot-control]] shows images retain spatial and relational information that language descriptions lose. And both can be enhanced by [[hierarchical-memory-action-decomposition]], which organizes memories at multiple temporal scales so that compression operates on the right granularity.

## The Deeper Question

Neither approach solves the fundamental problem: *what makes a memory important?* Proactive methods approximate importance with surprise. Reactive methods approximate it with redundancy (similar neighbors are less important). A future system might use task-conditioned importance — but that requires knowing the task structure in advance, which loops back to the planning problem these memory systems exist to support.
