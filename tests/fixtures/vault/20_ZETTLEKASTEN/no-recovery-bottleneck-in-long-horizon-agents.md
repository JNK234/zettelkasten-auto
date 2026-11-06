Created: 24-09-2026 16:44

Status: #baby

Tags: [[agentic-ai-systems]] [[error-propagation]] [[memory-management]]

# The No-Recovery Bottleneck in Long-Horizon Agents

## What This Is
The simple model of long-task failure is independent per-step errors: if each step succeeds with probability $p$, an $n$-step task succeeds with $p^n$. At $p=0.99$, 100 steps gives ~37% and 500 steps ~0.7% (illustration from the source note, not the paper). Real agent trajectories fall off *faster* than this, per the synthesis. The reason: errors are not independent. Once an agent commits to a wrong intermediate state, most architectures cannot detect and roll back, so the early error raises the failure risk of every later step. LEAD (2026) names this the **no-recovery bottleneck**.

## Why It Matters
It says where training signal is most valuable: in recovery, not in first-try accuracy. METR attributes its time-horizon trend mainly to better reliability and error recovery, not raw reasoning or tool skill. For RL environment design, this argues for tasks that inject or allow mid-trajectory errors and reward detecting and undoing them.

## Key Details
- METR: near-100% on tasks under ~4 human-minutes, <10% on tasks over ~4 hours; 50% horizon doubles ~every 7 months.
- Long-context web agents: 40-50% on short versions → <10% when the same task sits inside a long history, **even though the needed info is still in the window**. Trace analysis: the agent loses the objective and loops. It is goal drift, not retrieval failure, so bigger windows are not the fix.
- LoCoBench-Agent: agents re-read processed files, repeat failed calls, write redundant summaries that dilute context; ~3-5% efficiency loss per 50% length increase (the paper's "non-linear" wording does not fully match this number).
- Context-Folding: branch into a bounded sub-trajectory, then collapse it to a summary. Signals that context *management*, not length, is the binding constraint.
- "Faster than constant hazard" is asserted from the pattern across papers, not fitted.

## Tensions
- [[compounding-error-in-long-horizon-sequential-tasks]] and [[multi-step-compounding-error-in-sequential-tasks]] describe the independent-error model this note says underestimates failure.
- [[cascading-errors-in-long-horizon-tasks]] gives the robotics version of state-corrupting early errors.
- [[context-loss-in-long-horizon-agent-tasks]] is the symptom side; [[adversarial-self-play-for-error-recovery]] and [[recovery-rich-demonstrations-for-temporal-learning]] are training responses.
- [[human-time-horizon-as-grounded-difficulty-metric]] is the METR measurement this cluster leans on.

# References
1. [[2607.05775 - Beyond the Leaderboard Agent Failure Taxonomy]] (§3 Long-horizon degradation; §Cross-cutting patterns 1; §Limitations)
