Created: 20-04-2026 00:37
Updated: 20-04-2026 00:37
Type: idea
Status: #hypothesis
Confidence: high
Tags: [[robot-memory]] [[memory-mechanism]] [[attention-mechanism]] [[long-horizon-planning]]
Builds-on: [[learnable-keypoint-phase-embedding]] [[progress-aware-memory-for-d3-temporal-structure]] [[d1-d4-memory-content-taxonomy]] [[fifo-memory-eviction-failure-modes]] [[streaming-memory-bank-for-video-segmentation]]
Challenges: [[persistent-memory-tokens-rmt-style]]

# Step-Embedding-Keyed Memory Retrieval

## The Insight

The step embedding from PR #6 is the thesis's first intervention that actually worked — sas went from 0% to 100% (runs #31 vs. #34, clean ablation). But today it lives only inside the Stage 0 proprio stream, added once and never referenced again. Meanwhile, the Stage 2 memory is a FIFO of spatial features whose cross-attention scores entries by visual similarity only. On a long-horizon task, visual similarity is maximal for *recent* frames, not *relevant* ones — so a policy at waypoint 24 looking for "where block A was originally placed at waypoint 3" can't find it even when the frame is still in the bank.

The step embedding already encodes phase. The memory cross-attention already has a query. Nothing currently connects them.

## The Hypothesis

Inject the step embedding into both sides of memory cross-attention:
- **Write side:** When adding a frame to the memory bank, concatenate the step embedding with the stored features so each memory entry is tagged with "the phase I came from."
- **Read side:** When the policy queries memory, concatenate the current step embedding with the query features.

Cross-attention then scores memory entries by **visual similarity + phase proximity** jointly. At step 24 querying for phase-1 context, phase similarity pulls step 3's entry ahead of the recent-but-irrelevant frames at steps 20-23. The mechanism is Transformer-native; no new module needed.

## Supporting Evidence

- **Step embedding already proven.** Run #31 (ON) = 100% vs. run #34 (OFF) = 0% on sas. Per-wandb analysis, training loss was indistinguishable — the benefit is purely at generalization time, which is exactly what retrieval keying would exploit.
- **FIFO's core failure mode.** [[fifo-memory-eviction-failure-modes]] documents that FIFO is content-blind; visual similarity alone picks recency. Phase-keyed attention makes cross-attention content-aware along the temporal axis without changing eviction policy.
- **Cross-attention reads any key format.** [[streaming-memory-bank-for-video-segmentation]] shows SAM2's memory bank already accepts heterogeneous features. Appending 32-dim step embeddings to the K/V is architecturally cheap (<0.5% token dim increase).
- **Minimalism beats machinery.** [[progress-aware-memory-for-d3-temporal-structure]] proposes a separate GRU-updated progress token — more parameters, more hyperparameters, and training-instability risks documented for recurrent tokens in RoboMME. Reusing the already-learned step embedding avoids that.

## Counter-Evidence

- **Step embedding may encode "waypoint index," not "phase."** Sinusoidal-MLP on absolute keypoint index is a smooth function of time. If two phases have similar waypoint ranges across demos (e.g., phases at 6-9 and 18-21 both show a "hand over object" motion), phase proximity in embedding space may be ambiguous. A phase-id head trained on demo segmentation labels would be sharper — but needs labels.
- **Cross-attention may learn to ignore the phase channel.** If the network finds that visual features already discriminate phases well enough on the training set, it can set the phase-channel weights to zero. A gated injection (FiLM-style) would force use but adds parameters.
- **Write-time tagging permanently commits a single phase label per entry.** If phases are contiguous and non-repeating, this is fine. On cyclic tasks (repeat the same stacking motion 5 times), the step embedding increments but the phase identity repeats — the model sees 5 entries with near-identical features but monotonically increasing phase tags, which can confuse attention.
- **No direct precedent in robot memory.** Video segmentation's SAM2 uses object-pointer embeddings keyed on tracker state, not a task-progress signal. The transfer is plausible but untested.

## How To Test

Implemented as a flag `use_phase_keyed_memory` in `sam2act_plus.yaml`.

1. **Baseline:** #40 Stage 2 pipeline (22-task base + step emb + FIFO memory) on the long-horizon task.
2. **Phase-keyed variant:** Identical, but at both memory write and memory read, concatenate the 32-dim step embedding to the 64-dim spatial features → 96-dim K/V tokens. Retrain the memory attention's K/V projection to 256-dim from the new input dim.
3. **Ablations:**
   - (a) Phase-on-write only (entries tagged; query unchanged).
   - (b) Phase-on-read only (entries unchanged; query phase-aware).
   - (c) Both (full variant).
   - (d) Frozen step embedding weights during Stage 2 vs. joint fine-tune — tests whether the embedding needs to specialize for retrieval.
4. **Metric:** Task success rate on long-horizon task with explicit cross-phase dependency (needs info from phase 1 at phase 5). Baseline expected <30%; target ≥70%. Secondary: attention-weight heatmap over memory entries should show a bimodal distribution (recent + phase-matched) rather than pure recency.
5. **Regression check:** MemoryBench 4 tasks (pbb/rd/rab/sas) at ≥90% of #40 numbers.

## Relation to Other Ideas

Composes with [[phase-persistent-memory-anchors]] (anchors pinned, FIFO phase-keyed) and [[gripper-event-writes-with-phase-read]] (phase-tagged event log). Minimal viable package with phase anchors = first thing to test.
