# Exploring: `vault explore`, marks, link it, map history

[← README](../README.md) · [The shape of the vault](explore.md) · [Search and connections](search.md) · [Experiments](experiment.md) · [For assistants](agents.md)

`vault explore` runs the deepen and discover lenses ([The shape of the vault](explore.md)) through one registry and
shows a section per lens. It remembers what it showed and what you kept, in the engine's own store, never in your
notes. `vault link-it` turns a result into a real link in both notes, through the safe write path, and `vault unlink`
undoes it. `vault map --record` and `--since` say what changed between two looks at the vault. All of it answers in
JSON (`--json`) and over JSON-RPC: the contract is in [For assistants](agents.md); in the terminal app `/explore` is
interactive ([The terminal app](tui.md#explore)).

Examples use a copy of the sample vault, indexed and embedded: `cp -Rp tests/fixtures/vault /tmp/session-vault &&
vault index /tmp/session-vault && vault embed /tmp/session-vault`. Every output below is real, from one session
(`COLUMNS=120`), in the order it was run.

## `vault explore [vault] [note] [--fresh] [--lens L]… [--per-lens N] [--no-record] [--refresh] [--json] [-f FORMAT]`

From a note, or a missing note (a topic the vault links to but has no note for: `vault explore llm-as-a-judge-framework`),
every lens in `engine.explore.lenses` runs. Each gives its candidates in its own order; the section shows
`engine.explore.per_lens` of them (3 to 5, default 4). Every result has a ref (`2.3`: section 2, result 3), its lens,
and one line of evidence.

| lens | kind | a result is | evidence |
|---|---|---|---|
| `neighbourhood` | deepen | a related note not linked yet, at or above the 90th percentile of this note's scores | `related`'s reason and score |
| `cited` | deepen | a note cited together with it (not linked with it) | the notes citing both, co-citation and its percentile |
| `foundations` | deepen | a note linking to the same notes (not linked with it) | what both link to, coupling and its percentile |
| `backbone` | deepen | a note on the main chain through its community, nearest first | steps along the chain, what ties it |
| `cluster` | deepen | its community's topic page, hubs, and the communities next to it | the community, its weight, who carries it |
| `hidden` | discover | a note with no link either way but two or more middles in common | the middles and the score's percentile |
| `role` | discover | a note in another community in the same position | role distance and the features that stand out |
| `bridge` | discover | a pair of communities whose shortest paths run through it | the share of those paths |
| `gap` | discover | two communities close in meaning, linked less than chance predicts | closeness, links against chance, the pair that would join them |
| `surprise` | discover | a link of this note to a note far away in meaning | the cosine, how rarely the two communities link |
| `unwritten` | discover | a missing note it links that many communities want | who links it, from how many communities |
| `wander` | discover | a step of a seeded random walk | the link walked |
| `topicless` | map | a community without a topic page (only without a starting point) | its size, key phrases, hub |

The default list (`config/config.yaml`) has five of each kind: the deepen five and hidden, role, gap, surprise,
unwritten. Bridge and wander can be added; `--lens` picks for one run. A lens that needs a note's own links (cited,
foundations, backbone, role, bridge, wander) says "does not apply" to a missing note and takes no space.

**Equal space.** Deepen and discover get the same number of results: each kind's budget is `per_lens` × (the sections
that apply) / 2, split evenly over its sections, 3 to 5 per section. With the default list that is 4 everywhere. A
lens that finds fewer leaves its space empty; it is not handed to the other kind. The header says how many of each
were shown.

**Diversity.** Candidates are taken in the lens's order; one is left out of a section when it is about a note already
shown there, or when one of its notes is a near-duplicate of one already shown: a similarity at or above
`engine.explore.duplicate_percentile` (99) of all note pairs in the vault (centered embedding cosine; key-term cosine
without the model). On the sample that is a cosine of 0.51. The section header counts what was left out.

**Unlinked weights.** Explore looks for what is not linked yet, so it uses `engine.related.unlinked_weights`, the mix
[`vault experiment`](experiment.md) measured on exactly that task. Notes already linked to the start are left out of
neighbourhood, cited together, same foundations and hidden link; backbone and cluster describe the links that are
there and keep them (marked "linked"; link-it has nothing to add for those and says so).

```
$ vault explore /tmp/session-vault kl-penalty-in-rlhf -f brain
Explore kl-penalty-in-rlhf
  20_ZETTLEKASTEN/kl-penalty-in-rlhf.md · index 2026-11-10T19:28:59+00:00 (up to date)
  10 lenses (engine.explore.lenses) · equal space: deepen 13 results, discover 11 (up to 4 and 4 a section) · related: engine.related.unlinked_weights
  near-duplicates left out of a section: embedding cosine ≥ 0.51 (the 99th percentile of note pairs) or the same note
  sections ordered by the share of results you keep: (kept + 1) / (marked + 2), ties in config order
  memory /tmp/session-vault/.vaultengine/memory.sqlite · `vault mark keep|dismiss 1.2` · `vault link-it 1.2` · `vault kept`

  1. Neighbourhood  (deepen · related notes not linked yet, at or above the percentile of this note's scores; 4 of 16, 2 near-duplicates · no marks yet)
      1.1  RLHF
           neighbourhood · 2 hops via direct-preference-optimization-dpo +1 · similar text (rlhf, reward, policy); score 0.74 (p100 of this note's scores)
      1.2  covariance-targeted-entropy-intervention
           neighbourhood · 2 hops via trust-region-policy-optimization-monoto… +3 · shares [[policy-gradient-methods]] · both link trust-region-policy-optimization-monoto…, reward-model-overoptimization-hump (no note yet) · similar text (kl, token, grpo); score 0.71 (p99 of this note's scores)
      1.3  decoupling-clipping-unit-from-credit-assignment-unit
           neighbourhood · 2 hops via policy-gradient-methods (no note yet) · shares [[policy-gradient-methods]] · similar text (token, reward, gradient); score 0.70 (p99 of this note's scores)
      1.4  tool-output-masking-in-agentic-rl
           neighbourhood · 2 hops via the-environment-not-flops-bottlenecks-a… +1 · shares [[policy-gradient-methods]] · cited together by the-environment-not-flops-bottlenecks-a… · similar text (token, kl, aggregation); score 0.63 (p97 of this note's scores)

  2. Cited together  (deepen · notes the same notes link to, not linked with it; 1 of 1 · no marks yet)
      2.1  tool-output-masking-in-agentic-rl
           cited · cited together by the-environment-not-flops-bottlenecks-agentic-rl; co-citation 0.31 (p94 of the vault's pairs)

  3. Same foundations  (deepen · notes that link to the same notes, not linked with it; 0 of 0 · no marks yet)
       none stands out in this vault

  4. Backbone  (deepen · the main chain through its community, nearest first; 4 of 12, 2 near-duplicates · no marks yet)
      4.1  direct-preference-optimization-dpo
           backbone · where this note joins the backbone of community 6 (12 of its 18 notes): 1 step off it; linked
      4.2  DeepSeek R1 - Paper Notes
           backbone · 2 steps along the backbone of community 6 (12 of its 18 notes) from direct-preference-optimization-dpo, where this note joins it
      4.3  reinforcement-learning-in-language-models
           backbone · 2 steps along the backbone of community 6 (12 of its 18 notes) from direct-preference-optimization-dpo, where this note joins it
      4.4  decoupling-clipping-unit-from-credit-assignment-unit
           backbone · 3 steps along the backbone of community 6 (12 of its 18 notes) from direct-preference-optimization-dpo, where this note joins it

  5. Cluster  (deepen · its community: topic page, hubs, the communities next to it; 4 of 9, 1 near-duplicates · no marks yet)
      5.1  community 6: [[systems-engineering]] · [[reinforcement-learning-in-language-models]]
           cluster · 18 notes and no topic page; key phrases: [[systems-engineering]], [[reinforcement-learning-in-language-models]], [[distributed-systems]]
      5.2  reinforcement-learning-in-language-models
           cluster · a hub of community 6 ([[systems-engineering]] · [[reinforcement-learning-in-language-models]]): among its most tied-in notes
      5.3  agent-layer-inverts-who-owns-the-generation-loop
           cluster · a hub of community 6 ([[systems-engineering]] · [[reinforcement-learning-in-language-models]]): among its most tied-in notes
      5.4  next community 4: grpo-group-relative-policy-optimization · [[post-training]]
           cluster · weight 33 between the two communities, carried by reinforcement-learning-in-language-models, reinforcement-learning-from-human-feedback-rlhf

  6. Hidden link  (discover · no link either way, but two or more middles in common (Swanson's A-B-C); 1 of 1 · no marks yet)
      6.1  covariance-targeted-entropy-intervention
           hidden · no link either way; 8 middles in common: 'kl' (4 notes), 'region' (4 notes), trust-region-policy-optimization-monotonic-improvement-guarantee (6 notes), 'ppo' (7 notes) +4; score 3.86 (p95 of 2135 unlinked pairs)

  7. Same role elsewhere  (discover · notes in other communities holding the same position; 4 of 20 · no marks yet)
      7.1  rubric-agreement-does-not-establish-rubric-validity
           role · community 3 (this note: 6); role distance 0.076 (p2, lower is closer); hub 0.10 vs 0.11 (p81 vs p84)
      7.2  trajectory-as-the-atomic-unit-of-policy-optimization
           role · community 4 (this note: 6); role distance 0.108 (p3, lower is closer); hub 0.10 vs 0.23 (p81 vs p92)
      7.3  in-context-scheming-capability
           role · community 2 (this note: 6); role distance 0.116 (p4, lower is closer); hub 0.10 vs 0.01 (p81 vs p59)
      7.4  label-noise-dominates-saturated-benchmarks
           role · community 3 (this note: 6); role distance 0.127 (p5, lower is closer); hub 0.10 vs 0.02 (p81 vs p63)

  8. Gap  (discover · communities close in meaning with fewer links than chance predicts; 2 of 2 · no marks yet)
      8.1  communities 6 and 12: [[systems-engineering]] · [[reinforcement-learning-in-language-models]] / #prompt-eng…
           gap · closeness of their mean embeddings 0.38 (p97 of 66 community pairs); 0 links between them where chance predicts 0.2; join reward-computation-as-a-heavyweight-pipeline-stage ↔ persona-driven-synthetic-data-generation (via 'verifiable', rlvr-reinforcement-learning-with-verifiable-rewards (no note yet)) [meaning]
      8.2  communities 6 and 11: [[systems-engineering]] · [[reinforcement-learning-in-language-models]] / [[memory-ma…
           gap · closeness of their mean embeddings 0.24 (p91 of 66 community pairs); 0 links between them where chance predicts 0.1; no pair across shares two middles [meaning]

  9. Surprise  (discover · links between notes far apart in meaning, across communities; 0 of 0 · no marks yet)
       no link here joins notes this far apart

  10. Unwritten  (discover · missing notes linked from the most communities; 4 of 4 · no marks yet)
     10.1  [[reward-model-overoptimization-hump]]
           unwritten · 12 notes in 4 communities link it (4: 4, 5: 4, 2: 2, 6: 2)
     10.2  [[llm-as-judge-biases-catalog]]
           unwritten · 4 notes in 4 communities link it (3: 1, 5: 1, 6: 1, 10: 1)
     10.3  [[policy-gradient-methods]] +1
           unwritten · 13 notes in 2 communities link it (6: 7, 4: 6); merged by embedding (0.88): [[policy-gradient-methods]], [[policy-gradient]]
     10.4  [[PPO and RLHF - Deep Dive]]
           unwritten · 2 notes in 2 communities link it (4: 1, 6: 1)
```

### Without a starting point: where to start

`vault explore` alone proposes starting points from what the vault-wide lenses find (`engine.explore.start_lenses`):
the widest gaps, the most surprising links, the most wanted missing notes, the communities without a topic page.
Nothing is a fixed list: each proposal is a finding of its lens, with its evidence, and names where `vault explore`
would go (`→ explore …`). When the vault changes, so do the proposals.

```
$ vault explore /tmp/session-vault -f brain
Explore /tmp/session-vault: where to start
  the whole vault · index 2026-11-10T19:28:59+00:00 (up to date)
  proposals from 4 vault-wide lenses (engine.explore.start_lenses), up to 4 each; `vault explore <the start named>` goes there
  near-duplicates left out of a section: embedding cosine ≥ 0.51 (the 99th percentile of note pairs) or the same note
  sections ordered by the share of results you keep: (kept + 1) / (marked + 2), ties in config order
  memory /tmp/session-vault/.vaultengine/memory.sqlite · `vault mark keep|dismiss 1.2` · `vault link-it 1.2` · `vault kept`

  1. Gap  (discover · communities close in meaning with fewer links than chance predicts; 3 of 4, 1 near-duplicates · no marks yet)
      1.1  communities 8 and 9: [[fine-tuning-strategies-for-llms]] · [[con…  → explore fine-tuning-strategies-for-llms
           gap · closeness of their mean embeddings 0.52 (p98 of 66 community pairs); 0 links between them where chance predicts 0.5; join fine-tuning-strategies-for-llms ↔ importance-of-evaluation-in-ai-systems (via 'adjustments', 'tuner', critic-tuner-approach (no note yet)) [meaning]
      1.2  communities 6 and 12: [[systems-engineering]]…  → explore reward-computation-as-a-heavyweight-pipeline-stage
           gap · closeness of their mean embeddings 0.38 (p97 of 66 community pairs); 0 links between them where chance predicts 0.2; join reward-computation-as-a-heavyweight-pipeline-stage ↔ persona-driven-synthetic-data-generation (via 'verifiable', rlvr-reinforcement-learning-with-verifiable-rewards (no note yet)) [meaning]
      1.3  communities 6 and 11: [[systems-engineering]] · [[rein…  → explore reinforcement-learning-in-language-models
           gap · closeness of their mean embeddings 0.24 (p91 of 66 community pairs); 0 links between them where chance predicts 0.1; no pair across shares two middles [meaning]

  2. Surprise  (discover · links between notes far apart in meaning, across communities; 4 of 25, 4 near-duplicates · no marks yet)
      2.1  covariance-targeted-entropy-intervention → reward-hacki…  → explore covariance-targeted-entropy-intervention
           surprise · covariance-targeted-entropy-intervention links it (line 43); embedding cosine -0.00 (p8 of 368 linked pairs); communities 6 and 2 (3 links between them where chance predicts 31.3)
      2.2  reward-computation-as-a-heavyweight-pipeline-…  → explore reward-computation-as-a-heavyweight-pipeline-stage
           surprise · reward-computation-as-a-heavyweight-pipeline-stage links it (line 28); embedding cosine 0.02 (p9 of 368 linked pairs); communities 6 and 3 (1 links between them where chance predicts 10.0)
      2.3  2504.13958 - ToolRL Reward Desig…  → explore 2504.13958 - ToolRL Reward Design for Tool-Integrated Reasoning
           surprise · 2504.13958 - ToolRL Reward Design for Tool-Integrated Reasoning links it (line 111); embedding cosine -0.06 (p5 of 368 linked pairs); communities 4 and 2 (16 links between them where chance predicts 39.6)
      2.4  a-bounty-list-is-a-map-of-what-cannot-be-verif…  → explore a-bounty-list-is-a-map-of-what-cannot-be-verified
           surprise · a-bounty-list-is-a-map-of-what-cannot-be-verified links it (line 34); embedding cosine -0.16 (p2 of 368 linked pairs); communities 5 and 3 (7 links between them where chance predicts 9.5)

  3. Unwritten  (discover · missing notes linked from the most communities; 4 of 30 · no marks yet)
      3.1  [[llm-as-a-judge-framework]] +1  → explore [[llm-as-a-judge-framework]]
           unwritten · 19 notes in 7 communities link it (3: 8, 10: 4, 4: 3, 2: 1, 5: 1, 7: 1, 8: 1); merged by embedding (0.79): [[llm-as-a-judge-framework]], [[llm-as-judge]]
      3.2  [[evaluation-methodology]]  → explore [[evaluation-methodology]]
           unwritten · 31 notes in 6 communities link it (3: 13, 5: 6, 4: 5, 10: 4, 7: 2, 2: 1)
      3.3  [[experimental-design]]  → explore [[experimental-design]]
           unwritten · 20 notes in 6 communities link it (5: 8, 3: 4, 4: 3, 6: 2, 10: 2, 7: 1)
      3.4  [[agentic-ai-systems]] +1  → explore [[agentic-ai-systems]]
           unwritten · 14 notes in 6 communities link it (7: 5, 5: 3, 9: 3, 3: 1, 6: 1, 11: 1); merged by embedding (0.92): [[agentic-ai-systems]], [[agentic-ai]]

  4. No topic page  (map · communities without a topic page, largest first; 4 of 10 · no marks yet)
      4.1  community 1: [[discourse-graph]] · #evidence  → explore QUE 03 - Grade end state, trajectory or text
           topicless · 40 notes and no topic page; key phrases [[discourse-graph]], #evidence, [[rl-environment-design]]; hub QUE 03 - Grade end state, trajectory or text
      4.2  community 4: grpo-group-relative-policy-optimization · [[p…  → explore 2507.17746 - Rubrics as Rewards (RaR)
           topicless · 19 notes and no topic page; key phrases → grpo-group-relative-policy-optimization, [[post-training]], [[reinforcement-learning]]; hub 2507.17746 - Rubrics as Rewards (RaR)
      4.3  community 5: instance-specific-rubric-as-reward-fu…  → explore environment-as-installable-package-not-config
           topicless · 18 notes and no topic page; key phrases → instance-specific-rubric-as-reward-function, [[reward]], → Prime Intellect - Environments Hub; hub environment-as-installable-package-not-config
      4.4  community 6: [[systems-engineering]] · [[reinforcement…  → explore reinforcement-learning-in-language-models
           topicless · 18 notes and no topic page; key phrases [[systems-engineering]], [[reinforcement-learning-in-language-models]], [[distributed-systems]]; hub reinforcement-learning-in-language-models
```

## Memory: `vault mark`, `vault kept`, `--fresh`, keep rates

The store is `engine.memory` (default `<vault>/.vaultengine/memory.sqlite`), a file apart from the index, so a rebuilt
index loses nothing (`vaultengine/memory.py`). It holds each explore run and the results it showed (so refs resolve),
every result seen (per start), your marks, the links link-it made, and map snapshots. Your notes are never written by
any of this.

- `vault mark [vault] keep|dismiss|clear REF…`: marks results of the **last** explore run, by ref.
- `vault kept [vault] [--lens L] [--json]`: what you kept, with its lens, evidence, start and time.
- `--fresh`: leaves out what was seen in earlier runs from the same start, or dismissed there, and fills each section
  from further down its lens's list. Without `--fresh` a marked result shows `[kept]` or `[dismissed]`.
- **Keep rate.** Each section header says `kept K of M you marked`. Sections are ordered by (K + 1) / (M + 2) over
  all your marks, so a lens you never marked counts as one half, one you keep from rises, one you dismiss from sinks;
  ties keep the configured order. A link made with link-it counts as a keep.
- Explore records what it showed (seen, refs); `--no-record` does not. Running it twice gives the same output: the
  seen record changes nothing unless you ask for `--fresh`.

```
$ vault mark /tmp/session-vault keep 1.1 6.1
  1.1  kept  neighbourhood · RLHF
  6.1  kept  hidden · covariance-targeted-entropy-intervention
  hidden: kept 1 of 1 marked
  neighbourhood: kept 1 of 1 marked
```
```
$ vault mark /tmp/session-vault dismiss 7.1
  7.1  dismissed  role · rubric-agreement-does-not-establish-rubric-validity
  role: kept 0 of 1 marked
```

## `vault link-it [vault] REF [--write] [--reason TEXT] [-f FORMAT]` and `vault unlink [vault] NOTE NOTE [--write]`

Link it writes, in both notes, one line under `engine.explore.link_section` (default `See Also`):
`- [[other note]] — <reason>`, after the section's last line, or in a new section at the end of the note (at the level
of the note's last heading, at most 2, so it is not nested under it). The reason is the result's lens and evidence
without what the link makes false ("no link either way", "0 links between them") and without percentiles, cut to one
line of 160 characters, with nothing that would become a link, tag or math. A note result links the start and that
note; a gap links the pair that would join the two communities. A missing note, a community or a link that is already
there cannot be linked (exit 2, with why). `vault link-it NOTE NOTE --reason …` links two notes by name.

It goes through the safe write path ([Commands](commands.md#the-safe-write-path-vault-new-vault-link-vault-set)): the
format's gate refuses it if `vault check` would get worse, both files are committed atomically, the index is updated.
Without `--write` it is a dry run that prints the diff and writes nothing. `vault unlink` removes, in both notes, the
list lines under that section that link the other note, and the heading too when link-it created the section and
nothing else is left under it (with the blank line link-it put before it, if it put one); the two notes are then byte
for byte what they were. Both keep each note's line endings (LF or CRLF) and its ending: link-it adds no final line
break to a note that had none, so unlink can give it back as it was. What link-it created (the section, the blank
line) is remembered in the engine's memory; a link it did not write is removed line by line and its section stays.
Also a dry run unless `--write`.

```
$ vault link-it /tmp/session-vault 8.1 -f brain
--- a/20_ZETTLEKASTEN/reward-computation-as-a-heavyweight-pipeline-stage.md
+++ b/20_ZETTLEKASTEN/reward-computation-as-a-heavyweight-pipeline-stage.md
@@ -29,3 +29,6 @@
 
 # References
 1. [[Jiachen Liu - When LLMs Grow Hands and Feet, How to Design our Agentic RL Systems]]
+
+# See Also
+- [[persona-driven-synthetic-data-generation]] — gap: closeness of their mean embeddings 0.38; fewer links between their communities than chance predicts; shared: 'verifiable'…
--- a/20_ZETTLEKASTEN/persona-driven-synthetic-data-generation.md
+++ b/20_ZETTLEKASTEN/persona-driven-synthetic-data-generation.md
@@ -16,3 +16,6 @@
 - Related persona work, but for agent simulation rather than data synthesis: [[llm-agent-profiling-through-persona-prompts]], [[persona-alignment-in-agent-based-modeling]].
 ## References
 1. [[2411.15124 - Tulu 3 (RLVR Post-Training)]]
+
+## See Also
+- [[reward-computation-as-a-heavyweight-pipeline-stage]] — gap: closeness of their mean embeddings 0.38; fewer links between their communities than chance predicts; shared: 'verifiable'…

Would write 2 notes (link-it):
  ~ 20_ZETTLEKASTEN/reward-computation-as-a-heavyweight-pipeline-stage.md
      new section # See Also at the end (the level of the note's last heading, at most 2): - [[persona-driven-synthetic-data-generation]] — gap: closeness of their mean embeddings 0.38; fewer links between their communities than chance predicts; shared: 'verifiable'…
  ~ 20_ZETTLEKASTEN/persona-driven-synthetic-data-generation.md
      new section ## See Also at the end (the level of the note's last heading, at most 2): - [[reward-computation-as-a-heavyweight-pipeline-stage]] — gap: closeness of their mean embeddings 0.38; fewer links between their communities than chance predicts; shared: 'verifiable'…
  check    no new violations
           already there, not blocking: 1 missing-key (run vault check)
Dry run: nothing was written.

`--write` writes it, through the same gate; `vault unlink` undoes it.
```
```
$ vault link-it /tmp/session-vault 8.1 -f brain --write
Wrote 2 notes (link-it):
  ~ 20_ZETTLEKASTEN/reward-computation-as-a-heavyweight-pipeline-stage.md
      new section # See Also at the end (the level of the note's last heading, at most 2): - [[persona-driven-synthetic-data-generation]] — gap: closeness of their mean embeddings 0.38; fewer links between their communities than chance predicts; shared: 'verifiable'…
  ~ 20_ZETTLEKASTEN/persona-driven-synthetic-data-generation.md
      new section ## See Also at the end (the level of the note's last heading, at most 2): - [[reward-computation-as-a-heavyweight-pipeline-stage]] — gap: closeness of their mean embeddings 0.38; fewer links between their communities than chance predicts; shared: 'verifiable'…
  check    no new violations
           already there, not blocking: 1 missing-key (run vault check)
  index    updated 2 notes

Kept result 8.1 (gap): a link made counts as a keep.
```
```
$ vault check /tmp/session-vault -f brain | grep 'violations in'    # before the session: 102 violations in 77 notes
✗ 102 violations in 77 notes
```
```
$ vault kept /tmp/session-vault
Kept in /tmp/session-vault
  memory /tmp/session-vault/.vaultengine/memory.sqlite · 3 results

  covariance-targeted-entropy-intervention
    hidden · no link either way; 8 middles in common: 'kl' (4 notes), 'region' (4 notes), trust-region-policy-optimization-monotonic-improvement-guarantee (6 notes), 'ppo' (7 notes) +4; score 3.86 (p95 of 2135 unlinked pairs)
    from 20_ZETTLEKASTEN/kl-penalty-in-rlhf.md · kept 2026-11-10T19:29:10

  RLHF
    neighbourhood · 2 hops via direct-preference-optimization-dpo +1 · similar text (rlhf, reward, policy); score 0.74 (p100 of this note's scores)
    from 20_ZETTLEKASTEN/kl-penalty-in-rlhf.md · kept 2026-11-10T19:29:10

  communities 6 and 12: [[systems-engineering]] · [[reinforcement-learning-in-language-models]] / #prompt-engineering
    gap · closeness of their mean embeddings 0.38 (p97 of 66 community pairs); 0 links between them where chance predicts 0.2; join reward-computation-as-a-heavyweight-pipeline-stage ↔ persona-driven-synthetic-data-generation (via 'verifiable', rlvr-reinforcement-learning-with-verifiable-rewards (no note yet)) [meaning]
    from 20_ZETTLEKASTEN/kl-penalty-in-rlhf.md · kept 2026-11-10T19:29:12
```

Exploring again with `--fresh`: RLHF and the hidden link (kept, so seen), the dismissed role result and everything
else seen before are gone; the sections now lead with the lenses kept from (neighbourhood, hidden link, gap: 1 of 1)
and end with the one dismissed from (role: 0 of 1). The link just made moved `kl-penalty-in-rlhf` into another
community, so backbone and cluster changed too. Two notes changed since the vault-wide numbers were computed, so
this run reused them and says so in its header ([How fast, and why](#how-fast-and-why)); recomputing them
(`--refresh`) gives the same results, with one percentile different: the first role result is p6 instead of p5.

```
$ vault explore /tmp/session-vault kl-penalty-in-rlhf -f brain --fresh
Explore kl-penalty-in-rlhf
  20_ZETTLEKASTEN/kl-penalty-in-rlhf.md · index 2026-11-10T19:29:12+00:00 (up to date)
  10 lenses (engine.explore.lenses) · equal space: deepen 12 results, discover 4 (up to 4 and 4 a section) · related: engine.related.unlinked_weights
  near-duplicates left out of a section: embedding cosine ≥ 0.51 (the 99th percentile of note pairs) or the same note
  sections ordered by the share of results you keep: (kept + 1) / (marked + 2), ties in config order · --fresh: what you saw or dismissed before is left out
  memory /tmp/session-vault/.vaultengine/memory.sqlite · `vault mark keep|dismiss 1.2` · `vault link-it 1.2` · `vault kept`
  vault-wide numbers (distributions, key terms) from 2026-11-10T19:29:06: 2 notes changed since; recomputed when more than 3 have, or with --refresh
  2 notes had no cached vector: embedded in memory for this run only; `vault embed /tmp/session-vault` caches them

  1. Neighbourhood  (deepen · related notes not linked yet, at or above the percentile of this note's scores; 4 of 16, 5 seen or dismissed before, 3 near-duplicates · kept 1 of 1 you marked (100%))
      1.1  Cameron R. Wolfe - GRPO Tricks (Making RL Actually Work)
           neighbourhood · 2 hops via trust-region-policy-optimization-monoto… +2 · both link PPO and RLHF - Deep Dive (no note yet), trust-region-policy-optimization-monoto… · similar text (token, grpo, reward); score 0.69 (p98 of this note's scores)
      1.2  the-critic-deletion-tension-in-agentic-rl
           neighbourhood · 2 hops via policy-gradient-methods (no note yet) · shares [[policy-gradient-methods]] · similar text (reward, grpo, policy); score 0.53 (p96 of this note's scores)
      1.3  ppo-as-trpo-simplification
           neighbourhood · 2 hops via trust-region-policy-optimization-monoto… +1 · shares [[policy-gradient-methods]] · both link trust-region-policy-optimization-monoto… · similar text (ppo, policy, gradient); score 0.51 (p95 of this note's scores)
      1.4  goodhart-at-the-contribution-layer
           neighbourhood · 3 hops via reinforcement-learning-from-human-feedb… · similar text (reward, policy, low); score 0.37 (p93 of this note's scores)

  2. Hidden link  (discover · no link either way, but two or more middles in common (Swanson's A-B-C); 0 of 1, 1 seen or dismissed before · kept 1 of 1 you marked (100%))
       everything it found was seen or dismissed before (--fresh)

  3. Gap  (discover · communities close in meaning with fewer links than chance predicts; 0 of 0 · kept 1 of 1 you marked (100%))
       no pair of communities is close in meaning and linked less than chance predicts

  4. Cited together  (deepen · notes the same notes link to, not linked with it; 0 of 1, 1 seen or dismissed before · no marks yet)
       everything it found was seen or dismissed before (--fresh)

  5. Same foundations  (deepen · notes that link to the same notes, not linked with it; 0 of 0 · no marks yet)
       none stands out in this vault

  6. Backbone  (deepen · the main chain through its community, nearest first; 4 of 8, 1 seen or dismissed before, 3 near-duplicates · no marks yet)
      6.1  reinforcement-learning-from-human-feedback-rlhf
           backbone · 2 steps along the backbone of community 3 (9 of its 25 notes); linked
      6.2  Cameron R. Wolfe - GRPO Tricks (Making RL Actually Work)
           backbone · 3 steps along the backbone of community 3 (9 of its 25 notes)
      6.3  2507.17746 - Rubrics as Rewards (RaR)
           backbone · 5 steps along the backbone of community 3 (9 of its 25 notes)
      6.4  credit-assignment-problem
           backbone · 8 steps along the backbone of community 3 (9 of its 25 notes)

  7. Cluster  (deepen · its community: topic page, hubs, the communities next to it; 4 of 9, 1 near-duplicates · no marks yet)
      7.1  llm-judge-score-bias-and-contrastive-decoding
           cluster · topic page of community 3 ([[reinforcement-learning]] · grpo-group-relative-policy-optimization, 25 notes): links 1 of them
      7.2  2507.17746 - Rubrics as Rewards (RaR)
           cluster · a hub of community 3 ([[reinforcement-learning]] · grpo-group-relative-policy-optimization): among its most tied-in notes
      7.3  Cameron R. Wolfe - GRPO Tricks (Making RL Actually Work)
           cluster · a hub of community 3 ([[reinforcement-learning]] · grpo-group-relative-policy-optimization): among its most tied-in notes
      7.4  next community 6: [[systems-engineering]] · [[distributed-systems]]
           cluster · weight 32 between the two communities, carried by reinforcement-learning-from-human-feedback-rlhf, Cameron R. Wolfe - GRPO Tricks (Making RL Actually Work)

  8. Surprise  (discover · links between notes far apart in meaning, across communities; 0 of 0 · no marks yet)
       no link here joins notes this far apart

  9. Unwritten  (discover · missing notes linked from the most communities; 0 of 3, 3 seen or dismissed before · no marks yet)
       everything it found was seen or dismissed before (--fresh)

  10. Same role elsewhere  (discover · notes in other communities holding the same position; 4 of 19, 4 seen or dismissed before, 1 near-duplicates · kept 0 of 1 you marked (0%))
     10.1  shaping-term-ablation-against-true-completion-rate
           role · community 5 (this note: 3); role distance 0.135 (p5, lower is closer); hub 0.10 vs 0.11 (p81 vs p83)
     10.2  simulated-tool-execution-as-intent-proxy
           role · community 2 (this note: 3); role distance 0.138 (p6, lower is closer); hub 0.10 vs 0.02 (p81 vs p60)
     10.3  rubric-vs-checklist-vs-rating-scale
           role · community 4 (this note: 3); role distance 0.140 (p6, lower is closer); hub 0.10 vs 0.06 (p81 vs p75)
     10.4  trajectory-capture-by-instrumenting-the-llm-api-boundary
           role · community 6 (this note: 3); role distance 0.147 (p7, lower is closer); hub 0.10 vs 0.03 (p81 vs p63)
```

## Map history: `vault map --record` and `vault map --since DATE`

`vault map --record` prints the map and saves a small snapshot in the memory store (14 KB on the sample, compressed
JSON): the communities by their notes and topic pages, the gaps (with their sides' links and what chance predicted),
the missing notes and how many notes link each. Without `--record`, `vault map` writes nothing, not even the store.

`vault map --since DATE` (a date or `2026-11-10T14:30`) compares the vault now with the first snapshot at or after that
time (else the last one before it), and only reads (`vaultengine/history.py`):

- **gaps**: each recorded gap's two sides (the notes its communities had then) are measured now as the gap lens
  measures communities; it is *closed* when their links are no longer fewer than chance predicts, or both sides are
  one community now. The links added between them are listed. Measuring the recorded sides keeps renumbered and
  reshuffled communities out of it;
- **communities**: each now is matched with the recorded one it shares most notes with (Jaccard); below 0.5 it was
  *formed*, and a recorded one nothing matches is *gone*;
- **missing notes written**: recorded missing notes whose name is now a note;
- **topic pages**: recorded communities without one whose match now has one.

```
$ vault map /tmp/session-vault -f brain --record    # the map itself left out; its last line:
  recorded     snapshot 2026-11-10T19:29:03 (14,295 bytes) in /tmp/session-vault/.vaultengine/memory.sqlite: 12 communities, 4 gaps (meaning), 612 missing notes · `vault map --since 2026-11-10` compares
```
```
$ vault map /tmp/session-vault -f brain --since 2026-11-10
Map of /tmp/session-vault since 2026-11-10T19:29:03
  the first snapshot at or after 2026-11-10 · index 2026-11-10T19:29:12+00:00 (up to date)
  notes 180 → 180 · links between notes 439 → 441 · communities 12 → 11

  Gaps  (1 of 4 recorded gaps closed; by meaning)
    open    communities 8 and 9 then: [[fine-tuning-strategies-for-llms]] · [… / [[adas-mechanism]] · [[agentic-rag-appr…
            0 links between the two sides where chance predicts 0.5 (then 0, where chance predicted 0.5)
    closed  communities 6 and 12 then: [[systems-engineering]] · [[reinforceme… / #prompt-engineering
            2 links between the two sides where chance predicts 0.5: no longer fewer than chance (then 0, where chance predicted 0.2)
            new link persona-driven-synthetic-data-generation ↔ reward-computation-as-a-heavyweight-pipeline…
    open    communities 6 and 11 then: [[systems-engineering]] · [[reinforceme… / [[memory-management]] · multi-step-comp…
            0 links between the two sides where chance predicts 0.1 (then 0, where chance predicted 0.1)
    open    communities 8 and 12 then: [[fine-tuning-strategies-for-llms]] · [… / #prompt-engineering
            0 links between the two sides where chance predicts 0.2 (then 0, where chance predicted 0.1)

  Communities  (formed 0, gone 1: matched by shared notes, Jaccard below 0.5 = no match)
    gone    12. #prompt-engineering  2 notes then (best match now 0.08)

  Missing notes written  (0)
    none

  Topic pages  (1 community without one then has one now)
    community 3 (grpo-group-relative-policy-optimization…): llm-judge-score-bias-and-contrastive-decoding
```

The one link made closed the gap it was proposed for (2 links where chance predicts 0.5). It also moved notes
between communities: two-note community 12 (`#prompt-engineering`) was absorbed, and a community now holds the topic
page `llm-judge-score-bias-and-contrastive-decoding`. One link in a 180-note vault is enough to reshuffle Louvain's
communities, which is why gaps are measured on the recorded notes, not on community numbers.

## How fast, and why

`vault explore` used to take half a minute at 5,000 notes and up to three and a half minutes at 50,000. Profiled
(cProfile, one run from a note, on the generated vaults of `scripts/bench_shape.py`):

| | 5,000 notes (23 s profiled) | 50,000 notes (151 s profiled) |
|---|---|---|
| same role | 13.1 s: betweenness 10.1, HITS 1.5, constraint 1.2 | 66.6 s: HITS 33.7, betweenness 18.5, constraint 12.4 |
| hidden link: the distribution of every unlinked pair's score | 1.6 s | 16.3 s |
| key terms of every note | 0.9 s | 15.1 s |
| the near-duplicate cutoff | 2.3 s | |
| co-citation and coupling distributions | 1.3 s | |
| communities and their names, the middles of hidden link | | 9.6 s and 6.5 s |

So the time went into the vault-wide numbers and structures every lens compares against, recomputed from nothing on
every run. Same role did not compare every pair: past 707 notes its distribution already came from a seeded sample of
250,000 pairs, and one note's distances are one vectorised row. Its cost was its features: betweenness, HITS and
constraint of every note. What changed:

- **The vault-wide numbers are kept between runs** (`vaultengine/stats.py`, in `.vaultengine/stats.sqlite` next to
  the memory; derived, safe to delete). They are stored with the vault's state (every note's content hash) and the
  settings they depend on. Nothing changed: every number is reused, so the answer is the one a full computation
  gives. A few notes changed (at most `engine.explore.stats_refresh`, 2%): the distributions and the near-duplicate
  cutoff are reused, and the key terms of the notes that did not change; everything about particular notes (graph,
  communities, betweenness, roles) is computed for the vault as it is, kept for that state, and the output says
  `vault-wide numbers … from <time>: N notes changed since`. More changed, or `--refresh`: all computed again.
- **Betweenness and HITS run in numpy**, with the arithmetic of the Python loops in the same order (Python's `sum()`
  included, which compensates since 3.12), so the numbers are the same bit for bit; constraint computes its
  proportions once. `tests/test_vaultengine_shape.py` (`ReferenceTests`) compares them with the old loops
  (`tests/shape_reference.py`) on the sample and on random graphs, exact and sampled.
- The neighbourhood lens ranks once instead of twice, and writes reasons only for what it shows.

**The same output, proven.** `tests/fixtures/explore_golden.json.gz` holds what the old code answered on the sample
vault from all 180 notes, the 100 missing notes linked three or more times, and without a start, with all twelve
lenses (281 runs). `tests/test_vaultengine_stats.py` (`GoldenTests`) runs them all again, computing everything, then
a seventh of them from the numbers stored on disk: every answer is equal. Deepen, discover and map do not keep
numbers; their JSON on the sample was compared before and after the change: equal.

**Measured** (the whole command, wall clock; `scripts/bench_commands.py`, a copy of the generated vault, vectors
cached; this sandbox varies by up to half between identical runs):

| `vault explore` | 5,000 notes | 50,000 notes |
|---|---|---|
| first run: nothing stored | 10.0–11.9 s | 57.9–76.4 s |
| again: the numbers stored | 1.9–2.2 s | 17.2–24.3 s |
| after a link-it wrote 2 notes: numbers reused, the rest computed | 4.8–4.9 s | 35.3–44.7 s |
| again on that state | 1.7–1.9 s | 17.4–19.3 s |
| without a start | 2.0–2.3 s | 19.1–35.5 s |
| before this change, from a note | 18–34 s | 119–217 s |

What is left at 50,000 notes is reading the vault: building the graph from the index (4–6 s), loading 50,000 vectors
(2–6 s), the communities (5–7 s when they have to be found again), and the lenses' own work for one note (hidden link
2.5 s, neighbourhood 1.6 s, gap, surprise and unwritten about a second each). On the sample vault: about 3 s the first
time, 2 s after.

