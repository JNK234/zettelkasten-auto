# The shape of the vault and deepen lenses

[← Documentation index](../README.md#documentation)

Two commands look at the vault as a whole and around one note. Both use links and tags, not text
(`vault deepen`'s neighbourhood is `vault related`, which adds the embedding when the model is there).
Examples use a copy of the sample vault: `cp -Rp tests/fixtures/vault /tmp/sample-vault && vault index /tmp/sample-vault`.
`vault explore` runs these lenses and the discover lenses together, a section each, and remembers what you keep;
`vault map --record` / `--since` compare two looks at the map: see [Exploring](exploring.md).

## The graph, and which links are tags

Nodes are notes, *missing notes* (link targets with no note) and `#tags`; edges are links (their number is the
weight) and note–tag pairs. A `#tag` on more than a third of the notes (`#baby`, on 133 of 180) is left out: it
would join everything.

**Tag links.** `Tags: [[reward]] [[evaluation-methodology]]` tags a note with two topics. A link is a tag link when
it sits in a field whose role is `tag`: with a format (`--format`, or `engine.format`), a header field the note's
type declares with `link_role: tag` (every type in `formats/brain.yaml`); without one, a field named `tags`. The index
records which field each link is in, so this needs no file reading. In the graph a tag link is an edge to what it
tags with, like a `#tag`. In the pair signals of [`vault explain`](search.md#how-two-notes-relate-vault-explain) and
the tags signal of `vault related`, it is a tag: shared tag links count under shared tags, never as coupling or
co-citation. It still makes the two notes linked.

## `vault map [vault] [--limit N] [--json] [-f FORMAT]`

```
$ vault map /tmp/sample-vault -f brain --limit 0      # an excerpt
Map of /tmp/sample-vault
  index 2026-11-10T10:15:55+00:00 (up to date)

  graph        180 notes, 612 missing notes, 27 #tags (left out: #baby on 133 of 180 notes) · 1706 edges · tag links: format brain (Tags)
  communities  Louvain, seed 42, resolution 1 (the same every run) · modularity 0.67 · 12 communities hold 179 notes; 1 note in none
  cutoffs      at or above the 90th percentile of the vault (engine.explore.percentile)
  topic pages  notes the format brain types as topic

  Communities  (largest first; key phrases: its notes with the phrase / all notes with it)

    1. [[discourse-graph]] · #evidence  40 notes
       key phrases  [[discourse-graph]] 40/40 · #evidence 31/31 · [[rl-environment-design]] 40/52 · #supported 8/8 · → …
       hubs         QUE 03 - Grade end state, trajectory or text, CLM - Grade the observed end state rather than the tr…
       topic pages  none

    2. #ai-safety · control-evaluations-red-team-blue-team  32 notes
       key phrases  #ai-safety 14/15 · → control-evaluations-red-team-blue-team 12/12 · → Shlegeris Greenblatt - The Ca…
       hubs         scheming-as-threat-model, 2312.06942 - AI Control - Improving Safety Despite Intentional Subversion…
       topic pages  emergent-misalignment-from-reward-hacking (links 4)

    3. [[evaluation-methodology]] · construct-and-operationalization-gap  20 notes
       key phrases  [[evaluation-methodology]] 13/31 · → construct-and-operationalization-gap 6/7 · → llm-as-automated-…
       hubs         2411.00640 - Adding Error Bars to Evals, judge-agreement-against-human-ceiling, eval-precision-vs-e…
       topic pages  llm-judge-score-bias-and-contrastive-decoding (links 1)
  … 9 more communities

  No topic page  (10 of 12 communities; a topic page belongs to the community most of its direct links go to)
      1. [[discourse-graph]] · #evidence  40 notes
      4. grpo-group-relative-policy-optimization · [[post-training]]  19 notes
      5. instance-specific-rubric-as-reward-function · [[reward]]  18 notes
      6. [[systems-engineering]] · [[reinforcement-learning-in-language-models]]  18 notes
      7. 2511.01824 - Simia Simulating Environments with Reasoning Models for Agent Training · 2505.1097…  9 notes
      8. [[fine-tuning-strategies-for-llms]] · [[constitutional-ai]]  9 notes
      9. [[adas-mechanism]] · [[agentic-rag-approach]]  5 notes
     10. cross-family-judge-panel-with-unanimity · [[llm-as-a-judge-framework]]  4 notes
     11. [[memory-management]] · multi-step-compounding-error-in-sequential-tasks  3 notes
     12. #prompt-engineering  2 notes

  Bridge notes  (betweenness between notes, exact, from all 180 notes; 18 at or above the 90th percentile, 0.026; constraint: low = its contacts are not tied to each other)
         betweenness   constraint    note  → communities its links reach
      1. 0.106 (p100)  0.048 (p4)    reward-hacking-as-specification-gaming  → 4, 2, 5, 6, 3
      2. 0.090 (p99)   0.041 (p3)    reinforcement-learning-in-language-models  → 6, 4, 5, 8
      3. 0.085 (p99)   0.033 (p1)    emergent-misalignment-from-reward-hacking  → 2, 6, 1, 8, 4
      4. 0.057 (p98)   0.081 (p15)   llm-generated-reward-programs-need-validation  → 5, 4, 1, 8, 2
      …
```

- **Communities**: Louvain in plain Python. Nodes are visited in an order shuffled by `engine.explore.seed` and
  move only on a strict gain, so the same graph and seed give the same communities on every run. A community that
  Louvain left in pieces is split into its connected parts (what Leiden's refinement guarantees). Modularity is
  printed; `engine.explore.resolution` above 1 gives more, smaller communities.
- **Names**: the key phrases are the tags, tag links and link targets (`→`) of a community's notes that are both
  common in it and concentrated in it: (its notes with the phrase / its notes) × (its notes with it / all notes
  with it), at least two of its notes. Hubs are the notes with the most weight inside the community.
- **Topic pages**: a note whose type in the format is one of `engine.explore.topic_types` (`topic` and `page`;
  `formats/brain.yaml` has `topic`, the two notes in `50_MOC/topics`) is a topic page of the community most of its
  direct links go to, wherever Louvain placed it. Where a note sits says nothing: the discourse notes in `50_MOC` are
  claims, evidence and a question, not topic pages. A community no page calls home is listed under "No topic page";
  without a format there are no topic pages to find, and the map says so.
- **Bridge notes**: betweenness between notes (for every pair of notes, the share of their shortest paths that
  run through a note; Brandes, only notes as sources and targets) and Burt's constraint (high when a note's contacts
  are all tied to each other, low when it spans groups). Exact while notes × (nodes + edges) stays within
  `engine.explore.path_budget` edge visits (20,000,000); past that, a seeded sample of source notes, and the
  output says so and how many.
- **Hubs and authorities**: HITS on the body links (tag links are tags), missing notes included: a good hub links to
  good authorities, a good authority is linked from good hubs.

Every list stops at a percentile of the vault, `engine.explore.percentile` (90: the top tenth), never at a number
tuned on one vault.

## `vault deepen [vault] <note> [--lens L]… [--limit N] [--json] [-f FORMAT]`

Five lenses on one note; `--lens` picks some (`neighbourhood`, `cited`, `foundations`, `backbone`, `cluster`).

```
$ vault deepen /tmp/sample-vault reward-hacking-as-specification-gaming -f brain --limit 6
Deepen reward-hacking-as-specification-gaming
  20_ZETTLEKASTEN/reward-hacking-as-specification-gaming.md · index 2026-11-10T10:15:55+00:00
  cutoffs: at or above the 90th percentile of the vault (engine.explore.percentile)

  Neighbourhood  (vault related: 18 of 179 notes at or above the 90th percentile of this note's scores, 0.33)
      1. model-awareness-of-intent-violation-while-cheating  1.00  linked
         links to it · shares #emergent-behavior, #ai-safety · similar text (hacking, reward, cheating)
      2. obfuscated-reward-hacking  0.79  linked
         linked from it · shares #emergent-behavior, #ai-safety · similar text (hacking, reward, cheating)
      3. reward-hacking-as-eval-false-positive  0.66  linked
         links to it · shares #ai-safety · similar text (hacking, reward, hacks)
      4. reward-hacking-detection-via-at-scale-agent-qa  0.62  linked
         linked from it · shares #ai-safety · similar text (hacking, reward, agent)
      5. 2503.11926 - Monitoring Reasoning Models (CoT Obfuscation)  0.61  linked
         linked from it · similar text (hacking, reward, cheating)
      6. goodhart-at-the-contribution-layer  0.52  linked
         links to it · similar text (reward, hacking, optimizer)
         … 12 more (--limit 0 shows all)

  Cited together  (notes that link to both it and them; 34 found, 0 at or above the 90th percentile of the vault's 415 values (every pair), 0.29)
    none stands out in this vault; the strongest:
      1. instance-specific-rubric-as-reward-function  0.26 (p80)
         by verifier-failure-vs-rubric-design-failu…, environment-and-eval-are-the-same-object, llm-generated-reward-programs-need-vali… +1
      2. llm-judge-position-bias  0.26 (p79)
         by HUD - Verifier and Reward Design for RL…, 2511.19399 - DR Tulu (RLER, Evolving Ru…, a-bounty-list-is-a-map-of-what-cannot-b… +1
      3. four-layer-scoring-stack-for-rl-environments  0.25 (p77)
         by environment-and-eval-are-the-same-object, reward-computation-as-a-heavyweight-pip…, llm-generated-reward-programs-need-vali…

  Same foundations  (what it and they both link to; 24 found, 2 at or above the 90th percentile of the vault's 1097 values (every pair), 0.22)
      1. cot-monitor-beats-action-monitor  0.34 (p98)
         both link 2503.11926 - Monitoring Reasoning Model…, reward-hacking-detection-via-at-scale-a…, obfuscated-reward-hacking
      2. obfuscated-reward-hacking  0.28 (p96)
         both link 2503.11926 - Monitoring Reasoning Model…, scheming-as-threat-model, reward-model-overoptimization-hump (no …

  Backbone  (community 2: the longest path through the strongest ties of its 32 notes, 17 notes long)
    reliability-tax-80-vs-50-percent-time-horizon
      │ links
    task-messiness-as-external-validity-gap
      │ both #benchmark-design
    reward-hacking-detection-via-at-scale-agent-qa
      │ links
    reward-hacking-as-eval-false-positive
      │ links
    model-awareness-of-intent-violation-while-cheating
      │ links
    obfuscated-reward-hacking
      │ links
    monitorability-tax
      │ links
    cot-monitor-beats-action-monitor
      │ links
    2503.11926 - Monitoring Reasoning Models (CoT Obfuscation)
      │ links
    Shlegeris Greenblatt - The Case for Ensuring Powerful AIs are Controlled
      │ links
    2412.14093 - Alignment Faking in Large Language Models
      │ links
    2312.06942 - AI Control - Improving Safety Despite Intentional Subversion
      │ links
    2412.04984 - Frontier Models Capable of In-context Scheming
      │ links
    sandbagging-strategic-underperformance
      │ links
    scheming-as-threat-model
      │ links
    ai-control-vs-alignment
      │ links
    control-evaluations-red-team-blue-team
    this note is 1 step off it, joining at model-awareness-of-intent-violation-while-cheating

  Cluster  (community 2: #ai-safety · control-evaluations-red-team-blue-team, 32 notes)
    key phrases  #ai-safety 14/15 · → control-evaluations-red-team-blue-team 12/12 · → Shlegeris Greenblatt - The Case…
    hubs         scheming-as-threat-model, 2312.06942 - AI Control - Improving Safety Despite Intentional Subversion, control-evaluations-red-team-blue-team
    topic page   emergent-misalignment-from-reward-hacking (links 4)
    this note    weight 11 inside (p69 of the members); 66% of its weight goes to other communities (participation 0.73)
    next to      4. grpo-group-relative-policy-optimization · [[post-…  (weight 21, through reward-hacking-as-specification-gaming, 2411.15124 - Tulu 3 (RLVR Post-Training))
    next to      3. [[evaluation-methodology]] · construct-and-operat…  (weight 15, through reward-hacking-as-eval-false-positive, control-evaluations-red-team-blue-team)
    next to      5. instance-specific-rubric-as-reward-function · [[r…  (weight 7, through reward-hacking-as-specification-gaming, reward-hacking-as-eval-false-positive)
```

| Lens | What | Kept |
|---|---|---|
| neighbourhood | `vault related` (links, tags, text) | at or above the percentile of this note's scores over every other note |
| cited together | notes co-cited with it: other notes link to both (weighted as in `vault explain`) | at or above the percentile of every co-citation value in the vault; none kept: the strongest, marked with their percentile |
| same foundations | notes coupled with it: both link to the same notes or missing notes | at or above the percentile of every coupling value in the vault |
| backbone | in its community, two notes are tied by their direct links plus, for each missing note or tag both attach to, 1 / ln(1 + that node's notes); the longest path through the strongest tree of those ties is the main chain; the note is on it or so many steps off | — |
| cluster | its community: name, key phrases, hubs, topic page; how much of its weight stays inside (as a percentile of the members) and goes out (participation: 0 all in one community, higher spread); the neighbouring communities and the notes that carry the weight there | — |

The vault distributions for co-citation and coupling are exact (every pair); past 1,000 notes they come from a seeded
sample of 1,000 source notes.

## Meaning edges: `vault map --meaning`

A note with few links can end up in no community. `--meaning` adds, for the communities only, an edge between two
notes that are each other's nearest neighbour by embedding (centered cosine, as in `vault related`) when their
similarity is at or above `engine.explore.meaning_percentile` (50) of every note's nearest-neighbour similarity. Each
note gets at most one such edge; bridges, hubs and authorities stay on links and tags. The map then says what changed
against the map without them:

```
$ vault map /tmp/sample-vault -f brain --meaning      # an excerpt
  meaning      29 edges added for the communities: mutual nearest neighbours by embedding at or above the 50th percentile of every note's nearest-neighbour similarity (0.565)
  …
  What the meaning edges change  (communities 12 → 11; notes in none 1 → 2)
    joined  community 2 takes notes from communities 2, 12 of the map without meaning edges
    joined  community 3 takes notes from communities 2, 4, 5, 6, 12 of the map without meaning edges
    joined  community 4 takes notes from communities 3, 4, 5, 10 of the map without meaning edges
    joined  community 6 takes notes from communities 3, 5, 7 of the map without meaning edges
    joined  community 7 takes notes from communities 8, 9 of the map without meaning edges
    joined  community 8 takes notes from communities 5, 11 of the map without meaning edges
    joined  community 9 takes notes from communities 5, 8 of the map without meaning edges
    split   community 3 of the map without them is now 4, 11
    split   community 8 of the map without them is now 7, 9
    moved   34 notes changed community: llm-feedback-as-gradient-substitute (12 → 2), reward (5 → 3), decoupling-clipping-unit-from-credit-as… (6 → 3), a-bounty-list-is-a-map-of-what-cannot-b… (5 → 3), four-layer-scoring-stack-for-rl-environ… (5 → 3), 2411.15124 - Tulu 3 (RLVR Post-Training) (2 → 3) +28
```

On the sample this does not do what it is for. The one note in no community without meaning edges,
`emergent-misalignment-from-reward-hacking` (a topic page), is alone because of its ~30 links to notes that do not
exist yet, which only it links to: Louvain keeps it with them. It gets a meaning edge (to
`model-awareness-of-intent-violation-while-cheating`, 0.66), and one edge does not outweigh thirty. The 29 edges
reshuffle the rest instead: 34 notes move, two communities split and several merge, and
`distillation-to-single-step-policies` ends up alone. Louvain is sensitive to small changes of the graph, so the
meaning edges are an option to look at, not a default.

## `vault discover [vault] [note] [--lens L]… [--limit N] [--steps N] [--seed N] [--json] [-f FORMAT]`

Seven lenses for what the vault does not say yet. With a note: hidden link, same role elsewhere, bridge, wander, and
the gaps, surprises and unwritten notes around it; without one, gaps, surprises and unwritten notes over the whole
vault. Every result has a one-line reason with its evidence, every cutoff is a percentile of the vault
(`engine.explore.percentile`), and the same index, settings and seed give the same output. Embeddings only measure
distance (gap, surprise) and merge names (unwritten), never rank by nearness, and those three lenses also show what
they find without them. Word evidence (key terms) is each note's most distinctive words by tf × idf over this vault.

| Lens | What | Kept |
|---|---|---|
| hidden link | notes it has no link with (either way) that share middles with it: notes, missing notes, tags (the map's graph) or key terms; each middle weighs 1 / ln(1 + its notes) (Adamic–Adar, Swanson's A–B–C) | two or more middles, score at or above the percentile of every unlinked pair's score |
| same role elsewhere | notes in other communities with the same position: ten link features (linked from, links to, share to missing notes, tags, weight inside its community as a z-score, participation, betweenness, constraint, hub, authority), each as a percentile among the notes; distance = mean absolute difference | at or below the (100 − percentile)th percentile of all pairs' distances |
| bridge | its betweenness and constraint among the vault's, the communities its links reach, and for each pair of communities the share of their notes' shortest paths that run through it | the pairs with the largest share |
| gap | two communities close in meaning with fewer links between them than the configuration model predicts; closeness of their mean embeddings, or of their key-term profiles without; the notes that would join them are the unlinked pairs across with the best hidden-link score | closeness at or above the percentile of all community pairs and fewer links than chance; ordered by links, never by closeness |
| surprise | links between notes far apart in meaning (embedding or key-term similarity) in different communities | similarity at or below the (100 − percentile)th percentile of all linked pairs; ordered by how rarely their communities link |
| unwritten | missing notes linked from the most communities, after merging missing notes that mean the same: by name words without embeddings (`suggest`'s overlapping names), by embedding of the names with them (mutual nearest neighbours at or above the percentile) | linked from at least the percentile of communities over all missing notes |
| wander | a seeded random walk on links, a note not yet visited when there is one | `--steps` (8); `--seed` (`engine.explore.seed`) |

```
$ vault discover /tmp/sample-vault reward-hacking-as-specification-gaming -f brain --limit 2
Discover around reward-hacking-as-specification-gaming
  index 2026-11-10T10:15:55+00:00 (up to date)
  cutoffs: the 90th percentile of the vault (engine.explore.percentile) · seed 42 · tag links: format brain (Tags)

  Hidden link  (notes it has no link with, reached through two or more middles: notes, missing notes, tags or key terms both have; 3 of 29 at or above the 90th percentile of 2133 unlinked pairs, 2.81)
      1. reinforcement-learning-in-language-models
         no link either way; 9 middles in common: the-critic-deletion-tension-in-agentic-rl (3 notes), emergent-misalignment-from-reward-hacking (5 notes), ostensible-positives-from-low-entropy-answer-spaces (5 notes), a-bounty-list-is-a-map-of-what-cannot-be-verified (6 notes) +5; score 4.57 (p98 of 2133 unlinked pairs)
      2. reward
         no link either way; 7 middles in common: llm-generated-reward-programs-need-validation (5 notes), ostensible-positives-from-low-entropy-answer-spaces (5 notes), a-bounty-list-is-a-map-of-what-cannot-be-verified (6 notes), goodhart-at-the-contribution-layer (6 notes) +3; score 3.41 (p94 of 2133 unlinked pairs)

  Same role elsewhere  (notes in other communities whose ten link features sit at the same percentiles; 4 at or below the 10th percentile of 16110 pairs' role distances, 0.165)
      1. verifier-failure-vs-rubric-design-failure
         community 5 (this note: 2); role distance 0.135 (p5, lower is closer); linked from 19 vs 6 (p100 vs p93); betweenness 0.11 vs 0.02 (p100 vs p86); authority 1 vs 0.10 (p100 vs p88)
      2. four-layer-scoring-stack-for-rl-environments
         community 5 (this note: 2); role distance 0.147 (p7, lower is closer); linked from 19 vs 4 (p100 vs p86); betweenness 0.11 vs 0.01 (p100 vs p67); authority 1 vs 0.12 (p100 vs p91)

  Bridge  (betweenness 0.106, p100; constraint 0.048, p4; in community 2; its links and tags reach 4 (11), 2 (11), 5 (5), 6 (2), 3 (1), 1 (1); exact)
      1. 37% of the shortest paths between notes of communities 4 and 12 (38 pairs) run through it
      2. 37% of the shortest paths between notes of communities 2 and 4 (589 pairs) run through it

  Gap  (communities close in meaning with fewer links between them than chance predicts)
    with embeddings (closeness of mean embeddings at or above the 90th percentile of 66 community pairs, 0.23; 0 found)
         none
    without them (closeness of key-term profiles at or above the 90th percentile of 66 community pairs, 0.08; 0 found)
         none

  Surprise  (links between notes far apart in meaning, in different communities)
    with embeddings (embedding similarity at or below the 10th percentile of 368 linked pairs, 0.04; 5 found)
      1. covariance-targeted-entropy-intervention → reward-hacking-as-specification-gaming
         covariance-targeted-entropy-intervention links it (line 43); embedding cosine -0.00 (p8 of 368 linked pairs); communities 6 and 2 (3 links between them where chance predicts 31.3); about: cov, intervention, entropy / misspecification, exploiting, library
      2. rubric-agreement-does-not-establish-rubric-v… → reward-hacking-as-specification-gaming
         rubric-agreement-does-not-establish-rubric-validity links it (line 39); embedding cosine -0.16 (p2 of 368 linked pairs); communities 3 and 2 (9 links between them where chance predicts 24.2); about: consequential, substantive, validity / misspecification, exploiting, library
    without them (key-term similarity at or below the 10th percentile of 368 linked pairs, 0.00; 13 found)
      1. covariance-targeted-entropy-intervention → reward-hacking-as-specification-gaming
         covariance-targeted-entropy-intervention links it (line 43); key-term cosine 0.00 (p27 of 368 linked pairs); communities 6 and 2 (3 links between them where chance predicts 31.3); about: cov, intervention, entropy / misspecification, exploiting, library
      2. reward-computation-as-a-heavyweight-pipeline… → reward-hacking-as-specification-gaming
         reward-computation-as-a-heavyweight-pipeline-stage links it (line 28); key-term cosine 0.00 (p27 of 368 linked pairs); communities 6 and 2 (3 links between them where chance predicts 31.3); about: pipelining, computation, container / misspecification, exploiting, library

  Unwritten  (missing notes linked from the most communities; the same idea merged first)
    with embeddings (merged by embedding of the names at or above the 90th percentile, 0.73: 28 groups; linked from at least 2 communities, the 90th percentile of 584; 2 found)
      1. [[reward-model-overoptimization-hump]]
         12 notes in 4 communities link it (4: 4, 5: 4, 2: 2, 6: 2)
      2. [[1707.06347 - Proximal Policy Optimization Algorithms]] +1
         2 notes in 2 communities link it (2: 1, 4: 1); merged by embedding (0.94): [[1707.06347 - Proximal Policy Optimization Algorithms]], [[170706347-proximal-policy-optimization-algorithms]]
    without them (merged by name words: 27 groups; linked from at least 2 communities, the 90th percentile of 578; 2 found)
      1. [[reward-model-overoptimization-hump]]
         12 notes in 4 communities link it (4: 4, 5: 4, 2: 2, 6: 2)
      2. [[1707.06347 - Proximal Policy Optimization Algorithms]] +1
         2 notes in 2 communities link it (2: 1, 4: 1); merged by name words: [[1707.06347 - Proximal Policy Optimization Algorithms]], [[170706347-proximal-policy-optimization-algorithms]]

  Wander  (a random walk on links, seed 42: the same seed, the same walk; 8 steps through 4 communities)
    reward-hacking-as-specification-gaming
      → the-critic-deletion-tension-in-agentic-rl
        the-critic-deletion-tension-in-agentic-rl links to it (line 31); into community 4
      → credit-assignment-problem
        it links to credit-assignment-problem (line 27)
      → 2503.20783 - Understanding R1-Zero-Like Training
        2503.20783 - Understanding R1-Zero-Like Training links to it (line 195)
      → value-function
        it links to value-function (line 138); into community 5
      → reward
        it links to reward (a tag link, line 6)
      → four-layer-scoring-stack-for-rl-environments
        four-layer-scoring-stack-for-rl-environments links to it (a tag link, line 3)
      → environment-and-eval-are-the-same-object
        environment-and-eval-are-the-same-object links to it (line 31)
      → evaluation-unit-scales-from-answer-to-trace-to-world
        evaluation-unit-scales-from-answer-to-trace-to-world links to it (line 24); into community 3
```

Gaps over the whole vault, with and without embeddings (`vault discover /tmp/sample-vault -f brain --lens gap`):

```
  Gap  (communities close in meaning with fewer links between them than chance predicts)
    with embeddings (closeness of mean embeddings at or above the 90th percentile of 66 community pairs, 0.23; 4 found)
      1. communities 8 and 9: [[fine-tuning-strategies-for-llms]] · [… / [[adas-mechanism]] · [[agentic-rag-appr…
         closeness of their mean embeddings 0.52 (p98 of 66 community pairs); 0 links between them where chance predicts 0.5
         join fine-tuning-strategies-for-llms ↔ importance-of-evaluation-in-ai-systems (3.60, via 'adjustments', 'tuner', critic-tuner-approach (no not…)
         join constitutional-ai ↔ agentic-ai-usage-in-customer-service (2.44, via 'forge', knowledge-forge-method (no no…, 'responses')
         join constitutional-ai ↔ importance-of-evaluation-in-ai-systems (1.44, via 'tuner', critic-tuner-approach (no not…)
      2. communities 6 and 12: [[systems-engineering]] · [[reinforceme… / #prompt-engineering
         closeness of their mean embeddings 0.38 (p97 of 66 community pairs); 0 links between them where chance predicts 0.2
         join reward-computation-as-a-heavyweight-pip… ↔ persona-driven-synthetic-data-generation (0.87, via 'verifiable', rlvr-reinforcement-learning-w…)
      3. communities 6 and 11: [[systems-engineering]] · [[reinforceme… / [[memory-management]] · multi-step-comp…
         closeness of their mean embeddings 0.24 (p91 of 66 community pairs); 0 links between them where chance predicts 0.1
    without them (closeness of key-term profiles at or above the 90th percentile of 66 community pairs, 0.08; 2 found)
      1. communities 1 and 5: [[discourse-graph]] · #evidence / instance-specific-rubric-as-reward-func…
         closeness of their key-term profiles 0.12 (p98 of 66 community pairs); 3 links between them where chance predicts 16.0
         join EVD - Agents checked out future commits… ↔ environment-task-split-sets-the-cost-mo… (4.11, via 'barber', 'denain', Epoch AI (Denain & Barber) - …)
         join EVD - Verifiers makes an environment a … ↔ environment-as-installable-package-not-… (3.81, via 'package', 'installable', 'intellect')
         join EVD - Verifiers makes an environment a … ↔ environment-and-eval-are-the-same-object (2.88, via 'artifact', 'intellect', CLM - A fixed typed contract …)
      2. communities 1 and 7: [[discourse-graph]] · #evidence / 2511.01824 - Simia Simulating Environme…
         closeness of their key-term profiles 0.09 (p91 of 66 community pairs); 3 links between them where chance predicts 3.4
         join EVD - Agents checked out future commits… ↔ 2504.21798 - SWE-smith Scaling Data for… (2.09, via 'linked', 'repo', 'swe')
         join EVD - The ground truth deliberately lis… ↔ tools-as-database-read-write-for-verifi… (1.88, via 'write', 'read', rl-environment-design (no not…)
         join EVD - EnvScaler grades only the final d… ↔ tools-as-database-read-write-for-verifi… (1.56, via 'agentscaler', 'database', rl-environment-design (no not…)
```

What looks real and what looks like noise on the sample:

- hidden link: `reward-hacking-as-specification-gaming` → `cot-monitor-beats-action-monitor` (third, shared source and
  detection notes) and `kl-penalty-in-rlhf` → `covariance-targeted-entropy-intervention` (via 'kl', 'region', TRPO,
  'ppo': both are about holding a policy near where it was) are real missing links. The first result for
  reward-hacking, `reinforcement-learning-in-language-models`, is a hub that is near everything;
- gap: without embeddings it finds a real one, the discourse notes on RL environment design and the zettels on
  rewards and environments (3 links where chance predicts 16), and its best joiner, `EVD - Verifiers makes an
  environment a separate installable artifact` ↔ `environment-as-installable-package-not-config`, is the same idea
  written twice. With embeddings it prefers two small groups of older overview notes, close because they are written in
  the same general style; their joiners run through key terms like 'tuner' and 'forge';
- surprise: `covariance-targeted-entropy-intervention` → `reward-hacking-as-specification-gaming` is a plausible
  cross-topic link; several others lead into the group of older generic notes and are style, not insight. Without
  embeddings it is weak: a quarter of linked pairs share no key term, so everything at the cutoff ties at 0;
- unwritten: `[[llm-as-judge]]` and its variants (merged) and `[[evaluation-methodology]]`, each linked from 6–8
  communities, are the notes to write. Merging by names (words or embeddings) agrees on the clear cases; the
  embedding merge is more cautious (2 names for llm-as-judge, against 4 by words);
- same role: matches notes with similar percentiles (a much-linked concept in another community); true, rarely news;
- bridge and wander: descriptive, and correct (`reward-hacking-as-specification-gaming` carries 37% of the shortest
  paths between communities 2 and 4).

## What it costs, and what is incremental

Timed on generated vaults (`scripts/bench_shape.py`: topics of about 60 notes, each note tagged with its topic,
4–8 links, four in five inside its topic; seeded). The 3,000-note vault is real files, indexed; 5,000 and 50,000 are
the same shape built in memory.

| | 3,000 notes (3,259 nodes, 23,478 edges) | 5,000 (5,447, 39,448) | 50,000 (54,380, 393,979) |
|---|---|---|---|
| build the graph from the index | 0.23 s | 0.08 s | 1.2 s |
| communities (Louvain) | 0.32 s | 0.21 s | 5.4 s |
| betweenness, exact | 23 s | about 65 s (estimated) | about 4.5 hours (estimated: 0.32 s per source note) |
| betweenness, 200 sampled source notes | 2.1 s | 2.7 s | 67 s |
| Burt's constraint | 0.21 s | 0.34 s | 4.1 s |
| HITS | 1.0 s | 0.9 s | 17 s |
| co-citation / coupling distributions | 0.18 / 0.22 s | | |
| `vault map`, whole command | 12–14 s (748 sampled sources under the default budget) | | |

These were measured with the pure-Python betweenness, constraint and HITS. They now run in numpy with the same
arithmetic in the same order (the same numbers, bit for bit: `tests/test_vaultengine_shape.py`, `ReferenceTests`):
betweenness from 200 source notes takes about a quarter of the time above, HITS about a twentieth; constraint keeps
its loop with the proportions computed once (1.6 times faster). `vault explore` also keeps them between runs
([Exploring](exploring.md#how-fast-and-why)). Louvain, constraint and HITS grow with the edges. Exact betweenness grows with notes × edges, too slow in pure Python
past a few thousand notes, so the budget switches it to sampling. How good the sample is depends on how peaked
betweenness is. On the sample vault half the notes as sources find 8–10 of the true top 10. On the flat generated
vault (top note 5× the median) the overall order holds (Spearman 0.91 with 500 of 3,000 sources, 0.96 with 1,000) but
the exact top 20 does not (11 of 20 with 1,000 sources). Raise `engine.explore.path_budget` for more sources.

Incremental: the index is (only changed notes are re-parsed), and reading the graph from it is a fraction of a
second. Communities, betweenness and the distributions are recomputed from the graph on every run: nothing is cached,
because read commands do not write the index. A warm start of Louvain from the previous communities would make
re-runs after small edits cheaper; it is not done.

### `vault discover` and `vault map --meaning`

Timed on generated vaults of real files with words (`scripts/bench_shape.py discover N DIR`: the same links and tags,
plus 25 words per note, three in five from the note's topic, drawn from the embedding model's own vocabulary; missing
notes named by two topic words). Vectors are computed in memory (no `vault embed`), which is the slowest case.

| | 5,000 notes | 50,000 notes |
|---|---|---|
| index (full first run / unchanged) | 9 s / 1.3 s | 159 s / 3.4 s |
| embed every note in memory | 6–9 s | 51–70 s |
| communities / again with meaning edges | 0.6 / 0.4 s | 8–13 / 8–12 s |
| meaning edges (mutual nearest neighbours, in blocks) | 1.4–4.3 s | 21–33 s |
| key terms of every note | 1–1.7 s | 12–16 s |
| hidden link: vault distribution / one note | 2.5 s / 0.01 s | 19 s / 0.05 s |
| same role: betweenness / features and distribution | 15 s / 2.5 s | 23 s / 40 s |
| bridge, one note | 10 s | 14 s |
| gap with / without embeddings | 0.2 / 0.2 s | 1.9 / 7.6 s |
| surprise with / without | 0.5 / 0.7 s | 5.4 / 5.8 s |
| unwritten with / without | 0.2 / 0.1 s | 7.5 / 16 s |
| wander, 20 steps | 0.00 s | 0.1 s |

Every step is bounded: the vault distributions behind the cutoffs are exact up to 1,000 notes (role distances: up to
250,000 pairs) and past that come from a seeded sample (hidden link: as many source notes as `path_budget` allows;
role distances: 250,000 random pairs, a rank error of about 0.1 percentile points), and betweenness and the bridge lens
use the same budget as `vault map`. A note's own results (hidden link, same role, gap joiners) cost little once the
distributions exist. The numbers vary between runs by up to half: this sandbox is slow to hand out new memory (filling
a fresh 80 MB array takes 9.4 s here, a refill of the same array 0.03 s), which inflates the steps that allocate most
(embedding, the nearest-neighbour blocks); on an ordinary machine they are faster.
