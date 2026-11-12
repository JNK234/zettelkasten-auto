# Experiments: `vault experiment`

[← README](../README.md) · [Search and connections](search.md) · [The shape of the vault](explore.md)

`vault experiment` tests the engine's options on your own notes, with seeds so every run gives the same
numbers, and says which default it would change. Nobody labels anything by hand: the vault's own links,
`Tags:` topics and topic pages are the answers it checks against. Defaults in `config/config.yaml` were
changed only where a report showed a clear win; each changed value says why, next to it.

```
vault experiment [vault] [-f FORMAT] [--seeds 5] [--hide 0.15] [--queries 200] [--without PATH]…
                 [--question relatedness|text|communities]… [--output FILE] [--json] [--quiet]
```

Examples use a copy of the sample vault: `cp -Rp tests/fixtures/vault /tmp/sample-vault && vault index /tmp/sample-vault && vault embed /tmp/sample-vault`.

## The three questions

| question | options compared | scored by |
|---|---|---|
| **relatedness**: which signals rank related notes, with which weights | links only; links + co-citation + coupling; the same + tags; text only; tags + text; today's `related.unlinked_weights`; four mixes of all five signals and two of links, tags and text (12 options) | hidden links: MRR, recall@5, recall@10 |
| **text**: what a note is embedded as | `whole` (title, aliases, headings, body), `no_template` (without the template headings and the `Key: value` metadata lines), `first_section` (title and first section), each by itself and inside today's related mix | hidden links, with text alone; plus a diagnostic: how many of a note's 5 nearest neighbours share its note type, against chance |
| **communities**: Louvain with and without meaning edges | no meaning edges, and meaning edges with 1, 2 or 3 mutual nearest neighbours at or above the 25th, 50th or 75th percentile of similarity; each at resolution 0.5, 1 and 1.5 (30 options) | ARI and NMI with the `Tags:` topics and with the topic pages; stranded notes; stability across seeds; time |

**Datasets.** The whole vault, and the vault without each `--without` folder (links into the left-out
notes become links to missing notes). A default changes only when the option wins in every dataset:
`--without 50_MOC/discourse` checks that a result is not just the 40 templated discourse notes.

### Hidden links, and why they cannot leak

For each seed, a seeded 15% (`--hide`, 0.1 to 0.2 is meant) of the linked note pairs is hidden. For a
hidden pair, *every* link between the two notes goes, both ways, `Tags: [[…]]` tag links too, and with it
everything derived from it:

- the graph is rebuilt from the links that remain, so personalized PageRank, co-citation and coupling never
  see the hidden edge;
- the tag sets are rebuilt, so a hidden tag link is no shared tag;
- the links' own text (`[[B]]`, the display text of `[[B|…]]`, the `Tags:` entry) is blanked to spaces in
  both notes before they are parsed and embedded, so the words of a link do not make the two notes look
  alike.

Then, from the note that wrote the link, every note it is *not still linked with* is ranked by the option,
with the same functions `vault related` uses (`related.link_signals`, `tag_similarity`, `combine`), and
the hidden partner's rank is recorded (ties count at their expected rank). The test suite proves the hiding
is complete: the scores from the writing note to every other note, for all five signals and all three text
variants, equal those of a vault whose files never had the link
(`tests/test_vaultengine_experiment.py`, `NoLeakTests`; removing either the blanking or the tag-link
hiding makes it fail).

### Communities against the vault's own topics

Each reference is held out of the graph it is compared with: against the `Tags:` topics, the communities
are found without tag links and `#tags`; against the topic pages (the format's `topic`/`page` types),
without the pages' links. A note's tag label is its most widely used tag (tags on one note only, and
tags on most notes, are left out); a page label is the smallest page that links the note.
NMI alone rises with the number of communities and ARI against a coarse reference favours fewer, so a
community setting changes only when it wins on **both** ARI and NMI.

### When a difference counts

Every option is scored on the same seeds. A difference is *clear* when the option is ahead on **every**
seed and the mean per-seed difference is more than twice its standard error; otherwise the report says it
is within the seed-to-seed noise. The decision for a setting: the options that beat today's setting clearly
in every dataset, the best mean among them. The report has a summary table per question (winner, margin
over the runner-up, over today's setting, clear or not), the full tables, and the decisions.

## The run on the sample vault (2026-11-10)

Report: [docs/experiments/2026-11-10.md](experiments/2026-11-10.md). Run on a copy of the sample vault with the
settings as they were before (`--setting related.weights.graph=0.5 …`), 5 seeds, 43 s:

| question | whole vault (180 notes) | without `50_MOC/discourse` (140) | default |
|---|---|---|---|
| relatedness | all five, text 0.7: MRR 0.255 vs today's 0.221, +0.034 ± 0.008, 5 of 5 seeds | all five, text 0.7: 0.318 vs 0.276, +0.042 ± 0.010, 5 of 5 | **changed**, for unlinked notes only: `related.unlinked_weights` graph 0.1, cocitation 0.05, coupling 0.05, tags 0.1, text 0.7 (plain `related.weights` keeps graph 0.5, tags 0.2, text 0.3) |
| text | `no_template` 0.268 vs `whole` 0.236, +0.032 ± 0.012, 5 of 5 | `no_template` 0.310 vs 0.307, +0.003 ± 0.008, 2 of 5 | kept `whole`: the gain is the templated discourse notes, and goes with them |
| communities | resolution 0.5 (± meaning edges) wins on ARI (+0.19), not on NMI | the same: ARI +0.05, NMI not | kept: resolution 1, no meaning edges |

What the tables also say:

- Links alone are the weakest way to find a missing link (MRR 0.199 and 0.246), text alone is strong
  (0.236 and 0.307), and the mixes with text at 0.5 to 0.8 are best. Those three (text 0.5, 0.7, 0.8) are
  within the noise of each other: the data says *more text, less graph*, not exactly 0.7.
- Co-citation and coupling add nothing measurable here on top of the graph signal.
- `first_section` is the worst text in both datasets: it also makes neighbours share the note type more
  (0.73 against 0.42 by chance), i.e. it finds notes written from the same template, not about the same thing.
- Meaning edges at resolution 1: ARI +0.081 ± 0.014 (5 of 5) in the whole vault, but +0.010 (3 of 5)
  without the discourse notes and no gain in NMI in either: not a clear win.

The experiment only ever ranks notes that are *not* linked (a hidden link's partner among the unlinked notes),
so it says nothing about where linked notes should come. Its mix therefore went into its own setting,
`related.unlinked_weights`, used where you look for what you have not linked: `vault related --unlinked`,
`vault discover`, `vault explore`. Plain `vault related` keeps `related.weights` (graph 0.5, tags 0.2, text 0.3), so
the notes you already link stay first. (The report below was written when both were one setting; its decision line
says `related.weights`.) The experiment now decides `related.unlinked_weights`.

### Where 180 notes are too few

- **Topic pages**: the sample has 2, labelling 9 notes; the report marks them "too few to judge by", and no
  decision uses them.
- **Hidden links**: 55 (49 without discourse) per seed. One of them moving from rank 1 to rank 2 moves MRR
  by 0.009, so the differences between the text-heavy mixes, between meaning-edge settings, and between
  the text variants without discourse cannot be told apart.
- **Communities**: every option strands 0.4 to 2.4 notes on average, so the sample cannot show whether
  meaning edges rescue isolated notes. The `Tags:` reference labels 155 notes with 20 topics, but two of
  them cover 82 notes and eleven label 1 to 3 notes each.
- **Note types**: the sample has few, so two notes share a type by chance often (0.42, and 0.64 without
  the discourse notes); the same-type diagnostic has little room to show anything.

Run it on your own vault, with `--without` for any folder of templated or generated notes, before trusting
these defaults.

## What it costs

| vault | seeds | time | where it goes |
|---|---|---|---|
| sample, 180 notes, two datasets | 5 | 24 to 43 s | per dataset: hidden links 14 s (18 ranking options on 55 or 49 hidden links a seed), communities 6 to 8 s (30 options) |
| generated, 5,000 notes, 27,958 linked pairs | 5 | 295 s and 491 s in two runs (5 to 8 minutes) | in the 491 s run: hidden links 295 s (200 of the 4,194 hidden links scored per seed, `--queries`; the signals and 18 options for each, and the three texts embedded once, the hidden links' notes again per seed), the same-type diagnostic 15 s, communities 179 s (30 options × 5 seeds, Louvain on 5,000 notes twice each) |

The 5,000-note vault (`scripts/bench_shape.py`) has random words for text, so its winners mean nothing;
it is there for the time. The sandbox's timings vary a lot from run to run (the same command took 295 s and
491 s); the report prints its own. Relatedness and text are scored in one pass over the same hidden links, and
personalized PageRank is vectorized (numpy, one walk per held-out graph): the first version took 6 minutes
for one seed at 5,000 notes.

## Options

| option | meaning |
|---|---|
| `--seeds N` | runs, from `engine.explore.seed` on (2 or more; 5 by default) |
| `--hide F` | share of the linked note pairs hidden per seed (default 0.15) |
| `--queries N` | hidden links scored per seed at most (a seeded sample beyond; default 200) |
| `--without PATH` | also run without the notes under `PATH` (repeatable) |
| `--question Q` | only this question (repeatable) |
| `-f FORMAT` | the format: which links are tags, the template headings, the topic-page types |
| `--output FILE` | also write the report, e.g. `docs/experiments/<date>.md` |
| `--json` | the decisions as JSON instead of the report |

Every `--setting` works here too: the "today's setting" rows are the settings the run sees.
