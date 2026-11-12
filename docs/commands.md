# Commands

[← Documentation index](../README.md#documentation)

The examples are one session, in this order, on a fresh copy of the sample vault (see [Install](install.md); activate the
environment with `source .venv/bin/activate` so `vault` is on `PATH`):

```bash
rm -rf /tmp/sample-vault /tmp/inbox && cp -Rp tests/fixtures/vault /tmp/sample-vault && mkdir /tmp/inbox
vault index /tmp/sample-vault
```

Timings, dates and the `Created`/`Ingested` times differ on your machine; everything else should match. `-f` is short for
`--format`, a format name or file ([The format file](formats.md)); violations and their codes are listed in
[The format file: Violations](formats.md#violations).

| Exit code | Means |
|---|---|
| 0 | done (for `check`: no violations) |
| 1 | `check` or `formats --strict` found violations; a write was refused; `ingest` found a duplicate |
| 2 | bad arguments, no such note, a name already taken, no index for `--changed`, an invalid format file, invalid settings, an embedding model that cannot be loaded |
| 3 | a write failed and every file was restored |

## Settings: `vault config show [vault] [--db F] [--format F] [--setting K=V]… [--config FILE] [--json]`

Every engine setting is in one place, the `engine:` section of [`config/config.yaml`](../config/config.yaml): the vault,
the index file, the explore memory, the format, the embedding model, `related.weights` and `related.unlinked_weights`
(how [`vault related`](search.md#related-notes-links-to-add-notes-to-write) mixes its signals, with and without the
notes already linked; the second set from [`vault experiment`](experiment.md)), and the explore engine's settings
(the lenses of [`vault explore`](exploring.md), `explore.weights`: how much each
signal counts in [`vault explain`](search.md#how-two-notes-relate-vault-explain); the percentile that counts as
standing out, the topic page types, the seed and the rest of [`vault map`, `deepen` and `discover`](explore.md)). Each one can be overridden, and the first of these that sets it wins:

| Source | How | Example |
|---|---|---|
| flag | the vault argument, `--db`, `--format`, or `--setting KEY=VALUE` for any key (repeatable) | `--setting embedding.model_id=my-id` |
| env | `VAULTENGINE_` + the key in capitals, `__` between levels | `VAULTENGINE_EMBEDDING__MODEL_PATH=/models/potion-base-32M` |
| file | the `engine:` section of `--config FILE`, else `$VAULTENGINE_CONFIG`, else the repository's `config/config.yaml` | `format: brain` |
| default | built in | index: `<vault>/.vaultengine/index.sqlite`; model id: the model folder's name |

Relative paths in the file are relative to the file's folder (so the repository's file says
`model_path: ../models/potion-base-8M`); from a flag or the environment they are relative to the current folder.
Everything is validated (Pydantic) before a command runs: an unknown key, a wrong type or an unknown provider is an error
(exit 2) that names the key and where it was set. With `engine.format` set, `--format` can be left out everywhere.

```
$ vault config show /tmp/sample-vault
Engine settings · precedence: flag > env > file > default
  config file  /workspace/zettelkasten-auto/config/config.yaml  (the repository's; --config FILE or $VAULTENGINE_CONFIG for another)

  vault                                /tmp/sample-vault                                                                                                   flag <vault>
  index                                /tmp/sample-vault/.vaultengine/index.sqlite                                                                         default (<vault>/.vaultengine/index.sqlite)
  memory                               /tmp/sample-vault/.vaultengine/memory.sqlite                                                                        default (<vault>/.vaultengine/memory.sqlite)
  format                               (not set)                                                                                                           default
  embedding.provider                   model2vec                                                                                                           file
  embedding.model_path                 /workspace/zettelkasten-auto/models/potion-base-8M                                                                  file
  embedding.model_id                   potion-base-8M                                                                                                      default (the model folder's name)
  related.weights.graph                0.5                                                                                                                 file
  related.weights.cocitation           0.0                                                                                                                 file
  related.weights.coupling             0.0                                                                                                                 file
  related.weights.tags                 0.2                                                                                                                 file
  related.weights.text                 0.3                                                                                                                 file
  related.unlinked_weights.graph       0.1                                                                                                                 file
  related.unlinked_weights.cocitation  0.05                                                                                                                file
  related.unlinked_weights.coupling    0.05                                                                                                                file
  related.unlinked_weights.tags        0.1                                                                                                                 file
  related.unlinked_weights.text        0.7                                                                                                                 file
  explore.weights.direct               0.3                                                                                                                 file
  explore.weights.cocitation           0.15                                                                                                                file
  explore.weights.coupling             0.15                                                                                                                file
  explore.weights.tags                 0.1                                                                                                                 file
  explore.weights.text                 0.3                                                                                                                 file
  explore.percentile                   90.0                                                                                                                file
  explore.topic_types                  ['topic', 'page']                                                                                                   file
  explore.meaning                      False                                                                                                               file
  explore.meaning_neighbours           1                                                                                                                   file
  explore.meaning_percentile           50.0                                                                                                                file
  explore.seed                         42                                                                                                                  file
  explore.resolution                   1.0                                                                                                                 file
  explore.path_budget                  20000000                                                                                                            file
  explore.lenses                       ['neighbourhood', 'cited', 'foundations', 'backbone', 'cluster', 'hidden', 'role', 'gap', 'surprise', 'unwritten']  file
  explore.start_lenses                 ['gap', 'surprise', 'unwritten', 'topicless']                                                                       file
  explore.per_lens                     4                                                                                                                   file
  explore.duplicate_percentile         99.0                                                                                                                file
  explore.stats_refresh                0.02                                                                                                                file
  explore.link_section                 See Also                                                                                                            file

  env: VAULTENGINE_<KEY>, "__" between levels (VAULTENGINE_EMBEDDING__MODEL_PATH)
  flags: the vault argument, --db, --format, and --setting KEY=VALUE for any key
```

The same with a setting from the environment and one from a flag:

```
$ VAULTENGINE_FORMAT=brain vault config show /tmp/sample-vault --setting embedding.model_id=potion-8m-test
Engine settings · precedence: flag > env > file > default
  config file  /workspace/zettelkasten-auto/config/config.yaml  (the repository's; --config FILE or $VAULTENGINE_CONFIG for another)

  vault                                /tmp/sample-vault                                                                                                   flag <vault>
  index                                /tmp/sample-vault/.vaultengine/index.sqlite                                                                         default (<vault>/.vaultengine/index.sqlite)
  memory                               /tmp/sample-vault/.vaultengine/memory.sqlite                                                                        default (<vault>/.vaultengine/memory.sqlite)
  format                               brain                                                                                                               env VAULTENGINE_FORMAT
  embedding.provider                   model2vec                                                                                                           file
  embedding.model_path                 /workspace/zettelkasten-auto/models/potion-base-8M                                                                  file
  embedding.model_id                   potion-8m-test                                                                                                      flag --setting embedding.model_id
  related.weights.graph                0.5                                                                                                                 file
  related.weights.cocitation           0.0                                                                                                                 file
  related.weights.coupling             0.0                                                                                                                 file
  related.weights.tags                 0.2                                                                                                                 file
  related.weights.text                 0.3                                                                                                                 file
  related.unlinked_weights.graph       0.1                                                                                                                 file
  related.unlinked_weights.cocitation  0.05                                                                                                                file
  related.unlinked_weights.coupling    0.05                                                                                                                file
  related.unlinked_weights.tags        0.1                                                                                                                 file
  related.unlinked_weights.text        0.7                                                                                                                 file
  explore.weights.direct               0.3                                                                                                                 file
  explore.weights.cocitation           0.15                                                                                                                file
  explore.weights.coupling             0.15                                                                                                                file
  explore.weights.tags                 0.1                                                                                                                 file
  explore.weights.text                 0.3                                                                                                                 file
  explore.percentile                   90.0                                                                                                                file
  explore.topic_types                  ['topic', 'page']                                                                                                   file
  explore.meaning                      False                                                                                                               file
  explore.meaning_neighbours           1                                                                                                                   file
  explore.meaning_percentile           50.0                                                                                                                file
  explore.seed                         42                                                                                                                  file
  explore.resolution                   1.0                                                                                                                 file
  explore.path_budget                  20000000                                                                                                            file
  explore.lenses                       ['neighbourhood', 'cited', 'foundations', 'backbone', 'cluster', 'hidden', 'role', 'gap', 'surprise', 'unwritten']  file
  explore.start_lenses                 ['gap', 'surprise', 'unwritten', 'topicless']                                                                       file
  explore.per_lens                     4                                                                                                                   file
  explore.duplicate_percentile         99.0                                                                                                                file
  explore.stats_refresh                0.02                                                                                                                file
  explore.link_section                 See Also                                                                                                            file

  env: VAULTENGINE_<KEY>, "__" between levels (VAULTENGINE_EMBEDDING__MODEL_PATH)
  flags: the vault argument, --db, --format, and --setting KEY=VALUE for any key
```

```
$ VAULTENGINE_EMBEDDING__PROVIDER=openai vault related /tmp/sample-vault kl-penalty-in-rlhf; echo "exit $?"
error: invalid engine settings:
  engine.embedding.provider: unknown provider 'openai'; known: model2vec  [from env VAULTENGINE_EMBEDDING__PROVIDER]
  settings: vault, index, memory, format, embedding.provider, embedding.model_path, embedding.model_id, related.weights.graph, related.weights.cocitation, related.weights.coupling, related.weights.tags, related.weights.text, related.unlinked_weights.graph, related.unlinked_weights.cocitation, related.unlinked_weights.coupling, related.unlinked_weights.tags, related.unlinked_weights.text, explore.weights.direct, explore.weights.cocitation, explore.weights.coupling, explore.weights.tags, explore.weights.text, explore.percentile, explore.topic_types, explore.meaning, explore.meaning_neighbours, explore.meaning_percentile, explore.seed, explore.resolution, explore.path_budget, explore.lenses, explore.start_lenses, explore.per_lens, explore.duplicate_percentile, explore.stats_refresh, explore.link_section
exit 2
```

## `vault embed <vault> [--status] [--full] [--prune]`

Embeds the notes for `vault related` and the links of `vault suggest`, and caches the vectors in the index. It first
brings the index up to date (like `vault index`), then embeds every note whose text has no vector for the configured
model, and nothing else. The model is `engine.embedding` in the settings: `provider: model2vec` reads a static
[Model2Vec](https://github.com/MinishLab/model2vec) model folder (`tokenizer.json` + `model.safetensors`; the
repository ships `models/potion-base-8M`), with `tokenizers` and `numpy` only. Another Model2Vec model is a settings
change: point `model_path` at its folder.

How a note becomes a vector: its title, aliases, headings and body (no frontmatter, code, comments, tags, property keys,
URLs or TeX math `$…$`/`$$…$$`) are tokenized, unknown tokens are dropped, the rows of the remaining tokens are averaged and the mean is
normalised to length 1. Vectors are cached by **the SHA-256 of that text and the model id**, so:

- editing a note re-embeds that note only; an edit that leaves the text alone (a tag, blank lines) re-embeds nothing,
  and a rename re-embeds it (the title is part of the text);
- a new version of what goes into the text re-parses every note and re-embeds only those whose text changed (leaving
  out TeX math re-embedded 25 of the sample's 180 notes, the ones with math);
- changing the model id (or the model, whose folder name is the default id) re-embeds every note. The other id's
  vectors are kept, so switching back is free, until `--prune` drops them;
- a vector whose text no note has any more is dropped by the next `vault index`.

Only `vault embed` writes vectors. `vault related`, `vault suggest` and `vault explain` read the cached ones and embed
a note without one in memory, for that run only, and say so ("180 notes had no cached vector: embedded in memory for
this run only; `vault embed <vault>` caches them"). `--status`
only reports, and loads no model; `--full` embeds every note again; `--limit N` paths listed (0 = all).

```
$ vault embed /tmp/sample-vault --status
Embeddings: /tmp/sample-vault
  index 2026-11-09T20:27:13+00:00 (up to date)

  model      potion-base-8M · model2vec · /workspace/zettelkasten-auto/models/potion-base-8M (found)
  vectors    0 of 180 indexed notes · 180 missing: run `vault embed /tmp/sample-vault`
  other ids  none
  cache      0 vectors, 0 B in /tmp/sample-vault/.vaultengine/index.sqlite
$ vault embed /tmp/sample-vault --limit 2
Embedded /tmp/sample-vault in 1.69s
  model      potion-base-8M (model2vec, 256 dims) · /workspace/zettelkasten-auto/models/potion-base-8M
  index      already up to date
  vectors    180 embedded, 0 cached (180 notes)
    + 10_SOURCES/2312.06942 - AI Control - Improving Safety Despite Intentional Subversion.md
    + 10_SOURCES/2411.00640 - Adding Error Bars to Evals.md
  … 178 more (--limit 0 shows all)
```

After editing one note; with another model id; the status under the configured id again; and `--prune`:

```
$ echo "Edited." >> /tmp/sample-vault/20_ZETTLEKASTEN/kl-penalty-in-rlhf.md
$ vault embed /tmp/sample-vault
Embedded /tmp/sample-vault in 0.09s
  model      potion-base-8M (model2vec, 256 dims) · /workspace/zettelkasten-auto/models/potion-base-8M
  index      1 parsed (0 added, 1 changed, 0 deleted)
  vectors    1 embedded, 179 cached (180 notes)
    + 20_ZETTLEKASTEN/kl-penalty-in-rlhf.md
$ vault embed /tmp/sample-vault --setting embedding.model_id=potion-8m-test --limit 1
Embedded /tmp/sample-vault in 1.52s
  model      potion-8m-test (model2vec, 256 dims) · /workspace/zettelkasten-auto/models/potion-base-8M
  index      already up to date
  vectors    180 embedded, 0 cached (180 notes)
    + 10_SOURCES/2312.06942 - AI Control - Improving Safety Despite Intentional Subversion.md
  … 179 more (--limit 0 shows all)
$ vault embed /tmp/sample-vault --status
Embeddings: /tmp/sample-vault
  index 2026-11-09T20:27:19+00:00 (up to date)

  model      potion-base-8M · model2vec · /workspace/zettelkasten-auto/models/potion-base-8M (found)
  vectors    180 of 180 indexed notes (256 dims) · all up to date
  other ids  potion-8m-test: 180 vectors (256 dims)  (kept for switching back; `vault embed --prune` drops them)
  cache      360 vectors, 360.0 KiB in /tmp/sample-vault/.vaultengine/index.sqlite
$ vault embed /tmp/sample-vault --prune
Embedded /tmp/sample-vault in 0.03s
  model      potion-base-8M (model2vec, 256 dims) · /workspace/zettelkasten-auto/models/potion-base-8M
  index      already up to date
  vectors    0 embedded, 180 cached (180 notes)
  pruned     180 vectors of other model ids
```

A model that cannot be loaded is an error for `vault embed`, before anything is written (exit 2). Everywhere else it
only switches off the text signal: `vault related`, the links of `vault suggest`, drafts and `vault explain` still
answer from links and tags, and say that the text signal is unavailable and why:

```
$ vault related /tmp/sample-vault kl-penalty-in-rlhf --setting embedding.model_path=/nope --limit 1
Related to kl-penalty-in-rlhf
  20_ZETTLEKASTEN/kl-penalty-in-rlhf.md · index 2026-11-10T15:01:32+00:00

  score = 0.71 × graph + 0.29 × tags · 179 notes compared
  text signal unavailable: embedding model folder not found: /nope
    (the other signals still rank; their weights are rescaled to sum to 1)

   1. direct-preference-optimization-dpo                                            1.00      linked
      linked from it · shares [[reinforcement-learning-from-human-feedback-rlhf]], [[policy-gradient-methods]]
```

## `vault formats <vault> -f <format> [--type T] [--note N] [--limit N] [--strict] [--json]`

What the format sees: the notes of each type, the issues by rule, the relations and the link roles. `--note` shows one
note's type, metadata, relations (mirrored ones included) and every link with its role. Reads the files, not the index.

```
$ vault formats /tmp/sample-vault --format brain --limit 2
Format brain (/…/formats/brain.yaml) on /tmp/sample-vault

  notes      180   180 typed, 0 untyped
  issues     102   in 77 notes

  Types          notes  clean  issues   match
  question           1      1       0   in 50_MOC/discourse/ and named 'QUE *'
  claim              8      1      24   in 50_MOC/discourse/ and named 'CLM - *'
  evidence          31      4      35   in 50_MOC/discourse/ and named 'EVD - *'
  zettel           109     67      42   in 20_ZETTLEKASTEN/
  source            28     27       1   in 10_SOURCES/
  topic              2      2       0   in 50_MOC/topics/
  comparison         1      1       0   in 50_MOC/comparisons/
  …
Issues by rule (issues / notes)
  unresolved-target        59 / 34
  missing-heading          28 / 28
  missing-key              15 / 15
  …
  Relations            declared resolved unresolved mirrored   inverse
  supported-by               37       28          9        0   supports
  supports                   37       28          9        0   supported-by
  …
Link roles (links)
     505  tag
     481  tension
  …
```
(Each type also lists three example notes, `e.g. …`.)

## `vault check <vault> -f <format> [--changed] [--json]`

Every violation, per note, with its line. Exit 1 if there is any. `--changed` checks only notes added or changed since the
last `vault index`, plus what those changes break in other notes (a new `supports::` in A makes B's missing `supported-by::`
a violation; deleting a note breaks links to it). It never writes, not even the index: run `vault index` once the notes are
right. That makes it a hook: `vault check <vault> -f brain --changed || exit 1`.

```
$ vault check /tmp/sample-vault --format brain
vault check /tmp/sample-vault  (format brain: /…/formats/brain.yaml)
  checked 180 notes

10_SOURCES/RLHF.md  (source)
    138  missing-heading      missing heading 'Notes' (expected at the end of the note)

20_ZETTLEKASTEN/agentic-ai-usage-in-customer-service.md  (zettel)
     28  missing-heading      missing headings 'What This Is', 'Why It Matters', 'Key Details', 'Tensions' (expected before 'References')
…
✗ 102 violations in 77 notes
      59  unresolved-target
      28  missing-heading
      15  missing-key
$ echo $?
1
$ vault check /tmp/sample-vault --format brain --changed
vault check /tmp/sample-vault  (format brain: /…/formats/brain.yaml)
  checked 0 notes changed since the last index (indexed 2026-11-07T06:08:11+00:00)

✓ no violations in 0 notes checked
```

Break one note the way a script might (drop its `supported-by::` line, misspell its status), and only that note is reported:

```
$ f="/tmp/sample-vault/50_MOC/discourse/rl-environment-design/CLM - Without a real end state, grade the trajectory against path-agnostic subgoals.md"
$ sed -i.bak -e '/^supported-by::/d' -e 's/^Status: #supported/Status: #suported/' "$f" && rm "$f.bak"
$ vault check /tmp/sample-vault --format brain --changed
vault check /tmp/sample-vault  (format brain: /…/formats/brain.yaml)
  checked 1 note changed since the last index (indexed 2026-11-07T06:08:11+00:00)
    ~ 50_MOC/discourse/rl-environment-design/CLM - Without a real end state, grade the trajectory against path-agnostic subgoals.md

50_MOC/discourse/rl-environment-design/CLM - Without a real end state, grade the trajectory against path-agnostic subgoals.md  (claim)
      5  bad-value            Status: #suported is not an allowed value (supported, contested, weak)
     10  missing-relation     missing relation field `supported-by::` (required for claim)
     10  missing-mirror       missing `supported-by:: [[EVD - Where there is no real state to inspect, SynthAgent grades the trajectory against a]]` (mirror of `supports::` in 50_MOC/discourse/rl-environment-design/EVD - Where there is no real state to inspect, SynthAgent grades the trajectory against a.md:13)

✗ 3 violations in 1 note
       1  bad-value
       1  missing-mirror
       1  missing-relation
$ echo $?
1
```

`--json` prints `{"ok": …, "scope": …, "summary": …, "notes": [{"path", "type", "checked", "violations": [{"path", "line", "code", "message", "related"}]}]}`.

## `vault fix <vault> -f <format> [--dry-run] [--only missing-mirror|missing-heading] [--diff] [--json]`

Repairs only what has exactly one safe repair: a missing mirror (added to the existing field of that relation, or as a new
field after the last relation field, or under the type's `relations_heading`) and a missing required heading (inserted with an
empty section before the next required heading, or at the end, at the level it has in most notes of the type). Everything
else is left alone, and so is every byte the repair does not need. `--dry-run` prints the diff and writes nothing.

```
$ vault fix /tmp/sample-vault --format brain --dry-run
--- a/10_SOURCES/RLHF.md
+++ b/10_SOURCES/RLHF.md
…
--- a/50_MOC/discourse/rl-environment-design/CLM - Without a real end state, grade the trajectory against path-agnostic subgoals.md
+++ b/50_MOC/discourse/rl-environment-design/CLM - Without a real end state, grade the trajectory against path-agnostic subgoals.md
@@ -8,6 +8,7 @@
 
 ## Relations
 informs:: [[QUE 03 - Grade end state, trajectory or text]]
+supported-by:: [[EVD - Where there is no real state to inspect, SynthAgent grades the trajectory against a]]
 
 ## Scope
 One source (SynthAgent) with an LLM judge; it inherits the judge's unvalidated accuracy.

Would fix 29 notes:
  ~ 10_SOURCES/RLHF.md
      end of note: added 'Notes'
  …
  ~ 50_MOC/discourse/rl-environment-design/CLM - Without a real end state, grade the trajectory against path-agnostic subgoals.md
      after line 10: new field `supported-by:: [[EVD - Where there is no real state to inspect, SynthAgent grades the trajectory against a]]`
  check    no new violations
           already there, not blocking: 1 bad-value (run vault check)
Dry run: nothing was written.

Violations left after this fix, for you to fix: 59 unresolved-target, 15 missing-key, 1 bad-value
$ vault fix /tmp/sample-vault --format brain | tail -4
           already there, not blocking: 1 bad-value (run vault check)
  index    updated 28 notes; left pending (unindexed changes with violations): 50_MOC/discourse/rl-environment-design/CLM - Without a real end state, grade the trajectory against path-agnostic subgoals.md

Violations left after this fix, for you to fix: 59 unresolved-target, 15 missing-key, 1 bad-value
$ vault check /tmp/sample-vault --format brain | tail -4
✗ 75 violations in 50 notes
      59  unresolved-target
      15  missing-key
       1  bad-value
```

The claim stays pending in the index because its hand-made typo is still there, so `--changed` keeps reporting it; the other
28 notes are up to date. The typo is fixed with `vault set` below.

## The safe write path: `vault new`, `vault link`, `vault set`

All writes (these three, `fix` and `ingest`) go through one transaction:

- **The gate.** The whole vault is checked as it would be after the write. The write is refused (exit 1, nothing written) if
  that adds any violation anywhere. Violations already there do not block it ("already there, not blocking"), so a note can be
  repaired one problem at a time. A new note has no "before": it must be clean.
- **Only the lines needed change.** Every other byte of every file stays as it was (line endings, BOM, trailing spaces).
- **Atomic, as a group.** New contents go to temporary files (fsynced), originals are hard-linked to backups, a journal in
  `.vaultengine/journal/` lists them, then the files are renamed into place and read back. A failure before the journal is
  removed restores every file; a crash in between is rolled back by the next write (`recovered an interrupted write: …`).
  A file changed by someone else after planning is never overwritten. One writer at a time (a lock in the system temp folder,
  named after the vault, so a refused or dry-run write leaves the vault exactly as it was).
- **The index stays in sync, for what was written only.** Other pending edits stay "changed" for `check --changed`; a written
  note that had unindexed changes and still has violations stays pending too.

Options shared by all writes: `--dry-run` (the diff, nothing written), `--diff` (write and show the diff), `--json`.

```
$ vault set /tmp/sample-vault -f brain "CLM - Without a real end state, grade the trajectory against path-agnostic subgoals" Status "#supported"
Wrote 1 note (set):
  ~ 50_MOC/discourse/rl-environment-design/CLM - Without a real end state, grade the trajectory against path-agnostic subgoals.md
      line 5: Status: #supported
  check    no new violations
  index    updated 1 note
$ vault index /tmp/sample-vault | sed -n 4p
  parsed     0  (index already up to date)
```

`vault new <vault> -f F --type T --title "…" [--folder F] [--name N] [--set K=V]… [--rel RELATION NOTE]… [--section HEADING TEXT]…`
creates a note from the type's `new:` block. `--rel` adds relations to existing notes and writes their mirrors there, so a claim
can be created together with its evidence link; `--section` fills a required heading (`@file` reads a file, `-` stdin).

```
$ vault new /tmp/sample-vault -f brain --type zettel --title "Verifier Hacking Needs a Do-Nothing Baseline" \
    --set "Tags=[[reward-hacking-as-specification-gaming]] [[evaluation-methodology]]" \
    --section "What This Is" "A grader that a do-nothing agent passes is paying reward for inaction."
Wrote 1 note (new):
  + 20_ZETTLEKASTEN/verifier-hacking-needs-a-do-nothing-baseline.md
      new zettel: Verifier Hacking Needs a Do-Nothing Baseline
  check    no new violations
  index    updated 1 note
$ cat /tmp/sample-vault/20_ZETTLEKASTEN/verifier-hacking-needs-a-do-nothing-baseline.md
Created: 07-11-2026 06:08

Status: #baby

Tags: [[reward-hacking-as-specification-gaming]] [[evaluation-methodology]]

# Verifier Hacking Needs a Do-Nothing Baseline

## What This Is
A grader that a do-nothing agent passes is paying reward for inaction.

## Why It Matters

## Key Details

## Tensions

# References
$ vault new /tmp/sample-vault -f brain --type zettel --title "KL Penalty in RLHF" --set "Tags=[[x]]"
error: a note named 'kl-penalty-in-rlhf' already exists: 20_ZETTLEKASTEN/kl-penalty-in-rlhf.md
$ echo $?
2
```

`vault link <vault> -f F A B --as RELATION` writes `RELATION:: [[B]]` in A and, when the relation has an inverse, the mirror
in B. Notes are given by path or name.

```
$ vault link /tmp/sample-vault -f brain "EVD - Reward is a deterministic outcome check 1 only if the final database equals the data" \
    "CLM - Grade the observed end state rather than the trajectory" --as supports
Wrote 2 notes (link):
  ~ 50_MOC/discourse/rl-environment-design/EVD - Reward is a deterministic outcome check 1 only if the final database equals the data.md
      line 13: added [[CLM - Grade the observed end state rather than the trajectory]] to `supports::`
  ~ 50_MOC/discourse/rl-environment-design/CLM - Grade the observed end state rather than the trajectory.md
      line 11: added [[EVD - Reward is a deterministic outcome check 1 only if the final database equals the data]] to `supported-by::`
  check    no new violations
           already there, not blocking: 6 unresolved-target (run vault check)
  index    updated 2 notes
```

Run again, it writes nothing:

```
…/EVD - Reward is a deterministic outcome check 1 only if the final database equals the data.md already has `supports:: [[CLM - Grade the observed end state rather than the trajectory]]`
…/CLM - Grade the observed end state rather than the trajectory.md already has `supported-by:: [[EVD - Reward is a deterministic outcome check 1 only if the final database equals the data]]`
Nothing to write.
```

Refused writes show the violation and the change that was refused, and exit 1:

```
$ vault set /tmp/sample-vault -f brain "CLM - Grade the observed end state rather than the trajectory" Status "#settled"
✗ refused: the change would add 1 violation; nothing was written
  50_MOC/discourse/rl-environment-design/CLM - Grade the observed end state rather than the trajectory.md
      5  bad-value            Status: #settled is not an allowed value (supported, contested, weak)

The refused change:
--- a/50_MOC/discourse/rl-environment-design/CLM - Grade the observed end state rather than the trajectory.md
+++ b/50_MOC/discourse/rl-environment-design/CLM - Grade the observed end state rather than the trajectory.md
@@ -2,7 +2,7 @@
 
 Tags: [[discourse-graph]] [[rl-environment-design]]
 
-Status: #supported
+Status: #settled
 
 # Grading should check the observed end state of the environment rather than the path the agent took or its self-report.
 
$ vault link /tmp/sample-vault -f brain "EVD - Reward is a deterministic outcome check 1 only if the final database equals the data" kl-penalty-in-rlhf --as supports
✗ refused: the link would add 1 violation; nothing was written
  50_MOC/discourse/rl-environment-design/EVD - Reward is a deterministic outcome check 1 only if the final database equals the data.md
     13  relation-target-type `supports::` [[kl-penalty-in-rlhf]] is a zettel; expected claim
…
```

`vault set <vault> -f F NOTE KEY VALUE` sets a metadata key of the note's type: it replaces the value on its line, or adds the
line where the type's key order puts it (in frontmatter: one single-line property; a multi-line YAML value is refused, exit 2).
The value is written as given (`#supported`, `[[note]]`); `{now}` works for date keys.

## `vault ingest <vault> FILE --as TYPE -f F [--title T] [--folder F] [--set K=V]… [--dry-run]`

Turns a `.md`, `.txt` or saved `.html` file into a note of the type (it needs an `ingest:` block). HTML goes through
[markdownify](https://github.com/matthewwithanm/python-markdownify) after scripts, styles, navigation, forms and page
header/footer are removed, and links and images are made absolute against the page's canonical URL. The document's headings
move below the body heading, a first heading repeating the title is dropped, and `key:: value` lines are escaped so an imported
text cannot add relations. Provenance goes into the keys the `ingest:` block names.

Before anything is written, the document is compared with every note of the type, and it is a **duplicate** (exit 1,
nothing written) when any note has a **title similarity** of 0.9 or more, a **content overlap** of 0.8 or more, or the
**same SHA-256**:

- title similarity: word-level, 2·matched / all words, tolerant of typos, against the note's name (also without a leading
  `2411.00640 - `), its aliases, and the first link text under the type's `title_section` (for sources: the full paper title);
- content overlap: the share of the shorter text's five-word sequences that the other also has (texts under about 45 words
  are not compared);
- the SHA-256 of the file against the note file itself and the note's recorded content hash.

See [`vaultengine/formats/similarity.py`](../vaultengine/formats/similarity.py) for the exact definitions, and
[`tests/test_vaultengine_ingest.py`](../tests/test_vaultengine_ingest.py) for the near-duplicates that should and should
not match, with their scores.

```
$ cp "/tmp/sample-vault/10_SOURCES/2411.00640 - Adding Error Bars to Evals.md" /tmp/inbox/error-bars-copy.md
$ vault ingest /tmp/sample-vault /tmp/inbox/error-bars-copy.md --as source -f brain
✗ not ingested: error-bars-copy.md duplicates 1 existing note; nothing was written
  title    'error-bars-copy' (markdown, sha256 e1d99047fa55…)

  10_SOURCES/2411.00640 - Adding Error Bars to Evals.md
      title similarity  0.500  below (threshold 0.9)   'error-bars-copy' ~ 'Adding Error Bars to Evals'
      content overlap   1.000  duplicate (threshold 0.8)   7591 of 7591 five-word sequences shared
      same file         yes    identical SHA-256 (the note file, or its recorded content hash)
```

A new document (here a saved article with navigation, a `| Site` title suffix and relative links) becomes a source like the
others:

```
$ vault ingest /tmp/sample-vault /tmp/inbox/ensembles.html --as source -f brain
Wrote 1 note (ingest):
  + 10_SOURCES/Why Reward Model Ensembles Delay Overoptimization.md
      new source: Why Reward Model Ensembles Delay Overoptimization
  check    no new violations
  index    updated 1 note

  closest  10_SOURCES/2506.18032 - Why Do Some Language Models Fake Alignment While Others Don't.md  (title 0.235, content overlap 0.000)
$ cat "/tmp/sample-vault/10_SOURCES/Why Reward Model Ensembles Delay Overoptimization.md"
Created: 07-11-2026 06:08

Tags:

Status: #baby

Original: ensembles.html

Content hash: sha256:0587d8694bb790d0cb12ee383cc2363b64f006d8a1f43da617cd03d93f10b98a

Ingested: 07-11-2026 06:08

## References

1. [Why Reward Model Ensembles Delay Overoptimization](https://fieldnotes.example.org/posts/ensembles/)

# Notes

Optimizing a policy against a learned reward model works until it does not. Past a certain amount of optimization
…
## The proxy is right on average and wrong at the edges
…
See the [earlier post](https://fieldnotes.example.org/posts/kl/).
```

(The article: `<title>Why Reward Model Ensembles Delay Overoptimization | Field Notes</title>`, `<link rel="canonical"
href="https://fieldnotes.example.org/posts/ensembles/">`, a `<nav>`, an `<article>` with an `<h1>`, two paragraphs, an `<h2>`
and a link to `/posts/kl/`, and a `<footer>`.) Ingesting the same file again is stopped with title 1.000, content 1.000 and
the same SHA-256.

## `vault coverage <vault> -f F [--limit N] [--json]`

Sources nothing links to, and notes of citing types that link to no source, saying why: their citations point at notes that
do not exist, at notes that are not sources, or there are none. After the ingest above:

```
$ vault coverage /tmp/sample-vault -f brain --limit 2
vault coverage /tmp/sample-vault  (format brain)

  sources           29   23 linked from other notes, 6 that nothing links to
  citing notes     144   33 cite a source, 111 cite none (zettel, evidence, topic, comparison)
    of those     106 cite only missing notes, 0 cite only non-sources, 5 have no citation at all

Sources nothing links to (6)
  10_SOURCES/2501.10970 - The Alternative Annotator Test for LLM-as-a-Judge.md
  10_SOURCES/Benchmark Contamination - Detection and Defeat.md
  … 4 more (--limit 0 shows all)

Notes that cite no source (111)
  20_ZETTLEKASTEN/a-bounty-list-is-a-map-of-what-cannot-be-verified.md  (zettel) cites only missing notes: line 37 [[Prime Intellect - Environments Hub]]
  20_ZETTLEKASTEN/agent-layer-inverts-who-owns-the-generation-loop.md  (zettel) cites only missing notes: line 35 [[Jiachen Liu - When LLMs Grow Hands and Feet, How to Design our Agentic RL Systems]]
  … 109 more (--limit 0 shows all)

Most cited sources (citing notes)
      5  10_SOURCES/2503.11926 - Monitoring Reasoning Models (CoT Obfuscation).md
      4  10_SOURCES/2412.04984 - Frontier Models Capable of In-context Scheming.md
      3  10_SOURCES/2312.06942 - AI Control - Improving Safety Despite Intentional Subversion.md
      3  10_SOURCES/2504.03601 - APIGen-MT Multi-Turn Agent Data Generation.md
      3  10_SOURCES/HUD - Verifier and Reward Design for RL Environments.md
```

On a fresh copy the first lines read `28 … 23 linked …, 5 that nothing links to` and `143 … 33 cite a source, 110 cite none`
(106 / 0 / 4). The new zettel and the new source account for the difference.

Search, related notes, link suggestions and unlinked mentions: [Search and connections](search.md).
