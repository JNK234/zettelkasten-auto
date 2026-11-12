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

With a usable index it reads the notes from it (notes changed since the last `vault index` are parsed from disk), the way
`vault pipeline` does, and reports exactly what parsing every file reports (`tests/test_vaultengine_indexed.py` checks that
on the sample for both formats, also after random edits made without re-indexing). Without one it parses every file.

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

### A baseline: `--baseline`, `--update-baseline`

A vault that predates a format fails it everywhere. `--update-baseline` records the violations the whole vault has now in
`<vault>/.vaultengine/baseline-<format>.json`. It is the one option of `check` that writes, and it writes only that file.
`--baseline` then fails only on violations that are not in the record. `check` itself never writes: `--baseline` with nothing
recorded is an error (exit 2) that names the command to run.

- Violations are matched by what they are about, not by line, so editing a note with known violations does not make them new.
- **Renames:** a renamed or moved note keeps its baselined violations. The baseline lists every note path and a fingerprint of the
  text of each note with violations. A note that is gone is paired with a note that did not exist when the baseline was
  recorded and has the same text, or nearly (estimated shingle similarity ≥ 0.7), so a rename plus a small edit still follows.
  Ties (two equally good candidates) are not guessed: those violations count as new.
- **Re-recording:** `--update-baseline` again after fixing things locks the smaller baseline in, or accepts what is there now.
- **Hooks:** `--baseline --changed` checks only changed notes and reports only new violations.
- **Version control:** the file has one violation per line, so it diffs well if you keep it in git.

On a fresh copy of the sample vault with [`core`](formats.md#structure-rules-typed_folders-ignored_folders-subfolders-min_links):

```
$ vault check /tmp/core-demo -f core | tail -4
✗ 144 violations in 123 notes
      83  too-few-links
      43  missing-key
      18  missing-value
$ vault check /tmp/core-demo -f core --baseline; echo $?
error: no baseline for format 'core' at /tmp/core-demo/.vaultengine/baseline-core.json; record one with `vault check /tmp/core-demo -f core --update-baseline`
2
$ vault check /tmp/core-demo -f core --update-baseline
vault check /tmp/core-demo  (format core: /…/formats/core.yaml)
  checked 180 notes

Recorded a baseline of 144 violations: /tmp/core-demo/.vaultengine/baseline-core.json
  `vault check --baseline` now fails only on violations that are not in it
$ mv /tmp/core-demo/20_ZETTLEKASTEN/reward.md /tmp/core-demo/20_ZETTLEKASTEN/reward-signal.md
$ vault check /tmp/core-demo -f core --baseline
vault check /tmp/core-demo  (format core: /…/formats/core.yaml)
  checked 180 notes

✓ no new violations in 180 notes checked (144 known violations in the baseline not shown)
  baseline of 2026-11-10T09:12:12+00:00: /tmp/core-demo/.vaultengine/baseline-core.json
  followed a renamed note (same text): 20_ZETTLEKASTEN/reward.md → 20_ZETTLEKASTEN/reward-signal.md
```

Then a zettel that cites no source and a page that links to one zettel:

```
$ vault check /tmp/core-demo -f core --baseline; echo $?
vault check /tmp/core-demo  (format core: /…/formats/core.yaml)
  checked 182 notes

20_ZETTLEKASTEN/graders-should-fail-the-empty-policy.md  (zettel)
      1  too-few-links        links to 0 source notes; a zettel needs at least 1

50_MOC/topics/grader-sanity-checks.md  (page)
      1  too-few-links        links to 1 zettel note; a page needs at least 3

✗ 2 new violations in 2 notes (144 known violations in the baseline not shown)
       2  too-few-links
  baseline of 2026-11-10T09:12:12+00:00: /tmp/core-demo/.vaultengine/baseline-core.json
1
```

When baselined violations have been fixed, the summary says so (`3 baselined violations fixed since; --update-baseline records
the smaller baseline`). With `--json` the report has a `baseline` block: `path`, `recorded_at`, `recorded_now`, `violations`
(in the baseline), `new`, `known`, `fixed`, `fixed_items` and `renames` (`from`, `to`, `how`, `similarity`). A baseline that cannot
be read, was recorded by an older engine or for another format is an error (exit 2); record it again with `--update-baseline`.
The JSON-RPC `vault.check` does not take a baseline.

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

All writes (these three, `fix`, `ingest`, `page`, `idea`, `move` and `merge`) go through one transaction:

- **The gate.** The write is refused (exit 1, nothing written) if it would add a violation anywhere. Violations already
  there do not block it ("already there, not blocking"), so a note can be repaired one problem at a time. A new note has no
  "before": it must be clean. A moved note's violations follow it to its new path.
- **Only the notes a write can affect are checked, before and after:**
  - the notes it writes, moves or deletes;
  - the notes whose links name any of them (old and new names, aliases), since those links may now resolve differently;
  - files sharing those names;
  - the notes their relation fields point at (mirrors).

  No other note's violations can change, so the verdict equals a check of the whole vault. The tests check that on random
  writes and on writes whose violations land in other notes.
- **Reading through the index.** With an index, the vault is read through it: only notes changed since the last `vault index`
  are parsed, and the planned changes are laid over it in memory. A write costs what it touches.
- **Only the lines needed change.** Every other byte of every file stays as it was (line endings, BOM, trailing spaces).
- **Atomic, as a group.** New contents go to temporary files (fsynced), originals are hard-linked to backups, a journal in
  `.vaultengine/journal/` lists them, then the files are renamed into place, files to delete are removed (their backups kept
  until the end), and everything is read back. A failure before the journal is
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

`vault new <vault> -f F --type T --title "…" [--folder F] [--name N] [--set K=V]… [--rel RELATION NOTE]… [--section HEADING TEXT]… [--body TEXT]`
creates a note from the type's `new:` block (its `defaults` fill keys you do not set: core fills `Created` with the current time).
`--body` is text right after the title, for types without required headings: under core a zettel is refused unless it links a
source, so `--body "From [[RLHF]]."` is how a new one carries it. `--rel` adds relations to existing notes and writes their mirrors there, so a claim
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

With `formats/core.yaml` (`ingest: {url: Source}`), a source has no body heading: the document goes right after its
title, its headings start at level 2, and its URL (frontmatter `url`/`source`/`link`, an HTML page's canonical URL) fills
`Source`. Nothing else is recorded, so a source keeps exactly its three fields. Without a URL, give
`--set "Source=<url or citation>"`, and give `--set "Tags=…"` always; an empty one is refused. The whole flow is in
[The reading workflow](workflow.md).

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

## `vault pipeline <vault> -f F [--limit N] [--json]`

Where each source is, worked out from the links on every run; nothing is stored, so there is nothing to keep up to date. Per
the format's [`pipeline:` block](formats.md#pipeline-for-vault-pipeline-and-vault-note), in core:

- **Stages.** A source is **ingested**. It is **extracted** once a zettel links to it, and **compiled** once a page links to one of those zettels.
- **What "links to" means.** The same as the `min_links` rule: links in `Tags:` do not count, and neither do URLs or links to
  notes that are not in the vault. A page linking a source directly does not compile it. An idea does not compile anything.
  Notes in ignored folders play no part.
- **The gaps.** Sources nothing has been extracted from, and zettels no page includes.
- **Zettels on no source note.** These are counted with the reason from [`vault coverage`](#vault-coverage-vault--f-f---limit-n---json):
  they cite source notes that are not in the vault, notes that are not sources, only URLs, or nothing.

Exit 0; reads the files, no index needed. `--limit` caps each list (0 = all). On a fresh copy of the sample vault:

```
$ vault pipeline /tmp/core-demo -f core --limit 3
vault pipeline /tmp/core-demo  (format core: /…/formats/core.yaml)

  source notes             28
    compiled                3   a page links to a zettel that links to it
    extracted              12   a zettel links to it, no page uses those yet
    ingested               13   no zettel links to it
  zettel notes            109
    in a page               5
    resting on a source    28
    on no source note      81   78 cite source notes that are not in the vault, 3 cite nothing
  page notes                3   2 link to a zettel
  ignored                  40   notes in ignored folders, which play no part

Source notes by stage  (zettels linking to it, pages using those)
  compiled     5 z   1 p   10_SOURCES/2503.11926 - Monitoring Reasoning Models (CoT Obfuscation).md
  compiled     2 z   1 p   10_SOURCES/2412.14093 - Alignment Faking in Large Language Models.md
  compiled     1 z   1 p   10_SOURCES/2506.18032 - Why Do Some Language Models Fake Alignment While Others Don't.md
  … 25 more (--limit 0 shows all)

Source notes nothing has been extracted from (13)
  10_SOURCES/2501.10970 - The Alternative Annotator Test for LLM-as-a-Judge.md
  10_SOURCES/2503.20783 - Understanding R1-Zero-Like Training.md
  10_SOURCES/2504.03601 - APIGen-MT Multi-Turn Agent Data Generation.md
  … 10 more (--limit 0 shows all)

Zettel notes no page includes (104)
  20_ZETTLEKASTEN/a-bounty-list-is-a-map-of-what-cannot-be-verified.md
  20_ZETTLEKASTEN/agent-layer-inverts-who-owns-the-generation-loop.md
  20_ZETTLEKASTEN/agent-test-time-scaling-limited-by-verification-gap.md
  … 101 more (--limit 0 shows all)
```

The sample looks like a slice of a larger vault, and it shows. The 78 zettels cite, under References, 60 different notes that
are not in it. The two topic pages link about 28 notes each, of which 2 and 3 are zettels in the sample. All three compiled
sources come from one topic page.

`--json`: `stages`, `sources` (each with `stage`, `zettels`, and `pages` with the zettels `via` which each uses it), `extract`
(totals, and `unsourced` with `by_reason` and each note's citations), `compile`, `gaps` (`not_extracted`, `not_compiled`), `ignored`.

## `vault note <vault> NOTE -f F [--json]`

What the engine knows about one note, instead of fields kept by hand. The note is given by path or name.

- **Stage.** A source's stage as above. A zettel is extracted, or compiled once a page includes it; a zettel that rests on no
  source note has no stage.
- **Modified.** The file's own modification time.
- **Rests on.** Sources: linked directly, or through the zettels it links to (`via`). Zettels: the ones it links to.
- **Used by.** The zettels that link to it; pages that link to it (for a source, the pages that use its zettels, `via` which).
- **Every note linking to it and every note it links to,** with their types (ignored notes marked), and its links that match no note.
- **Its violations.**

```
$ vault note /tmp/core-demo monitorability-tax -f core
20_ZETTLEKASTEN/monitorability-tax.md  (zettel)
  stage      extracted
  modified   2026-11-09T20:57:22+00:00  (file)

Rests on
  sources    2 sources
      10_SOURCES/2503.11926 - Monitoring Reasoning Models (CoT Obfuscation).md  (direct; also via cot-monitor-beats-action-monitor, cot-monitorability-vs-faithfulness, obfuscated-reward-hacking)
      10_SOURCES/2503.14499 - Measuring AI Ability to Complete Long Tasks (METR Time Horizon).md  (via reliability-tax-80-vs-50-percent-time-horizon)
  zettels    5 zettels
      …

Used by
  zettels    3 zettels
      …
  pages      0 pages

Links
  from       3 notes: 3 zettel
      …
  to         6 notes: 5 zettel, 1 source
      …
  dangling   1 that match no note or several: [[2507.11473 - Chain of Thought Monitorability]]

Violations (1)
      1  missing-key          missing metadata 'Created'
```

`--json` has the same facts: `stage`, `modified`, `rests_on` (`sources`, `extract`), `used_by` (`extract`, `compile`), each item
with `path`, `direct` and `via`; `linked_from`, `links_to` (`path`, `type`, `ignored`), `links_to_nothing`, `violations`.

## `vault page new|refresh|check|exclude`

Pages are the compiled reading notes under `50_MOC` (the format's pipeline `compile` type). A page the engine manages has two
generated blocks between markers, `<!-- vault:zettels -->` and `<!-- vault:sources -->`. Everything else in it is yours.
`vault page new` and `vault page refresh` print what they would write. Only `--write` writes, through the
[safe write path](#the-safe-write-path-vault-new-vault-link-vault-set): the page, the index note's pages part and a log line
go as one atomic group, refused if they would add a violation.

**The blocks.**

- The zettels on a page are the zettels it links to now, anywhere in it: the block's own lines and any zettel you link in your text.
  To drop one, delete its line (and your links to it). To add one, link it and refresh.
- Zettels are grouped under the tag most of the page's zettels share. Each shows its first line of prose as plain text, so
  the blocks link nothing but what they list.
- The sources block lists every source note those zettels link to, with the zettels citing it, and names the zettels that
  rest on no source note.

**Engine state.** In `.vaultengine/pages.json`, never in the note. For each page it holds:

- when the page was last written or refreshed, and the SHA-256 of each of its zettels then;
- the zettels excluded from it;
- a fingerprint of its text.

A renamed or moved page keeps its record, paired the way the [baseline](#a-baseline---baseline---update-baseline) pairs
renamed notes: same or similar text (≥ 0.7), ties not guessed. The next refresh writes the record under the new path.

### `vault page new <vault> "<title>" --from QUERY_OR_NOTES … -f F [--tags T] [--folder F] [--limit N] [--write] [--json]`

`--from '[[a]] [[b]]'` (only links) takes those zettels. Any other `--from` is a [search query](search.md); its best-ranked
zettels are taken, at most `--limit` (10) per query, and it needs `vault index`. `--from` repeats. Tags default to the tags
at least half of the zettels share. A page with fewer zettels than the format's `min_links` is refused. On a copy of the
sample vault:

```
$ vault page new /tmp/page-demo "Monitoring Reasoning" -f core --write \
    --from "[[cot-monitor-beats-action-monitor]] [[cot-monitorability-vs-faithfulness]] [[obfuscated-reward-hacking]] [[monitorability-tax]] [[reward-hacking-as-specification-gaming]]"
Wrote 3 notes (page new):
  + 50_MOC/monitoring-reasoning.md
      new page: Monitoring Reasoning (5 zettels)
  + 50_MOC/index.md
      new index: 4 pages
  + 50_MOC/log.md
      new log
  state    .vaultengine/pages.json: 50_MOC/monitoring-reasoning.md: 5 zettels recorded
  check    no new violations
$ cat /tmp/page-demo/50_MOC/monitoring-reasoning.md
Created: 10-11-2026 09:45
Tags: #ai-safety

# Monitoring Reasoning

<!-- vault:zettels -->
## Zettels

### ai-safety
- [[cot-monitor-beats-action-monitor]]: You can catch a frontier reasoning model cheating by having *another, weaker LLM read its chain-of-thought*. A deliberately cheap monitor — zero-shot prompted …
- [[cot-monitorability-vs-faithfulness]]: A reasoning model emits a long chain-of-thought (CoT) before its answer, written in natural language inherited from pretraining. The strict notion of …
- [[monitorability-tax]]: The monitorability tax is the cost you pay to keep a reasoning model's chain-of-thought legible as a safety instrument: shipping "slightly less performant …
- [[obfuscated-reward-hacking]]: If a CoT monitor is good at *detecting* cheating, the obvious move is to *prevent* cheating by baking the monitor into the training reward …
- [[reward-hacking-as-specification-gaming]]: Reward hacking is when an RL policy earns high reward by exploiting a flaw or misspecification in the objective rather than by doing the task you actually …
<!-- /vault:zettels -->

<!-- vault:sources -->
## Sources

- [[2503.11926 - Monitoring Reasoning Models (CoT Obfuscation)]] (cot-monitor-beats-action-monitor, cot-monitorability-vs-faithfulness, monitorability-tax, obfuscated-reward-hacking, reward-hacking-as-specification-gaming)
<!-- /vault:sources -->

$ grep monitoring /tmp/page-demo/50_MOC/index.md /tmp/page-demo/50_MOC/log.md
/tmp/page-demo/50_MOC/index.md:- [[monitoring-reasoning]]: Monitoring Reasoning (5 zettels, 1 source)
/tmp/page-demo/50_MOC/log.md:- 2026-11-10 09:45 page new [[monitoring-reasoning]] (5 zettels, 1 source)
```

The index note (`pages.index`, a type of its own so it is not a page) gets one line per page, rewritten on every page write.
If it has no markers yet, the part is appended. The log note (`pages.log`, `50_MOC/log.md` in core, also a type of its own)
gets a line appended.

### `vault page refresh <vault> PAGE -f F [--write] [--json]`

Rebuilds both blocks from the zettels the page links now and records the zettels in the engine state. Only the lines between
the markers change. Every other byte of the page stays as it was, line endings included. If only the state is out of date (a
zettel changed below its first line), only the state file is written. If a block's markers are missing, doubled or out of
order, it stops (exit 2) and says so.

### `vault page check <vault> [PAGE] -f F [--limit N] [--json]`

What needs attention on one page, or on every page made by `vault page new` (pages without blocks are listed, not checked):

- **Blocks out of date**, with the lines that would change.
- **Zettels changed since the last refresh**, compared with the engine state: `changed`, `deleted`, or `not recorded`.
- **Behind**: zettels not on the page, and not excluded from it, that have at least two things in common with it, most first.
  Each of the page's tags a zettel carries counts (`#x` and `[[x]]` are the same tag), and so does each note the page links to
  (one of its zettels or sources) that the zettel links to as well.
- **Its index line**, missing or out of date.
- **Its violations.**

A page with none of these is `✓ ready to write`. Exit 0 when every page checked is ready, 1 otherwise. The same page after
editing the first line of `monitorability-tax`:

```
$ vault page check /tmp/page-demo monitoring-reasoning -f core --limit 3
vault page check /tmp/page-demo  (format core: /…/formats/core.yaml)

50_MOC/monitoring-reasoning.md  (Monitoring Reasoning; refreshed 2026-11-10T09:45:14)
  ✗ zettels block out of date
      - - [[monitorability-tax]]: The monitorability tax is the cost you pay to keep a reasoning model's chain-of-thought legible as a safety instrument: shipping "slightly less performant …
      + - [[monitorability-tax]]: The monitorability tax is the capability you give up to keep a reasoning model's chain-of-thought legible as a safety instrument: shipping "slightly less …
  ✗ 1 zettel changed since the last refresh
      changed       20_ZETTLEKASTEN/monitorability-tax.md
  ✗ behind: 2 zettels with at least two of its tags or links in common, not on it
      20_ZETTLEKASTEN/model-awareness-of-intent-violation-while-cheating.md   tags ai-safety · links obfuscated-reward-hacking, reward-hacking-as-specification-gaming
      20_ZETTLEKASTEN/reward-hacking-as-eval-false-positive.md   tags ai-safety · links obfuscated-reward-hacking, reward-hacking-as-specification-gaming
  → vault page refresh /tmp/page-demo '50_MOC/monitoring-reasoning.md' -f core --write
  → link the behind zettels that belong in the text and refresh; `vault page exclude` the ones that do not

✗ 0 of 1 managed page ready to write
```

With one thing in common counting, this page was behind on 26 zettels. Mostly these came through one broad tag (`ai-safety`)
or one hub zettel (`reward-hacking-as-specification-gaming`). With two in common it is behind on the two that belong.

### `vault page exclude <vault> PAGE ZETTEL… -f F [--undo] [--write] [--json]`

Records in the engine state that these zettels are not for the page, so it is no longer behind on them; `--undo` brings them
back. The page itself is not touched. It is refused for a zettel that is on the page (delete its line instead), for a note that
is not a zettel, and for a page with no record yet. Continuing the example: exclude one behind zettel, link the other in
your own text, refresh. The page links one more zettel now, and that makes it behind on two others. Then move the page:

```
$ vault page exclude /tmp/page-demo monitoring-reasoning model-awareness-of-intent-violation-while-cheating -f core --write
Recorded in the engine state (page exclude); no note changes:
  state    .vaultengine/pages.json: 50_MOC/monitoring-reasoning.md: 5 zettels recorded, 1 excluded
$ echo 'The eval side of this: [[reward-hacking-as-eval-false-positive]].' >> /tmp/page-demo/50_MOC/monitoring-reasoning.md
$ vault page refresh /tmp/page-demo monitoring-reasoning -f core --write | head -3
Wrote 3 notes (page refresh):
  ~ 50_MOC/monitoring-reasoning.md
      rebuilt the zettels and sources blocks (6 zettels)
$ mv /tmp/page-demo/50_MOC/monitoring-reasoning.md /tmp/page-demo/50_MOC/topics/cot-monitoring.md
$ vault page check /tmp/page-demo cot-monitoring -f core
…
50_MOC/topics/cot-monitoring.md  (Monitoring Reasoning; refreshed 2026-11-10T12:28:53)
  = renamed from 50_MOC/monitoring-reasoning.md: its record followed it
  = 1 zettel excluded from it (`vault page exclude … --undo` to bring one back)
  ✗ behind: 2 zettels with at least two of its tags or links in common, not on it
      20_ZETTLEKASTEN/environment-and-eval-are-the-same-object.md   links reward-hacking-as-eval-false-positive, reward-hacking-as-specification-gaming
      20_ZETTLEKASTEN/ostensible-positives-from-low-entropy-answer-spaces.md   links reward-hacking-as-eval-false-positive, reward-hacking-as-specification-gaming
  ✗ index line missing
  → vault page refresh /tmp/page-demo '50_MOC/topics/cot-monitoring.md' -f core --write
  → link the behind zettels that belong in the text and refresh; `vault page exclude` the ones that do not
```

### How `vault pipeline` and `vault page check` scale

`vault pipeline`, `vault page check` and `vault note` read the SQLite index (`vault index` builds it). They parse only the
notes changed since the last `vault index`, in memory, and never write. Opening compares the vault with the index the way
`vault index` does: a `stat` of every file, then a hash of those whose size or mtime changed. Notes changed since the last
index are parsed from disk, and deleted ones are dropped. Stored link resolutions are reused except the ones an edit can
affect (a link whose target name is the name or alias of an added, deleted or changed file), which are resolved again.
The answers are the same as from parsing every file. The tests check that on the sample vault for both formats, and again
after random edits made without re-indexing. Without a usable index, these commands read every file, as before, and say so.

### How long each command takes

Measured with `scripts/bench_pages.py time-every VAULT --build` on vaults it generates, shaped like the sample: 20% sources
of ~26 KB, 75% zettels of ~3 KB, 5% pages of 8–25 zettels, 150 tags. The commands run in the order of
[the reading workflow](workflow.md), each a fresh process, as from the shell. Reads and dry runs are the best of 3
runs; writes ran once, after their dry run.

On this shared machine the same read varied by up to 2× between runs: `vault next` at 50,000 notes took 17–58 s over
the day. "Before" is the earlier version, where there is one: parsing every file, or findings scoring pair by pair.

| Command | 5,000 notes (39 MB) | 50,000 notes (392 MB) | before: 5,000 / 50,000 |
|---|---|---|---|
| `vault index`, full build (once) | 38.5 s | 522 s | |
| `vault index`, nothing changed | 0.4 s | 1.8 s | |
| `vault index`, 20 notes changed | 0.9 s | 4.3 s | |
| `vault check` (whole vault) | 2.5 s | 31.4 s | 17 s / 187 s (every file parsed) |
| `vault check --update-baseline` | 3.0 s | 31.7 s | |
| `vault check --baseline` | 2.5 s | 26.2 s | |
| `vault check --baseline`, 20 notes changed since the index | 2.9 s | 26.2 s | |
| `vault pipeline` | 0.8 s | 7.7 s | 9.4–21 s / 136 s |
| `vault note ZETTEL` (with its trust, a percentile over the vault) | 1.0 s | 7.9 s | |
| `vault page check PAGE` | 0.5 s | 2.1 s | 9.7–21 s / 106 s |
| `vault page check` (every page: 250 / 2,500) | 3.5 s | 38.9 s | 10–17 s / 230–359 s |
| `vault page new` dry run / `--write` | 1.3 s / 1.8 s | 9.6 s / 13.1 s | |
| `vault page refresh` dry run | 1.3 s | 7.6 s | a write before the local gate: 16 s / 245 s |
| `vault page exclude` dry run / `--write` | 0.5 s / 0.6 s | 1.9 s / 2.0 s | |
| `vault findings` | 3.6 s | 28.7 s | 11.9–17 s / 382–456 s |
| `vault idea new` (a link finding) dry run / `--write` | 3.8 s / 4.0 s | 28.8 s / 41.1 s | |
| `vault idea set` dry run / `--write` | 0.5 s / 0.9 s | 2.9 s / 6.1 s | |
| `vault move` dry run / `--write` (the zettel most notes link to) | 0.8 s / 1.7 s | 3.9 s / 7.2 s | |
| `vault merge` dry run / `--write` | 1.0 s / 1.9 s | 5.0 s / 8.6 s | |
| `vault next` | 2.7 s | 21.7 s | |

Where the time goes at 50,000 notes:

- **Every read** starts with about 1.5 s to `stat` every file and compare with the index (the floor without a file
  watcher), and about 1 s to load the list of files and build the name lookup.
- **One page check** (2.1 s) then takes about 0.2 s for the page itself. The page and its zettels are read from the
  index, and the zettels sharing its tags or links come from two indexed queries.
- **`vault check`, the pipeline, `vault next`, every page check** are about the whole vault: every note's links (about
  700,000 rows) are read in one pass, and the notes in a few batches.
- **`vault findings`** (28.7 s): about half goes to reading the index's links and term profiles. The other half scores
  every pair that shares a neighbour or a citer: about 5.2 million contributions from neighbours with at most 50 links. See
  [how link findings are computed](#vault-findings-vault--f-f---method-m---id-id---limit-n---json). `vault idea new`
  from a link finding computes the findings once.
- **Writes** read through the index too, and the gate checks only the notes a write can affect (see
  [the safe write path](#the-safe-write-path-vault-new-vault-link-vault-set)).

## `vault move <vault> NOTE TO -f F [--write] [--json]` and `vault merge <vault> A B -f F [--write] [--json]`

`vault move` gives a note a new place or name. `TO` is one of:

- a folder (ending in `/`, or an existing folder);
- a new name (same folder);
- a new vault path.

`vault merge A B` merges B into A. A survives and keeps its own text. B's text is appended between `<!-- vault:merged -->`
markers under `## Merged from B`, for you to tidy: its frontmatter goes in a YAML code block, and engine block markers are
removed. Then B is deleted. B may also be a name that links use but no note has (the `existing-name` and `same-idea`
findings): then only those links are rewritten, to A (or to A's name, when neither exists).

Both show the change (a diff; a moved note as `rename old -> new`) and write only with `--write`, through the
[safe write path](#the-safe-write-path-vault-new-vault-link-vault-set): journaled, atomic, refused if it would add a violation.

**Links**, rewritten with the resolver, Obsidian's rules:

- **What is kept.** Every link, embed, heading link and block link that reached the note still reaches it, at its new place.
  A heading or block that was found is still found. This holds for wikilinks and markdown links, in the body and in YAML
  frontmatter.
- **What is rewritten.** Only links that would no longer resolve, and each keeps its style:
  - a bare name stays a bare name when the name reaches only that note, else it becomes the vault path;
  - a path stays a path, a relative one relative;
  - markdown links keep `%20` or `<…>`;
  - display text, subpaths and the embed `!` are kept.
- **Links to other notes** that a new name would make ambiguous get their path, so they keep their target.
- **Links that matched no note** and match the new name now reach it, and the output says so.
- **Aliases.** A link that reached B through one of its aliases gets the alias as display text (`[[A|alias]]`).
- **Refusal.** If some link cannot be spelled to reach its note, nothing is written.

**Self-links a merge would create are dropped.** After `vault merge A B`, a link in A to B, or a link in B's text to A,
would point at A itself. Each one is dropped, and the output lists it (`self  N links would point at the note it is in:
dropped`; `--json`: `self_links` with `path`, `line`, `link`, `action`, `result`):

| The link | becomes | `action` |
|---|---|---|
| alone on its line, after a list, task or quote marker (`- [[B]]`, `- [ ] [[B\|x]]`, `> [[B]]`) | the line is removed | `line removed` |
| with a heading or block (`[[B#Part\|the part]]`, `[t](B.md#part)`, `![[B#^id]]`) | a link inside the note: `[[#Part\|the part]]`, `[t](#part)`, `![[#^id]]` | `same-note link` |
| in a field value (`Tags: [[ml]] [[B]]`, `related:: [[z]], [[B]]`) | removed with its separator; an inline field left empty goes as a line | `removed` (`line removed`) |
| an embed (`![[B]]`) | removed, with its separator (what it showed is in A now) | `removed` |
| anywhere else (`see [[B]] here`, `[[B\|home]]`, `[md](B.md)`) | its visible text: `see B here`, `home`, `md` | `unlinked` |
| in A's frontmatter | a list item holding only the link goes; elsewhere the link becomes its visible text (`up: "B"`) | `line removed` / `unlinked` |

Self-links a note already had are not dropped: A's own stay, B's `[[B#Heading]]` is re-spelled `[[A#Heading]]`, and
`[[#Heading]]` links stay as they are.
The check of the write then runs on the result: the gate refuses the merge if dropping a link would add a violation.

**What the engine remembers follows the note:**

- **Baselines.** The note's violations, fingerprint and path.
- **Page records.** A page's own record; the zettels in others' records; exclusions. A zettel whose only change was the
  engine re-spelling a link is not "changed since the last refresh"; a merge survivor is, because it holds new text.
- **Finding blocks of ideas.** Their links are rewritten, and their id is recomputed from the moved notes.

The merge of the duplicate `vault findings` spotted on the sample (`name-5553fbc478`), on a fresh copy:

```
$ vault merge /tmp/round6 on-policy-versus-off-policy-learning on-policy-vs-off-policy-learning -f core
--- a/10_SOURCES/2505.10978 - GiGPO Group-in-Group Policy Optimization for LLM Agent Training.md
+++ b/10_SOURCES/2505.10978 - GiGPO Group-in-Group Policy Optimization for LLM Agent Training.md
@@ -330,7 +330,7 @@
…
-- Fully [[on-policy-vs-off-policy-learning]] on-policy: $\pi_{\theta_\text{old}}$ generates, $\rho_\theta$ corrects for the small drift within an update.
+- Fully [[on-policy-versus-off-policy-learning]] on-policy: $\pi_{\theta_\text{old}}$ generates, $\rho_\theta$ corrects for the small drift within an update.
…  (four more notes, one line each)

Would merge: links to [[on-policy-vs-off-policy-learning]] (no note) -> 20_ZETTLEKASTEN/on-policy-versus-off-policy-learning.md
  links    5 links rewritten in 5 notes
           now points at the note it is in (tidy it): 20_ZETTLEKASTEN/on-policy-versus-off-policy-learning.md:28  [[on-policy-versus-off-policy-learning]]
  check    no new violations
           already there, not blocking: 1 missing-key, 3 too-few-links
Not written: add --write to write it.
$ vault merge … --write | tail -2
           already there, not blocking: 1 missing-key, 3 too-few-links
  index    updated 5 notes
$ vault health /tmp/round6          # before → after
  unresolved links                      1315   across 612 missing targets   →   1310   across 611 missing targets
  ambiguous links                          0                                 →      0
  missing heading/block targets            0                                 →      0
```

The existing note's own "See Also" linked the other name, so it now links to itself, which the output points out. The
finding is gone from `vault findings`.

## `vault next <vault> -f F [--limit N] [--json]`

One queue of what needs doing, worked out from the vault on every run. Each item has a kind, a reason, and the exact command
that resolves it. `<…>` marks what only you can supply: what a new zettel says, where a source lives, and a source note when
none of the linking notes cites one.

| Kind | What | Command |
|---|---|---|
| `refresh` | a page whose zettels changed, were linked or dropped since its last refresh, or whose index line is out of date | `vault page refresh … --write` |
| `behind` | a zettel a page is behind on (two or more tags or links in common) | `vault page exclude … --write`, or link it in the page's text and refresh |
| `merge` | two names for one idea (findings `same-idea`, `existing-name`) | `vault merge A B … --write` |
| `write` | a missing note many notes link to (finding `missing-note`) | `vault new … --name NAME …` (a zettel citing the source its linking notes cite most; a source when most links to it are citations) |
| `extract` | a source no zettel links to | `vault new … --type zettel --body "From [[source]]."` |
| `page` | zettels in no page that share a tag, as many as a page needs | `vault page new … --from "[[a]] [[b]] …" --write` |
| `idea` | an open idea no note links to | `vault idea set … dropped --write` |

**Trust.** Items are ranked by the trust of what they are about: how much the vault relies on it, worked out from use and
never stored.

- **For a note:** `log2(1 + notes linking to it) + pages citing it + ideas citing it`. For a source, pages citing its zettels
  count too. The index and log notes the engine keeps are not counted.
- **For a missing note:** the same, from the notes linking to the name.
- **As shown:** a percentile, the share of the vault's notes trusted less (`vault note` shows it, `--json` under `trust`).
- **Ties:** by kind, in the order of the table, then by name. So the queue is the same on every run.

**Cutoffs are relative.** Findings use their percentiles. Sources no zettel links to, and clusters of zettels in no page, are
listed when their trust is at least the median of their kind; the rest are counted. On a copy of the sample after the merge
above (commands shortened):

```
$ vault next /tmp/round6 -f core
vault next /tmp/round6  (format core; 97 items, most trusted first)

 1. write    write [[transcript-spotting-heuristics-for-reward-hacking]]   (trust 97)
    why  6 notes link to it and it does not exist
    run  vault new /tmp/round6 -f core --type zettel --name transcript-spotting-heuristics-for-reward-hacking --title … --set 'Tags=[[reinforcement-learning]] [[post-training]] [[reward-modeling]]' --body 'From [[HUD - Verifier and Reward Design for RL Environments]].'
 2. write    write [[reward-model-overoptimization-hump]]   (trust 96)
    why  12 notes link to it and it does not exist
 3. write    write [[rlvr-reinforcement-learning-with-verifiable-rewards]]   (trust 96)
 4. write    write [[verifiable-rule-based-rewards-prevent-reward-hacking]]   (trust 95)
 5. write    write [[grpo-group-relative-policy-optimization]]   (trust 93)
 6. write    write [[llm-as-judge-evaluation-pattern]]   (trust 93)
 7. write    write [[llm-as-judge-biases-catalog]]   (trust 92)
 8. merge    merge [[AI Control - Improving Safety Despite Intentional Subversion]] into [[2312.06942 - AI Control - Improving Safety Despite Intentional Subversion]]   (trust 90)
    why  [[AI Control - Improving Safety Despite Intentional Subversion]] does not exist; 3 notes link to it: …
    run  vault merge /tmp/round6 '10_SOURCES/2312.06942 - AI Control - Improving Safety Despite Intentional Subversion.md' 'AI Control - Improving Safety Despite Intentional Subversion' -f core --write
 9. write    write [[construct-and-operationalization-gap]]   (trust 87)
10. write    write [[cross-family-judge-panel-with-unanimity]]   (trust 87)
  … 87 more (--limit 0 shows all)

By kind (listed / all):
  refresh     0 / 0    pages out of date with their zettels
  behind      0 / 0    zettels a page is behind on
  merge       8 / 8    two names for one idea
  write      71 / 71   missing notes many notes link to
  extract    13 / 13   sources no zettel links to
  page        5 / 9    zettels in no page that share a tag (104 zettels in no page); below the median trust not listed
  idea        0 / 0    open ideas nothing builds on
```

Item 1 outranks notes 12 notes link to because a topic page also cites it: being cited by a page counts as use.

## `vault findings <vault> -f F [--method M]… [--id ID] [--limit N] [--json]`

What the engine discovered, worked out from the index on every run (it reads; it never writes, and it needs `vault index`).
Four methods, from `vault suggest` and the link suggestions of `vault suggest --links`:

| Method | A finding is | Listed when |
|---|---|---|
| `missing-note` | a note that does not exist that notes link to | as many notes link to it as the 90th percentile of all missing notes (never fewer than 2) |
| `same-idea` | two missing notes whose names say the same (one is the other plus at most two words) | both are linked from somewhere |
| `existing-name` | a missing note whose name is probably an existing note's | the 75th percentile of missing notes' linking notes (at least 2) |
| `link` | two notes that are not linked but probably belong together | shared neighbours or citers (never text alone), and a score at the 99th percentile of such pairs (at least `engine.findings.link_floor`, 0.2); at most `engine.findings.link_limit` (500) of them, best first |

The thresholds are nearest-rank percentiles of this vault, with a floor, so they move with the vault and are the same on
every run of it. The header line says what they came to (on the sample: 3 linking notes, 2, and a score of 0.30), and `--json`
has them under `thresholds`. When the cap cuts the list the header says so (`links: score 0.20+ (the best 500)`): on a
generated 5,000-note vault the floor alone still lets 1,315 links through, and each listed link costs about 7 ms to explain.

**The id** is derived from what the finding is: its method and the notes it is about (by lookup key; an existing note by its
path, a missing one by its name), `<prefix>-<10 hex of SHA-256>`. Its evidence and its rank are not part of it. So the same
finding has the same id on every run, whatever else changes in the vault, and even when new evidence for it turns up.

**Whether a finding is listed** depends on its thresholds, not on its rank among the others. The thresholds and link scores
are relative, though: a percentile moves as the vault grows, and a neighbour shared by many notes weighs less. So an edit can
move a borderline finding across its threshold. Its id does not change if it comes back.

**Under a format** these notes take no part: notes in ignored folders, ideas (an idea links its finding's notes and must not
become a finding itself), and the index and log notes. Links in metadata values do not count either, as in `min_links` and the
pipeline. A tag nobody wrote a note for is a tag, not a missing note: on the sample, 21 of the 93 "missing notes" were only
tags. On a copy of the sample (`--limit 1`, evidence shortened):

```
$ vault findings /tmp/round5 -f core --limit 1
vault findings /tmp/round5  (94 findings; from the index of 2026-11-10T12:28:21+00:00)
  thresholds (percentiles of this vault, with floors): missing notes: 3+ linking notes, names: 2+, links: score 0.30+

missing-note: a note many notes link to that does not exist (72)
  missing-32eacad6a1  [[reward-model-overoptimization-hump]] does not exist; 12 notes link to it
      linked from 12 notes: 2507.17746 - Rubrics as Rewards (RaR), covariance-targeted-entropy-intervention, … +7
      mostly in 20_ZETTLEKASTEN/ (11), 10_SOURCES/ (1)
  … 71 more (--limit 0 shows all)

same-idea: two names for the same idea (6)
  same-41d5352c29  [[Patronus AI - Generative Simulators]] and [[Patronus AI - Introducing Generative Simulators]] may be one idea under two names
      neither note exists; [[Patronus AI - Generative Simulators]] is linked from 3 notes, [[Patronus AI - Introducing Generative Simulators]] from 2
      [[Patronus AI - Introducing Generative Simulators]] is [[Patronus AI - Generative Simulators]] plus: introducing
  … 5 more (--limit 0 shows all)

existing-name: a link name that is probably an existing note (3)
  name-5553fbc478  [[on-policy-vs-off-policy-learning]] is probably [[on-policy-versus-off-policy-learning]]
      [[on-policy-vs-off-policy-learning]] does not exist; 5 notes link to it: 2505.10978 - GiGPO …, on-policy-versus-off-policy-learning, … +1
      its words are all in the name of 20_ZETTLEKASTEN/on-policy-versus-off-policy-learning.md
  … 2 more (--limit 0 shows all)

link: two notes that probably belong linked (13)
  link-6b21be7696  [[in-context-scheming-capability]] and [[scheming-inability-safety-case-closed]] are not linked but probably belong together
      both link to: 2412.04984 - Frontier Models Capable of In-context Scheming, ai-control-vs-alignment, control-evaluations-red-team-blue-team
      cited together by: scheming-as-threat-model
      similar text: scheming, context, reasoning
      shared tags: #frontier-models-in-ai, #ai-safety
      score 0.38 (shared neighbours 0.27, cited together 0.30, tags 0.44, text 0.58)
  … 12 more (--limit 0 shows all)
```

`--id ID` shows one finding. `--json` gives each with `id`, `method`, `title`, `notes` (`name`, `path`, `linking_notes`),
`evidence`, `score` and `data`.

**How link findings are computed.** The score of a pair is the pair-by-pair definition of `vault suggest --links`. The
threshold is a percentile over every pair with a shared neighbour or citer, so every such pair is scored. But no pair's
neighbour sets are intersected:

- **Shared neighbours and citers.** Note by note, each shared neighbour adds its weight to the pairs it connects, and
  each sum is rounded once (`math.fsum`). A neighbour with more than 50 links only adds to pairs found otherwise.
- **Text.** Each note's 5 closest notes by embedding (the centered cosine, as in `vault suggest --links`), and the pairs
  sharing a tag on at most 50 notes; their evidence is worked out pair by pair. Neither alone makes a pair structural.
- **The percentile** is read from the top 1% without sorting every score. Names and shared words, the explanation, are
  worked out only for the pairs listed.

On the sample the findings, thresholds and every link suggestion are those of a pair-by-pair implementation of the same
scoring: the same ids, titles, notes and evidence in the same order, and the same scores
([`tests/fixtures/findings/sample_reference.json`](../tests/fixtures/findings/sample_reference.json), recorded with it,
checked by `tests/test_vaultengine_findings.py`; the text evidence needs `models/potion-base-8M`). The `vault findings`
and `vault next` timings above were measured before link scores took in tags and embeddings (term profiles then).

## `vault idea new <vault> --from ID -f F [--title T] [--tags T] [--write] [--json]` and `vault idea set <vault> IDEA STATUS -f F [--write]`

`vault idea new` makes an idea from a finding. It writes to the format's `ideas.type` folder (`50_MOC/ideas` in core):

- `Created`, and `Tags` (the tags the finding's notes share; `--tags` to choose);
- `Status: #open`;
- the finding's id, method, notes and evidence between `<!-- vault:finding -->` markers;
- an empty `## Claim` for you.

It prints by default; `--write` writes it through the safe write path. The default title says what the finding is about without taking a missing note's
name, which would turn the idea into that note. `vault idea set` moves an idea's status to one of the format's values (core:
`open`, `dropped`, `done`, with or without `#`), through the same gate: any other value is refused.

An idea without its finding block fails `vault check` (`missing-block`), and `vault new --type idea` is refused for it:

```
$ vault idea new /tmp/round5 --from link-5ec0974343 -f core --write
Wrote 1 note (idea new):
  + 50_MOC/ideas/in-context-scheming-capability and sandbagging-strategic-underperformance.md
      new idea from link-5ec0974343: in-context-scheming-capability and sandbagging-strategic-underperformance
  check    no new violations
  index    updated 1 note
$ cat "/tmp/round5/50_MOC/ideas/in-context-scheming-capability and sandbagging-strategic-underperformance.md"
Created: 10-11-2026 12:28
Tags: #ai-safety #frontier-models-in-ai
Status: #open

# in-context-scheming-capability and sandbagging-strategic-underperformance

<!-- vault:finding -->
## Finding

- id: link-5ec0974343
- method: link (two notes that probably belong linked)
- notes: [[in-context-scheming-capability]], [[sandbagging-strategic-underperformance]]
- evidence:
  - both link to: 2412.04984 - Frontier Models Capable of In-context Scheming, ai-control-vs-alignment, control-evaluations-red-team-blue-team
  - cited together by: scheming-as-threat-model
  - similar text: scheming, goal, sonnet
  - score 0.24 (shared neighbours 0.31, cited together 0.30, text 0.10)
- found: 2026-11-10 12:28
<!-- /vault:finding -->

## Claim

$ vault check /tmp/round5 -f core | tail -4          # 144 before the idea: it adds none
✗ 144 violations in 123 notes
      83  too-few-links
      43  missing-key
      18  missing-value
$ vault new /tmp/round5 -f core --type idea --title "Just a thought" --set "Tags=[[x]]"
✗ refused: the new note would add 1 violation; nothing was written
  50_MOC/ideas/Just a thought.md
      1  missing-block        no finding block (<!-- vault:finding --> … <!-- /vault:finding -->)
…
$ vault idea set /tmp/round5 "in-context-scheming-capability and sandbagging-strategic-underperformance" parked -f core --write
…
✗ refused: the idea set would add 1 violation; nothing was written
  50_MOC/ideas/…:3  bad-value        Status: #parked is not an allowed value (open, dropped, done)
```

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
