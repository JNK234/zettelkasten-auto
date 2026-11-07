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
| 2 | bad arguments, no such note, a name already taken, no index for `--changed`, an invalid format file |
| 3 | a write failed and every file was restored |

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

## `vault coverage <vault> -f F [--limit N] [--suggest] [--json]`

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
(106 / 0 / 4). The new zettel and the new source account for the difference. `--suggest` adds a likely source for each note that
cites none (experimental, see below):

```
  20_ZETTLEKASTEN/a-bounty-list-is-a-map-of-what-cannot-be-verified.md  (zettel) cites only missing notes: line 37 [[Prime Intellect - Environments Hub]]
      perhaps: 10_SOURCES/HUD - Verifier and Reward Design for RL Environments.md  (lexical 0.214, experimental)
```

## `vault suggest <vault> -f F (--note N [--as RELATION] | --file DOC [--type T]) [--method lexical|semantic|fused] [--limit N]` (experimental)

Ranks notes by text similarity for the three questions the commands above leave open: which claim an evidence note may
support (`--as supports`), which sources a note may draw on (no `--as`), which existing notes a document resembles (`--file`,
which catches a reworded copy the duplicate gate scores near 0). Advisory only; it writes nothing and nothing depends on it.

```
$ vault suggest /tmp/sample-vault -f brain --note "EVD - Reward is a deterministic outcome check 1 only if the final database equals the data" --as supports
Suggestions (lexical, experimental): notes 50_MOC/discourse/rl-environment-design/EVD - Reward is a deterministic outcome check 1 only if the final database equals the data.md may `supports`
  1  50_MOC/discourse/rl-environment-design/CLM - State graders should check what must not change as well as what must.md   lexical 0.112
  2  50_MOC/discourse/rl-environment-design/CLM - Hack-robustness is the first requirement of a verifier.md   lexical 0.098
  3  50_MOC/discourse/rl-environment-design/CLM - Grading end state lets the action space open safely.md   lexical 0.087

to link the first: vault link /tmp/sample-vault -f brain "EVD - Reward is a deterministic outcome check 1 only if the final database equals the data" "CLM - State graders should check what must not change as well as what must" --as supports
```

(The claim it was just linked to is not suggested again.) `--method lexical` (default) is TF-IDF over the whole vault;
`semantic` uses [WordLlama](https://github.com/dleemiller/WordLlama) static embeddings (offline; its model ships in the wheel);
`fused` combines both rankings. [`python scripts/semantic_eval.py`](../scripts/semantic_eval.py) measures them on the vault's own links:

```
task                         queries           lexical          semantic             fused
A  note -> its source             33   33/33 · 33 top3   26/33 · 29 top3   28/33 · 30 top3
B  evidence -> its claim          26   16/26 · 24 top3   11/26 · 20 top3   13/26 · 23 top3
C  paraphrase -> its paper         8    7/8 ·  8 top3    6/8 ·  8 top3    7/8 ·  8 top3
```

On the sample vault the embeddings did not beat words, and their absolute scores cannot be thresholded (unrelated long notes
score about 0.6 against each other), so nothing gates on them.

The same operations over JSON-RPC: [The JSON-RPC API](api.md#format-methods). In the terminal app: [The terminal app](tui.md).
