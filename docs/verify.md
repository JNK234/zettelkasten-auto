# Verify it yourself

[← Documentation index](../README.md#documentation)

Everything at once, on scratch copies of the sample vault (nothing in the repo is changed):

```bash
scripts/verify.sh            # about 11 minutes: both test suites, then every command end to end
scripts/verify.sh --quick    # about 4 minutes: every command, the JSON-RPC API and the app end to end (369 checks), without the two long test suites
```

From a clean checkout (no `.venv`, no `tui/node_modules`):

```bash
source scripts/setup.sh       # installs everything (Python and the app), puts .venv/bin on PATH
bash scripts/verify.sh        # ends with ALL CHECKS PASSED
```

Each step prints `== <step>` and one `  ok  …` line per thing it checked; the last line is `ALL CHECKS PASSED`.
On the first mismatch it prints `FAIL`, what it expected, and what it got, and stops. The steps:

| Step | What it proves |
|---|---|
| Preflight | the [installs](install.md) are there (`.venv`, `markdownify`, `tui/node_modules`, a built `node-pty`, Node 22+) |
| Documentation | [`scripts/check_docs.py`](../scripts/check_docs.py): every link in `README.md`, `docs/` and `tui/README.md` resolves, anchors included; every doc is linked from `README.md` |
| Python | `pytest`: `636 passed, 17 warnings, 1046 subtests passed` |
| Index, stats and health | incremental indexing: 180 notes, then 0 re-parsed, then exactly the edited note |
| JSON-RPC 2.0 | a pick, an unknown method (-32601), an invalid request (-32600) |
| The format file | `vault formats` on `brain.yaml`; a typo in a copy of it is an error with its place (exit 2) |
| Check and its exit codes | 102 violations (exit 1); `--changed` without an index (exit 2), with nothing changed (exit 0), after one note is broken: 3 violations in exactly that note (exit 1) |
| The core format, a baseline, the pipeline | 144 violations under core; `--baseline` without a baseline (exit 2, nothing written); `--update-baseline` records them and `--baseline` passes; a renamed note keeps its baselined violations; a zettel with no source, a page linking one zettel and a note in `10_SOURCES/old/` are exactly the 3 new violations; re-recording; `vault pipeline` and `vault note` on the sample; `vault new` refuses a zettel without a source and writes one with |
| Pages | `vault page new` prints and writes nothing without `--write`; written, with its index and log lines, it adds no violation; behind on 2 zettels; an edited zettel makes its block out of date; `refresh` keeps your text byte for byte; `exclude` (a dry run writes nothing); a moved page keeps its record and exclusions |
| Pages and the pipeline from the index | reads parse only what changed; a stale index gives the same answers as reading every file |
| Findings and ideas | `vault findings` (one of each method, their ids); `vault idea new` prints, then writes; an idea without a finding block is refused; `vault idea set` (a status the format does not have is refused) |
| Move, merge and the queue | a merge of a missing name into a note and a rename keep every link (`vault health` unchanged); a move into a folder no type matches is refused; `vault next`; a merge that drops the line `- [[B]]` in A (it would link A to itself), listed as dropped, and no self-link left on disk |
| Findings are the pair-by-pair implementation's | `tests/test_vaultengine_findings.py -k SameAsPairByPair`: on the sample, every finding (ids, titles, notes, evidence, order), the thresholds and every link suggestion (scores to 1e-9) equal `tests/fixtures/findings/sample_reference.json`, recorded with the pair-by-pair implementation |
| The reading workflow | the commands of [The reading workflow](workflow.md) on a fresh copy: an ingest refused for empty Tags, then written with Source from the document's URL; a zettel without a source refused; three that cite it; the source *extracted*; a page, behind on one zettel, excluded, ready; the link finding between two new zettels; an idea from it; `vault next` lists it; `vault check --baseline` passes with 188 notes |
| Fix | `--dry-run` prints the diff and leaves every file byte-identical; applying repairs 29 notes; 75 violations left, none repairable; a second fix has nothing to do; the index keeps the hand-broken note pending |
| Safe writes | a new zettel with the right structure; a link that writes `supports::` in one note and `supported-by::` in the other (checked in both files); the same link again is a no-op; `set` repairs a missing `Created`; a refused `set`, a refused `link`, a refused `new`, a taken name and an unknown note (exit 1 or 2) leave every file unchanged; the index is in sync after every write |
| The duplicate gate | a copy of a sample source (content 1.000, same SHA-256) and a saved page with the paper's full title (title 1.000) are stopped, files unchanged; a near-miss title (0.857) and a new article are ingested; ingesting that article again is stopped |
| Coverage | 30 sources, 7 nothing links to (the 5 of the sample and the 2 just ingested) |
| JSON-RPC format methods | `vault.check`, `vault.coverage`, a refused `note.set`, -32002, -32004, the vault unchanged (not even a lock file); a `note.link` that writes the mirror; -32003 without `--format` |
| JSON-RPC reading-workflow methods | with `--format core`: `vault.check` `baseline` without one (-32007), `vault.pipeline`, `findings.list`, a `note.merge` dry run listing its dropped self-link, `page.check`, `vault.next`, the vault unchanged; then `update_baseline` and a passing `baseline` |
| App | `npm test` (`Tests  107 passed (107)`), the type check, and `npm run test:e2e` (`Tests  5 passed (5)`): the built app in a real pseudo-terminal running /health; /check, /coverage, a /link that writes both notes, /check changed; and with `--format core` /pipeline, /findings link, a /merge as a dry run then written (the self-link gone on disk), /check baseline, /next 3; every printed line fits the terminal |

One at a time, with what each prints at the end:

| Command (repo root) | Prints at the end |
|---|---|
| `.venv/bin/python -m pytest -q` | `636 passed, 17 warnings, 1046 subtests passed` (about 8 minutes) |
| `cd tui && npm test` | `src/rpc/schema.ts is up to date`, then `Test Files  6 passed (6)` and `Tests  107 passed (107)` |
| `cd tui && npm run typecheck` | nothing (exit code 0) |
| `cd tui && npm run test:e2e` | builds, then `Test Files  3 passed (3)` and `Tests  5 passed (5)` |
| `.venv/bin/vault rpc-schema \| diff - vaultengine/rpc/schema.json` | nothing: the committed schema matches the models |

And by hand, on a fresh copy (with `source .venv/bin/activate` done; [Commands](commands.md) shows the full output of each):

```bash
rm -rf /tmp/sample-vault && cp -Rp tests/fixtures/vault /tmp/sample-vault && vault index /tmp/sample-vault >/dev/null
vault check /tmp/sample-vault -f brain > /tmp/check.txt; echo "exit $?"; tail -4 /tmp/check.txt
```
```
exit 1
✗ 102 violations in 77 notes
      59  unresolved-target
      28  missing-heading
      15  missing-key
```

| Then run | Expect |
|---|---|
| `vault check /tmp/sample-vault -f brain --changed; echo $?` | `✓ no violations in 0 notes checked`, `0` |
| `vault fix /tmp/sample-vault -f brain --dry-run \| tail -3` | `Dry run: nothing was written.`, a blank line, `Violations left after this fix, for you to fix: 59 unresolved-target, 15 missing-key` |
| `vault link /tmp/sample-vault -f brain "EVD - Reward is a deterministic outcome check 1 only if the final database equals the data" "CLM - Grade the observed end state rather than the trajectory" --as supports` | `Wrote 2 notes (link):`, one line ending ``to `supports::` ``, one ending ``to `supported-by::` ``, `index    updated 2 notes` |
| `grep -c "Reward is a deterministic outcome check 1" "/tmp/sample-vault/50_MOC/discourse/rl-environment-design/CLM - Grade the observed end state rather than the trajectory.md"` | `1`: the mirror is in the claim |
| `vault set /tmp/sample-vault -f brain "CLM - Grade the observed end state rather than the trajectory" Status "#settled"; echo $?` | `✗ refused: the change would add 1 violation; nothing was written`, the violation, the refused diff, `1` |
| `cp "/tmp/sample-vault/10_SOURCES/2411.00640 - Adding Error Bars to Evals.md" /tmp/copy.md && vault ingest /tmp/sample-vault /tmp/copy.md --as source -f brain; echo $?` | `✗ not ingested: copy.md duplicates 1 existing note; nothing was written`, `content overlap   1.000  duplicate (threshold 0.8)`, `same file         yes`, `1` |
| `vault coverage /tmp/sample-vault -f brain \| sed -n 3,5p` | `sources           28   23 linked from other notes, 5 that nothing links to` and `citing notes     143   33 cite a source, 110 cite none …` |
| `vault-tui --vault /tmp/sample-vault --format brain`, then `/check` and `/coverage` | the same numbers, as in [The terminal app](tui.md) |


## Tests and expected results

| Command (from the repo root) | Prints at the end |
|---|---|
| `.venv/bin/python -m pytest -q` | `636 passed, 17 warnings, 1046 subtests passed`; the 17 warnings come from chromadb in the legacy `test_integration.py` |
| `.venv/bin/python -m unittest discover -s tests` | `Ran 636 tests` … `OK` |
| `cd tui && npm test` | `src/rpc/schema.ts is up to date`, then `Test Files  6 passed (6)` and `Tests  107 passed (107)` |
| `cd tui && npm run typecheck` | nothing (exit code 0) |
| `cd tui && npm run test:e2e` | builds, then `Test Files  3 passed (3)` and `Tests  5 passed (5)` |
| `scripts/verify.sh` | all of the above and every command end to end; the last line is `ALL CHECKS PASSED` |

What they cover:

- `tests/test_vaultengine_parser.py` (65): parser: frontmatter, tags, links, embeds, headings, blocks, code
- `tests/test_vaultengine_resolver.py` (55): link resolution and the link graph
- `tests/test_vaultengine_index.py` (29): SQLite index, incremental updates, CLI
- `tests/test_vaultengine_rpc.py` (44): JSON-RPC API: recorded request/response sessions replayed against a real `vault serve` (`sample_vault_format` covers every format method, refusals, duplicates and each new error code; `sample_vault_core`, served with `--format core`, every reading-workflow method: the baseline (-32007 without one), pipeline, note facts, pages, findings, ideas, a merge listing its dropped self-link, a move, next), checks on what that recording shows, the read methods returning what the CLI prints with `--json`, a failed commit (-32005), the format file reloading, the parse cache never serving a changed note, error codes, JSON Schema
- `tests/test_vaultengine_formats.py` (47): header and `key::` fields in the parser, format file validation, typing, metadata, headings, relations and mirrors, link roles, `brain.yaml` on the sample vault, `vault formats`
- `tests/test_vaultengine_check_fix.py` (25): `vault check` (all notes, `--changed` incl. violations caused in other notes and by deletions) and `vault fix` (mirror and heading repairs, byte-exact edits with CRLF/BOM/no final newline, the re-check safety net, refusal of files changed on disk, dry-run writes nothing)
- `tests/test_vaultengine_core_format.py` (55): `formats/core.yaml` (its four types and their homes, only the decided fields, each
  field enforced, idea statuses, a URL is not a source, a vault that follows it is clean); `subfolders: false`; `typed_folders` and
  `untyped-note`; `ignored_folders`; `min_links` and `too-few-links` (what counts: resolved, distinct, relation fields, embeds; what
  does not: other types, unresolved and ambiguous links, `Tags:` links, code, self-links); `vault new` filling `Created` and
  `--body`; the write gate under core; `vault check --baseline` and `--update-baseline` (check never writes, compare, moved lines,
  fixed, counts, `--changed`, broken and old files; renames: same text, edited, moved out of its home, ties not guessed, only new
  paths are candidates); core on the sample vault; brain unchanged
- `tests/test_vaultengine_pipeline.py` (16): `vault pipeline` and `vault note` on a synthetic vault with each case worked out by
  hand (every stage, both gaps, why zettels rest on no source, what each note rests on and is used by, tag links and ignored
  notes counting for nothing, stages moving when links change, nothing written); the sample vault's numbers
- `tests/test_vaultengine_pages.py` (27): `vault page new` (prints and writes nothing without `--write`; the whole page; it
  passes `vault check`; index, log and engine state; refused when too thin; bad input; `--from` a query; `--tags`), `refresh`
  (nothing to do on a fresh page; editing a listed zettel makes it stale and refresh clears it; a change only the engine state
  sees; every byte outside the markers kept, CRLF included; dropping a line drops the zettel; broken markers), `check` (the
  behind list with two things in common, exactly; a clean page is ready to write; every page; no record; index line),
  `exclude` (and `--undo`, refusals), a renamed page keeping its record (and a different page not taking it), the pieces (excerpts, markers,
  grouping, format validation), and that `scripts/bench_pages.py` makes a clean core vault
- `tests/test_vaultengine_indexed.py` (7): the index-backed vault: every note rebuilt from the index equals a fresh parse
  (sample, resolver fixture, frontmatter, CRLF/BOM, fields, empty note); a format's answers through the index equal those
  from parsing every file (issues, links and roles, relations and mirrors, metadata, note links, pipeline, page checks; brain
  and core); the same after random edit sequences made without re-indexing; reads write nothing; a missing or other-schema
  index is left alone and every file is read
- `tests/test_vaultengine_findings.py` (16): on the sample, the findings, thresholds and every link suggestion equal those of the pair-by-pair implementation (`tests/fixtures/findings/sample_reference.json`); thresholds are percentiles of the vault (they move with it; floors); one finding of each method on a built vault (tags are not missing notes, ignored
  notes take no part); ids derived from method and notes, the same across runs, after unrelated edits and with new evidence;
  a resolved finding goes; reading writes nothing; `vault idea new` (nothing without `--write`, the whole idea, it passes
  `vault check`, it is not a finding itself), an idea without a finding block refused and failing check (absent, empty,
  unclosed, in code), `vault idea set` (open, dropped, done; anything else refused)
- `tests/test_vaultengine_moves.py` (16): `vault move` and `vault merge` on a vault with every link shape (names, paths, relative
  and markdown links with `%20` and `<…>`, headings, blocks, embeds, aliases, frontmatter, a Tags line, CRLF, self-links, a
  dangling name): every resolved link still reaches its note and `vault health` shows no new broken link; a new name another
  note has; bytes outside links kept; dry runs write nothing; refusals (the gate, bad destinations); a crash after renames or
  after the deletion rolled back; a merge's marked section, a missing name merged into a note, two missing names; the links a merge would point at the note they are in dropped and reported (a line holding only the link, a task, a quote, a field value, an inline field, an embed, a heading link made a same-note link, prose and aliases unlinked to their text, frontmatter list items and values); the baseline,
  page records and an idea's finding block follow; the local gate equals a check of the whole vault on random writes and on
  writes whose violations land in other notes
- `tests/test_vaultengine_next.py` (5): `vault next` lists exactly the planted items (one of each kind) in a stable order, the
  same across hash seeds; each item's command resolves it; the median cutoff; trust shown by `vault note` and never written
- `tests/test_vaultengine_write.py` (36): `vault new`/`link`/`set`: exact output files, mirrors (also for a note a dangling link already names), the gate, scoped index sync (other pending edits stay pending; partial updates equal a full rebuild), no index, index needing a rebuild; refusals (each checks that no byte, mtime or index row changed and nothing is left behind, not even an engine folder); the commit under a failing rename, a bad read-back, a file changed meanwhile, a process killed between renames (rolled back by the next write); the lock; and a random sequence of links and sets checked from disk after every step
- `tests/test_vaultengine_ingest.py` (32): the duplicate gate: title pairs and content pairs that should and should not match, with their scores pinned on both sides of each threshold (content cases built from a real 6,800-word sample source), no two sample sources near a threshold; conversion of HTML, markdown and text; whole ingests into a copy of the sample (an exact copy, an edited and renamed copy, a saved page of a paper already in the vault, a near-miss title, a new article, the same article again and with other page chrome), each refusal leaving every file untouched; `vault coverage` on the sample (its unlinked sources are exactly the index's orphan sources) and each reason; under core (`ingest: {url: Source}`): the URL fills Source, the document follows the title, a given Source wins, none is refused
- `tests/test_vaultengine_search.py` (75), `tests/test_vaultengine_query.py` (38): full-text search (tokenizer, fields,
  postings, incremental upkeep, BM25F) and the query language (syntax trees, errors, every filter on a hand-built vault, the
  sample vault)
- `tests/test_vaultengine_related.py` (23), `tests/test_vaultengine_connect.py` (18): related notes (personalized PageRank
  against an exact solution, signals, reasons) and connecting notes (unlinked mentions, links to add, the rules neither may
  break); the link suggestions' scores are also pinned against the pair-by-pair reference in `test_vaultengine_findings.py`
- `tests/test_integration.py` (7): the original [zettel tool](zettel.md)
- `tui/test/units.test.ts` (14); `tui/test/app.test.tsx` (25 frame tests, incl. keys that arrive in one chunk); `tui/test/format.test.tsx` (21 frame tests of /check /fix /new /link /ingest /coverage, fed with the recorded `sample_vault_format` responses); `tui/test/pipeline.test.tsx` (17 frame tests of /pipeline /page /findings /idea /move /merge /next /check baseline, fed with the recorded `sample_vault_core` responses: what each shows, the params it sends, bad arguments, a format without a `pipeline:` block, no line wider than the terminal); `tui/test/integration.test.tsx` (8, real Python server, one started with `--format brain`); `tui/test/search.test.tsx` (22 frame tests of /search, /related, /suggest, /unlinked)
- `tui/test/e2e/health.pty.test.ts` (3), `tui/test/e2e/format.pty.test.ts` (1) and `tui/test/e2e/pipeline.pty.test.ts` (1): the built `vault-tui` in a real pseudo-terminal (node-pty): `/health`; `/check`, `/coverage`, a `/link` that writes both notes (checked in the files), `/check changed`; with `--format core` `/pipeline`, `/findings link`, a `/merge` shown then written (the file on disk has no self-link, the other note is gone), `/check baseline`, `/next 3`; every printed line fits the terminal; quitting stops the Python server
