# Verify it yourself

[← Documentation index](../README.md#documentation)

Everything at once, on scratch copies of the sample vault (nothing in the repo is changed):

```bash
scripts/verify.sh            # about 8 minutes: both test suites, then every command end to end
scripts/verify.sh --quick    # 1–3 minutes: every command, the JSON-RPC API and the app end to end (135 checks), without the two long test suites
```

Each step prints `== <step>` and one `  ok  …` line per thing it checked; the last line is `ALL CHECKS PASSED`.
On the first mismatch it prints `FAIL`, what it expected, and what it got, and stops. The steps:

| Step | What it proves |
|---|---|
| Preflight | the [installs](install.md) are there (`.venv`, `markdownify`, `tui/node_modules`, a built `node-pty`, Node 22+) |
| Documentation | [`scripts/check_docs.py`](../scripts/check_docs.py): every link in `README.md`, `docs/` and `tui/README.md` resolves, anchors included; every doc is linked from `README.md` |
| Python | `pytest`: `336 passed, 3 skipped, 17 warnings, 271 subtests passed` (3 WordLlama tests skip without it) |
| Index, stats and health | incremental indexing: 180 notes, then 0 re-parsed, then exactly the edited note |
| JSON-RPC 2.0 | a pick, an unknown method (-32601), an invalid request (-32600) |
| The format file | `vault formats` on `brain.yaml`; a typo in a copy of it is an error with its place (exit 2) |
| Check and its exit codes | 102 violations (exit 1); `--changed` without an index (exit 2), with nothing changed (exit 0), after one note is broken: 3 violations in exactly that note (exit 1) |
| Fix | `--dry-run` prints the diff and leaves every file byte-identical; applying repairs 29 notes; 75 violations left, none repairable; a second fix has nothing to do; the index keeps the hand-broken note pending |
| Safe writes | a new zettel with the right structure; a link that writes `supports::` in one note and `supported-by::` in the other (checked in both files); the same link again is a no-op; `set` repairs a missing `Created`; a refused `set`, a refused `link`, a refused `new`, a taken name and an unknown note (exit 1 or 2) leave every file unchanged; the index is in sync after every write |
| The duplicate gate | a copy of a sample source (content 1.000, same SHA-256) and a saved page with the paper's full title (title 1.000) are stopped, files unchanged; a near-miss title (0.857) and a new article are ingested; ingesting that article again is stopped |
| Coverage and suggestions | 30 sources, 7 nothing links to (the 5 of the sample and the 2 just ingested); suggestions print |
| JSON-RPC format methods | `vault.check`, `vault.coverage`, a refused `note.set`, -32002, -32004, the vault unchanged (not even a lock file); a `note.link` that writes the mirror; -32003 without `--format` |
| App | `npm test` (`Tests  64 passed (64)`), the type check, and `npm run test:e2e` (`Tests  3 passed (3)`): the built app in a real pseudo-terminal running /health, and /check, /coverage, a /link that writes both notes, /check changed, with every printed line fitting the terminal |

One at a time, with what each prints at the end:

| Command (repo root) | Prints at the end |
|---|---|
| `.venv/bin/python -m pytest -q` | `336 passed, 3 skipped, 17 warnings, 271 subtests passed` (with WordLlama installed: `339 passed, 17 warnings, 273 subtests passed`) |
| `cd tui && npm test` | `src/rpc/schema.ts is up to date`, then `Test Files  4 passed (4)` and `Tests  64 passed (64)` |
| `cd tui && npm run typecheck` | nothing (exit code 0) |
| `cd tui && npm run test:e2e` | builds, then `Test Files  2 passed (2)` and `Tests  3 passed (3)` |
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
| `.venv/bin/python -m pytest -q` | `336 passed, 3 skipped, 17 warnings, 271 subtests passed`: the 3 skips are the WordLlama tests (with it installed: `339 passed, 17 warnings, 273 subtests passed`); the 17 warnings come from chromadb in the legacy `test_integration.py` |
| `.venv/bin/python -m unittest discover -s tests` | `Ran 339 tests` … `OK (skipped=3)` |
| `cd tui && npm test` | `src/rpc/schema.ts is up to date`, then `Test Files  4 passed (4)` and `Tests  64 passed (64)` |
| `cd tui && npm run typecheck` | nothing (exit code 0) |
| `cd tui && npm run test:e2e` | builds, then `Test Files  2 passed (2)` and `Tests  3 passed (3)` |
| `scripts/verify.sh` | all of the above and every command end to end; the last line is `ALL CHECKS PASSED` |

What they cover:

- `tests/test_vaultengine_parser.py` (65): parser: frontmatter, tags, links, embeds, headings, blocks, code
- `tests/test_vaultengine_resolver.py` (55): link resolution and the link graph
- `tests/test_vaultengine_index.py` (29): SQLite index, incremental updates, CLI
- `tests/test_vaultengine_rpc.py` (35): JSON-RPC API: recorded request/response sessions replayed against a real `vault serve` (`sample_vault_format` covers every format method, refusals, duplicates and each new error code), a failed commit (-32005), the format file reloading, the parse cache never serving a changed note, error codes, JSON Schema
- `tests/test_vaultengine_formats.py` (47): header and `key::` fields in the parser, format file validation, typing, metadata, headings, relations and mirrors, link roles, `brain.yaml` on the sample vault, `vault formats`
- `tests/test_vaultengine_check_fix.py` (25): `vault check` (all notes, `--changed` incl. violations caused in other notes and by deletions) and `vault fix` (mirror and heading repairs, byte-exact edits with CRLF/BOM/no final newline, the re-check safety net, refusal of files changed on disk, dry-run writes nothing)
- `tests/test_vaultengine_write.py` (36): `vault new`/`link`/`set`: exact output files, mirrors (also for a note a dangling link already names), the gate, scoped index sync (other pending edits stay pending; partial updates equal a full rebuild), no index, index needing a rebuild; refusals (each checks that no byte, mtime or index row changed and nothing is left behind, not even an engine folder); the commit under a failing rename, a bad read-back, a file changed meanwhile, a process killed between renames (rolled back by the next write); the lock; and a random sequence of links and sets checked from disk after every step
- `tests/test_vaultengine_ingest.py` (29): the duplicate gate: title pairs and content pairs that should and should not match, with their scores pinned on both sides of each threshold (content cases built from a real 6,800-word sample source), no two sample sources near a threshold; conversion of HTML, markdown and text; whole ingests into a copy of the sample (an exact copy, an edited and renamed copy, a saved page of a paper already in the vault, a near-miss title, a new article, the same article again and with other page chrome), each refusal leaving every file untouched; `vault coverage` on the sample (its unlinked sources are exactly the index's orphan sources) and each reason
- `tests/test_vaultengine_semantic.py` (11): the experimental ranking: lexical results pinned on the sample vault, suggestions exclude what is already linked, a paraphrase finds its paper; WordLlama loads with the network blocked (skipped without it); `vault suggest`, `vault coverage --suggest`
- `tests/test_integration.py` (7): the original [zettel tool](zettel.md)
- `tui/test/units.test.ts` (14); `tui/test/app.test.tsx` (23 frame tests, incl. keys that arrive in one chunk); `tui/test/format.test.tsx` (21 frame tests of /check /fix /new /link /ingest /coverage, fed with the recorded `sample_vault_format` responses); `tui/test/integration.test.tsx` (6, real Python server, one started with `--format brain`)
- `tui/test/e2e/health.pty.test.ts` (2) and `tui/test/e2e/format.pty.test.ts` (1): the built `vault-tui` in a real pseudo-terminal (node-pty): `/health`; `/check`, `/coverage`, a `/link` that writes both notes (checked in the files), `/check changed`, and every printed line fits the terminal; quitting stops the Python server
