# The terminal app

[← Documentation index](../README.md#documentation)

Install it with the [install](install.md) steps (`npm ci && npm run build && npm link` in `tui/`). The examples use
`/tmp/sample-vault` from [Getting started](getting-started.md).

```bash
vault-tui --vault /tmp/sample-vault --format brain
```

A banner appears, the index is brought up to date, and the status line reads
`◆ sample-vault  ·  format brain  ·  180 notes · 1,787 links  ·  ● index fresh`. On a fresh copy, `/check 3` and
`/coverage 3` print (a 100-column terminal; captured from the running app):

```
> /check 3
⏺ Format check  brain · all 180 notes
  ⎿  ✗ 102 violations in 77 notes
     59 unresolved-target · 28 missing-heading · 15 missing-key
     28 can be repaired automatically: /fix shows the diff first

     RLHF  10_SOURCES  (source)
       L138  missing-heading       missing heading 'Notes' (expected at the end of the note)

     agentic-ai-usage-in-customer-service  20_ZETTLEKASTEN  (zettel)
       L28   missing-heading       missing headings 'What This Is', 'Why It Matters', 'Key Details',
                                    'Tensions' (expected before 'References')
     …
     … 74 more notes with violations · /check all lists every note

> /coverage 3
⏺ Coverage  brain · sources: source · citing: zettel, evidence, topic, comparison
  ⎿  ! 28 sources: 23 linked, 5 that nothing links to
     ! 143 citing notes: 33 cite a source, 110 cite none
       106 cite only notes that do not exist · 0 cite notes that are not sources · 4 cite nothing at
       all

     Sources nothing links to (5)
       2501.10970 - The Alternative Annotator Test for LLM-as-a-Judge  10_SOURCES
       …
     Notes that cite nothing at all (4)
       q-value-function  20_ZETTLEKASTEN  (zettel)
       …
     Most cited sources (citing notes)
          5  2503.11926 - Monitoring Reasoning Models (CoT Obfuscation)  10_SOURCES
       …
```

| Type | You should see |
|---|---|
| `/health` Enter | `⏺ Vault health` with `✗ 1,315 unresolved links (612 missing targets)`, `! 41 orphans (no backlinks)`, `! 21 dead ends …`, `✓ 0 empty notes` and the top 5 of each list |
| `/stats` Enter | `⏺ Vault stats` with `180 notes · 0 attachments · 0 with frontmatter` and top tags / most linked / most wanted |
| `/note @klpen`, then Enter twice | the picker shows `❯ kl-penalty-in-rlhf  20_ZETTLEKASTEN`; the first Enter inserts it, the second shows the note: `Links out (10: 4 resolved)`, `Links in (1)` |
| `/check [changed] [all \| <notes>]` | as above; `/check changed` checks what changed since the last index (`✓ no violations in 0 notes checked` on a fresh index) |
| `/coverage [all \| <per list>]` | as above |
| `/fix [apply] [mirrors \| headings] [all]` | `⏺ Fix preview: 28 notes would change  nothing written yet` with the diff of the first three and the changes of the rest; `/fix apply` writes them: `⏺ Fixed 28 notes`, `✓ index: updated 28 notes` |
| `/link @EVD… supports @CLM…` | `⏺ Wrote 2 notes  link` and both diffs; type `@` to pick each note. Linking evidence to a zettel shows `⏺ Refused: the link would add 1 violation  nothing was written`, the violation and the refused diff |
| `/new zettel "A Title" Tags=[[x]]` | `⏺ Created a-title  20_ZETTLEKASTEN`, the check, the index, and the whole new file in a box; `relation @note` pairs add relations |
| `/ingest ~/file.html [as source]` | the source note it created, or `⏺ Not ingested: … duplicates an existing note`, each score against its threshold as a bar, and `/note @…` to open the existing note |
| `/` | the list of commands; Tab completes, Enter runs the highlighted one |
| ↑ | the previous command |
| `/help` | all commands, their usage, and the keys |
| ctrl+c | quits and stops the Python server |

All write commands take `--dry-run`. Without `--format`, the format commands say
`No format configured. Start vault-tui with --format <name>, e.g. --format brain (error -32003)`.
Options: `--format <name or file>`, `--python <interpreter>` (default: `$VAULT_TUI_PYTHON`, then this repo's `.venv`,
then `python3`), `--db <index file>`, `--no-index`. `vault-tui` needs an interactive terminal. Development (layout, tests): [tui/README.md](../tui/README.md).
