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
| `/explore @note` (or `/explore <name>`, or `/explore` for where to start) | the explore sections, below; `/explore @note --fresh` leaves out what you saw or dismissed |
| `/kept [lens]` | `⏺ Kept  2 results`, each with its lens, evidence, start and time |
| `/map`, `/map record`, `/map since <date>` | `⏺ Map · 12 communities` with the communities, those without a topic page and the bridge notes; `record` adds `✓ recorded a snapshot (14,295 bytes, 4 gaps)`; `since` shows `Gaps: 1 of 4 closed`, each gap's links against chance and the links added, communities formed and gone, missing notes written |
| `/config` | `⏺ Engine settings  precedence: flag > env > file > default`, every setting, its value and source |
| `/` | the list of commands; Tab completes, Enter runs the highlighted one |
| ↑ | the previous command |
| `/help` | all commands, their usage, and the keys |
| ctrl+c | quits and stops the Python server |

All write commands take `--dry-run`. Without `--format`, the format commands say
`No format configured. Start vault-tui with --format <name>, e.g. --format brain (error -32003)`.
Options: `--format <name or file>`, `--python <interpreter>` (default: `$VAULT_TUI_PYTHON`, then this repo's `.venv`,
then `python3`), `--db <index file>`, `--no-index`. `vault-tui` needs an interactive terminal. Development (layout, tests): [tui/README.md](../tui/README.md).

## /explore

`/explore` runs `vault explore` ([Exploring](exploring.md)) and keeps the keyboard: ↑↓ move through the results of
every section, Enter opens one (its full evidence, its notes, what link it would join), `k` keeps, `d` dismisses, `c`
clears the mark (the engine's memory, as `vault mark`), `l` links it: the dry run's diff of both notes first, `w`
writes it through the safe write path (a link made counts as a keep), Esc cancels. `p` pivots: explores from the
selected result (its note, or the start a proposal names), `b` goes back to the previous start, `r` runs it again, `f`
runs it again with fresh. Esc closes it and leaves a summary in the transcript. From the recorded engine replies on
the sample vault (a 100×24 terminal; `test/explore.test.tsx` checks these frames):

```
> /explore @20_ZETTLEKASTEN/kl-penalty-in-rlhf.md
⏺ Explore kl-penalty-in-rlhf  10 lenses · deepen 13, discover 11

1. Neighbourhood  deepen · 4 of 16, 2 near-duplicates · no marks yet
❯   1.1  RLHF
         neighbourhood · 2 hops via direct-preference-optimization-dpo +1 · similar text (rlhf, rew…
    1.2  covariance-targeted-entropy-intervention
         neighbourhood · 2 hops via trust-region-policy-optimization-monoto… +3 · shares [[policy-g…
    1.3  decoupling-clipping-unit-from-credit-assignment-unit
         neighbourhood · 2 hops via policy-gradient-methods (no note yet) · shares [[policy-gradien…
    1.4  tool-output-masking-in-agentic-rl
         neighbourhood · 2 hops via the-environment-not-flops-bottlenecks-a… +1 · shares [[policy-g…

2. Cited together  deepen · 1 of 1 · no marks yet

     1 of 24
╭──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ ↑↓ Enter open · k keep · d dismiss · l link · p pivot · b back · r again · f fresh · Esc close   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Enter on the hidden-link result:

```
> /explore @20_ZETTLEKASTEN/kl-penalty-in-rlhf.md
⏺ 6.1  covariance-targeted-entropy-intervention  Hidden link · discover

     no link either way, but two or more middles in common (Swanson's A-B-C)

     hidden · no link either way; 8 middles in common: 'kl' (4 notes), 'region' (4 notes),
     trust-region-policy-optimization-monotonic-improvement-guarantee (6 notes), 'ppo' (7 notes) +4;
      score 3.86 (p95 of 2135 unlinked pairs)

     covariance-targeted-entropy-intervention  20_ZETTLEKASTEN

     link it: [[covariance-targeted-entropy-intervention]] in kl-penalty-in-rlhf, and back, under
     See Also
╭──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ k keep · d dismiss · c clear · l link it · p pivot · Esc back to the list                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

`l` on the gap result, the dry run:

```
> /explore @20_ZETTLEKASTEN/kl-penalty-in-rlhf.md
⏺ Dry run: would write 2 notes  nothing was written
  ⎿  ✓ check: no new violations  (already there, not blocking: 1 missing-key)
     w writes it in both notes; Esc cancels

     ~ reward-computation-as-a-heavyweight-pipeline-stage  20_ZETTLEKASTEN
       new section # See Also at the end (the level of the note's last heading, at most 2): -
       [[persona-driven-synthetic-data-generation]] — gap: closeness of their mean embeddings 0.38;
       fewer links between their communities than chance predicts; shared: 'verifiable'…
       --- a/20_ZETTLEKASTEN/reward-computation-as-a-heavyweight-pipeline-stage.md
       +++ b/20_ZETTLEKASTEN/reward-computation-as-a-heavyweight-pipeline-stage.md
       @@ -29,3 +29,6 @@

        # References
        1. [[Jiachen Liu - When LLMs Grow Hands and Feet, How to Design our Agentic RL Systems]]
       +
       +# See Also
       +- [[persona-driven-synthetic-data-generation]] — gap: closeness of their mean embeddings
       0.38; fewer links between their communities than chance predicts; shared: 'verifiable'…

       …
╭──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ w write it (both notes) · Esc cancel                                                             │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Searching, related notes, link suggestions and unlinked mentions in the app (`/search`, `/related`, `/suggest`, `/unlinked`): [Search and connections](search.md#in-the-terminal-app).
