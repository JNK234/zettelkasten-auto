# The reading workflow, step by step

[← Documentation index](../README.md#documentation)

From a paper you just read to the pages and ideas built on it, with [`formats/core.yaml`](../formats/core.yaml):

1. ingest the source;
2. write zettels that cite it;
3. see where it stands (`vault pipeline`, `vault note`);
4. make a page from the zettels and bring it to ready (`vault page new`, `check`, `refresh`, `exclude`);
5. read what the engine found (`vault findings`);
6. turn a finding into an idea (`vault idea new`);
7. ask what to do next (`vault next`).

Every command below was run in this order on a fresh copy of the sample vault, and the output is what it printed (dates
and times will differ). `scripts/verify.sh` runs the same commands and checks the same lines. The last section is for an
AI assistant driving the vault: [Driving it from an assistant](#driving-it-from-an-assistant).

**Two kinds of write.** `vault new`, `link`, `set`, `fix` and `ingest` write unless you add `--dry-run`. `vault page`,
`idea`, `move` and `merge` only show what they would do unless you add `--write`. Either way the write is checked first
and refused (exit 1) if it would add a violation. Nothing is half-written: a crash is rolled back by the next write.

## 0. Once: an index and a baseline

```bash
source scripts/setup.sh                 # the installs, and .venv/bin on PATH (see install.md)
cp -R tests/fixtures/vault /tmp/wf-vault
vault index /tmp/wf-vault
vault check /tmp/wf-vault -f core --update-baseline
```
```
  checked 180 notes

Recorded a baseline of 144 violations: /tmp/wf-vault/.vaultengine/baseline-core.json
  `vault check --baseline` now fails only on violations that are not in it
```

The sample's older notes do not fit the core format. The baseline records today's 144 violations, so from now on
`vault check --baseline` fails only on new ones. The index makes every read below fast. Writes keep it up to date.
After edits made outside the engine, run `vault index` again. Until then, reads still see the changed notes, parsed
from disk.

## 1. Ingest the source

The paper, saved as markdown (`.txt` and saved `.html` work too):

```bash
cat > /tmp/paper.md <<'EOF'
---
title: Verifier Gaming in Agentic RL
url: https://example.org/verifier-gaming
---
Agents trained against automatic verifiers learn to satisfy the verifier rather than the task. The paper
collects 40 cases across coding and web environments and sorts them into three kinds: tests the agent can
edit, graders that read only the final message, and rubrics scored by a model that can be flattered.

A do-nothing baseline catches most of them: if an agent that takes no action scores above zero, the
grader is checking the wrong thing.
EOF
vault ingest /tmp/wf-vault /tmp/paper.md -f core --as source --dry-run
```
```
✗ refused: the ingest would add 1 violation; nothing was written
  10_SOURCES/Verifier Gaming in Agentic RL.md
      2  missing-value        'Tags' is empty
…
  closest  10_SOURCES/HUD - Verifier and Reward Design for RL Environments.md  (title 0.308, content overlap 0.000)
```

A source needs Created, Tags and Source. Created is filled in, and the document's `url` fills Source (for a file
without one, add `--set "Source=<url or citation>"`). The tags are yours to give:

```bash
vault ingest /tmp/wf-vault /tmp/paper.md -f core --as source --set "Tags=[[reward-hacking]] [[evaluation-methodology]]" --dry-run
vault ingest /tmp/wf-vault /tmp/paper.md -f core --as source --set "Tags=[[reward-hacking]] [[evaluation-methodology]]"
```
```
+Created: 10-11-2026 16:03
+Tags: [[reward-hacking]] [[evaluation-methodology]]
+Source: https://example.org/verifier-gaming
+
+# Verifier Gaming in Agentic RL
+
+Agents trained against automatic verifiers learn …
…
Wrote 1 note (ingest):
  + 10_SOURCES/Verifier Gaming in Agentic RL.md
      new source: Verifier Gaming in Agentic RL
  check    no new violations
  index    updated 1 note
```

The ingest is also stopped if the paper is already in the vault: the same file, a similar title, or overlapping text.
The closest source is shown either way (see [the duplicate gate](commands.md)).

## 2. Zettels that cite it

A zettel must link to at least one source note, so one that cites nothing is refused:

```bash
vault new /tmp/wf-vault -f core --type zettel --title "Graders should see the final state" \
  --set "Tags=[[reward-hacking]]" --body "Check what the agent left, not what it said." --dry-run
```
```
✗ refused: the new note would add 1 violation; nothing was written
  20_ZETTLEKASTEN/graders-should-see-the-final-state.md
      1  too-few-links        links to 0 source notes; a zettel needs at least 1
```

Cite the source with a link, anywhere in the text:

```bash
T="[[reward-hacking]] [[evaluation-methodology]]"
vault new /tmp/wf-vault -f core --type zettel --title "A do-nothing agent is the cheapest grader test" --set "Tags=$T" \
  --body "If an agent that takes no action scores above zero, the grader checks the wrong thing. From [[Verifier Gaming in Agentic RL]]."
vault new /tmp/wf-vault -f core --type zettel --title "Editable tests are a reward channel" --set "Tags=$T" \
  --body "When the agent can write to the test files, passing the tests stops meaning the task is done. From [[Verifier Gaming in Agentic RL]]."
vault new /tmp/wf-vault -f core --type zettel --title "Final-message graders reward confident summaries" --set "Tags=$T" \
  --body "A grader that reads only the last message pays for claims, not for the state the agent left. From [[Verifier Gaming in Agentic RL]]."
```
```
Wrote 1 note (new):
  + 20_ZETTLEKASTEN/a-do-nothing-agent-is-the-cheapest-grader-test.md
      new zettel: A do-nothing agent is the cheapest grader test
  check    no new violations
  index    updated 1 note
…
```

The file name comes from the title (`{slug}`), Created from the clock. Run each with `--dry-run` first if you want to
see the note before it is written.

## 3. Where the source stands

```bash
vault note /tmp/wf-vault "Verifier Gaming in Agentic RL" -f core
vault pipeline /tmp/wf-vault -f core --limit 3
```
```
10_SOURCES/Verifier Gaming in Agentic RL.md  (source)
  stage      extracted
  trust      70th percentile  (linked from 3, cited by 0 pages and 0 ideas)
…
Used by
  zettels    3 zettels
      20_ZETTLEKASTEN/a-do-nothing-agent-is-the-cheapest-grader-test.md
      20_ZETTLEKASTEN/editable-tests-are-a-reward-channel.md
      20_ZETTLEKASTEN/final-message-graders-reward-confident-summaries.md
  pages      0 pages

  source notes             29
    compiled                3   a page links to a zettel that links to it
    extracted              13   a zettel links to it, no page uses those yet
    ingested               13   no zettel links to it
…
Zettel notes no page includes (107)
```

The stage is worked out from the links on every run, never stored. The source is *extracted*: zettels cite it, but no
page uses them yet.

## 4. A page, until it is ready

```bash
Z="[[a-do-nothing-agent-is-the-cheapest-grader-test]] [[editable-tests-are-a-reward-channel]] [[final-message-graders-reward-confident-summaries]]"
vault page new /tmp/wf-vault "Grader gaming" --from "$Z" -f core
```
```
Created: 10-11-2026 16:03
Tags: [[reward-hacking]] [[evaluation-methodology]]

# Grader gaming

<!-- vault:zettels -->
## Zettels

### reward-hacking
- [[a-do-nothing-agent-is-the-cheapest-grader-test]]: If an agent that takes no action scores above zero, the grader checks the wrong thing. From Verifier Gaming in Agentic RL.
…
<!-- /vault:zettels -->

<!-- vault:sources -->
## Sources

- [[Verifier Gaming in Agentic RL]] (a-do-nothing-agent-is-the-cheapest-grader-test, editable-tests-are-a-reward-channel, final-message-graders-reward-confident-summaries)
<!-- /vault:sources -->

────────────────────────────────────────────────────────────
Would write 3 notes (page new):
  + 50_MOC/grader-gaming.md
      new page: Grader gaming (3 zettels)
  + 50_MOC/index.md
      new index: 4 pages
  + 50_MOC/log.md
      new log
  state    .vaultengine/pages.json: 50_MOC/grader-gaming.md: 3 zettels recorded
  check    no new violations
Not written: add --write to write it.
```

`--from` also takes a search query instead of links (`--from "verifier gaming" --limit 10`: its best-ranked zettels).
The blocks between `<!-- vault:… -->` markers belong to the engine. Everything else on the page is yours to write. Then:

```bash
vault page new /tmp/wf-vault "Grader gaming" --from "$Z" -f core --write
vault page check /tmp/wf-vault grader-gaming -f core          # exit 1 until ready, then 0
```
```
50_MOC/grader-gaming.md  (Grader gaming; refreshed 2026-11-10T16:03:06)
  ✗ behind: 1 zettel with at least two of its tags or links in common, not on it
      20_ZETTLEKASTEN/reward-hacking-as-eval-false-positive.md   tags reward-hacking, evaluation-methodology
  → link the behind zettels that belong in the text and refresh; `vault page exclude` the ones that do not

✗ 0 of 1 managed page ready to write
```

Now the loop: `vault page check`, act on what it says, again until ✓. Each problem names its command:

| `vault page check` says | do |
|---|---|
| `behind: N zettels … not on it` | write the ones that belong into your text as links, then refresh; `vault page exclude PAGE Z… --write` the others |
| `zettels block out of date`, `N zettels changed since the last refresh`, `index line out of date` | `vault page refresh PAGE -f core --write` |
| a violation | fix the note (the message says what is wrong) |

Here, your paragraph under the title links the behind zettel:

```
Every grader here checks a proxy the agent can reach, so a pass is a false positive of the eval:
[[reward-hacking-as-eval-false-positive]].
```

`vault page check` now wants a refresh (the blocks lack the new zettel) and lists 7 more zettels that share a tag and a
link with the page. They are not for this page, so exclude them:

```bash
vault page refresh /tmp/wf-vault grader-gaming -f core --write
vault page exclude /tmp/wf-vault grader-gaming environment-and-eval-are-the-same-object eval-precision-vs-eval-validity \
  grader-failure-routing llm-judge-position-bias measurement-correction-inflates-apparent-agent-progress \
  model-awareness-of-intent-violation-while-cheating ostensible-positives-from-low-entropy-answer-spaces -f core --write
vault page check /tmp/wf-vault grader-gaming -f core
```
```
Wrote 3 notes (page refresh):
  ~ 50_MOC/grader-gaming.md
      rebuilt the zettels and sources blocks (4 zettels)
…
Recorded in the engine state (page exclude); no note changes:
  state    .vaultengine/pages.json: 50_MOC/grader-gaming.md: 4 zettels recorded, 7 excluded

50_MOC/grader-gaming.md  (Grader gaming; refreshed 2026-11-10T16:03:07)
  = 7 zettels excluded from it (`vault page exclude … --undo` to bring one back)
  ✓ ready to write: blocks up to date, zettels as recorded, nothing behind, in the index, no violations

✓ 1 of 1 managed page ready to write
```

Exclusions are kept in the engine state (`.vaultengine/pages.json`), not in the note. Later, when one of the page's
zettels changes or a new zettel shares its tags, `vault page check` says so again.

## 5. What the engine found

```bash
vault findings /tmp/wf-vault -f core --method link --limit 2
```
```
vault findings /tmp/wf-vault  (13 findings; from the index of 2026-11-10T16:03:07+00:00)
  thresholds (percentiles of this vault, with floors): links: score 0.17+

link: two notes that probably belong linked (13)
  link-e0e1ed4ecd  [[a-do-nothing-agent-is-the-cheapest-grader-test]] and [[final-message-graders-reward-confident-summaries]] are not linked but probably belong together
      both link to: Verifier Gaming in Agentic RL
      cited together by: grader-gaming
      similar text: grader, gaming, verifier
      score 0.25 (shared neighbours 0.29, cited together 0.36, text 0.09)
…
```

Without `--method`, you also get missing notes many notes link to, two names for one idea, and link names that are
probably an existing note. The id of a finding is derived from its method and notes, so it is the same on every run and
survives edits elsewhere.

## 6. An idea from a finding

```bash
vault idea new /tmp/wf-vault --from link-e0e1ed4ecd --title "Graders that never look at the state" -f core
vault idea new /tmp/wf-vault --from link-e0e1ed4ecd --title "Graders that never look at the state" -f core --write
```
```
Created: 10-11-2026 16:03
Tags: [[reward-hacking]] [[evaluation-methodology]]
Status: #open

# Graders that never look at the state

<!-- vault:finding -->
## Finding

- id: link-e0e1ed4ecd
- method: link (two notes that probably belong linked)
- notes: [[a-do-nothing-agent-is-the-cheapest-grader-test]], [[final-message-graders-reward-confident-summaries]]
…
<!-- /vault:finding -->

## Claim
…
Wrote 1 note (idea new):
  + 50_MOC/ideas/Graders that never look at the state.md
```

Write your claim under `## Claim`. An idea must keep its finding block (`vault check` reports `missing-block`
otherwise). Its status moves with `vault idea set /tmp/wf-vault "Graders that never look at the state" done -f core
--write` (`open`, `dropped`, `done`).

## 7. What to do next

```bash
vault next /tmp/wf-vault -f core --limit 3
```
```
vault next /tmp/wf-vault  (format core; 100 items, most trusted first)

1. write    write [[transcript-spotting-heuristics-for-reward-hacking]]   (trust 96)
   why  6 notes link to it and it does not exist
   run  vault new /tmp/wf-vault -f core --type zettel --name transcript-spotting-heuristics-for-reward-hacking …
…
By kind (listed / all):
  refresh     0 / 0    pages out of date with their zettels
  behind      0 / 0    zettels a page is behind on
  merge       9 / 9    two names for one idea
  write      72 / 72   missing notes many notes link to
  extract    13 / 13   sources no zettel links to
  page        5 / 9    zettels in no page that share a tag (103 zettels in no page); below the median trust not listed
  idea        1 / 1    open ideas nothing builds on
```

One queue for everything above: pages to refresh or that are behind, names to merge, notes to write, sources to extract
from, zettels that could make a page, open ideas. Most trusted first: how much the vault relies on the note (links to it,
pages and ideas citing it). Each item has a `run` line, the command that resolves it. Show it first where it writes:
`vault new … --dry-run`, `vault merge …` without `--write`.

To end a session:

```bash
vault check /tmp/wf-vault -f core --baseline      # exit 0: no violation the baseline does not have
```
```
✓ no new violations in 188 notes checked (144 known violations in the baseline not shown)
```

## In the app

`vault-tui --vault /tmp/wf-vault --format core` has the same steps as commands, with a dry run by default and `write` at
the end to write ([The app](tui.md)):

| Step | In the app |
|---|---|
| where sources stand | `/pipeline` |
| a page | `/page new "Grader gaming" @zettel @zettel @zettel` (then the same line with `write`), `/page check @grader-gaming`, `/page refresh @grader-gaming write`, `/page exclude @grader-gaming @zettel… write` |
| findings | `/findings`, `/findings link all`, `/findings link-e0e1ed4ecd` |
| an idea | `/idea new link-e0e1ed4ecd "Graders that never look at the state"` (then `write`), `/idea set @idea done write` |
| move, merge | `/move @note new-name`, `/merge @A @B` (then `write`) |
| next | `/next`, `/next all` |
| the baseline | `/check baseline` |

## Driving it from an assistant

For an AI assistant (Claude Code, a script) working on the vault for you.

**Rules.**

1. **Never edit a note file directly.** Not to add a link, fix a tag, rename a note or tidy a page. The commands check
   every write against the format and refuse one that adds a violation. They keep every link reaching its note when a
   note moves, and keep the engine's records right: page records, exclusions, the baseline, ideas' finding blocks, the
   index. An edit behind their back bypasses all of that. It also gets overwritten: the blocks between
   `<!-- vault:… -->` markers are rebuilt by `vault page refresh`. The one exception is the prose a person writes on
   a page, under an idea's `## Claim`, or in a zettel, and that is the person's to write.
2. **Dry run first, then write.** For `new`, `link`, `set`, `fix` and `ingest`, pass `--dry-run`; for `page`, `idea`,
   `move` and `merge`, leave out `--write`. Show the person the diff or note it would write, then run the same command
   with the write flag. Do not change the arguments between the two.
3. **A refusal is an answer, not an obstacle.** Exit 1 with `✗ refused` (or `"status": "refused"` in JSON) lists the
   violations the write would add. Tell the person. Do not retry with made-up values to get past it, such as a tag
   nobody chose or a source the note does not rest on.
4. **Use `--json`** and read the fields below, not the text output.
5. **End with `vault check VAULT -f core --baseline`**: exit 0 means nothing new is wrong.

**Which command for what.**

| To … | run | `--json` gives |
|---|---|---|
| know what is wrong | `vault check V -f core --baseline` | `ok`, `summary` (`violations`, `by_code`), `notes[].violations[]` (`path`, `line`, `code`, `message`), `baseline` (`new`, `known`, `fixed`) |
| see where sources are | `vault pipeline V -f core` | `stages`, `sources[]` (`path`, `stage`, `zettels`, `pages`), `gaps` (`not_extracted`, `not_compiled`) |
| know one note | `vault note V NOTE -f core` | `type`, `stage`, `rests_on`, `used_by`, `links_to_nothing`, `violations`, `trust` |
| add a source | `vault ingest V FILE -f core --as source --set "Tags=…" [--set "Source=…"]` | a write result (below), plus `duplicates` and `candidates` (`status: "duplicate"` when it is already there) |
| write a zettel | `vault new V -f core --type zettel --title … --set "Tags=…" --body "… [[Source]] …"` | a write result |
| make a page | `vault page new V TITLE --from "[[a]] [[b]] [[c]]" -f core` | a write result, plus `page` (`path`, `content`) and `state` |
| know if a page is ready | `vault page check V [PAGE] -f core` | `ready`, `pages[]` (`blocks`, `changed`, `behind`, `excluded`, `violations`, `index`, `ready`); exit 0 when ready |
| bring a page up to date | `vault page refresh V PAGE -f core` | a write result (`status: "unchanged"` when there is nothing to do) |
| take zettels off a page's list | `vault page exclude V PAGE Z… -f core [--undo]` | a write result, with `state` |
| see what the engine found | `vault findings V -f core [--method M] [--id ID]` | `findings[]` (`id`, `method`, `title`, `notes`, `evidence`, `data`), `thresholds` |
| write an idea from one | `vault idea new V --from ID [--title …] -f core` | a write result, plus `finding` and `idea` (`path`, `content`) |
| change an idea's status | `vault idea set V IDEA open\|dropped\|done -f core` | a write result |
| rename or move a note | `vault move V NOTE TO -f core` | a write result, plus `moved`, `rewritten`, `now_resolving`, `state` |
| make two notes one | `vault merge V A B -f core` | the same, plus `self_links` (`path`, `line`, `link`, `action`, `result`): links dropped because they would point at A itself |
| know what to do next | `vault next V -f core [--limit N]` | `items[]` (`kind`, `title`, `why`, `command`, `trust`, `id`), `counts` |

**A write result** (any command that writes, with `--json`):

- `status`: `written`, `dry-run`, `refused` or `unchanged`.
- `files[]`: each with `path`, `action` (`create`, `modify`, `delete`), `changes`, a unified `diff`, and `content` for
  a created file.
- `new_violations`: why a write was refused.
- `tolerated`: violations already in those notes; they do not block it.
- `index`: how the index was updated.
- `ok`: false when refused.

**Exit codes.** 0 done (or nothing to do); 1 refused, violations found, or a page not ready; 2 bad arguments, or a note
or finding that does not exist; 3 the write failed and was rolled back.

**The same over JSON-RPC.** `vault serve V --format core` answers one JSON request per line with the same operations:

- `vault.check` (`baseline`, `update_baseline`);
- `vault.pipeline`, `note.facts`;
- `page.new`, `page.check`, `page.refresh`, `page.exclude`;
- `findings.list`, `idea.new`, `idea.set`;
- `note.move`, `note.merge`;
- `vault.next`.

There, writes take `"dry_run": true` for the dry run, and without it they write. See [the API](api.md).
