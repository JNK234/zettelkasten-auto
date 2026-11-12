# For assistants: the explore commands' JSON

[← README](../README.md) · [Exploring](exploring.md) · [The JSON-RPC API](api.md)

Every explore command answers in JSON with `--json`, and the same answer comes from the JSON-RPC server
(`vault serve`): one function builds both (`vaultengine/operations.py`), one Pydantic model describes it
(`vaultengine/rpc/models.py`), and the JSON Schema (`vaultengine/rpc/schema.json`) and the TypeScript types
(`tui/src/rpc/schema.ts`) are generated from the models. The CLI adds `vault` (and, for reads, `index`: the index
file, when it was built, whether it is out of date) around the result; the method returns the result alone.

| command | JSON-RPC method | reads or writes |
|---|---|---|
| `vault explore [note] [--fresh] [--lens L]… [--per-lens N] [--no-record] [--refresh]` | `explore.run` | reads the notes; remembers the run (refs, seen) in the engine's memory |
| `vault mark keep\|dismiss\|clear REF…` | `explore.mark` | the memory only |
| `vault kept [--lens L]` | `explore.kept` | reads the memory |
| `vault link-it REF [--write] [--reason R]` (or `NOTE NOTE`) | `explore.link` | **writes two notes with `--write` / `dry_run: false`**; a dry run otherwise |
| `vault unlink NOTE NOTE [--write]` | `explore.unlink` | **writes two notes with `--write` / `dry_run: false`**; a dry run otherwise |
| `vault map [--record]` | `map.get` | reads; `--record` saves a snapshot in the memory |
| `vault map --since DATE` | `map.since` | reads |
| `vault deepen NOTE` | `note.deepen` | reads |
| `vault discover [NOTE]` | `vault.discover` | reads |
| `vault explain A B` | `notes.explain` | reads |
| `vault embed --status` | `embed.status` | reads |
| `vault config show` | `config.show` | reads |

Exit codes of the CLI: 0 done (a dry run too); 1 a write refused because it would add a violation (nothing
written); 2 a usage error, a note that does not exist, or nothing to act on (the message says why, and what to do);
3 a write that failed and was rolled back. JSON-RPC errors: -32002 no such note, -32003 no format (start the server
with `--format`), -32004 a write rejected before it was tried, -32005 a write rolled back, -32007 nothing to act on
(no explore run yet, a ref not in the last run, a result that is not two notes, no map snapshot, a lens that does not
run from this start).

## How to use them

1. **Never edit note files directly.** Every change goes through `vault link-it` / `vault unlink` (or the other
   write commands, `vault new`, `vault link`, `vault set`), which check the format before writing, write both notes
   or neither, and keep the index up to date.
2. **Start** with `vault explore --json` (no note: where to start, from the vault-wide lenses) or
   `vault explore NOTE --json`. Read `sections[].results[]`: `title`, `lens`, `evidence`. Each result's `ref`
   (`"2.3"`) names it for the next commands; refs belong to the **last** explore run, so mark and link right after
   the run they came from.
3. **Keep or dismiss** what you show the person and they decide on: `vault mark keep 1.1 --json`,
   `vault mark dismiss 7.1 --json`. Keeps reorder the sections next time (`keep_score`); dismissed results stay out
   of `--fresh` runs.
4. **Link with a dry run first**: `vault link-it 8.1 -f brain --json`. Check `status` (`"dry-run"`),
   `new_violations` (empty), and `files[].diff` (the `- [[other]] — reason` line in both notes); show the diff. Only
   then `vault link-it 8.1 -f brain --write --json`; check `status == "written"`. `vault unlink A B --write` undoes
   it: both notes byte for byte as they were, line endings and final line break included.
5. **Come back** with `vault explore NOTE --fresh --json`: what was seen or dismissed is left out, the next results
   move up. `vault kept --json` lists what was kept.
6. To see what changed over days: `vault map --record` once, then later `vault map --since DATE --json`.

Results are deterministic: the same vault, settings and memory give the same JSON (`--timing` adds each lens's
seconds, the only part that varies).

## `vault explore` / `explore.run`

Params (RPC): `note` (a name, path or alias, or a missing note's name; null: where to start), `lenses`, `per_lens`
(3–5), `fresh`, `record` (default true), `refresh`.

| field | meaning |
|---|---|
| `start`, `start_kind` | the start's path (`"[[name]]"` for a missing note, null without a start); `"note"`, `"missing"` or null |
| `fresh` | whether seen and dismissed results were left out |
| `duplicate_cutoff`, `similarity` | two notes at or above this similarity (embedding cosine, or key-term cosine without the model) are near-duplicates: a section shows one of them |
| `memory` | the memory file |
| `numbers` | `source: "current"`, or `"reused"`: the vault-wide distributions are from `computed_at`, `changed_since` notes ago (at most `threshold`) |
| `text_signal` | whether embeddings were used (`available`), the model, why not, notes embedded in memory (`uncached`) |
| `sections[]` | one per lens, ordered by `keep_score` (ties in the configured order) |
| `sections[].lens`, `kind`, `title`, `what` | the lens (registry name), deepen / discover / map, its title, what a result is |
| `sections[].budget`, `found`, `hidden`, `duplicates` | results it may show (3–5; deepen and discover get the same total), candidates found, left out as seen or dismissed, left out as near-duplicates |
| `sections[].kept`, `dismissed`, `keep_score` | your marks of this lens over every start, and (kept + 1) / (marked + 2) |
| `sections[].applies`, `note` | false when the lens does not apply (e.g. to a missing note); why there are no results |
| `results[].ref` | `"section.rank"`: what `mark` and `link-it` take |
| `results[].kind`, `target` | `note`, `pair` (two notes not linked, a gap's joiners), `link` (a link that is there), `missing`, `community`; what it is about, as remembered |
| `results[].title`, `evidence` | what to show; one line of why, with its numbers |
| `results[].notes`, `link` | the notes it is about (paths); the two notes `link-it` would link, or null |
| `results[].start_at`, `mark` | without a start: where to start for it (pass it as the note); your earlier mark for it, `"keep"` / `"dismiss"` / null |

```
$ vault explore /tmp/agent-vault kl-penalty-in-rlhf -f brain --json     # one section and one result shown
{
  "vault": "/tmp/agent-vault",
  "index": {"db": "…", "last_indexed_at": "…", "stale": false, "pending": "…"},
  "start": "20_ZETTLEKASTEN/kl-penalty-in-rlhf.md",
  "start_kind": "note",
  "fresh": false,
  "duplicate_cutoff": 0.5069471937417984,
  "similarity": "embedding",
  "memory": "/tmp/agent-vault/.vaultengine/memory.sqlite",
  "numbers": {"source": "current"},
  "text_signal": {"available": true, "model_id": "potion-base-8M", "unavailable": null, "uncached": 0},
  "sections": [
    {
      "lens": "neighbourhood", "kind": "deepen", "title": "Neighbourhood",
      "what": "related notes not linked yet, at or above the percentile of this note's scores",
      "budget": 4, "found": 16, "hidden": 0, "duplicates": 2, "kept": 0, "dismissed": 0, "keep_score": 0.5,
      "applies": true, "note": null,
      "results": [
        {
          "ref": "1.1", "lens": "neighbourhood", "kind": "note", "target": "10_SOURCES/RLHF.md", "title": "RLHF",
          "evidence": "2 hops via direct-preference-optimization-dpo +1 · similar text (rlhf, reward, policy); score 0.74 (p100 of this note's scores)",
          "notes": ["10_SOURCES/RLHF.md"],
          "link": ["20_ZETTLEKASTEN/kl-penalty-in-rlhf.md", "10_SOURCES/RLHF.md"],
          "start_at": null, "mark": null
        }
      ]
    },
    "… 9 more sections"
  ]
}
```

Without a note, each result is a proposal with `start_at`:

```
{"ref": "1.1", "lens": "gap", "kind": "pair",
 "target": "gap: 20_ZETTLEKASTEN/fine-tuning-strategies-for-llms.md ↔ 20_ZETTLEKASTEN/importance-of-evaluation-in-ai-systems.md",
 "title": "communities 8 and 9: [[fine-tuning-strategies-for-llms]] · [[constitutional-ai]] / [[adas-mechanism]] · [[agentic-rag-approach]]",
 "evidence": "closeness of their mean embeddings 0.52 (p98 of 66 community pairs); 0 links between them where chance predicts 0.5; join fine-tuning-strategies-for-llms ↔ importance-of-evaluation-in-ai-systems (via 'adjustments', 'tuner', critic-tuner-approach (no note yet)) [meaning]",
 "notes": ["20_ZETTLEKASTEN/fine-tuning-strategies-for-llms.md", "20_ZETTLEKASTEN/importance-of-evaluation-in-ai-systems.md"],
 "link": ["20_ZETTLEKASTEN/fine-tuning-strategies-for-llms.md", "20_ZETTLEKASTEN/importance-of-evaluation-in-ai-systems.md"],
 "start_at": "fine-tuning-strategies-for-llms", "mark": null}
```

## `vault mark` / `explore.mark`

Params: `verdict` (`keep`, `dismiss`, `clear`), `refs` (of the last run). Result: `marked[]` (`ref`, `lens`, `target`,
`title`, `verdict`, `changed`: false for clearing a result that was not marked), `keep_rates[]` (`lens`, `kept`,
`dismissed`, over every start, for the lenses marked).

```
$ vault mark /tmp/agent-vault keep 1.1 --json
{"vault": "/tmp/agent-vault",
 "marked": [{"ref": "1.1", "lens": "neighbourhood", "target": "10_SOURCES/RLHF.md", "title": "RLHF", "verdict": "keep", "changed": true}],
 "keep_rates": [{"lens": "neighbourhood", "kept": 1, "dismissed": 0}]}
```

## `vault kept` / `explore.kept`

Params: `lens` (optional). Result: `memory`, `kept[]`, oldest first: `start` (null: no start), `lens`, `target`,
`kind`, `title`, `evidence` (as shown when kept), `link`, `at`.

```
$ vault kept /tmp/agent-vault --json
{"vault": "/tmp/agent-vault", "memory": "/tmp/agent-vault/.vaultengine/memory.sqlite",
 "kept": [{"start": "20_ZETTLEKASTEN/kl-penalty-in-rlhf.md", "lens": "neighbourhood", "target": "10_SOURCES/RLHF.md",
           "kind": "note", "title": "RLHF",
           "evidence": "2 hops via direct-preference-optimization-dpo +1 · similar text (rlhf, reward, policy); score 0.74 (p100 of this note's scores)",
           "link": ["20_ZETTLEKASTEN/kl-penalty-in-rlhf.md", "10_SOURCES/RLHF.md"], "at": "2026-11-10T17:55:20"}]}
```

## `vault link-it` / `explore.link` and `vault unlink` / `explore.unlink`

Params (RPC): `ref`, or `a` and `b`; `reason` (optional); `dry_run` (**default true**). `explore.unlink`: `a`, `b`,
`dry_run` (default true). They need a format (`-f brain`, `engine.format`, or `vault serve --format brain`).

The result is the write result every write command has, plus what link-it adds:

| field | meaning |
|---|---|
| `operation` | `"link-it"` or `"unlink"` |
| `status` | `"dry-run"`, `"written"`, `"refused"` (it would add a violation: see `new_violations`, nothing written), `"unchanged"` (already linked there, see `already`) |
| `dry_run`, `written` | what was asked; whether files changed |
| `files[]` | `path`, `action` (`modify`), `changes` (what was done, for people), `diff` (unified diff), `content` |
| `already` | parts already so |
| `new_violations`, `tolerated` | violations the write would add (then refused) / that were there already and do not block it |
| `index` | after a write: the index updated for the notes written |
| `recovered` | files restored from an interrupted earlier write |
| `ref`, `lens`, `notes`, `reason`, `kept` | link-it: the result it came from, the two notes, the line's reason, whether it counted as a keep |
| `recorded` | unlink: link-it made this link (its section goes too when nothing else is left in it) |

```
$ vault link-it /tmp/agent-vault 8.1 -f brain --json      # a dry run: one of the two files shown, its diff cut
{"vault": "/tmp/agent-vault", "operation": "link-it", "status": "dry-run", "dry_run": true, "written": false,
 "files": [{"path": "20_ZETTLEKASTEN/reward-computation-as-a-heavyweight-pipeline-stage.md", "action": "modify",
            "changes": ["new section # See Also at the end (the level of the note's last heading, at most 2): - [[persona-driven-synthetic-data-generation]] — gap: closeness of their mean embeddings 0.38; fewer links between their communities than chance predicts; shared: 'verifiable'…"],
            "diff": "--- a/20_ZETTLEKASTEN/reward-computation-as-a-heavyweight-pipeline-stage.md\n+++ b/…\n@@ -29,3 +29,6 @@\n …\n+\n+# See Also\n+- [[persona-driven-synthetic-data-generation]] — gap: …",
            "content": null}, "… 1 more file"],
 "already": [], "new_violations": [], "tolerated": ["…"], "index": null, "recovered": [], "ok": true,
 "ref": "8.1", "lens": "gap",
 "notes": ["20_ZETTLEKASTEN/reward-computation-as-a-heavyweight-pipeline-stage.md", "20_ZETTLEKASTEN/persona-driven-synthetic-data-generation.md"],
 "reason": "gap: closeness of their mean embeddings 0.38; fewer links between their communities than chance predicts; shared: 'verifiable', rlvr-reinforcement-learning-with-verifiable-rewards (no note yet)",
 "kept": false}
$ vault link-it /tmp/agent-vault 8.1 -f brain --write --json     # then: status "written", kept true, index updated for both notes
$ vault unlink /tmp/agent-vault reward-computation-as-a-heavyweight-pipeline-stage persona-driven-synthetic-data-generation -f brain --write --json
                                                                  # status "written", recorded true: both notes as they were
```

## `vault map` / `map.get` and `vault map --since` / `map.since`

`map.get` params: `limit` (items per list), `meaning`, `record`. Result: the map, as in
[The shape of the vault](explore.md): `communities[]` (`number`, `name`, `notes`, `phrases`, `hubs`, `topic_pages`),
`bridges[]`, `hubs[]`, `authorities[]`, the counts (`notes`, `missing_notes`, `tags`, `nodes`, `edges`,
`modularity`, …), and with `record` a `recorded` object: `at`, `bytes`, `memory`, `communities`, `gaps`, `basis`,
`missing`.

`map.since` params: `since` (a date or a time). Result: `snapshot` (when it was recorded) and `chosen` (how),
`notes`/`links`/`communities` as `[then, now]`, `gaps[]` (`communities` then, `names`, `links_then`, `expected_then`,
`links_now`, `expected_now`, `added` links between the two sides, `status` `open`/`closed`/`gone`, `why`),
`formed[]` and `gone[]` communities, `written[]` missing notes that are notes now, `topic_pages[]`.

```
$ vault map /tmp/agent-vault -f brain --since 2026-11-10 --json      # after linking the gap's pair; 3 open gaps left out
{"since": "2026-11-10", "snapshot": "2026-11-10T17:55:22", "chosen": "the first snapshot at or after 2026-11-10",
 "notes": [180, 180], "links": [439, 441], "communities": [12, 11], "formed": [],
 "gone": [{"community": 12, "name": "#prompt-engineering", "notes": 2, "overlap": 0.08333333333333333}],
 "gaps": [{"communities": [6, 12], "names": ["[[systems-engineering]] · [[reinforcement-learning-in-language-models]]", "#prompt-engineering"],
           "links_then": 0.0, "expected_then": 0.2345415778251599, "links_now": 2.0, "expected_now": 0.47558386411889597,
           "added": [["20_ZETTLEKASTEN/persona-driven-synthetic-data-generation.md", "20_ZETTLEKASTEN/reward-computation-as-a-heavyweight-pipeline-stage.md"]],
           "status": "closed", "why": "2 links between the two sides where chance predicts 0.5: no longer fewer than chance"}, "…"],
 "gap_basis": "meaning", "written": [], "topic_pages": ["…"]}
```

## `vault deepen` / `note.deepen`, `vault discover` / `vault.discover`, `vault explain` / `notes.explain`

The lenses explore runs, each on its own, with every number behind them (explore shows the results; these show the
cutoffs and distributions too). `note.deepen` params: `note`, `lenses`, `limit`; result `note`, `percentile`,
`lenses.{neighbourhood, cited, foundations, backbone, cluster}`. `vault.discover` params: `note` (null: the vault-wide
lenses), `lenses`, `limit`, `steps`, `seed`; result `note`, `percentile`, `seed`, `text_signal`,
`lenses.{hidden, role, bridge, gap, surprise, unwritten, wander}` (gap, surprise and unwritten have
`with_embeddings` and `without_embeddings`, each the lens's answer or `{"unavailable": why}`). `notes.explain` params:
`a`, `b`, `words`; result `a`, `b`, `score` (the sum of the contributions), `signals[]` (`name`, `value`, `weight`,
`contribution`, `available`, `unavailable`, `evidence` by signal: the link lines, the notes citing both, what both link
to, the tags, the words behind the text score), `configured_weights`, `model_id`, `uncached`.

```
$ vault discover /tmp/agent-vault reward-hacking-as-specification-gaming -f brain --limit 1 --json    # its first hidden link, middles cut
{"note": "20_ZETTLEKASTEN/reinforcement-learning-in-language-models.md", "score": 4.566860051645819, "percentile": 97.79859484777518,
 "middles": [{"middle": "the-critic-deletion-tension-in-agentic-rl", "kind": "note", "notes": 3, "weight": 0.7213475204444817}, "… 8 more"],
 "reason": "no link either way; 9 middles in common: the-critic-deletion-tension-in-agentic-rl (3 notes), … +5; score 4.57 (p98 of 2135 unlinked pairs)"}
```

## `vault embed --status` / `embed.status`, `vault config show` / `config.show`

`embed.status`: `model_id`, `notes`, `with_vector`, `missing` (and `missing_paths`: run `vault embed` to cache them;
explore embeds them in memory meanwhile), `dim`, `other_models`, `cache_vectors`, `cache_bytes`, `provider`,
`model_path`, `model_found`. `config.show`: `config_file`, `config_found_by`, `precedence`, `settings[]` (`key`,
`value`, `source`: default, file, env NAME or flag, `env`: the variable that sets it).

```
$ vault embed /tmp/agent-vault --status --json      # after the link: the two edited notes have no cached vector yet
{"model_id": "potion-base-8M", "notes": 180, "with_vector": 178, "missing": 2,
 "missing_paths": ["20_ZETTLEKASTEN/persona-driven-synthetic-data-generation.md", "20_ZETTLEKASTEN/reward-computation-as-a-heavyweight-pipeline-stage.md"],
 "dim": 256, "other_models": [], "cache_vectors": 178, "cache_bytes": 182272, "provider": "model2vec",
 "model_path": "/workspace/zettelkasten-auto/models/potion-base-8M", "model_found": true, "…": "vault and index"}
$ vault config show /tmp/agent-vault --json
{"config_file": "/workspace/zettelkasten-auto/config/config.yaml", "config_found_by": "repository default",
 "precedence": ["flag", "env", "file", "default"],
 "settings": [{"key": "vault", "value": "/tmp/agent-vault", "source": "flag <vault>", "env": "VAULTENGINE_VAULT"}, "…"]}
```
