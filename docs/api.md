# The JSON-RPC API

[← Documentation index](../README.md#documentation)

The examples use `/tmp/sample-vault` from [Getting started](getting-started.md) (a copy of the sample vault, indexed).

## The protocol

`vault serve` reads one JSON request per line on stdin and writes one response per line on stdout
(methods: `index.build`, `index.status`, `vault.stats`, `health.report`, `notes.pick`, `note.get`, the
[search and connection methods](#search-and-connection-methods), and with `--format` the [format methods](#format-methods)
and the [reading-workflow methods](#reading-workflow-methods); see `vault rpc-schema`).

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"notes.pick","params":{"query":"klpen","limit":1}}' \
  '{"jsonrpc":"2.0","id":2,"method":"notes.delete"}' \
  '{"jsonrpc":"1.0","id":3,"method":"index.status"}' \
  | vault serve /tmp/sample-vault
```
```
{"jsonrpc":"2.0","id":1,"result":{"query":"klpen","total":3,"results":[{"path":"20_ZETTLEKASTEN/kl-penalty-in-rlhf.md","title":"kl-penalty-in-rlhf","matched_field":"title","matched_text":"kl-penalty-in-rlhf","score":141,"positions":[0,1,3,4,5]}]}}
{"jsonrpc":"2.0","id":2,"error":{"code":-32601,"message":"Method not found","data":{"method":"notes.delete","available":["findings.list","health.report","idea.new","idea.set","index.build","index.status","links.suggest","mentions.unlinked","note.facts","note.get","note.ingest","note.link","note.merge","note.move","note.new","note.preview","note.related","note.set","notes.pick","notes.suggest","page.check","page.exclude","page.new","page.refresh","search.query","vault.check","vault.coverage","vault.fix","vault.next","vault.pipeline","vault.stats"]}}}
{"jsonrpc":"2.0","id":3,"error":{"code":-32600,"message":"Invalid Request","data":[{"loc":["jsonrpc"],"msg":"Input should be '2.0'","type":"literal_error"}]}}
```
On stderr it also prints `vaultengine: serving /tmp/sample-vault (index …) on stdio`.

## Format methods

`vault rpc-schema` prints the JSON Schema of every method (also committed as
`vaultengine/rpc/schema.json`; the app's TypeScript types are generated from it with `cd tui && npm run gen:types`).

Started with `vault serve <vault> --format brain`, the server also answers the format methods, the same operations as the
CLI commands, with the same checks and the same safe write path:

| Method | Does | Params (all optional unless marked) |
|---|---|---|
| `vault.check` | violations per note, with lines | `changed`, `limit` (notes listed), `baseline`, `update_baseline` (see [below](#reading-workflow-methods)) |
| `vault.fix` | the safe repairs; `dry_run` returns the diffs | `dry_run`, `only` (`missing-mirror` / `missing-heading`) |
| `vault.coverage` | sources nothing links to, notes citing no source | `limit` |
| `note.new` | create a note of a type | **`type`**, **`title`**, `folder`, `name`, `values`, `relations` (`[{relation, note}]`), `sections`, `dry_run` |
| `note.link` | relate A to B, mirror in B | **`source`**, **`target`**, **`relation`**, `dry_run` |
| `note.set` | set a metadata value | **`note`**, **`key`**, **`value`**, `dry_run` |
| `note.ingest` | a document into a note, stopped by the duplicate gate | **`path`** or **`content`** + **`filename`**, `type` (default `source`), `title`, `folder`, `values`, `dry_run` |

Every write returns `status` (`written`, `refused`, `dry-run`, `unchanged`; `duplicate` for `note.ingest`), the files with
their `changes`, a unified `diff` per file and the whole `content` of a created file, the `new_violations` that made a write
refused, the violations that were already there (`tolerated`), and how the index was updated (`index`). A refused write or a
duplicate is a result, not an error. Errors: `-32003` no format (or an invalid format file; it is reloaded when it changes;
or the format lacks a block the method needs, such as `pipeline:`), `-32004` write rejected (bad arguments, name taken;
nothing written), `-32005` write failed and was rolled back, `-32002` a note (or finding) named in the request does not
exist, `-32008` `vault.check` with `baseline` and no baseline recorded.

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"note.set","params":{"note":"CLM - Grade the observed end state rather than the trajectory","key":"Status","value":"#settled","dry_run":true}}' \
  | vault serve /tmp/sample-vault --format brain 2>/dev/null | python3 -c "import sys, json; r = json.load(sys.stdin)['result']; print(r['status'], [v['message'] for v in r['new_violations']])"
```
```
refused ['Status: #settled is not an allowed value (supported, contested, weak)']
```

The server keeps parsed notes between calls and re-parses only notes whose content changed, so after the first call
`vault.check` takes about 0.1 s on the sample vault and a write about 0.3 s.

## Reading-workflow methods

Every command of [the reading workflow](workflow.md), with the same checks and results as the CLI's `--json` (less its
`vault` and `ok` keys; the recorded session `tests/fixtures/rpc/sample_vault_core.jsonl` has one of each, and a test
compares them with the CLI). Served with a format that has the blocks they use (`vault serve <vault> --format core`):

| Method | CLI | Params (all optional unless marked) |
|---|---|---|
| `vault.check` | `vault check [--baseline \| --update-baseline]` | `baseline` (only violations not in the baseline count; `-32008` when none is recorded), `update_baseline` (record it; the only write of a check), `changed`, `limit`. The result's `baseline` has `violations`, `new`, `known`, `fixed`, `renames` |
| `vault.pipeline` | `vault pipeline` | `limit` (rows per list) |
| `note.facts` | `vault note` | **`note`** |
| `page.new` | `vault page new` | **`title`**, `zettels` (names or paths), `queries` (search queries; their best-ranked zettels), `limit` (per query), `tags`, `folder`, `dry_run`. One of `zettels`, `queries` is needed |
| `page.refresh` | `vault page refresh` | **`page`**, `dry_run` |
| `page.exclude` | `vault page exclude` | **`page`**, **`zettels`**, `undo`, `dry_run` |
| `page.check` | `vault page check` | `page` (default: every page). `ready` is the CLI's exit code |
| `findings.list` | `vault findings` | `methods`, `id` (`-32002` when there is no such finding now), `limit` (per method) |
| `idea.new` | `vault idea new` | **`finding`** (an id), `title`, `tags`, `dry_run` |
| `idea.set` | `vault idea set` | **`idea`**, **`status`**, `dry_run` |
| `note.move` | `vault move` | **`note`**, **`to`**, `dry_run` |
| `note.merge` | `vault merge` | **`survivor`**, **`other`**, `dry_run`. `self_links`: the links dropped because they would point at the survivor itself |
| `vault.next` | `vault next` | `limit` (default 10) |

The writes return the write result above plus `state` (what they record in `.vaultengine/`). `page.new` adds `page` and
`from`, `idea.new` adds `finding` and `idea`, and `note.move`/`note.merge` add `moved`, `rewritten`, `now_resolving` and
`self_links`. Unlike the CLI's `page`, `idea`, `move` and `merge`, which need `--write`, these methods write unless
`"dry_run": true`, like every write method of the API: send the dry run first.

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"vault.check","params":{"baseline":true}}' \
  '{"jsonrpc":"2.0","id":2,"method":"vault.check","params":{"update_baseline":true,"limit":1}}' \
  '{"jsonrpc":"2.0","id":3,"method":"vault.check","params":{"baseline":true}}' \
  '{"jsonrpc":"2.0","id":4,"method":"findings.list","params":{"methods":["link"],"limit":1}}' \
  '{"jsonrpc":"2.0","id":5,"method":"note.merge","params":{"survivor":"enterprise-agent-evaluation-challenges","other":"pass-hat-k-reliability-metric","dry_run":true}}' \
  '{"jsonrpc":"2.0","id":6,"method":"vault.next","params":{"limit":1}}' \
  | vault serve /tmp/api-vault --format core 2>/dev/null
```

(`/tmp/api-vault`: a fresh copy of the sample vault, indexed.) Shortened to the fields that matter here:

```
1 error -32008 No baseline for format 'core'; record one with vault.check update_baseline=true
2 ok True new 0 known 144
3 ok True new 0 known 144
4 13 {'link': {'score': 0.2969, 'percentile': 99, 'floor': 0.1}} link-6b21be7696
5 dry-run [{'path': '20_ZETTLEKASTEN/enterprise-agent-evaluation-challenges.md', 'line': 31, 'link': '[[pass-hat-k-reliability-metric]]', 'action': 'line removed', 'result': ''}]
6 99 write write [[transcript-spotting-heuristics-for-reward-hacking]]
```

## Search and connection methods

`search.query`, `note.preview`, `note.related`, `links.suggest`, `notes.suggest` and `mentions.unlinked` back `vault search`, `related`, `suggest` and `unlinked`. A query syntax error is `-32006` (invalid query), with `data.start`/`data.end` offsets and a `hint`. `note.related` and `links.suggest` use the embedding model (`engine.embedding` in [the settings](commands.md#settings-vault-config-show-vault---db-f---format-f---setting-kv---config-file---json)) for their text signal; when it cannot be loaded they still answer from links and tags, and their `text_signal` says so (`available: false` and why); `weights` gives the weights used. Vectors the index lacks are computed for the answer and never written (`text_signal.uncached` counts them). Examples and result shapes: [Search and connections](search.md#over-json-rpc).

## Explore methods

`explore.run`, `explore.mark`, `explore.kept`, `explore.link`, `explore.unlink`, `map.get`, `map.since`,
`note.deepen`, `vault.discover`, `notes.explain`, `embed.status` and `config.show` back `vault explore`, `mark`,
`kept`, `link-it`, `unlink`, `map` (`--record`, `--since`), `deepen`, `discover`, `explain`, `embed --status` and
`config show`, and return exactly what those print with `--json` (without the `vault` and `index` around it): one
function builds both. Every field, with an example each and how an assistant should use them: [For
assistants](agents.md).

| method | what | params (bold = required) |
|---|---|---|
| `explore.run` | a section per lens from a note, or where to start; remembered for mark and link | `note`, `lenses`, `per_lens` (3–5), `fresh`, `record` (default true), `refresh` |
| `explore.mark` | keep, dismiss or clear results of the last run | **`verdict`**, **`refs`** |
| `explore.kept` | what was kept, with lens and evidence | `lens` |
| `explore.link` | a result (or two notes) made a See Also link in both notes; needs `--format` | `ref` or `a` + `b`, `reason`, `dry_run` (**default true**) |
| `explore.unlink` | undo it | **`a`**, **`b`**, `dry_run` (**default true**) |
| `map.get` | the map; `record` saves a snapshot | `limit`, `meaning`, `record` |
| `map.since` | what changed since a snapshot | **`since`** |
| `note.deepen` | the five deepen lenses with their numbers | **`note`**, `lenses`, `limit` |
| `vault.discover` | the seven discover lenses with their numbers | `note`, `lenses`, `limit`, `steps`, `seed` |
| `notes.explain` | how two notes relate, signal by signal | **`a`**, **`b`**, `words` |
| `embed.status` | cached vectors for the configured model | |
| `config.show` | every engine setting and its source | |

`-32007` (not available) is what there is nothing to act on: no explore run yet, a ref that is not in the last run, a
result that is not two notes to link, no map snapshot, a lens that does not run from this start; `data.hint` says
what to do. The recorded session `tests/fixtures/rpc/sample_vault_explore.jsonl` (replayed through `vault serve` by
`tests/test_vaultengine_rpc.py`) calls every one of them on the sample vault: explore twice, keep, dismiss, a ref
that is not there, kept, map with a snapshot, link it (dry run, then written), explore with fresh, map since (the
gap closed), unlink (dry run, then written), pivot, deepen, discover, explain, and the errors.

