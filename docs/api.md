# The JSON-RPC API

[← Documentation index](../README.md#documentation)

The examples use `/tmp/sample-vault` from [Getting started](getting-started.md) (a copy of the sample vault, indexed).

## The protocol

`vault serve` reads one JSON request per line on stdin and writes one response per line on stdout
(methods: `index.build`, `index.status`, `vault.stats`, `health.report`, `notes.pick`, `note.get`, and with `--format`
the [format methods](#format-methods); see `vault rpc-schema`).

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"notes.pick","params":{"query":"klpen","limit":1}}' \
  '{"jsonrpc":"2.0","id":2,"method":"notes.delete"}' \
  '{"jsonrpc":"1.0","id":3,"method":"index.status"}' \
  | vault serve /tmp/sample-vault
```
```
{"jsonrpc":"2.0","id":1,"result":{"query":"klpen","total":3,"results":[{"path":"20_ZETTLEKASTEN/kl-penalty-in-rlhf.md","title":"kl-penalty-in-rlhf","matched_field":"title","matched_text":"kl-penalty-in-rlhf","score":141,"positions":[0,1,3,4,5]}]}}
{"jsonrpc":"2.0","id":2,"error":{"code":-32601,"message":"Method not found","data":{"method":"notes.delete","available":["health.report","index.build","index.status","note.get","note.ingest","note.link","note.new","note.set","notes.pick","vault.check","vault.coverage","vault.fix","vault.stats"]}}}
{"jsonrpc":"2.0","id":3,"error":{"code":-32600,"message":"Invalid Request","data":[{"loc":["jsonrpc"],"msg":"Input should be '2.0'","type":"literal_error"}]}}
```
On stderr it also prints `vaultengine: serving /tmp/sample-vault (index …) on stdio`.

Started with `vault serve <vault> --format brain`, the server also answers the format methods, the same operations as the
CLI commands, with the same checks and the same safe write path:

| Method | Does | Params (all optional unless marked) |
|---|---|---|
| `vault.check` | violations per note, with lines | `changed`, `limit` (notes listed) |
| `vault.fix` | the safe repairs; `dry_run` returns the diffs | `dry_run`, `only` (`missing-mirror` / `missing-heading`) |
| `vault.coverage` | sources nothing links to, notes citing no source | `limit` |
| `note.new` | create a note of a type | **`type`**, **`title`**, `folder`, `name`, `values`, `relations` (`[{relation, note}]`), `sections`, `dry_run` |
| `note.link` | relate A to B, mirror in B | **`source`**, **`target`**, **`relation`**, `dry_run` |
| `note.set` | set a metadata value | **`note`**, **`key`**, **`value`**, `dry_run` |
| `note.ingest` | a document into a note, stopped by the duplicate gate | **`path`** or **`content`** + **`filename`**, `type` (default `source`), `title`, `folder`, `values`, `dry_run` |

Every write returns `status` (`written`, `refused`, `dry-run`, `unchanged`; `duplicate` for `note.ingest`), the files with
their `changes`, a unified `diff` per file and the whole `content` of a created file, the `new_violations` that made a write
refused, the violations that were already there (`tolerated`), and how the index was updated (`index`). A refused write or a
duplicate is a result, not an error. Errors: `-32003` no format (or an invalid format file; it is reloaded when it changes),
`-32004` write rejected (bad arguments, name taken; nothing written), `-32005` write failed and was rolled back, `-32002`
a note named in the request does not exist.

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"note.set","params":{"note":"CLM - Grade the observed end state rather than the trajectory","key":"Status","value":"#settled","dry_run":true}}' \
  | vault serve /tmp/sample-vault --format brain 2>/dev/null | python3 -c "import sys, json; r = json.load(sys.stdin)['result']; print(r['status'], [v['message'] for v in r['new_violations']])"
```
```
refused ['Status: #settled is not an allowed value (supported, contested, weak)']
```

The server keeps parsed notes between calls and re-parses only notes whose content changed, so after the first call
`vault.check` takes about 0.1 s on the sample vault and a write about 0.3 s.

## Format methods

`vault rpc-schema` prints the JSON Schema of every method (also committed as
`vaultengine/rpc/schema.json`; the app's TypeScript types are generated from it with `cd tui && npm run gen:types`).

Started with `vault serve <vault> --format brain`, the server also answers the format methods, the same operations as the
CLI commands, with the same checks and the same safe write path:

| Method | Does | Params (all optional unless marked) |
|---|---|---|
| `vault.check` | violations per note, with lines | `changed`, `limit` (notes listed) |
| `vault.fix` | the safe repairs; `dry_run` returns the diffs | `dry_run`, `only` (`missing-mirror` / `missing-heading`) |
| `vault.coverage` | sources nothing links to, notes citing no source | `limit` |
| `note.new` | create a note of a type | **`type`**, **`title`**, `folder`, `name`, `values`, `relations` (`[{relation, note}]`), `sections`, `dry_run` |
| `note.link` | relate A to B, mirror in B | **`source`**, **`target`**, **`relation`**, `dry_run` |
| `note.set` | set a metadata value | **`note`**, **`key`**, **`value`**, `dry_run` |
| `note.ingest` | a document into a note, stopped by the duplicate gate | **`path`** or **`content`** + **`filename`**, `type` (default `source`), `title`, `folder`, `values`, `dry_run` |

Every write returns `status` (`written`, `refused`, `dry-run`, `unchanged`; `duplicate` for `note.ingest`), the files with
their `changes`, a unified `diff` per file and the whole `content` of a created file, the `new_violations` that made a write
refused, the violations that were already there (`tolerated`), and how the index was updated (`index`). A refused write or a
duplicate is a result, not an error. Errors: `-32003` no format (or an invalid format file; it is reloaded when it changes),
`-32004` write rejected (bad arguments, name taken; nothing written), `-32005` write failed and was rolled back, `-32002`
a note named in the request does not exist.

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"note.set","params":{"note":"CLM - Grade the observed end state rather than the trajectory","key":"Status","value":"#settled","dry_run":true}}' \
  | vault serve /tmp/sample-vault --format brain 2>/dev/null | python3 -c "import sys, json; r = json.load(sys.stdin)['result']; print(r['status'], [v['message'] for v in r['new_violations']])"
```
```
refused ['Status: #settled is not an allowed value (supported, contested, weak)']
```

The server keeps parsed notes between calls and re-parses only notes whose content changed, so after the first call
`vault.check` takes about 0.1 s on the sample vault and a write about 0.3 s.
