# Zettelkasten Automation

Convert substantive source notes into atomic Zettels, keep the knowledge graph local, and assemble exportable bundles around a source or note.

This repository has two parts:

- **Vault engine, `vault` CLI and `vault-tui`** (no LLM): read an Obsidian vault the way Obsidian does (frontmatter, tags, wikilinks, embeds, headings, blocks), resolve links, keep an incremental SQLite index, and work with it from the command line, over JSON-RPC, or in a Claude Code–style terminal app. See [Vault engine, CLI and terminal app](#vault-engine-cli-and-terminal-app).
- **`zettel`**: the original LLM tool that turns source notes into Zettels. Documented from [What The Tool Does](#what-the-tool-does) on.

## Vault engine, CLI and terminal app

| Piece | What it is | Run it |
|---|---|---|
| `vaultengine/` | Python package: parser, link resolver, link graph, SQLite index, stats/health | `from vaultengine import Vault, LinkGraph` |
| `vault` | CLI on top of the engine | `vault index|stats|health <vault>` |
| `vault serve` | the engine as a JSON-RPC 2.0 server on stdio (schema: `vaultengine/rpc/schema.json`) | `vault serve <vault>` |
| `tui/` → `vault-tui` | terminal app (TypeScript + Ink) that starts `vault serve` and talks to it | `vault-tui --vault <vault>` |

### Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Node.js 22 or newer (for `vault-tui`)
- Only for the terminal end-to-end test: a C/C++ toolchain for `node-pty` (Linux: `build-essential` and `python3`; macOS uses a prebuilt binary)

### Install

From the repository root:

```bash
uv sync                          # Python env in .venv (engine, CLI, tests, and the zettel tool)
source .venv/bin/activate        # puts `vault` on PATH
cd tui && npm ci && npm run build && npm link && cd ..   # puts `vault-tui` on PATH
```

`uv sync` also installs `torch`/`sentence-transformers` for the zettel tool's embeddings (several GB on Linux). The engine, CLI and app don't need them.

### Try it on the sample vault

Work on a copy, so no index is written into the test fixtures (`-p` keeps file times, which keeps the outputs below exact):

```bash
cp -Rp tests/fixtures/vault /tmp/sample-vault
```

In the outputs below, timings, timestamps and the index date differ on your machine; everything else should match.
The sample vault's file names `ΓÇö` and `┬╖` are really `—` and `·`, garbled when the vault was zipped.
That is why those two notes show up as orphans: links to them use the real characters.

#### 1. Build the index

```bash
vault index /tmp/sample-vault
```
```
Indexed /tmp/sample-vault in 1.16s
  index      /tmp/sample-vault/.vaultengine/index.sqlite
  files      179 notes, 0 attachments
  parsed     179  (179 added, 0 changed, 0 deleted)
  read       179 notes hashed, 0 skipped by size/mtime
  links      1607 resolved this run, 0 changed status in notes that were not re-parsed
    + 10_SOURCES/2310.11667 - SOTOPIA Interactive Evaluation for Social Intelligence (COMPREHENSIVE + NETLOGO).md
    …
  … 169 more (--limit 0 shows all)
```

Run it again: nothing is read or parsed.

```bash
vault index /tmp/sample-vault
```
```
Indexed /tmp/sample-vault in 0.02s
  index      /tmp/sample-vault/.vaultengine/index.sqlite
  files      179 notes, 0 attachments
  parsed     0  (index already up to date)
  read       0 notes hashed, 179 skipped by size/mtime
  links      0 resolved this run, 0 changed status in notes that were not re-parsed
```

Edit one note and index again: exactly that note is parsed.

```bash
echo "Edited." >> "/tmp/sample-vault/20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"
vault index /tmp/sample-vault
```
```
  parsed     1  (0 added, 1 changed, 0 deleted)
  read       1 note hashed, 178 skipped by size/mtime
  links      11 resolved this run, 0 changed status in notes that were not re-parsed
    ~ 20_ZETTLEKASTEN/kl-penalty-in-rlhf.md
```

#### 2. Stats and health

```bash
vault stats /tmp/sample-vault --top 3
```
```
Vault stats: /tmp/sample-vault
  index 2026-11-06T03:43:23+00:00 (up to date)

  notes             179   (1 with frontmatter, 0 frontmatter errors)
  attachments         0
  links            1607   wikilinks 1607 · markdown 0 · embeds 3
    resolved        552   unresolved 1055 (581 distinct targets) · ambiguous 0
  note edges        519   distinct note→note connections
  tags              221   uses of 29 distinct tags
  properties          4   distinct keys
  aliases             0
  headings         1499
  blocks              0

Notes per top-level folder
     70  20_ZETTLEKASTEN
     67  50_MOC
     42  10_SOURCES

Top tags (notes)
    100  #baby
     48  #evidence
     11  #supported
…
Most wanted missing notes (linking notes / links)
   63 / 63  rl-environment-design
   62 / 62  discourse-graph
   11 / 15  2406.12045 - Tau-Bench (Tool-Agent-User Interaction)
```

```bash
vault health /tmp/sample-vault --limit 3
```
```
Vault health: /tmp/sample-vault
  index 2026-11-06T03:43:23+00:00 (up to date)

  unresolved links                      1055   across 581 missing targets
  ambiguous links                          0
  missing heading/block targets            0
  orphans (no backlinks)                   7
  dead ends (no outgoing note links)      22
  duplicate basenames                      0
  empty notes                              0

Unresolved link targets (581)
  rl-environment-design  — 63 links from 63 notes, e.g. 50_MOC/discourse/rl-environment-design/CLM - A fixed typed contract should separate environment from trainer.md:3 (+62 more notes)
  …
```

Add `--json` to `index`, `stats` or `health` for machine-readable output, e.g.
`vault health /tmp/sample-vault --json | python3 -c "import sys, json; print(json.load(sys.stdin)['summary'])"` prints
`{'unresolved_links': 1055, 'unresolved_targets': 581, 'ambiguous_links': 0, 'missing_subpaths': 0, 'orphans': 7, 'dead_ends': 22, 'duplicate_basenames': 0, 'empty_notes': 0}`.

If files change after indexing, `stats` and `health` still answer from the index and print a warning on stderr:
``warning: index is out of date (1 added, 0 changed, 0 deleted); run `vault index` to refresh``.

#### 3. The engine as a Python library

```bash
python - <<'EOF'
from vaultengine import LinkGraph, Vault

vault = Vault.load("/tmp/sample-vault")
graph = LinkGraph(vault)
print(len(vault.notes), "notes,", len(graph.edges), "links,", sum(e.status == "resolved" for e in graph.edges), "resolved")
note = "20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"
print("tags:", vault.notes[note].tag_names)
print("backlinks:", graph.backlink_sources(note))
EOF
```
```
179 notes, 1607 links, 552 resolved
tags: ['baby']
backlinks: ['20_ZETTLEKASTEN/1707.06347 - Proximal Policy Optimization Algorithms.md']
```

#### 4. The engine as a JSON-RPC 2.0 server

`vault serve` reads one JSON request per line on stdin and writes one response per line on stdout
(methods: `index.build`, `index.status`, `vault.stats`, `health.report`, `notes.pick`, `note.get`; see `vault rpc-schema`).

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"notes.pick","params":{"query":"klpen","limit":1}}' \
  '{"jsonrpc":"2.0","id":2,"method":"notes.delete"}' \
  '{"jsonrpc":"1.0","id":3,"method":"index.status"}' \
  | vault serve /tmp/sample-vault
```
```
{"jsonrpc":"2.0","id":1,"result":{"query":"klpen","total":7,"results":[{"path":"20_ZETTLEKASTEN/kl-penalty-in-rlhf.md","title":"kl-penalty-in-rlhf","matched_field":"title","matched_text":"kl-penalty-in-rlhf","score":141,"positions":[0,1,3,4,5]}]}}
{"jsonrpc":"2.0","id":2,"error":{"code":-32601,"message":"Method not found","data":{"method":"notes.delete","available":["health.report","index.build","index.status","note.get","notes.pick","vault.stats"]}}}
{"jsonrpc":"2.0","id":3,"error":{"code":-32600,"message":"Invalid Request","data":[{"loc":["jsonrpc"],"msg":"Input should be '2.0'","type":"literal_error"}]}}
```
On stderr it also prints `vaultengine: serving /tmp/sample-vault (index …) on stdio`.

#### 5. The terminal app

```bash
vault-tui --vault /tmp/sample-vault
```

A banner appears, the index is brought up to date, and the status line reads
`◆ sample-vault  ·  179 notes · 1,607 links  ·  ● index fresh`. Then try:

| Type | You should see |
|---|---|
| `/health` Enter | `⏺ Vault health` with `✗ 1,055 unresolved links (581 missing targets)`, `! 7 orphans (no backlinks)`, `! 22 dead ends …`, `✓ 0 empty notes` and the top 5 of each list |
| `/stats` Enter | `⏺ Vault stats` with `179 notes · 0 attachments · 1 with frontmatter` and top tags / most linked / most wanted |
| `/note @klpen`, then Enter twice | the picker shows `❯ kl-penalty-in-rlhf  20_ZETTLEKASTEN`; the first Enter inserts it, the second shows the note: `Links out (10: 6 resolved)`, `Links in (1)` |
| `/` | the list of commands; Tab completes, Enter runs the highlighted one |
| ↑ | the previous command |
| `/help` | all commands and keys |
| ctrl+c | quits and stops the Python server |

Options: `--python <interpreter>` (default: `$VAULT_TUI_PYTHON`, then this repo's `.venv`, then `python3`), `--db <index file>`, `--no-index`.
`vault-tui` needs an interactive terminal. Details: [tui/README.md](tui/README.md).

### Tests and expected results

| Command (from the repo root) | Prints at the end |
|---|---|
| `python -m pytest -q` | `183 passed, 17 warnings, 125 subtests passed` (the 17 warnings come from chromadb in the legacy `test_integration.py`) |
| `python -m unittest discover -s tests` | `Ran 183 tests` … `OK` |
| `cd tui && npm test` | `src/rpc/schema.ts is up to date`, then `Test Files  3 passed (3)` and `Tests  41 passed (41)` |
| `cd tui && npm run typecheck` | no errors (exit code 0) |
| `cd tui && npm run test:e2e` | builds, then `✓ /health on the sample vault` and `Tests  2 passed (2)` |

What they cover:

- `tests/test_vaultengine_parser.py` (64): parser: frontmatter, tags, links, embeds, headings, blocks, code
- `tests/test_vaultengine_resolver.py` (55): link resolution and the link graph
- `tests/test_vaultengine_index.py` (29): SQLite index, incremental updates, CLI
- `tests/test_vaultengine_rpc.py` (28): JSON-RPC API: recorded request/response sessions replayed against a real `vault serve`, error codes, JSON Schema
- `tests/test_integration.py` (7): the original zettel tool
- `tui/test/units.test.ts` (14), `tui/test/app.test.tsx` (22 frame tests), `tui/test/integration.test.tsx` (5, real Python server)
- `tui/test/e2e/health.pty.test.ts` (2): the built `vault-tui` in a real pseudo-terminal (node-pty): runs `/health`, checks the report, quits, checks the server is gone

### All checks in one command

```bash
scripts/verify.sh
```

It runs the Python tests, the CLI and JSON-RPC checks on a scratch copy of the sample vault (asserting the key lines shown above),
the TUI tests, the type check and the end-to-end test, and ends with `ALL CHECKS PASSED`. It stops at the first mismatch and
prints the expected line and the actual output.

## What The Tool Does

- Processes markdown source notes from a configured source folder.
- Extracts reusable concepts with a structured LLM response.
- Writes new Zettels from a template.
- Stores semantic embeddings in a local ChromaDB collection scoped to the active embedding provider and model.
- Builds context bundles around a source or note.

The tool is intentionally CLI-first and local-first. Source files are never deleted. Successful processing marks them as processed through a `Status: #processed` marker.

## Installation

Using `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
pip install -e .
```

## Configuration

1. Copy `.env.example` to `.env`.
2. Add the API key(s) for the providers you plan to use.
3. Edit `config/config.yaml`.

Portable example:

```yaml
vault:
  root: "/path/to/your/obsidian/vault"
  sources_dir: "10_SOURCES"
  zettel_dir: "20_ZETTLEKASTEN"
  bundles_dir: "110_BUNDLES"
  template_path: "60_TEMPLATES/Note Template.md"

llm:
  provider: "anthropic"  # "openai", "ollama", "gemini", or "anthropic"
  model: "claude-sonnet-4-5"  # Model name (provider-specific)
  base_url: "http://localhost:11434"  # Only for Ollama
  # OpenAI models: gpt-4o, gpt-4o-mini
  # Ollama models: llama3.1:8b, mistral:7b, qwen2.5:7b
  # Gemini models: gemini-1.5-flash, gemini-1.5-pro
  # Anthropic models: claude-sonnet-4-5, claude-haiku-4-5, claude-opus-4-6

embeddings:
  db_path: "chromastores/zettledb"
  provider: "sentence-transformer"  # "openai" or "sentence-transformer"
  model: "all-MiniLM-L6-v2"
  top_k: 5
  max_distance: 0.5

bundle:
  default_depth: 1
  default_top_k: 10

processing:
  min_source_chars: 100
  duplicate_max_distance: 0.25
  chunk_target_chars: 12000
  chunk_max_chars: 16000
```

`template_path` should point to a markdown template containing these anchors:

- `Tags:`
- `# {{Title}}`
- `# References`

Including `# See Also` in the template is recommended but optional. The writer inserts it if it is missing.

## CLI

Show help:

```bash
zettel --help
```

Main commands:

```bash
zettel process [--dry-run] [--limit N]
zettel rebuild-index [--dry-run]
zettel bundle --source SOURCE [--depth N] [--top-k N] [--dry-run]
zettel bundle --note "Note Title" [--depth N] [--dry-run]
```

Legacy maintenance commands:

```bash
zettel normalize-links [--dry-run]
zettel rename-files [--dry-run]
zettel sync-backlinks [--dry-run]
```

## Recommended Workflow

1. Put substantive, reusable source notes into `sources_dir`.
2. Preview extraction:

```bash
zettel process --dry-run --limit 5
```

3. Run the real ingestion:

```bash
zettel process --limit 5
```

4. Rebuild the active embedding collection whenever notes are renamed, merged, or heavily edited by hand:

```bash
zettel rebuild-index
```

5. Generate a context bundle:

```bash
zettel bundle --source source-file-name
zettel bundle --note "core-idea"
```

`bundle --dry-run` is read-only and, for source bundles, requires a fresh index. If the index is stale it exits with a clear error instead of silently using inaccurate semantic results.

## How Processing Works

For each source note, the tool:

1. Skips notes already marked as processed through `Status: #processed` or a legacy terminal `#processed` line.
2. Skips notes shorter than `processing.min_source_chars`.
3. Splits long sources by markdown headings and then paragraphs.
4. Runs extraction per chunk and validates the provider output.
5. Deduplicates candidate concepts by normalized title and semantic similarity.
6. Writes new Zettels and indexes them in the active embedding collection.
7. Marks the source as processed when the note was successfully handled or intentionally skipped as non-extractable.

Per-source outcomes are summarized with categories such as `processed`, `too_short`, `non_extractable`, `provider_error`, `invalid_output`, and `no_valid_concepts`.

## Note Format

Newly generated notes separate semantic relationships from aboutness:

```markdown
Created: DD-MM-YYYY HH:MM

Tags: [[alpha-tag]] [[topic-label]]

# Concept Title

Concept content...

# See Also

- [[related-idea]]
- [[another-note]]

# References

1. [[Source File Name]]
```

- `Tags:` contains only the extracted aboutness links.
- `# See Also` contains semantically similar notes.
- Bundle traversal follows `Tags` beyond degree `0`; it does not recursively follow `See Also`.

## Project Structure

```text
zettelkasten-auto/
├── vaultengine/             # Vault engine (no LLM): parser, resolver, graph, SQLite index
│   ├── parser.py            # Obsidian-flavoured markdown parsing
│   ├── resolver.py          # Link resolution rules
│   ├── graph.py             # Link graph and backlinks (in memory)
│   ├── index.py             # Incremental SQLite index
│   ├── queries.py           # stats / health / note queries on the index
│   ├── service.py           # Operations used by the JSON-RPC server
│   ├── cli.py               # `vault` command
│   └── rpc/                 # JSON-RPC 2.0 server, Pydantic models, schema.json
├── tui/                     # `vault-tui` terminal app (TypeScript + Ink), see tui/README.md
├── scripts/verify.sh        # Runs every check of the engine, CLI and app
├── config/
│   └── config.yaml          # zettel tool configuration
├── src/                     # zettel tool (LLM extraction)
│   ├── main.py              # CLI entry point, orchestration, file operations
│   ├── prompts.py           # LLM prompt templates
│   ├── bundle.py            # Context bundle assembly and export
│   ├── db/
│   │   ├── client.py        # ChromaDB collection management, indexing, search
│   │   └── embeddings.py    # Embedding function factory (OpenAI, SentenceTransformer)
│   └── llm/
│       ├── extraction.py    # Concept extraction with validation
│       └── providers.py     # LLM providers (OpenAI, Anthropic, Ollama, Gemini)
└── tests/                   # Python tests; fixtures/vault is the sample vault
```

## Tests

Everything (engine, CLI, JSON-RPC, zettel tool, terminal app): `scripts/verify.sh`.
Python only: `python -m pytest -q` (or `python -m unittest discover -s tests`).
See [Tests and expected results](#tests-and-expected-results) for what each prints.
