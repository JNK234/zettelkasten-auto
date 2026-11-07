# Zettelkasten Automation

Convert substantive source notes into atomic Zettels, keep the knowledge graph local, and assemble exportable bundles around a source or note.

This repository has two parts:

- **Vault engine, `vault` CLI and `vault-tui`** (no LLM): read an Obsidian vault the way Obsidian does (frontmatter, tags,
  wikilinks, embeds, headings, blocks), resolve links, keep an incremental SQLite index, and work with it from the command
  line, over JSON-RPC, or in a Claude Code–style terminal app. On top of that, a **formats layer** checks notes against your
  own conventions, written in a YAML file under `formats/`, and gives scripts a safe way to write notes.
- **`zettel`**: the original LLM tool that turns source notes into Zettels. See [docs/zettel.md](docs/zettel.md).

## Quick start

```bash
uv sync && source .venv/bin/activate                        # Python env with the `vault` command
cp -Rp tests/fixtures/vault /tmp/sample-vault                # work on a copy of the sample vault
vault index /tmp/sample-vault                                # 180 notes, 1787 links
vault check /tmp/sample-vault --format brain                 # ✗ 102 violations in 77 notes, each with file and line
```

Details: [Install](docs/install.md) and [Getting started](docs/getting-started.md).

## What is in it

| Piece | What it is | Run it | Docs |
|---|---|---|---|
| `vaultengine/` | Python package: parser, link resolver, link graph, SQLite index, stats/health | `from vaultengine import Vault, LinkGraph` | [Getting started](docs/getting-started.md) |
| `vault` | CLI on top of the engine | `vault index\|stats\|health <vault>` | [Getting started](docs/getting-started.md) |
| `vaultengine/formats/` + `formats/*.yaml` | your note conventions as data: note types, required headings and metadata, allowed values, typed relations with mirror pairs, link roles | `vault formats <vault> --format brain` | [The format file](docs/formats.md) |
| `vault check` / `vault fix` | enforce a format: every violation with file and line (exit 1 if any; `--changed` for hooks), and the safe automatic repairs | `vault check <vault> --format brain --changed` | [Commands](docs/commands.md#vault-check-vault--f-format---changed---json) |
| `vault new` / `link` / `set` | the safe write path for scripts: create a note of a type, relate two notes (mirror included), set a metadata value; atomic, index kept in sync, refused if it would add a violation | `vault link <vault> -f brain A B --as supports` | [Commands](docs/commands.md#the-safe-write-path-vault-new-vault-link-vault-set) |
| `vault ingest` / `coverage` | bring a markdown, text or HTML file in as a source (with provenance, refused if it duplicates one), and see which sources nothing links to and which notes cite none | `vault ingest <vault> paper.html --as source -f brain` | [Commands](docs/commands.md#vault-ingest-vault-file---as-type--f-f---title-t---folder-f---set-kv---dry-run) |
| `vault suggest` | experimental: rank likely sources, relation targets, or notes a document resembles | `vault suggest <vault> -f brain --note X --as supports` | [Commands](docs/commands.md#vault-suggest-vault--f-f---note-n---as-relation----file-doc---type-t---method-lexicalsemanticfused---limit-n-experimental) |
| `vault serve` | the engine as a JSON-RPC 2.0 server on stdio (schema: `vaultengine/rpc/schema.json`); with `--format`, also the format methods | `vault serve <vault> --format brain` | [The JSON-RPC API](docs/api.md) |
| `tui/` → `vault-tui` | terminal app (TypeScript + Ink) that starts `vault serve` and talks to it | `vault-tui --vault <vault> --format brain` | [The terminal app](docs/tui.md) |

## Documentation

| Document | What it covers |
|---|---|
| [Install](docs/install.md) | requirements, `uv sync` and `npm ci`, installing offline |
| [Getting started](docs/getting-started.md) | the sample vault step by step: building the index, stats and health, the engine as a Python library |
| [The format file](docs/formats.md) | every key of a format file (note types, headings, metadata, relations and mirrors, link roles, `new`, `ingest`, `coverage`), the violation codes, the Python API |
| [Commands](docs/commands.md) | `vault formats`, `check`, `fix`, `new`, `link`, `set`, `ingest`, `coverage`, `suggest`: options, exit codes and real output; the safe write path and the duplicate gate |
| [The JSON-RPC API](docs/api.md) | the protocol, the error codes, and the format methods (`vault.check`, `note.link`, `note.ingest`, …) |
| [The terminal app](docs/tui.md) | `vault-tui`: `/check`, `/fix`, `/new`, `/link`, `/ingest`, `/coverage` and the other commands, with a captured screen |
| [Verify it yourself](docs/verify.md) | `scripts/verify.sh`, each test suite and what it prints, and the commands to check by hand |
| [Project structure](docs/project-structure.md) | where everything is in the repository |
| [The zettel tool](docs/zettel.md) | the original LLM extraction tool: configuration, CLI, workflow |
| [tui/README.md](tui/README.md) | developing the terminal app |

## Verify

```bash
bash scripts/verify.sh    # about 8 minutes; the last line is ALL CHECKS PASSED
```

It runs both test suites, then every command, the JSON-RPC API and the terminal app end to end on scratch copies of the
sample vault. What each step proves, and what every test command prints: [Verify it yourself](docs/verify.md).

## License

See [LICENSE](LICENSE).
