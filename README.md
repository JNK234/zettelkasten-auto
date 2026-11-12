# Zettelkasten Automation

Convert substantive source notes into atomic Zettels, keep the knowledge graph local, and assemble exportable bundles around a source or note.

This repository has two parts:

- **Vault engine, `vault` CLI and `vault-tui`** (no LLM): read an Obsidian vault the way Obsidian does (frontmatter, tags,
  wikilinks, embeds, headings, blocks), resolve links, keep an incremental SQLite index, search it with a query language, find related notes and missing links, and work with it from the command
  line, over JSON-RPC, or in a Claude Code–style terminal app. On top of that, a **formats layer** checks notes against your
  own conventions, written in a YAML file under `formats/`, and gives scripts a safe way to write notes.
- **`zettel`**: the original LLM tool that turns source notes into Zettels. See [docs/zettel.md](docs/zettel.md).

## Quick start

```bash
uv sync && source .venv/bin/activate                        # Python env with the `vault` command
cp -Rp tests/fixtures/vault /tmp/sample-vault                # work on a copy of the sample vault
vault index /tmp/sample-vault                                # 180 notes, 1787 links
vault check /tmp/sample-vault --format brain                 # ✗ 102 violations in 77 notes, each with file and line
vault search /tmp/sample-vault '"reward hacking" -path:10_SOURCES'   # ranked hits with snippets
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
| `vault pipeline` / `note` | where each source stands (ingested, extracted, compiled) and the gaps, and what the engine knows about one note (what links to it, what it rests on, what uses it), all computed from the links, never stored | `vault pipeline <vault> -f core` | [Commands](docs/commands.md#vault-pipeline-vault--f-f---limit-n---json) |
| `vault page new` / `refresh` / `check` / `exclude` | pages made from zettels: a scaffold with generated zettels and sources blocks between markers (yours around them), rebuilt on refresh, and what needs attention (stale blocks, changed zettels, zettels it is behind on); the index and a log kept in step; read from the index, so a check takes seconds on a large vault | `vault page new <vault> "Title" --from '[[a]] [[b]] [[c]]' -f core --write` | [Commands](docs/commands.md#vault-page-newrefreshcheckexclude) |
| `vault findings` / `vault idea new` / `set` | what the engine discovered (missing notes many notes link to, two names for one idea, names that are probably an existing note, links worth adding), each with a stable id; an idea made from one, with the finding's method, notes and evidence in a block the format requires | `vault findings <vault> -f core` | [Commands](docs/commands.md#vault-findings-vault--f-f---method-m---id-id---limit-n---json) |
| `vault move` / `vault merge` | move or rename a note, or merge two (or a name with no note into a note); every link, embed, heading and block link is rewritten so it still resolves, and baselines, page records and ideas' finding blocks follow | `vault merge <vault> A B -f core --write` | [Commands](docs/commands.md#vault-move-vault-note-to--f-f---write---json-and-vault-merge-vault-a-b--f-f---write---json) |
| `vault next` | one queue of what needs doing, ranked by trust (worked out from use), each item with its reason and the command that resolves it | `vault next <vault> -f core` | [Commands](docs/commands.md#vault-next-vault--f-f---limit-n---json) |
| `vault search` | ranked, typo-tolerant search with a query language: text, `"phrases"`, `tag:`, `path:`, properties, `links-to:`, `within:N of:`, `has:`/`missing:`, dates, `-`, `OR` | `vault search <vault> 'tag:ai-safety links-to:[[x]] -path:10_SOURCES'` | [Search and connections](docs/search.md) |
| `vault related` / `suggest` / `unlinked` | related notes with a reason each; links to add and notes to write (with drafts); plain-text mentions that aren't links yet | `vault related <vault> kl-penalty-in-rlhf` | [Search and connections](docs/search.md#related-notes-links-to-add-notes-to-write) |
| `vault serve` | the engine as a JSON-RPC 2.0 server on stdio (schema: `vaultengine/rpc/schema.json`); with `--format`, also the format methods, and every command of the reading workflow (baseline check, pipeline, pages, findings, ideas, move, merge, next) | `vault serve <vault> --format core` | [The JSON-RPC API](docs/api.md) |
| `tui/` → `vault-tui` | terminal app (TypeScript + Ink) that starts `vault serve` and talks to it | `vault-tui --vault <vault> --format brain` | [The terminal app](docs/tui.md) |

## Documentation

| Document | What it covers |
|---|---|
| [Install](docs/install.md) | requirements, `uv sync` and `npm ci`, installing offline |
| [Getting started](docs/getting-started.md) | the sample vault step by step: building the index, stats and health, the engine as a Python library |
| [The format file](docs/formats.md) | every key of a format file (note types, headings, metadata, relations and mirrors, link roles, `new`, `ingest`, `coverage`), the violation codes, the Python API |
| [Search and connections](docs/search.md) | `vault search` and the query language, `related`, `suggest`, `unlinked`: options, real output, the JSON-RPC methods and `/search` in the app |
| [Commands](docs/commands.md) | `vault formats`, `check`, `fix`, `new`, `link`, `set`, `ingest`, `coverage`: options, exit codes and real output; the safe write path and the duplicate gate |
| [The reading workflow](docs/workflow.md) | from a paper to pages and ideas with `formats/core.yaml`, step by step with the exact commands (ingest, zettels, pipeline, pages until ready, findings, ideas, next); and how an AI assistant should drive it |
| [The JSON-RPC API](docs/api.md) | the protocol, the error codes, the format methods (`vault.check`, `note.link`, `note.ingest`, …) and the reading-pipeline methods (`vault.pipeline`, `page.*`, `findings.list`, `idea.*`, `note.move`, `note.merge`, `vault.next`) |
| [The terminal app](docs/tui.md) | `vault-tui`: `/search`, `/related`, `/suggest`, `/unlinked`, `/check`, `/fix`, `/new`, `/link`, `/ingest`, `/coverage`, `/pipeline`, `/page`, `/findings`, `/idea`, `/move`, `/merge`, `/next` and the other commands, with a captured screen |
| [Verify it yourself](docs/verify.md) | `scripts/verify.sh`, each test suite and what it prints, and the commands to check by hand |
| [Project structure](docs/project-structure.md) | where everything is in the repository |
| [The zettel tool](docs/zettel.md) | the original LLM extraction tool: configuration, CLI, workflow |
| [tui/README.md](tui/README.md) | developing the terminal app |

## Verify

```bash
bash scripts/verify.sh    # about 11 minutes; the last line is ALL CHECKS PASSED
```

It runs both test suites, then every command, the JSON-RPC API and the terminal app end to end on scratch copies of the
sample vault. What each step proves, and what every test command prints: [Verify it yourself](docs/verify.md).

## License

See [LICENSE](LICENSE).
