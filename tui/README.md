# vault-tui

Terminal workbench for an Obsidian vault, in the style of Claude Code, built with
TypeScript + Ink. It starts the Python engine (`python -m vaultengine serve`) and
talks JSON-RPC 2.0 to it over stdio. Requirements: [REQUIREMENTS.md](REQUIREMENTS.md).

```bash
cd tui && npm install && npm run build && npm link   # once
vault-tui --vault /path/to/vault --format brain
```

The Python side must be installed (`uv sync` at the repo root). The interpreter is
found via `--python`, `$VAULT_TUI_PYTHON`, the repo's `.venv`, then `python3`.
`o` opens `obsidian://open?vault=<vault folder name>&file=<path>` with `open` / `xdg-open` / `start`,
or with `$VAULT_TUI_OPENER` (any program taking the URL as its argument).

| Input | Does |
|---|---|
| `/stats` `/health [limit]` `/note @note` `/note <search>` `/help` | commands |
| `/check [changed] [all] [baseline]` `/fix [apply] [mirrors\|headings] [all]` `/coverage [all]` | the format: violations by note (`baseline`: only those not in the recorded baseline), the safe repairs as diffs, sources and citations (need `--format`) |
| `/new <type> "<title>" [Key=Value]… [<relation> @note]…` `/link @A <relation> @B` `/ingest <file> [as <type>]` | safe writes; add `--dry-run` to see the diff only |
| `/search <query>` | the query language of `vault search`; opens a result list |
| in the list | ↑↓ PgUp/PgDn Home/End move · Enter preview · `o` open in Obsidian · Esc (or q) close |
| in the preview | ↑↓ PgUp/PgDn scroll · `n`/`N` next/previous match · `o` open in Obsidian · Esc back to the list |
| `/related @note` `/suggest [links\|notes\|@note]` `/unlinked [@note]` | related notes, links to add and notes to write, unlinked mentions |
| `/pipeline [all]` `/next [all\|<n>]` | the reading pipeline with `--format core`: each source's stage and the gaps; one queue of what needs doing, most trusted first, with the command for each |
| `/page new "<title>" @zettel… [from "<query>"]…` `/page check [@page]` `/page refresh @page` `/page exclude @page @zettel… [undo]` | pages made from zettels: a dry run shows the page (and its index and log lines); add `write` to write |
| `/findings [<method>]… [all] [<id>]` `/idea new <id>` `/idea set @idea <status>` | what the engine discovered, each with a stable id; an idea from one; its status. Dry run first, `write` writes |
| `/move @note <folder/\|name\|path>` `/merge @A @B` | every link keeps its target; a merge drops links that would point at the note they are in and lists them. Dry run first, `write` writes |
| `/` | command suggestions (Tab completes, Enter runs) |
| `@` | note picker (↑↓, Tab/Enter insert, Esc close) |
| ↑ ↓ | history · ctrl+a/e/u/k edit · ctrl+c/ctrl+d quit |

## Layout

```
src/cli.tsx                 argument parsing, starts server + Ink app
src/rpc/client.ts           JSON-RPC 2.0 client (newline-delimited JSON)
src/rpc/server.ts           spawns python -m vaultengine serve, exit reasons
src/rpc/api.ts              typed VaultApi facade (what commands use)
src/rpc/schema.ts           GENERATED from vaultengine/rpc/schema.json (npm run gen:types)
src/commands/registry.ts    Command type + CommandRegistry (the extension point); a command returns
                            transcript output, or an InteractiveView that has the keys until closed
src/commands/mentions.ts    @-mention parsing
src/commands/builtin/       /help /stats /health /note /search /related /suggest /unlinked /check /fix /new /link /ingest /coverage
                            /pipeline /page /findings /idea /move /merge /next
src/commands/args.ts        quote-aware argument splitting (words, "quoted", @mentions)
src/system/open.ts          obsidian:// URLs and the system opener
src/ui/                     App, Prompt (history, / suggestions, @ picker), StatusLine, views
src/ui/views/SearchView.tsx the /search list and preview (an InteractiveView)
```

## Tests

| Command | Runs | Ends with |
|---|---|---|
| `npm test` | generated-types check, unit tests, ink-testing-library frame tests (the App fed with responses recorded from the real server; `test/pipeline.test.tsx` for /pipeline /page /findings /idea /move /merge /next /check baseline), integration tests against the real Python server | `Tests  107 passed (107)` |
| `npm run typecheck` | `tsc` over sources and tests | exit code 0 |
| `npm run test:e2e` | builds, then launches `dist/cli.js` in a real pseudo-terminal with node-pty: the `/health` report; `/search` with move, preview, `o` and back; with `--format brain` `/check`, `/coverage`, a `/link` that writes both notes and `/check changed`; with `--format core` `/pipeline`, `/findings link`, a `/merge` as a dry run then written (the self-link dropped on disk), `/check baseline`, `/next 3` | `Tests  5 passed (5)` |

`node-pty` is a native module: on Linux `npm ci` compiles it (needs `build-essential` and `python3`);
macOS uses a prebuilt binary. Only the end-to-end tests use it. Offline, node-gyp cannot download the Node.js headers;
`npm_config_nodedir=/usr npm ci` uses the installed ones (the folder with `include/node/node.h`).
