# vault-tui

Terminal workbench for an Obsidian vault, in the style of Claude Code, built with
TypeScript + Ink. It starts the Python engine (`python -m vaultengine serve`) and
talks JSON-RPC 2.0 to it over stdio. Requirements: [REQUIREMENTS.md](REQUIREMENTS.md).

```bash
cd tui && npm install && npm run build && npm link   # once
vault-tui --vault /path/to/vault
```

The Python side must be installed (`uv sync` at the repo root). The interpreter is
found via `--python`, `$VAULT_TUI_PYTHON`, the repo's `.venv`, then `python3`.
`o` opens `obsidian://open?vault=<vault folder name>&file=<path>` with `open` / `xdg-open` / `start`,
or with `$VAULT_TUI_OPENER` (any program taking the URL as its argument).

| Input | Does |
|---|---|
| `/stats` `/health [limit]` `/note @note` `/note <search>` `/help` | commands |
| `/search <query>` | the query language of `vault search`; opens a result list |
| in the list | ↑↓ PgUp/PgDn Home/End move · Enter preview · `o` open in Obsidian · Esc (or q) close |
| in the preview | ↑↓ PgUp/PgDn scroll · `n`/`N` next/previous match · `o` open in Obsidian · Esc back to the list |
| `/related @note` `/suggest [links\|notes\|@note]` `/unlinked [@note]` | related notes, links to add and notes to write, unlinked mentions |
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
src/commands/builtin/       /help /stats /health /note /search /related /suggest /unlinked
src/system/open.ts          obsidian:// URLs and the system opener
src/ui/                     App, Prompt (history, / suggestions, @ picker), StatusLine, views
src/ui/views/SearchView.tsx the /search list and preview (an InteractiveView)
```

## Tests

| Command | Runs | Ends with |
|---|---|---|
| `npm test` | generated-types check, unit tests, ink-testing-library frame tests (the App fed with responses recorded from the real server), integration tests against the real Python server | `Tests  67 passed (67)` |
| `npm run typecheck` | `tsc` over sources and tests | exit code 0 |
| `npm run test:e2e` | builds, then launches `dist/cli.js` in a real pseudo-terminal with node-pty: types `/health` and checks the report; runs `/search`, moves, opens the preview, presses `o` (with `VAULT_TUI_OPENER` set to a script that records the URL), goes back and closes; quits with ctrl+c and checks the Python server is gone | `Tests  3 passed (3)` |

`node-pty` is a native module: on Linux `npm ci` compiles it (needs `build-essential` and `python3`);
macOS uses a prebuilt binary. Only the end-to-end test uses it.
