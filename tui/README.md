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

| Input | Does |
|---|---|
| `/stats` `/health [limit]` `/note @note` `/note <search>` `/help` | commands |
| `/check [changed] [all]` `/fix [apply] [mirrors\|headings] [all]` `/coverage [all]` | the format: violations by note, the safe repairs as diffs, sources and citations (need `--format`) |
| `/new <type> "<title>" [Key=Value]… [<relation> @note]…` `/link @A <relation> @B` `/ingest <file> [as <type>]` | safe writes; add `--dry-run` to see the diff only |
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
src/commands/registry.ts    Command type + CommandRegistry (the extension point)
src/commands/mentions.ts    @-mention parsing
src/commands/builtin/       /help /stats /health /note /check /fix /new /link /ingest /coverage
src/commands/args.ts        quote-aware argument splitting (words, "quoted", @mentions)
src/ui/                     App, Prompt (history, / suggestions, @ picker), StatusLine, views
```

## Tests

| Command | Runs | Ends with |
|---|---|---|
| `npm test` | generated-types check, unit tests, ink-testing-library frame tests (the App fed with responses recorded from the real server), integration tests against the real Python server | `Tests  64 passed (64)` |
| `npm run typecheck` | `tsc` over sources and tests | exit code 0 |
| `npm run test:e2e` | builds, then launches `dist/cli.js` in a real pseudo-terminal with node-pty: `/health`; and with `--format brain` `/check`, `/coverage`, a `/link` that writes both notes, `/check changed`, every printed line fitting the terminal; quits with ctrl+c and checks the Python server is gone | `Tests  3 passed (3)` |

`node-pty` is a native module: on Linux `npm ci` compiles it (needs `build-essential` and `python3`);
macOS uses a prebuilt binary. Only the end-to-end tests use it. Offline, node-gyp cannot download the Node.js headers;
`npm_config_nodedir=/usr npm ci` uses the installed ones (the folder with `include/node/node.h`).
