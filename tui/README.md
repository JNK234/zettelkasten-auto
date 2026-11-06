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

| Input | Does |
|---|---|
| `/stats` `/health [limit]` `/note @note` `/note <search>` `/help` | commands |
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
src/commands/builtin/       /help /stats /health /note
src/ui/                     App, Prompt (history, / suggestions, @ picker), StatusLine, views
```

## Tests

| Command | Runs | Ends with |
|---|---|---|
| `npm test` | generated-types check, unit tests, ink-testing-library frame tests (the App fed with responses recorded from the real server), integration tests against the real Python server | `Tests  41 passed (41)` |
| `npm run typecheck` | `tsc` over sources and tests | exit code 0 |
| `npm run test:e2e` | builds, then launches `dist/cli.js` in a real pseudo-terminal with node-pty, types `/health`, checks the report, quits with ctrl+c and checks the Python server is gone | `Tests  2 passed (2)` |

`node-pty` is a native module: on Linux `npm ci` compiles it (needs `build-essential` and `python3`);
macOS uses a prebuilt binary. Only the end-to-end test uses it.
