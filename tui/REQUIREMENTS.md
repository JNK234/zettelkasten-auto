# vault-tui — POC requirements

A terminal workbench for an Obsidian vault, in the style of Claude Code. It is
**not a chatbot**: you type commands that run operations on notes, and the
output stays in the transcript. This POC is the shell that later tools plug into.

## Scope of the POC

| # | Requirement | Done when |
|---|---|---|
| R1 | `vault-tui --vault <path>` starts the Python engine (`python -m vaultengine serve`) and talks JSON-RPC 2.0 to it over stdio | the app starts against the sample vault; killing the app stops the server |
| R2 | On start, the index is brought up to date (incremental `index.build`) | status line shows note/link counts right after start |
| R3 | Prompt with editing and **history** (↑/↓ recall, draft kept) | frame tests |
| R4 | **Status line**: vault name, note count, link count, index freshness (fresh / stale / indexing / not built), server state | frame tests |
| R5 | **Command registry**: a command is one object `{name, summary, usage, run}` registered once; `/help`, suggestions and dispatch read from the registry | adding a command needs no change to the UI code |
| R6 | Commands: `/stats`, `/health`, `/note`, `/help` | frame tests + live run |
| R7 | **`@` note picker**: typing `@` opens a fuzzy picker (engine `notes.pick`); ↑/↓ select, Tab/Enter insert, Esc close. `/note @note` opens that note. Any future command can read the mentions | frame tests |
| R8 | Typing `/` lists matching commands; Tab completes, Enter runs the highlighted one | frame tests |
| R9 | Errors from the engine (e.g. note not found) are shown in the transcript, never crash the app | frame tests |
| R10 | Request/result types come from the engine's exported JSON Schema (generated, not hand-written) | `npm run gen:types:check` |

## Out of scope (later)

Editing notes, graph algorithms, multiple vaults, persistent prompt history,
cancelling long requests, mouse support, themes.

## Extending

```ts
// src/commands/builtin/<name>.tsx
export const reindex: Command = {
	name: 'reindex',
	summary: 'Bring the index up to date',
	async run(_input, ctx) {
		const report = await ctx.api.build();
		return <Text>{report.parsed.length} notes re-parsed</Text>;
	},
};
// then add it to builtinCommands in src/commands/builtin/index.ts
```
