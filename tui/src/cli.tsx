#!/usr/bin/env node
/**
 * vault-tui --vault <path> [--python <interpreter>] [--db <index.sqlite>] [--no-index]
 */
import {statSync} from 'node:fs';
import {parseArgs} from 'node:util';
import {render} from 'ink';
import {builtinCommands} from './commands/builtin/index.js';
import {CommandRegistry} from './commands/registry.js';
import {createVaultApi} from './rpc/api.js';
import {startVaultServer} from './rpc/server.js';
import {App} from './ui/App.js';

const USAGE = `Usage: vault-tui --vault <path> [options]

Options:
  -v, --vault <path>     Obsidian vault folder (required)
      --python <path>    Python interpreter with vaultengine (default: $VAULT_TUI_PYTHON, repo .venv, python3)
      --db <path>        index file (default: <vault>/.vaultengine/index.sqlite)
      --no-index         do not update the index on start
  -h, --help             show this help`;

function fail(message: string): never {
	process.stderr.write(`vault-tui: ${message}\n\n${USAGE}\n`);
	process.exit(2);
}

let options;
try {
	options = parseArgs({
		options: {
			vault: {type: 'string', short: 'v'},
			python: {type: 'string'},
			db: {type: 'string'},
			'no-index': {type: 'boolean', default: false},
			help: {type: 'boolean', short: 'h', default: false},
		},
		allowPositionals: false,
	}).values;
} catch (error) {
	fail(error instanceof Error ? error.message : String(error));
}

if (options.help) {
	process.stdout.write(`${USAGE}\n`);
	process.exit(0);
}

const vault = options.vault ?? fail('--vault is required');
if (!statSync(vault, {throwIfNoEntry: false})?.isDirectory()) {
	fail(`vault folder not found: ${vault}`);
}

if (!process.stdin.isTTY) {
	fail('needs an interactive terminal (stdin is not a TTY)');
}

const server = startVaultServer({vault, python: options.python, db: options.db});
const api = createVaultApi(server.client);
const registry = new CommandRegistry().register(...builtinCommands);

const app = render(
	<App
		api={api}
		registry={registry}
		vaultPath={vault}
		autoIndex={!options['no-index']}
		onDisconnect={(listener) => {
			server.client.events.on('close', listener);
			return () => server.client.events.off('close', listener);
		}}
	/>,
);

try {
	await app.waitUntilExit();
} finally {
	server.stop();
}
