/**
 * Start `python -m vaultengine serve <vault>` and connect a JsonRpcClient to it.
 */
import {spawn, type ChildProcess} from 'node:child_process';
import {existsSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {ConnectionClosedError, JsonRpcClient} from './client.js';

export type ServerOptions = {
	vault: string;
	/** Python interpreter; default: $VAULT_TUI_PYTHON, the repo's .venv, then python3. */
	python?: string;
	/** Index file; default: <vault>/.vaultengine/index.sqlite (decided by the server). */
	db?: string;
	/** Format (name in formats/ or a file) for /check, /fix, /new, /link, /ingest, /coverage. */
	format?: string;
};

export type VaultServer = {
	client: JsonRpcClient;
	process: ChildProcess;
	python: string;
	/** Last lines the server wrote to stderr (for error messages). */
	stderrTail: () => string[];
	stop: () => void;
};

/** The repository root: tui/dist/rpc/server.js or tui/src/rpc/server.ts -> ../../.. */
export const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

export function resolvePython(explicit?: string): string {
	if (explicit) {
		return explicit;
	}

	if (process.env['VAULT_TUI_PYTHON']) {
		return process.env['VAULT_TUI_PYTHON'];
	}

	const venvPython = join(repoRoot, '.venv', 'bin', 'python');
	return existsSync(venvPython) ? venvPython : 'python3';
}

export function startVaultServer(options: ServerOptions): VaultServer {
	const python = resolvePython(options.python);
	const args = ['-m', 'vaultengine', 'serve', resolve(options.vault)];
	if (options.db) {
		args.push('--db', resolve(options.db));
	}

	if (options.format) {
		// A bare name ("brain") is looked up by the server; a path is made absolute here.
		args.push('--format', /[\\/]|\.ya?ml$/i.test(options.format) ? resolve(options.format) : options.format);
	}

	// cwd = repo root, so `-m vaultengine` works even if the package is not installed.
	const child = spawn(python, args, {
		cwd: existsSync(join(repoRoot, 'vaultengine')) ? repoRoot : undefined,
		env: {...process.env, PYTHONUNBUFFERED: '1'},
		stdio: ['pipe', 'pipe', 'pipe'],
	});
	// The process owner reports the end, with the exit reason (see 'close' below).
	const client = new JsonRpcClient(child.stdout, child.stdin, {failOnEnd: false});

	const tail: string[] = [];
	child.stderr.setEncoding('utf8');
	child.stderr.on('data', (chunk: string) => {
		tail.push(...chunk.split('\n').filter(Boolean));
		tail.splice(0, Math.max(0, tail.length - 20));
	});

	const describeExit = (reason: string) => {
		const last = tail.at(-1);
		return new ConnectionClosedError(`Vault server ${reason}${last ? `: ${last}` : ''}`);
	};

	child.on('error', (error) => {
		client.fail(describeExit(`could not start (${python}): ${error.message}`));
	});
	// 'close' comes after the process ended AND its stdout was drained, so every
	// response already sent has been delivered before pending requests are failed.
	child.on('close', (code, signal) => {
		client.fail(describeExit(signal ? `was stopped (${signal})` : `exited with code ${code}`));
	});

	return {
		client,
		process: child,
		python,
		stderrTail: () => [...tail],
		stop() {
			client.close();
			if (child.exitCode === null && child.signalCode === null) {
				child.kill();
			}
		},
	};
}
