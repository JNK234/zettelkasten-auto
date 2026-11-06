/**
 * End to end in a real pseudo-terminal (node-pty): launch the built
 * `vault-tui --vault <copy of the sample vault>`, wait for the index, type
 * /health like a user, check the report, quit with ctrl+c and make sure the
 * Python server is gone.
 */
import {execFileSync} from 'node:child_process';
import {cpSync, existsSync, mkdtempSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {stripVTControlCharacters} from 'node:util';
import * as pty from 'node-pty';
import {afterAll, beforeAll, describe, expect, test} from 'vitest';

const TUI_ROOT = fileURLToPath(new URL('../..', import.meta.url));
const REPO_ROOT = join(TUI_ROOT, '..');
const CLI = join(TUI_ROOT, 'dist', 'cli.js');

let workdir: string;
let vault: string;

beforeAll(() => {
	if (!existsSync(CLI)) {
		throw new Error('dist/cli.js missing: run `npm run test:e2e` (it builds first)');
	}

	workdir = mkdtempSync(join(tmpdir(), 'vault-tui-e2e-'));
	vault = join(workdir, 'sample-vault');
	cpSync(join(REPO_ROOT, 'tests', 'fixtures', 'vault'), vault, {
		recursive: true,
		preserveTimestamps: true,
		filter: (source) => !source.split(/[\\/]/).includes('.vaultengine'), // never reuse an index
	});
});

afterAll(() => {
	rmSync(workdir, {recursive: true, force: true});
});

/** A running terminal session with helpers to wait for text and type. */
function launch(args: string[]) {
	const term = pty.spawn(process.execPath, [CLI, ...args], {
		name: 'xterm-256color',
		cols: 100,
		rows: 50,
		cwd: workdir,
		env: {...process.env, FORCE_COLOR: '1'},
	});
	let raw = '';
	term.onData((data) => {
		raw += data;
	});
	const exited = new Promise<number>((resolve) => {
		term.onExit(({exitCode}) => {
			resolve(exitCode);
		});
	});

	/** Everything the app printed, without escape codes (colors, cursor moves). */
	const text = () => stripVTControlCharacters(raw);

	const waitFor = async (wanted: string, timeout = 30_000) => {
		const started = Date.now();
		while (!text().includes(wanted)) {
			if (Date.now() - started > timeout) {
				throw new Error(`Timed out after ${timeout} ms waiting for ${JSON.stringify(wanted)}. Output tail:\n${text().slice(-2500)}`);
			}

			await new Promise((resolve) => {
				setTimeout(resolve, 50);
			});
		}

		return text();
	};

	const type = async (keys: string) => {
		for (const key of keys) {
			term.write(key);
			await new Promise((resolve) => {
				setTimeout(resolve, 20);
			});
		}
	};

	return {term, text, waitFor, type, exited};
}

function serverProcesses(): string[] {
	try {
		return execFileSync('pgrep', ['-f', `vaultengine serve ${vault}`], {encoding: 'utf8'}).split('\n').filter(Boolean);
	} catch {
		return []; // pgrep exits 1 when nothing matches
	}
}

describe('vault-tui in a real terminal', () => {
	test('/health on the sample vault', async () => {
		const app = launch(['--vault', vault]);

		// Start-up: banner, then the server indexes the vault and the status line fills in.
		await app.waitFor('✻ vault-tui');
		await app.waitFor('180 notes · 1,787 links  ·  ● index fresh');
		expect(serverProcesses()).toHaveLength(1);

		await app.type('/health');
		await app.type('\r');
		const output = await app.waitFor('Dead ends (showing 5 of 21)');

		const report = output.slice(output.lastIndexOf('⏺ Vault health'));
		for (const line of [
			'✗ 1,315 unresolved links (612 missing targets)',
			'✓ 0 ambiguous links',
			'✓ 0 missing heading/block targets',
			'! 41 orphans (no backlinks)',
			'! 21 dead ends (no outgoing note links)',
			'✓ 0 duplicate basenames',
			'✓ 0 empty notes',
			'Unresolved targets (showing 5 of 612)',
			'rl-environment-design  52 links from 52 notes',
			'discourse-graph  40 links from 40 notes',
			'Orphans (showing 5 of 41)',
		]) {
			expect(report).toContain(line);
		}

		// Quit: the app exits cleanly and takes the Python server with it.
		app.term.write('\u0003');
		expect(await app.exited).toBe(0);
		await new Promise((resolve) => {
			setTimeout(resolve, 300);
		});
		expect(serverProcesses()).toEqual([]);
	});

	test('a missing vault folder is rejected before anything starts', async () => {
		const app = launch(['--vault', join(workdir, 'does-not-exist')]);
		expect(await app.exited).toBe(2);
		expect(app.text()).toContain('vault-tui: vault folder not found');
	});
});
