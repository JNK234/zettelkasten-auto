/**
 * End to end in a real pseudo-terminal (node-pty): the built `vault-tui --format brain`
 * on a copy of the sample vault. /check, /coverage, a /link that writes two notes,
 * /check changed, then quit. Also checks what the frame tests cannot: that every line
 * the app prints fits the terminal (output in the transcript is laid out against the
 * terminal width, not wrapped by the terminal at its edge).
 */
import {execFileSync} from 'node:child_process';
import {cpSync, existsSync, mkdtempSync, readFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {stripVTControlCharacters} from 'node:util';
import * as pty from 'node-pty';
import {afterAll, beforeAll, describe, expect, test} from 'vitest';

const TUI_ROOT = fileURLToPath(new URL('../..', import.meta.url));
const REPO_ROOT = join(TUI_ROOT, '..');
const CLI = join(TUI_ROOT, 'dist', 'cli.js');
const COLS = 100;
const D = '50_MOC/discourse/rl-environment-design';
const EVD = 'EVD - Reward is a deterministic outcome check 1 only if the final database equals the data';
const CLM = 'CLM - Grade the observed end state rather than the trajectory';

let workdir: string;
let vault: string;

beforeAll(() => {
	if (!existsSync(CLI)) {
		throw new Error('dist/cli.js missing: run `npm run test:e2e` (it builds first)');
	}

	workdir = mkdtempSync(join(tmpdir(), 'vault-tui-e2e-format-'));
	vault = join(workdir, 'sample-vault');
	cpSync(join(REPO_ROOT, 'tests', 'fixtures', 'vault'), vault, {
		recursive: true,
		preserveTimestamps: true,
		filter: (source) => !source.split(/[\\/]/).includes('.vaultengine'),
	});
});

afterAll(() => {
	rmSync(workdir, {recursive: true, force: true});
});

function launch(args: string[]) {
	const term = pty.spawn(process.execPath, [CLI, ...args], {
		name: 'xterm-256color',
		cols: COLS,
		rows: 50,
		cwd: REPO_ROOT,
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
	const text = () => stripVTControlCharacters(raw);
	const sleep = async (ms: number) =>
		new Promise((resolve) => {
			setTimeout(resolve, ms);
		});
	const waitFor = async (wanted: string, timeout = 30_000) => {
		const started = Date.now();
		while (!text().includes(wanted)) {
			if (Date.now() - started > timeout) {
				throw new Error(`Timed out after ${timeout} ms waiting for ${JSON.stringify(wanted)}. Output tail:\n${text().slice(-3000)}`);
			}

			await sleep(50);
		}

		return text();
	};

	/** Type a command like a person: characters, Esc to close a suggestion list, Enter. */
	const run = async (line: string) => {
		for (const key of line) {
			term.write(key);
			await sleep(10);
		}

		term.write('\u001B');
		await sleep(80);
		term.write('\r');
	};

	return {term, text, waitFor, run, exited};
}

function serverProcesses(): string[] {
	try {
		return execFileSync('pgrep', ['-f', `vaultengine serve ${vault}`], {encoding: 'utf8'}).split('\n').filter(Boolean);
	} catch {
		return [];
	}
}

describe('vault-tui --format brain in a real terminal', () => {
	test('/check, /coverage, /link with its mirror, /check changed', async () => {
		const app = launch(['--vault', vault, '--format', 'brain']);
		await app.waitFor('format brain  ·  180 notes · 1,787 links  ·  ● index fresh');

		await app.run('/check');
		let output = await app.waitFor('more notes with violations');
		const check = output.slice(output.lastIndexOf('⏺ Format check'));
		for (const line of [
			'brain · all 180 notes',
			'✗ 102 violations in 77 notes',
			'59 unresolved-target · 28 missing-heading · 15 missing-key',
			'28 can be repaired automatically: /fix shows the diff first',
			"missing heading 'Notes' (expected at the end of the note)",
		]) {
			expect(check).toContain(line);
		}

		await app.run('/coverage');
		output = await app.waitFor('Most cited sources');
		expect(output.slice(output.lastIndexOf('⏺ Coverage'))).toContain('28 sources: 23 linked, 5 that nothing links to');

		await app.run(`/link @"${EVD}" supports @"${CLM}"`);
		output = await app.waitFor('index: updated 2 notes');
		expect(output).toContain('⏺ Wrote 2 notes  link');
		const claim = readFileSync(join(vault, D, `${CLM}.md`), 'utf8');
		const evidence = readFileSync(join(vault, D, `${EVD}.md`), 'utf8');
		expect(evidence).toMatch(new RegExp(`^supports:: .*\\[\\[${CLM}\\]\\]$`, 'm'));
		expect(claim).toMatch(new RegExp(`^supported-by:: .*\\[\\[${EVD}\\]\\]$`, 'm')); // the mirror

		await app.run('/check changed');
		await app.waitFor('✓ no violations in 0 notes checked');

		// Nothing the app printed is wider than the terminal. (Ink's word wrap can leave the
		// space it broke at after the last column; it is invisible, so it is not counted.)
		const tooWide = app
			.text()
			.split(/\r?\n/)
			.filter((line) => [...line.trimEnd()].length > COLS);
		expect(tooWide).toEqual([]);

		app.term.write('\u0003');
		expect(await app.exited).toBe(0);
		await new Promise((resolve) => {
			setTimeout(resolve, 300);
		});
		expect(serverProcesses()).toEqual([]);
	});
});
