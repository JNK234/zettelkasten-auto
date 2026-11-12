/**
 * End to end in a real pseudo-terminal (node-pty): the built `vault-tui --format core` on a copy of the
 * sample vault. /pipeline, /findings link, a /merge shown as a dry run and then written (the link that
 * would point at the note it is in is dropped, on disk too), /check baseline, /next, then quit. As in
 * format.pty.test.ts, every line the app prints must fit the terminal.
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
const PYTHON = join(REPO_ROOT, '.venv', 'bin', 'python');
const COLS = 100;
const SURVIVOR = 'enterprise-agent-evaluation-challenges';
const OTHER = 'pass-hat-k-reliability-metric';

let workdir: string;
let vault: string;

beforeAll(() => {
	if (!existsSync(CLI)) {
		throw new Error('dist/cli.js missing: run `npm run test:e2e` (it builds first)');
	}

	workdir = mkdtempSync(join(tmpdir(), 'vault-tui-e2e-pipeline-'));
	vault = join(workdir, 'sample-vault');
	cpSync(join(REPO_ROOT, 'tests', 'fixtures', 'vault'), vault, {
		recursive: true,
		preserveTimestamps: true,
		filter: (source) => !source.split(/[\\/]/).includes('.vaultengine'),
	});
	// Today's violations become the baseline (the sample's notes predate the core format).
	const python = existsSync(PYTHON) ? PYTHON : 'python3';
	execFileSync(python, ['-m', 'vaultengine', 'check', vault, '-f', 'core', '--update-baseline'], {cwd: REPO_ROOT});
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
	/** Wait until `wanted` appears after position `from` of the output. */
	const waitFor = async (wanted: string, from = 0, timeout = 60_000) => {
		const started = Date.now();
		while (!text().slice(from).includes(wanted)) {
			if (Date.now() - started > timeout) {
				throw new Error(`Timed out after ${timeout} ms waiting for ${JSON.stringify(wanted)}. Output tail:\n${text().slice(-3000)}`);
			}

			await sleep(50);
		}

		return text().slice(from);
	};

	/** Type a command like a person: characters, Esc to close a suggestion list, Enter. Returns where its output starts. */
	const run = async (line: string) => {
		const start = text().length;
		for (const key of line) {
			term.write(key);
			await sleep(10);
		}

		term.write('\u001B');
		await sleep(80);
		term.write('\r');
		return start;
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

describe('vault-tui --format core in a real terminal', () => {
	test('/pipeline, /findings, /merge dry run then write, /check baseline, /next', async () => {
		const app = launch(['--vault', vault, '--format', 'core']);
		await app.waitFor('format core  ·  180 notes · 1,787 links  ·  ● index fresh');

		let start = await app.run('/pipeline');
		let output = await app.waitFor('Zettels in no page (104)', start);
		for (const line of ['⏺ Pipeline  core · 28 source notes', '109 zettel notes: 5 in a page · 28 resting on a source', 'Not extracted yet (13)']) {
			expect(output).toContain(line);
		}

		start = await app.run('/findings link');
		output = await app.waitFor('/findings link all', start);
		expect(output).toContain('link: two notes that probably belong linked (13)');
		expect(output).toContain('link-5ec0974343');

		const before = readFileSync(join(vault, '20_ZETTLEKASTEN', `${SURVIVOR}.md`), 'utf8');
		start = await app.run(`/merge @${SURVIVOR} @${OTHER}`);
		output = await app.waitFor('add write to the command to write it', start);
		expect(output).toContain('⏺ Dry run: would write 4 notes');
		expect(output).toContain(`[[${OTHER}]] → line removed`);
		expect(readFileSync(join(vault, '20_ZETTLEKASTEN', `${SURVIVOR}.md`), 'utf8')).toBe(before); // a dry run writes nothing

		start = await app.run(`/merge @${SURVIVOR} @${OTHER} write`);
		output = await app.waitFor('index: updated', start);
		expect(output).toContain('⏺ Wrote 4 notes  merge');
		const merged = readFileSync(join(vault, '20_ZETTLEKASTEN', `${SURVIVOR}.md`), 'utf8');
		expect(merged).toContain(`## Merged from ${OTHER}`);
		expect(merged).not.toContain(`[[${OTHER}]]`); // the line that would link the note to itself is gone
		expect(merged).not.toContain(`[[${SURVIVOR}]]`);
		expect(existsSync(join(vault, '20_ZETTLEKASTEN', `${OTHER}.md`))).toBe(false);

		start = await app.run('/check baseline');
		output = await app.waitFor('still there (not listed)', start);
		expect(output).toContain('✓ no new violations in 179 notes checked');
		expect(output).toContain('144 violations · 143 still there (not listed) · 1 fixed'); // the merged note's own

		start = await app.run('/next 3');
		output = await app.waitFor('By kind (listed / all)', start);
		expect(output).toContain('most trusted first');
		expect(output).toContain('1. ');
		expect(output).toContain('run vault ');

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
