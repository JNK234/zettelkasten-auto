/**
 * End to end in a real pseudo-terminal (node-pty): the built `vault-tui --format brain` on a copy of the sample
 * vault, against the real engine. /map record, /explore from a note, keep the first result, open and link the gap
 * result (dry run, then w), run again with fresh, close, /kept, /map since the record, then quit. The link is checked in
 * both note files; every printed line must fit the terminal.
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
const COLS = 110;
const Z = '20_ZETTLEKASTEN';
const A = 'reward-computation-as-a-heavyweight-pipeline-stage';
const B = 'persona-driven-synthetic-data-generation';

let workdir: string;
let vault: string;

beforeAll(() => {
	if (!existsSync(CLI)) {
		throw new Error('dist/cli.js missing: run `npm run test:e2e` (it builds first)');
	}

	workdir = mkdtempSync(join(tmpdir(), 'vault-tui-e2e-explore-'));
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
		rows: 60,
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
	/** Wait until `wanted` appears in what was printed after `from` (an offset in text()). */
	const waitFor = async (wanted: string, from = 0, timeout = 60_000) => {
		const started = Date.now();
		while (!text().slice(from).includes(wanted)) {
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

	const press = async (key: string, times = 1) => {
		for (let count = 0; count < times; count++) {
			term.write(key);
			await sleep(60);
		}
	};

	return {term, text, waitFor, run, press, exited};
}

function serverProcesses(): string[] {
	try {
		return execFileSync('pgrep', ['-f', `vaultengine serve ${vault}`], {encoding: 'utf8'}).split('\n').filter(Boolean);
	} catch {
		return [];
	}
}

describe('vault-tui /explore in a real terminal', () => {
	test('/map record, /explore, keep, link it (dry run, then w), fresh, /kept, /map since', async () => {
		const app = launch(['--vault', vault, '--format', 'brain']);
		await app.waitFor('● index fresh');

		await app.run('/map record');
		await app.waitFor('✓ recorded a snapshot');

		let mark = app.text().length;
		await app.run('/explore kl-penalty-in-rlhf');
		await app.waitFor('Esc close', mark);
		expect(app.text().slice(mark)).toContain('⏺ Explore kl-penalty-in-rlhf');

		mark = app.text().length;
		await app.press('k');
		await app.waitFor('✓ Kept 1.1 RLHF', mark);

		// The gap result: 4 neighbourhood, 1 cited, 4 backbone, 4 cluster, 1 hidden, 4 role results before it.
		await app.press('\u001B[B', 18);
		mark = app.text().length;
		await app.press('l');
		await app.waitFor('w writes it in both notes', mark);
		expect(app.text().slice(mark)).toContain('Dry run: would write 2 notes');
		expect(readFileSync(join(vault, Z, `${A}.md`), 'utf8')).not.toContain('See Also'); // a dry run writes nothing

		mark = app.text().length;
		await app.press('w');
		await app.waitFor('Wrote 2 notes', mark);
		expect(readFileSync(join(vault, Z, `${A}.md`), 'utf8')).toMatch(new RegExp(`^- \\[\\[${B}\\]\\] — gap: `, 'm'));
		expect(readFileSync(join(vault, Z, `${B}.md`), 'utf8')).toMatch(new RegExp(`^- \\[\\[${A}\\]\\] — gap: `, 'm'));

		await app.press('\u001B');
		await new Promise((resolve) => {
			setTimeout(resolve, 200);
		});
		mark = app.text().length;
		await app.press('f');
		await app.waitFor('· fresh', mark);

		mark = app.text().length;
		await app.press('\u001B');
		await app.waitFor('Try /help', mark);

		mark = app.text().length;
		await app.run('/kept');
		await app.waitFor('⏺ Kept  2 results', mark);

		mark = app.text().length;
		await app.run('/map since 2000-01-01');
		await app.waitFor('Gaps: 1 of 4 closed', mark);

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
