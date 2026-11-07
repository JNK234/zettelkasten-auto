/**
 * Frame tests: render the whole App with ink-testing-library, type keys,
 * assert on the rendered terminal frame. The API replays recorded engine
 * responses (see helpers.tsx).
 */
import {Text} from 'ink';
import {afterEach, describe, expect, test} from 'vitest';
import {cleanup} from 'ink-testing-library';
import {builtinCommands} from '../src/commands/builtin/index.js';
import {CommandRegistry} from '../src/commands/registry.js';
import {ConnectionClosedError} from '../src/rpc/client.js';
import {delay, fakeApi, freshStatus, keys, promptLine, renderApp} from './helpers.js';

afterEach(() => {
	cleanup();
});

describe('start-up and status line (R2, R4)', () => {
	test('indexes on start, then shows vault name, counts and freshness', async () => {
		const app = renderApp();
		const frame = await app.waitFor('index fresh');
		expect(frame).toContain('✻ vault-tui');
		expect(frame).toContain('◆ resolver_vault');
		expect(frame).toContain('6 notes · 20 links');
		expect(app.api.calls.slice(0, 2)).toEqual(['index.build', 'index.status']);
		expect(promptLine(frame)).toContain('Try /help');
	});

	test('shows "indexing" while the build runs', async () => {
		let finish: () => void = () => {};
		const api = fakeApi({
			build: async () =>
				new Promise((resolve) => {
					finish = () => resolve(undefined as never);
				}),
		});
		const app = renderApp({api});
		await app.waitFor('◌ indexing…');
		finish();
		await app.waitFor('index fresh');
	});

	test('stale index shows how many files changed', async () => {
		const api = fakeApi({
			status: async () => ({...freshStatus, stale: true, pending: {added: ['New.md'], changed: ['A.md', 'B.md'], deleted: [], reason: null}}),
		});
		const app = renderApp({api});
		await app.waitFor('● index stale (3 files changed)');
	});

	test('status is refreshed after every command (notes edited meanwhile)', async () => {
		let calls = 0;
		const api = fakeApi({
			status: async () => {
				calls += 1;
				return calls === 1 ? freshStatus : {...freshStatus, stale: true, pending: {added: [], changed: ['A.md'], deleted: [], reason: null}};
			},
		});
		const app = renderApp({api});
		await app.waitFor('index fresh');
		await app.type('/help');
		await app.press(keys.enter);
		await app.waitFor('● index stale (1 file changed)');
	});

	test('not built and start-up errors', async () => {
		const notBuilt = renderApp({api: fakeApi({status: async () => ({...freshStatus, built: false, stale: null, notes: 0, links: 0})}), autoIndex: false});
		await notBuilt.waitFor('● index not built');
		const broken = renderApp({api: fakeApi({build: async () => Promise.reject(new Error('disk full'))})});
		await broken.waitFor('✗ disk full');
	});

	test('server disconnect is shown and commands fail cleanly', async () => {
		let disconnect: (error: Error) => void = () => {};
		const api = fakeApi({stats: async () => Promise.reject(new ConnectionClosedError('Vault server exited with code 1'))});
		const app = renderApp({
			api,
			onDisconnect(listener) {
				disconnect = listener;
				return () => {};
			},
		});
		await app.waitFor('index fresh');
		disconnect(new ConnectionClosedError('Vault server exited with code 1'));
		await app.waitFor('✗ server disconnected');
		await app.type('/stats');
		await app.press(keys.enter);
		const frame = await app.waitFor('⏺ Vault server exited with code 1\n');
		expect(frame).toContain('Restart vault-tui to reconnect');
	});
});

describe('commands (R6)', () => {
	test('/help lists every registered command and the keys', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/help');
		await app.press(keys.enter);
		const frame = await app.waitFor('Keys');
		for (const usage of ['/health [limit]', '/help', '/note @note | /note <search>', '/stats']) {
			expect(frame).toContain(usage);
		}

		expect(frame).toContain('ctrl+c · ctrl+d');
	});

	test('/stats renders counts and top lists', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/stats');
		await app.press(keys.enter);
		const frame = await app.waitFor('⏺ Vault stats');
		expect(frame).toContain('6 notes · 2 attachments');
		expect(frame).toContain('20 links: 15 resolved · 4 unresolved (4 targets) · 1 ambiguous · 2 embeds');
		expect(frame).toContain('Most linked (linking notes)');
		expect(frame).toContain('Proximal Policy Optimization  Concepts');
		expect(frame).toContain('Most wanted missing notes');
	});

	test('/health renders every check with severity marks', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/health');
		await app.press(keys.enter);
		const frame = await app.waitFor('⏺ Vault health');
		expect(frame).toContain('✗ 4 unresolved links (4 missing targets)');
		expect(frame).toContain('✗ 1 ambiguous links');
		expect(frame).toContain('✗ 1 missing heading/block targets');
		expect(frame).toContain('! 4 orphans (no backlinks)');
		expect(frame).toContain('✓ 0 empty notes');
		expect(frame).toContain('[[Index]]  Notes/Open questions.md:7 → Archive/Index.md | Projects/Index.md');
		expect(frame).toContain('[[Proximal Policy Optimization#Value Function]]  heading not found');
	});

	test('/health validates its argument', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/health zero');
		await app.press(keys.enter);
		await app.waitFor('Usage: /health [limit]');
	});

	test('/note by search shows outline, links out with statuses, and backlinks', async () => {
		const api = fakeApi({
			pick: async ({query}) => ({
				query,
				total: 1,
				results: [{path: 'Notes/Reading notes.md', title: 'Reading notes', matched_field: 'title', matched_text: 'Reading notes', score: 1, positions: []}],
			}),
		});
		const app = renderApp({api});
		await app.waitFor('index fresh');
		await app.type('/note reading');
		await app.press(keys.enter);
		const frame = await app.waitFor('Links in (0)');
		expect(frame).toContain('⏺ Reading notes  Notes/Reading notes.md');
		expect(frame).toContain('best match for "reading"');
		expect(frame).toContain('Links out (9: 9 resolved)');
		expect(frame).toContain('→ Proximal Policy Optimization#Value Function  (not found)');
		expect(frame).toContain('→ (this note)#Local section');
		expect(frame).toContain('→ embed diagram.png');
		expect(frame).toContain('none: this note is an orphan');
	});

	test('/note shows ambiguous and unresolved links', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/note @"Notes/Open questions.md"');
		await app.press(keys.enter);
		const frame = await app.waitFor('Links out (6: 1 resolved)');
		expect(frame).toContain('? Index  ambiguous: Archive/Index.md | Projects/Index.md');
		expect(frame).toContain('✗ GRPO  not in vault');
		expect(frame).toContain('→ Proximal Policy Optimization  via alias "PPO"');
	});
});

describe('@ note picker (R7)', () => {
	test('type @, pick with Enter, run /note', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/note @');
		await app.waitFor('Type to search notes by title or alias…');
		await app.type('ppo');
		const picker = await app.waitFor('alias of Proximal Policy Optimization');
		expect(picker).toContain('❯ PPO');
		await app.press(keys.enter);
		await app.waitFor((frame) => promptLine(frame) === '> /note @"Concepts/Proximal Policy Optimization.md"');
		await app.press(keys.enter);
		const frame = await app.waitFor('Links in (11)');
		expect(frame).toContain('aliases: PPO, Proximal Policy Optimisation');
		expect(frame).toContain('## Clipped Objective');
		expect(frame).toContain('← Reading notes  Notes  #Value Function  (not found here)  L6');
		expect(app.api.calls).toContain('note.get Concepts/Proximal Policy Optimization.md');
	});

	test('↓ and Tab choose another result; Esc closes the picker', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/note @index');
		await app.waitFor((frame) => frame.includes('Archive') && frame.includes('Projects'));
		await app.press(keys.down, keys.tab);
		await app.waitFor((frame) => promptLine(frame) === '> /note @Projects/Index.md');

		await app.type('@opt');
		await app.waitFor('No notes match "opt"');
		await app.press(keys.escape);
		await app.waitFor((frame) => !frame.includes('No notes match'));
		expect(promptLine(app.frame())).toBe('> /note @Projects/Index.md @opt');
	});

	test('long titles stay on one line per result', async () => {
		const long = 'EVD - A Terminal-Bench audit found a task whose tests assumed a file path the spec never stated';
		const api = fakeApi({
			pick: async ({query}) => ({
				query,
				total: 2,
				results: [
					{path: `50_MOC/discourse/${long}.md`, title: long, matched_field: 'title', matched_text: long, score: 2, positions: [0]},
					{path: 'short.md', title: 'short', matched_field: 'title', matched_text: 'short', score: 1, positions: [0]},
				],
			}),
		});
		const app = renderApp({api});
		await app.waitFor('index fresh');
		await app.type('/note @evd');
		const frame = await app.waitFor('  short');
		const rows = frame.split('\n');
		const first = rows.findIndex((row) => row.startsWith('❯ EVD'));
		expect(rows[first]!.length).toBeLessThanOrEqual(100);
		expect(rows[first]!.endsWith('…')).toBe(true);
		expect(rows[first + 1]).toBe('  short');
	});

	test('unknown @path is reported as an engine error and the app keeps working', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/note @Missing.md');
		await app.press(keys.escape, keys.enter);
		await app.waitFor("⏺ No indexed note at 'Missing.md' (error -32002)");
		await app.type('/help');
		await app.press(keys.enter);
		await app.waitFor('Keys');
	});
});

describe('prompt: suggestions and history (R3, R8)', () => {
	test('"/" lists commands; ↓ moves; Tab completes; Enter runs the highlighted one', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/');
		let frame = await app.waitFor('/stats');
		for (const name of ['/health', '/help', '/note', '/stats']) {
			expect(frame).toContain(name);
		}

		await app.type('he');
		frame = await app.waitFor((f) => !f.includes('/stats  '));
		expect(frame).toContain('❯ /health');
		await app.press(keys.down, keys.tab);
		await app.waitFor((f) => promptLine(f) === '> /help');

		await app.press(keys.ctrlU);
		await app.type('/st');
		await app.press(keys.enter);
		await app.waitFor('⏺ Vault stats');
	});

	test('unknown commands and plain text get a hint', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/hx');
		await app.press(keys.escape, keys.enter);
		await app.waitFor('Unknown command /hx. Did you mean /health, /help?');
		await app.type('hello');
		await app.press(keys.enter);
		await app.waitFor('Not a command. Commands start with "/"; try /help.');
	});

	test('keys that arrive in one chunk (typed fast or pasted): a trailing Enter still runs the command', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		app.stdin.write('/health\r'); // one chunk, as a pty delivers fast typing
		await app.waitFor('⏺ Vault health');
		app.stdin.write('/he\r'); // Enter picks the highlighted suggestion, as a keypress would
		await app.waitFor((frame) => frame.split('⏺ Vault health').length === 3);
		app.stdin.write('/note one\ntwo'); // a line break inside a paste is a space, not Enter
		await app.waitFor((frame) => promptLine(frame) === '> /note one two');
	});

	test('↑/↓ recall history and restore the draft', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		for (const line of ['/help', '/stats']) {
			await app.type(line);
			await app.press(keys.escape, keys.enter);
			await app.waitFor((f) => promptLine(f).includes('Try /help'));
		}

		await app.type('draft');
		await app.press(keys.up);
		await app.waitFor((f) => promptLine(f) === '> /stats');
		await app.press(keys.up);
		await app.waitFor((f) => promptLine(f) === '> /help');
		await app.press(keys.down, keys.down);
		await app.waitFor((f) => promptLine(f) === '> draft');
	});

	test('line editing: cursor moves, backspace, ctrl+u', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/stts');
		await app.press(keys.left, keys.left); // "/st|ts"
		await app.type('a');
		await app.waitFor((f) => promptLine(f) === '> /stats');
		await app.press(keys.backspace);
		await app.waitFor((f) => promptLine(f) === '> /stts');
		await app.press(keys.ctrlU); // deletes "/st" before the cursor
		await app.waitFor((f) => promptLine(f) === '> ts');
	});

	test('Enter is ignored while a command runs; a spinner shows', async () => {
		let finish: () => void = () => {};
		const api = fakeApi({
			health: async () =>
				new Promise((resolve) => {
					finish = () => resolve(fakeApi().health());
				}),
		});
		const app = renderApp({api});
		await app.waitFor('index fresh');
		await app.type('/health');
		await app.press(keys.enter);
		await app.waitFor('Working…');
		await app.type('/stats');
		await app.press(keys.enter);
		await delay(50);
		expect(app.api.calls).not.toContain('vault.stats');
		finish();
		await app.waitFor('⏺ Vault health');
		expect(promptLine(app.frame())).toBe('> /stats'); // kept, can be sent now
	});
});

describe('command registry is the extension point (R5)', () => {
	test('a newly registered command appears in /help and suggestions, and runs', async () => {
		const registry = new CommandRegistry().register(...builtinCommands, {
			name: 'orphans',
			summary: 'List notes nobody links to',
			async run(_input, {api}) {
				const report = await api.health({});
				return <Text>orphans: {report.orphans.join(', ')}</Text>;
			},
		});
		const app = renderApp({registry});
		await app.waitFor('index fresh');
		await app.type('/or');
		await app.waitFor((f) => /❯ \/orphans\s+List notes nobody links to/.test(f));
		await app.press(keys.enter);
		await app.waitFor('orphans: Archive/Index.md, Notes/Open questions.md');
		await app.type('/help');
		await app.press(keys.enter);
		await app.waitFor((f) => f.split('Keys')[0]!.includes('/orphans'));
	});
});
