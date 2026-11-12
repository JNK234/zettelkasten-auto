/**
 * Frame tests for /search (interactive list, preview, open in Obsidian) and
 * /related, /suggest, /unlinked. The API replays responses recorded from the
 * real server on the sample vault (tests/fixtures/rpc/sample_vault.jsonl).
 * Matches render as «match» here: frame tests have no colour (see Marked.tsx).
 */
import {afterEach, describe, expect, test} from 'vitest';
import {cleanup} from 'ink-testing-library';
import type {LinksSuggestResult, NoteRelatedResult} from '../src/rpc/schema.js';
import {fakeApi, keys, promptLine, renderApp, sampleReply} from './helpers.js';

afterEach(() => {
	cleanup();
});

const QUERY = 'reward hacking';
const FIRST = '50_MOC/topics/emergent-misalignment-from-reward-hacking.md';
const SECOND = '20_ZETTLEKASTEN/reward-hacking-as-eval-false-positive.md';

/** The frame with every run of whitespace (line wraps included) as one space. */
function flat(frame: string): string {
	return frame.replaceAll(/\s+/g, ' ');
}

/** The row of the selected hit ("❯   2. title  score"). */
function selectedRow(frame: string): string {
	return frame.split('\n').find((row) => row.startsWith('❯ ')) ?? '';
}

async function openSearch(query = QUERY) {
	const app = renderApp({vaultPath: '/vaults/sample-vault'});
	await app.waitFor('index fresh');
	await app.type(`/search ${query}`);
	await app.press(keys.escape, keys.enter);
	await app.waitFor('Enter preview');
	return app;
}

describe('/search: a list to move through', () => {
	test('shows the ranked hits with matches marked, and the keys', async () => {
		const app = await openSearch();
		const frame = app.frame();
		expect(frame).toContain('⏺ Search “reward hacking”  106 of 180 notes · exact matches first, then score');
		expect(selectedRow(frame)).toBe('❯  1. emergent-misalignment-from-«reward»-«hacking»  3.93');
		expect(frame).toContain('      50_MOC/topics  ·  reward: title, heading, body ×31 · hacking: title, heading, body ×38');
		expect(frame).toContain('«reward» «hacking». - The SDF «hacking» model attempted sabotage');
		expect(frame).toContain('1 of 30 shown (106 match)');
		expect(frame).toContain('↑↓ move · Enter preview · o open in Obsidian · Esc close');
		expect(promptLine(frame)).toBe(''); // the prompt is hidden while the list has the keys
		expect(app.api.calls).toContain('search.query reward hacking');
	});

	test('↓ ↑ move the selection; PgDn jumps a page; the list scrolls with it', async () => {
		const app = await openSearch();
		await app.press(keys.down);
		await app.waitFor((f) => selectedRow(f).startsWith('❯  2. «reward»-«hacking»-as-eval-false-positive'));
		expect(app.frame()).toContain('2 of 30 shown');
		await app.press(keys.up, keys.up);
		await app.waitFor('30 of 30 shown'); // wraps around to the last hit
		expect(selectedRow(app.frame())).toMatch(/^❯ 30\. /);
		expect(app.frame()).not.toContain('  1. emergent-misalignment'); // scrolled: the first page is gone
		await app.press(keys.down); // back to the first
		await app.waitFor('1 of 30 shown');
		await app.press(keys.pageDown);
		await app.waitFor((f) => /\n\s+(\d+) of 30 shown/.exec(f)?.[1] !== '1');
		const position = Number(/\n\s+(\d+) of 30 shown/.exec(app.frame())?.[1]);
		expect(position).toBeGreaterThan(2);
	});

	test('typing letters does not reach the prompt while the list is open', async () => {
		const app = await openSearch();
		await app.type('xyz');
		await app.press(keys.escape);
		await app.waitFor((f) => promptLine(f).includes('Try /help'));
		expect(promptLine(app.frame())).not.toContain('xyz');
	});
});

describe('/search: preview with highlighted matches', () => {
	test('Enter opens the preview at the first match; n jumps to the next; Esc returns to the same hit', async () => {
		const app = await openSearch();
		await app.press(keys.down, keys.enter);
		const preview = await app.waitFor('Esc back to the list');
		expect(preview).toContain(`⏺ reward-hacking-as-eval-false-positive  ${SECOND}`);
		expect(preview).toMatch(/\d+ matches · lines \d+–\d+ of \d+/);
		expect(preview).toContain('│ # «Reward» «Hacking» as an Eval False Positive (Weak Model Looks Capable)'); // the heading, marked
		expect(app.api.calls).toContain(`note.preview ${SECOND}`);

		const before = /lines (\d+)–/.exec(preview)?.[1];
		await app.type('n');
		await app.waitFor((f) => /lines (\d+)–/.exec(f)?.[1] !== before);
		await app.press(keys.escape);
		await app.waitFor('Enter preview');
		expect(selectedRow(app.frame())).toMatch(/^❯ {2}2\. «reward»-«hacking»-as-eval-false-positive/);
	});

	test('a preview that takes a while still arrives (and a stale one is not shown)', async () => {
		const slow = fakeApi();
		const answer = slow.preview.bind(slow);
		const api = fakeApi({
			preview: async (params) =>
				new Promise((resolve, reject) => {
					setTimeout(() => {
						answer(params).then(resolve, reject);
					}, 80);
				}),
		});
		const app = renderApp({api, vaultPath: '/vaults/sample-vault'});
		await app.waitFor('index fresh');
		await app.type(`/search ${QUERY}`);
		await app.press(keys.escape, keys.enter);
		await app.waitFor('Enter preview');
		await app.press(keys.enter);
		await app.waitFor('Loading');
		await app.press(keys.escape, keys.down, keys.enter); // another note while the first is loading
		const frame = await app.waitFor((f) => /\d+ matches · lines/.test(f));
		expect(frame).toContain('⏺ reward-hacking-as-eval-false-positive');
	});

	test('↓ scrolls the preview one line, ↑ scrolls back', async () => {
		const app = await openSearch();
		await app.press(keys.enter);
		const first = await app.waitFor('Esc back to the list');
		expect(first).toContain('⏺ emergent-misalignment-from-reward-hacking');
		expect(first.replaceAll(/\s+/g, '')).toContain(FIRST); // the path, possibly wrapped
		const top = Number(/lines (\d+)–/.exec(first)?.[1]);
		await app.press(keys.down);
		await app.waitFor(`lines ${top + 1}–`);
		await app.press(keys.up);
		await app.waitFor(`lines ${top}–`);
	});
});

describe('/search: open in Obsidian', () => {
	test('o opens obsidian://open?vault=<vault folder>&file=<path> from the list and from the preview', async () => {
		const app = await openSearch();
		await app.type('o');
		const url = 'obsidian://open?vault=sample-vault&file=50_MOC%2Ftopics%2Femergent-misalignment-from-reward-hacking.md';
		const frame = await app.waitFor(`↗ Opened ${FIRST} in Obsidian`);
		expect(frame.replaceAll(/\s+/g, '')).toContain(url); // the URL, wrapped under the notice
		expect(app.opened).toEqual([url]);

		await app.press(keys.down, keys.enter);
		await app.waitFor('Esc back to the list');
		await app.type('o');
		await app.waitFor(`↗ Opened ${SECOND} in Obsidian`);
		expect(app.opened).toEqual([url, `obsidian://open?vault=sample-vault&file=${encodeURIComponent(SECOND)}`]);
	});

	test('a failing opener is reported, and the list keeps working', async () => {
		const app = renderApp({
			vaultPath: '/vaults/sample-vault',
			openUrl: async () => {
				throw new Error('could not run xdg-open: spawn xdg-open ENOENT');
			},
		});
		await app.waitFor('index fresh');
		await app.type(`/search ${QUERY}`);
		await app.press(keys.escape, keys.enter);
		await app.waitFor('Enter preview');
		await app.type('o');
		await app.waitFor('✗ Could not open Obsidian: could not run xdg-open: spawn xdg-open ENOENT');
		await app.press(keys.down);
		await app.waitFor('2 of 30 shown');
	});
});

describe('/search: closing, errors, empty results', () => {
	test('Esc closes the list: a summary stays in the transcript and the prompt is back', async () => {
		const app = await openSearch();
		await app.press(keys.escape);
		const frame = await app.waitFor((f) => promptLine(f).includes('Try /help'));
		expect(frame).toContain('> /search reward hacking');
		expect(frame).toContain('⏺ Search “reward hacking”  106 of 180 notes');
		expect(frame).toContain('… 96 more (narrow the query, e.g. add tag:… or path:…)');
		expect(frame).toContain('tags: #baby 84 · #evidence 10 · #ai-safety 9');
		expect(frame).not.toContain('Enter preview');
		await app.type('/help');
		await app.press(keys.enter);
		await app.waitFor('Keys');
	});

	test('a query syntax error shows the query with a caret under the problem', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/search reward (tag:ai-safety OR path:50_MOC');
		await app.press(keys.escape, keys.enter);
		const frame = await app.waitFor('Invalid query');
		const rows = frame.split('\n');
		const at = rows.findIndex((row) => row.includes('⏺ Invalid query: this ( is never closed'));
		expect(rows[at + 1]).toBe('     reward (tag:ai-safety OR path:50_MOC');
		expect(rows[at + 2]).toBe('            ^');
		expect(rows[at + 3]).toBe('     add ) after the group');
		expect(rows[at + 2]!.indexOf('^')).toBe(rows[at + 1]!.indexOf('('));
	});

	test('an unknown field suggests the right one', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/search tga:ai-safety reward');
		await app.press(keys.escape, keys.enter);
		const frame = await app.waitFor('Invalid query: unknown field "tga:"');
		expect(frame).toContain('     ^^^^\n');
		expect(frame).toContain('did you mean tag:?');
	});

	test('no results: a plain answer, no list to close', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/search zetelkasten');
		await app.press(keys.escape, keys.enter);
		const frame = await app.waitFor('no notes match');
		expect(frame).toContain('⏺ Search “zetelkasten”  0 of 180 notes · by score');
		expect(promptLine(frame)).toContain('Try /help');
	});

	test('filters with a phrase: warnings and the filtered hits', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/search "reward hacking" links-to:[[grpo-group-relative-policy-optimization]] -path:10_SOURCES');
		await app.press(keys.escape, keys.enter);
		const frame = await app.waitFor('Enter preview');
		expect(flat(frame)).toContain('note: [[grpo-group-relative-policy-optimization]] is not a file in the vault; using the links written to it (10 notes)');
		expect(frame).toContain('1 of 2');
		expect(selectedRow(frame)).toContain('instance-specific-rubric-as-reward-function');
	});

	test('/search without a query explains the usage', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/search');
		await app.press(keys.escape, keys.enter);
		await app.waitFor('Usage: /search <query>');
	});
});

describe('/related, /suggest, /unlinked', () => {
	test('/related @note lists related notes with linked / not linked and the reason', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/related @20_ZETTLEKASTEN/kl-penalty-in-rlhf.md');
		await app.press(keys.escape, keys.enter);
		const frame = await app.waitFor('⏺ Related to kl-penalty-in-rlhf');
		expect(frame).toContain('score = 0.5 × graph + 0.3 × text + 0.2 × tags · 179 notes compared');
		expect(frame).toMatch(/ 1\. direct-preference-optimization-dpo {2}20_ZETTLEKASTEN {2}0\.87 {2}linked/);
		expect(flat(frame)).toContain('linked from it · shares [[reinforcement-learning-from-human-feedback-rlhf]], [[policy-gradient-methods]]');
		expect(frame).toMatch(/ 3\. RLHF {2}10_SOURCES {2}0\.50 {2}not linked/);
		expect(frame).toContain('2 hops via direct-preference-optimization-dpo +1 · similar text (rlhf, reward, policy)');
		// Recorded from a vault that was never `vault embed`-ed: the server computed the vectors without writing them.
		expect(flat(frame)).toContain('180 notes had no cached vector: embedded for this answer only; `vault embed` caches them');
	});

	test('/related without the embedding model: links and tags still rank, the text signal says why it is off', async () => {
		const off = {available: false, model_id: null, unavailable: 'embedding model folder not found: /no/model', uncached: 0};
		const api = fakeApi({
			related: async (params) => ({...sampleReply<NoteRelatedResult>('note.related', params), text_signal: off, weights: {graph: 0.5 / 0.7, text: 0, tags: 0.2 / 0.7}}),
			suggestLinks: async (params) => ({...sampleReply<LinksSuggestResult>('links.suggest', params ?? {}), text_signal: off, weights: {shared: 0.4 / 0.7, cited: 0.3 / 0.7, text: 0}}),
		});
		const app = renderApp({api});
		await app.waitFor('index fresh');
		await app.type('/related @20_ZETTLEKASTEN/kl-penalty-in-rlhf.md');
		await app.press(keys.escape, keys.enter);
		const frame = await app.waitFor('⏺ Related to kl-penalty-in-rlhf');
		expect(frame).toContain('score = 0.71 × graph + 0.29 × tags · 179 notes compared');
		expect(flat(frame)).toContain('text signal unavailable: embedding model folder not found: /no/model (the other signals still rank)');
		await app.type('/suggest links');
		await app.press(keys.escape, keys.enter);
		const links = await app.waitFor('⏺ Links to add');
		expect(flat(links)).toContain('score = 0.57 × shared neighbours + 0.43 × cited together');
		expect(flat(links)).toContain('text signal unavailable: embedding model folder not found: /no/model');
	});

	test('/related on a missing note, and on a name that is nothing', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/related rl-environment-design');
		await app.press(keys.escape, keys.enter);
		await app.waitFor('⏺ Related to rl-environment-design  no note yet; 52 notes link to it');
		await app.type('/related nope');
		await app.press(keys.escape, keys.enter);
		await app.waitFor('No note matches "nope"');
	});

	test('/suggest shows links to add and notes to write', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/suggest');
		await app.press(keys.escape, keys.enter);
		const frame = await app.waitFor('⏺ Notes to write');
		expect(flat(frame)).toContain('⏺ Links to add score = 0.33 × shared neighbours + 0.25 × cited together + 0.17 × shared tags + 0.25 × similar text');
		expect(flat(frame)).toContain('EVD - AppWorld fails any database change outside an allow-list ↔ EVD - AppWorld grades only the database');
		// Tags: links are tags: discourse-graph is a shared tag now, not a shared neighbour.
		expect(flat(frame)).toContain('shared neighbours: StonyBrookNLP-appworld - Controllable World… · shares #evidence, [[discourse-graph]], [[rl-environment-design]] · cited together by');
		expect(frame).toContain('1. rl-environment-design  52 notes link here · 1 more mention it');
		expect(flat(frame)).toContain('[[on-policy-vs-off-policy-learning]] (5 notes) → 20_ZETTLEKASTEN/on-policy-versus-off-policy-learning.md');
	});

	test('/suggest @note, /suggest links, /suggest notes, and a bad argument', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/suggest @20_ZETTLEKASTEN/kl-penalty-in-rlhf.md');
		await app.press(keys.escape, keys.enter);
		await app.waitFor('⏺ Links to add with kl-penalty-in-rlhf');
		await app.type('/suggest notes');
		await app.press(keys.escape, keys.enter);
		await app.waitFor('10. ');
		await app.type('/suggest nonsense');
		await app.press(keys.escape, keys.enter);
		await app.waitFor('Usage: /suggest, /suggest links, /suggest notes or /suggest @note');
		expect(app.api.calls).toContain('notes.suggest {"limit":10}');
	});

	test('/unlinked @note marks each plain-text mention', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/unlinked @20_ZETTLEKASTEN/monitorability-tax.md');
		await app.press(keys.escape, keys.enter);
		const frame = await app.waitFor('⏺ Unlinked mentions of monitorability-tax');
		expect(frame).toContain('4 mentions in 1 note that does not link to it yet');
		expect(frame).toContain('10_SOURCES/2503.11926 - Monitoring Reasoning Models (CoT Obfuscation).md:22');
		expect(flat(frame)).toContain('They call the cost of avoiding this the **«monitorability tax»**.');
		expect(flat(frame)).toContain('## The «monitorability tax»');
	});

	test('/unlinked lists the most mentioned notes', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/unlinked');
		await app.press(keys.escape, keys.enter);
		const frame = await app.waitFor('⏺ Unlinked mentions of any note');
		expect(frame).toContain('595 mentions in 79 notes that do not link to it yet');
		expect(frame).toContain('reward  71 notes');
		expect(frame).toContain('… 509 more');
	});

	test('/help lists the new commands and the search keys', async () => {
		const app = renderApp();
		await app.waitFor('index fresh');
		await app.type('/help');
		await app.press(keys.enter);
		const frame = await app.waitFor('Keys');
		for (const usage of ['/search <query>', '/related @note | /related <name>', '/suggest [links | notes | @note]', '/unlinked [@note | <name>]']) {
			expect(frame).toContain(usage);
		}

		expect(frame).toContain('↑↓ move · Enter preview · o open in Obsidian · Esc back / close');
	});
});

test('the fake API names any call that was never recorded', async () => {
	const api = fakeApi();
	await expect(api.search({query: 'never recorded', limit: 30})).rejects.toThrow('no recorded sample_vault reply for search.query');
});
