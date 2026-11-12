/**
 * Frame tests for /explore (interactive: sections, open a result, pivot, back, keep, dismiss, link it, run again),
 * /kept, /map (record, since) and /config. The API replays the responses recorded from the real server on the
 * sample vault (tests/fixtures/rpc/sample_vault_explore.jsonl, replayed through `vault serve` by the Python tests).
 */
import {afterEach, describe, expect, test} from 'vitest';
import {cleanup} from 'ink-testing-library';
import {pivotTarget, windowOf} from '../src/ui/views/ExploreView.js';
import type {ExploreRunResult} from '../src/rpc/schema.js';
import {exploreReply, keys, promptLine, renderApp} from './helpers.js';

afterEach(() => {
	cleanup();
});

const NOTE = '20_ZETTLEKASTEN/kl-penalty-in-rlhf.md';

/** The frame with every run of whitespace (line wraps included) as one space. */
function flat(frame: string): string {
	return frame.replaceAll(/\s+/g, ' ');
}

/** The row of the selected result ("❯   1.1  RLHF …"). */
function selectedRow(frame: string): string {
	return frame.split('\n').find((row) => row.startsWith('❯ ')) ?? '';
}

async function openExplore(args = `@${NOTE}`) {
	const app = renderApp({vaultPath: '/vaults/sample-vault'});
	await app.waitFor('index fresh');
	await app.type(`/explore ${args}`);
	await app.press(keys.escape, keys.enter);
	await app.waitFor('Esc close');
	return app;
}

/** Move the selection down until the selected row starts with `ref`. */
async function selectRef(app: Awaited<ReturnType<typeof openExplore>>, ref: string) {
	for (let step = 0; step < 40 && !selectedRow(app.frame()).includes(` ${ref}  `); step++) {
		await app.press(keys.down);
	}

	expect(selectedRow(app.frame())).toContain(` ${ref}  `);
}

describe('/explore: sections to move through', () => {
	test('shows a section per lens, every result with its lens and evidence, and the keys', async () => {
		const app = await openExplore();
		const frame = app.frame();
		expect(frame).toContain('⏺ Explore kl-penalty-in-rlhf  10 lenses · deepen 13, discover 11');
		expect(frame).toContain('1. Neighbourhood  deepen · 4 of 16, 2 near-duplicates · no marks yet');
		expect(selectedRow(frame)).toBe('❯   1.1  RLHF');
		expect(frame).toContain('         neighbourhood · 2 hops via direct-preference-optimization-dpo +1 · similar text (rlhf, rew…');
		expect(frame).toContain('1 of 24');
		expect(frame).toContain('↑↓ Enter open · k keep · d dismiss · l link · p pivot · b back · r again · f fresh · Esc close');
		expect(promptLine(frame)).toBe(''); // the keys go to the list
		expect(app.api.calls).toContain(`explore.run {"note":"${NOTE}"}`);
	});

	test('Enter opens the result: the full evidence, its notes, what link it would join; Esc goes back', async () => {
		const app = await openExplore();
		await selectRef(app, '6.1');
		await app.press(keys.enter);
		const frame = await app.waitFor('Esc back to the list');
		expect(frame).toContain('⏺ 6.1  covariance-targeted-entropy-intervention  Hidden link · discover');
		expect(flat(frame)).toContain("hidden · no link either way; 8 middles in common: 'kl' (4 notes), 'region' (4 notes),");
		expect(flat(frame)).toContain('link it: [[covariance-targeted-entropy-intervention]] in kl-penalty-in-rlhf, and back, under See Also');
		await app.press(keys.escape);
		await app.waitFor('Esc close');
		expect(selectedRow(app.frame())).toContain(' 6.1  covariance-targeted-entropy-intervention');
	});

	test('the window follows the selection', () => {
		const data = exploreReply<ExploreRunResult>('explore.run', {note: NOTE});
		const rows = data.sections.flatMap((section, number) => [
			{kind: 'section' as const, section, number: number + 1},
			...section.results.map((item) => ({kind: 'item' as const, item, index: 0})),
		]);
		expect(windowOf(rows, 0, 100).length).toBe(rows.length);
		expect(pivotTarget(data.sections[0]!.results[0]!)).toBe('10_SOURCES/RLHF.md');
	});
});

describe('/explore: keep, dismiss, link it, pivot, back, run again', () => {
	test('k keeps and d dismisses the selected result; the engine remembers them (explore.mark)', async () => {
		const app = await openExplore();
		await app.press('k');
		await app.waitFor('[kept]');
		expect(selectedRow(app.frame())).toBe('❯   1.1  [kept] RLHF');
		expect(app.frame()).toContain('✓ Kept 1.1 RLHF · neighbourhood: kept 1 of 1');
		await selectRef(app, '7.1');
		await app.press('d');
		await app.waitFor('[dismissed]');
		expect(app.frame()).toContain('✓ Dismissed 7.1 rubric-agreement-does-not-establish-rubric-validity · role: kept 0 of 1');
		expect(app.api.calls).toEqual(expect.arrayContaining(['explore.mark {"verdict":"keep","refs":["1.1"]}', 'explore.mark {"verdict":"dismiss","refs":["7.1"]}']));
	});

	test('l shows the dry run of the link in both notes; w writes it; the result counts as kept', async () => {
		const app = await openExplore();
		await selectRef(app, '8.1');
		await app.press('l');
		let frame = await app.waitFor('w writes it in both notes; Esc cancels');
		expect(frame).toContain('⏺ Dry run: would write 2 notes  nothing was written');
		expect(flat(frame)).toContain('+- [[persona-driven-synthetic-data-generation]] — gap: closeness of their mean embeddings 0.38');
		expect(flat(frame)).toContain('+- [[reward-computation-as-a-heavyweight-pipeline-stage]] — gap:');
		expect(frame).toContain('w write it (both notes) · Esc cancel');
		await app.press('w');
		frame = await app.waitFor('⏺ Wrote 2 notes');
		expect(frame).toContain('✓ check: no new violations');
		expect(flat(frame)).toContain('✓ Linked reward-computation-as-a-heavyweight-pipeline-stage ↔ persona-driven-synthetic-data-generation; kept 8.1');
		await app.press(keys.escape);
		await app.waitFor('Esc close');
		expect(selectedRow(app.frame())).toContain(' 8.1  [kept] communities 6 and 12');
		expect(app.api.calls).toEqual(expect.arrayContaining(['explore.link {"ref":"8.1"}', 'explore.link {"ref":"8.1","dry_run":false}']));
	});

	test('p explores from the selected result; b goes back to the previous start', async () => {
		const app = await openExplore();
		await app.press('p');
		let frame = await app.waitFor('⏺ Explore RLHF');
		expect(selectedRow(frame)).toBe('❯   1.1  kl-penalty-in-rlhf');
		expect(app.api.calls).toContain('explore.run {"note":"10_SOURCES/RLHF.md"}');
		await app.press('b');
		frame = await app.waitFor('⏺ Explore kl-penalty-in-rlhf');
		expect(selectedRow(frame)).toBe('❯   1.1  RLHF');
	});

	test('f runs again with --fresh: what was seen or dismissed is left out', async () => {
		const app = await openExplore();
		await app.press('f');
		const frame = await app.waitFor('· fresh');
		expect(frame).toContain('1. Neighbourhood  deepen · 4 of 16, 5 seen before, 3 near-duplicates · kept 1 of 1 (100%)');
		expect(selectedRow(frame)).toBe('❯   1.1  Cameron R. Wolfe - GRPO Tricks (Making RL Actually Work)');
		expect(frame).toContain('vault-wide numbers from');
		expect(app.api.calls).toContain(`explore.run {"note":"${NOTE}","fresh":true}`);
	});

	test('Esc closes it and leaves a summary in the transcript; the prompt is back', async () => {
		const app = await openExplore();
		await app.press(keys.escape);
		const frame = await app.waitFor((f) => promptLine(f).includes('Try /help'));
		expect(frame).toContain('> /explore @20_ZETTLEKASTEN/kl-penalty-in-rlhf.md');
		expect(frame).toContain('⏺ Explore kl-penalty-in-rlhf  10 lenses · deepen 13, discover 11');
		expect(flat(frame)).toContain('1. Neighbourhood RLHF · covariance-targeted-entropy-intervention ·');
	});

	test('without a note: where to start, each proposal from a lens', async () => {
		const app = await openExplore('');
		const frame = app.frame();
		expect(frame).toContain('⏺ Explore where to start  proposals from 4 lenses');
		expect(frame).toContain('1. Gap  discover · 3 of 4, 1 near-duplicate · no marks yet');
		expect(selectedRow(frame)).toContain('   1.1  communities 8 and 9: [[fine-tuning-strategies-for-llms]]');
		expect(app.api.calls).toContain('explore.run {}');
	});
});

describe('/kept, /map, /config', () => {
	test('/kept lists what was kept, with its lens and evidence', async () => {
		const app = renderApp({vaultPath: '/vaults/sample-vault'});
		await app.waitFor('index fresh');
		await app.type('/kept');
		await app.press(keys.escape, keys.enter);
		const frame = await app.waitFor('⏺ Kept  2 results');
		expect(frame).toContain('RLHF  from kl-penalty-in-rlhf');
		expect(flat(frame)).toContain("hidden · no link either way; 8 middles in common: 'kl' (4 notes)");
	});

	test('/map, /map record, /map since', async () => {
		const app = renderApp({vaultPath: '/vaults/sample-vault'});
		await app.waitFor('index fresh');
		await app.type('/map');
		await app.press(keys.escape, keys.enter);
		let frame = await app.waitFor('⏺ Map · 12 communities');
		expect(frame).toContain('1. [[discourse-graph]] · #evidence  40 notes');
		expect(frame).toContain('Bridge notes');
		await app.type('/map record');
		await app.press(keys.escape, keys.enter);
		frame = await app.waitFor('✓ recorded a snapshot');
		expect(frame).toContain('4 gaps): /map since');
		await app.type('/map since 2000-01-01');
		await app.press(keys.escape, keys.enter);
		frame = await app.waitFor('Gaps: 1 of 4 closed');
		expect(frame).toContain('closed  [[systems-engineering]] · [[reinforcement-learning-in-language-models]] / #prompt-en…');
		expect(frame).toContain('2 links between the two sides where chance predicts 0.5: no longer fewer than chance');
		expect(frame).toContain('new link persona-driven-synthetic-data-generation ↔ reward-computation-as-a-heavywei…');
		expect(frame).toContain('gone   12. #prompt-engineering (2 notes then)');
	});

	test('/config shows every setting and where it came from', async () => {
		const app = renderApp({vaultPath: '/vaults/sample-vault'});
		await app.waitFor('index fresh');
		await app.type('/config');
		await app.press(keys.escape, keys.enter);
		const frame = await app.waitFor('⏺ Engine settings  precedence: flag > env > file > default');
		expect(frame).toContain('related.unlinked_weights.text');
		expect(frame).toContain('explore.lenses');
		expect(frame).toMatch(/format +brain {2}flag --format/);
	});
});
