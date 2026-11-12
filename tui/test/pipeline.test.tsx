/**
 * Frame tests for the reading-pipeline commands: /pipeline, /page, /findings, /idea, /move, /merge, /next,
 * and /check baseline. Results are real engine responses recorded in tests/fixtures/rpc/sample_vault_core.jsonl
 * (served with --format core; request ids are noted next to each use).
 */
import {afterEach, describe, expect, test} from 'vitest';
import {cleanup} from 'ink-testing-library';
import type {VaultApi} from '../src/rpc/api.js';
import {coreResult, fakeApi, keys, renderApp} from './helpers.js';

afterEach(() => {
	cleanup();
});

/** The frame with wrapped lines joined: entries are laid out at the terminal width (100 in tests). */
const flat = (frame: string) => frame.replaceAll(/\s*\n\s*/g, ' ');

/** Lines wider than the 100 columns ink-testing-library lays out at (none should be). */
const tooWide = (frame: string) => frame.split('\n').filter((line) => [...line.trimEnd()].length > 100);

const Z1 = 'a-bounty-list-is-a-map-of-what-cannot-be-verified';
const Z3 = 'agent-test-time-scaling-limited-by-verification-gap';

/** An api whose `method` answers with recorded request `id` and remembers the params it was given. */
function answering<K extends keyof VaultApi>(method: K, id: number | ((params: unknown) => number)) {
	const seen: unknown[] = [];
	const api = fakeApi({
		[method]: async (params: unknown) => {
			seen.push(params);
			return coreResult(typeof id === 'number' ? id : id(params));
		},
	} as Partial<VaultApi>);
	return {api, seen};
}

async function run(app: ReturnType<typeof renderApp>, line: string) {
	await app.waitFor('index fresh');
	await app.type(line);
	await app.press(keys.escape, keys.enter);
}

describe('/pipeline', () => {
	test('stages with what they mean, counts, sources by stage, and the gaps', async () => {
		const {api, seen} = answering('pipeline', 7); // vault.pipeline {limit: 3}
		const app = renderApp({api});
		await run(app, '/pipeline');
		const frame = await app.waitFor('Zettels in no page (104)');
		expect(seen).toEqual([{limit: 8}]);
		expect(frame).toContain('⏺ Pipeline  core · 28 source notes');
		expect(frame).toMatch(/compiled +3 +a page links to a zettel that links to it/);
		expect(frame).toMatch(/extracted +12 +a zettel links to it, no page uses those yet/);
		expect(frame).toMatch(/ingested +13 +no zettel links to it/);
		expect(frame).toContain('109 zettel notes: 5 in a page · 28 resting on a source · 81 on no source');
		expect(frame).toContain('on no source: 78 cite sources not in the vault, 3 cite nothing');
		expect(frame).toMatch(/compiled +5 z +1 p 2503\.11926 - Monitoring Reasoning Models \(CoT Obfuscation\) {2}10_SOURCES/);
		expect(frame).toContain('Not extracted yet (13)');
		expect(frame).toContain('… 25 more · /pipeline all');
		expect(frame).toContain(`${Z1}  20_ZETTLEKASTEN`);
	});

	test('a format without a pipeline: block says which format has one', async () => {
		const {RpcError} = await import('../src/rpc/client.js');
		const api = fakeApi({
			pipeline: async () => {
				throw new RpcError(-32_003, "Format 'brain' has no `pipeline:` block", {format: '/x/brain.yaml'});
			},
		});
		const app = renderApp({api});
		await run(app, '/pipeline');
		await app.waitFor((frame) =>
			flat(frame).includes("Format 'brain' has no `pipeline:` block: start vault-tui with a format that has one, e.g. --format core (error -32003)"),
		);
	});

	test('/pipeline all asks for every row; bad arguments', async () => {
		const {api, seen} = answering('pipeline', 7);
		const app = renderApp({api});
		await run(app, '/pipeline all');
		await app.waitFor('Not extracted yet');
		expect(seen).toEqual([{limit: null}]);
		await app.type('/pipeline a b');
		await app.press(keys.escape, keys.enter);
		await app.waitFor('Usage: /pipeline [all | <rows>]');
	});
});

describe('/page', () => {
	test('new: a dry run shows the page it would write, with its index and log lines', async () => {
		const {api, seen} = answering('pageNew', 10); // page.new {zettels, dry_run: true}
		const app = renderApp({api});
		await run(app, `/page new "Verification gaps" @${Z1} @agent-layer-inverts-who-owns-the-generation-loop @${Z3}`);
		const frame = await app.waitFor('new index: 4 pages');
		expect(seen).toEqual([
			{
				title: 'Verification gaps',
				zettels: [Z1, 'agent-layer-inverts-who-owns-the-generation-loop', Z3],
				queries: null,
				tags: null,
				dry_run: true,
			},
		]);
		expect(frame).toContain('⏺ Dry run: would write 3 notes  nothing was written');
		expect(frame).toContain('state .vaultengine/pages.json: 50_MOC/verification-gaps.md: 3 zettels recorded');
		expect(frame).toContain('add write to the command to write it');
		expect(frame).toContain('+ verification-gaps  50_MOC');
		expect(frame).toContain('new page: Verification gaps (3 zettels)');
		expect(frame).toContain('<!-- vault:zettels -->');
		expect(frame).toContain(`- [[${Z1}]]: A reading technique`);
		expect(frame).toContain('- [[verification-gaps]]: Verification gaps (3 zettels, 0 sources)');
	});

	test('new from a query, written with write', async () => {
		const {api, seen} = answering('pageNew', (params) => ((params as {dry_run: boolean}).dry_run ? 11 : 13));
		const app = renderApp({api});
		await run(app, '/page new "Verification gaps" from "verification" limit 3');
		let frame = await app.waitFor('from "verification": 3 of 16 matching zettels (26 notes matched)');
		expect(frame).toContain('Tags: [[agentic-ai-systems]]');
		await app.type(`/page new "Verification gaps" @${Z1} write`);
		await app.press(keys.escape, keys.enter);
		frame = await app.waitFor('index: updated');
		expect(seen).toEqual([
			{title: 'Verification gaps', zettels: null, queries: ['verification'], limit: 3, tags: null, dry_run: true},
			{title: 'Verification gaps', zettels: [Z1], queries: null, tags: null, dry_run: false},
		]);
		expect(frame).toContain('⏺ Wrote 3 notes  page new');
	});

	test('check: not ready, why, and the commands that fix it', async () => {
		const {api, seen} = answering('pageCheck', 14); // page.check {page: verification-gaps}
		const app = renderApp({api});
		await run(app, '/page check @verification-gaps');
		const frame = await app.waitFor('for the others');
		expect(seen).toEqual([{page: 'verification-gaps'}]);
		expect(frame).toContain('⏺ Page check  core · 1 managed page');
		expect(frame).toContain('✗ 0 of 1 managed page ready to write');
		expect(frame).toContain('verification-gaps  50_MOC  Verification gaps · refreshed');
		expect(frame).toContain('✗ behind: 8 zettels with two or more of its tags or links in common, not on it');
		expect(frame).toContain('colocated-environments-starve-the-inference-engine  20_ZETTLEKASTEN');
		expect(frame).toContain('tags reinforcement-learning-in-language-models · links');
		expect(flat(frame)).toContain('/page exclude @verification-gaps @zettel… write for the others');
		expect(frame).not.toContain('/page refresh'); // blocks current, zettels as recorded, index line current
	});

	test('check every page; exclude records in the engine state only; refresh with nothing to do', async () => {
		const api = fakeApi({
			pageCheck: async () => coreResult(15), // page.check {}
			pageExclude: async () => coreResult(16), // page.exclude {…} (written)
			pageRefresh: async () => coreResult(17), // page.refresh {dry_run: true}: unchanged
		});
		const app = renderApp({api});
		await run(app, '/page check');
		let frame = await app.waitFor('without vault blocks');
		expect(frame).toContain('3 pages without vault blocks (not made by /page new)');
		await app.type('/page exclude @verification-gaps @colocated-environments-starve-the-inference-engine write');
		await app.press(keys.escape, keys.enter);
		frame = await app.waitFor('1 excluded');
		expect(frame).toContain('⏺ Recorded in the engine state  page exclude');
		await app.type('/page refresh @verification-gaps');
		await app.press(keys.escape, keys.enter);
		frame = await app.waitFor('⏺ Nothing to write');
	});

	test('bad arguments', async () => {
		const app = renderApp();
		await run(app, '/page new "T"');
		await app.waitFor((frame) => flat(frame).includes('name zettels with @, or give from "<query>"'));
		await app.type('/page exclude @p');
		await app.press(keys.escape, keys.enter);
		await app.waitFor((frame) => flat(frame).includes('a page, then the zettels'));
		await app.type('/page frobnicate');
		await app.press(keys.escape, keys.enter);
		await app.waitFor('Usage: /page new');
	});
});

describe('/findings and /idea', () => {
	test('findings by method, each with its id, title and evidence', async () => {
		const {api, seen} = answering('findings', 18); // findings.list {limit: 1}
		const app = renderApp({api});
		await run(app, '/findings');
		const frame = await app.waitFor('/findings link all');
		expect(seen).toEqual([{limit: 3}]);
		expect(frame).toContain('⏺ Findings  94 findings · from the index');
		expect(flat(frame)).toContain('thresholds (percentiles of this vault, with floors): missing notes 3+ linking notes · names 2+ · links score 0.16+');
		expect(frame).toContain('missing-note: a note many notes link to that does not exist (72)');
		expect(frame).toContain('missing-32eacad6a1  [[reward-model-overoptimization-hump]] does not exist; 12 notes link to');
		expect(frame).toContain('mostly in 20_ZETTLEKASTEN/ (11), 10_SOURCES/ (1)');
		expect(frame).toContain('link-5ec0974343  [[in-context-scheming-capability]] and');
		expect(frame).toContain('score 0.24 (shared neighbours 0.31, cited together 0.30, text 0.10)');
		expect(frame).toContain('… 71 more · /findings missing-note all');
		expect(tooWide(frame)).toEqual([]);
	});

	test('a method, all of them, one id', async () => {
		const {api, seen} = answering('findings', 19);
		const app = renderApp({api});
		await run(app, '/findings link all');
		await app.waitFor('link: two notes that probably belong linked (13)');
		await app.type('/findings link-5ec0974343');
		await app.press(keys.escape, keys.enter);
		await app.waitFor((frame) => frame.split('⏺ Findings').length === 3);
		expect(seen).toEqual([{limit: null, methods: ['link']}, {limit: null, id: 'link-5ec0974343'}]);
	});

	test('idea new: a dry run shows the idea with its finding block', async () => {
		const {api, seen} = answering('ideaNew', 21); // idea.new {finding, dry_run: true}
		const app = renderApp({api});
		await run(app, '/idea new missing-32eacad6a1');
		const frame = await app.waitFor('## Claim');
		expect(seen).toEqual([{finding: 'missing-32eacad6a1', title: null, tags: null, dry_run: true}]);
		expect(frame).toContain('+ Missing note reward-model-overoptimization-hump  50_MOC/ideas');
		expect(frame).toContain('Status: #open');
		expect(frame).toContain('<!-- vault:finding -->');
		expect(frame).toContain('- id: missing-32eacad6a1');
	});

	test('idea set: a status the format does not have is refused', async () => {
		const {api, seen} = answering('ideaSet', 23); // idea.set {status: maybe, dry_run: true}
		const app = renderApp({api});
		await run(app, '/idea set @"Missing note reward-model-overoptimization-hump" maybe');
		const frame = await app.waitFor('The refused change');
		expect(seen).toEqual([{idea: 'Missing note reward-model-overoptimization-hump', status: 'maybe', dry_run: true}]);
		expect(frame).toContain('⏺ Refused: the status change would add 1 violation  nothing was written');
		expect(frame).toMatch(/L3 +bad-value +Status: #maybe is not an allowed value \(open, dropped, done\)/);
	});
});

describe('/move, /merge', () => {
	test('merge: the self-link it would create is dropped and shown', async () => {
		const {api, seen} = answering('merge', 25); // note.merge {dry_run: true}
		const app = renderApp({api});
		await run(app, '/merge @enterprise-agent-evaluation-challenges @pass-hat-k-reliability-metric');
		const frame = await app.waitFor('Self-links dropped');
		expect(seen).toEqual([{survivor: 'enterprise-agent-evaluation-challenges', other: 'pass-hat-k-reliability-metric', dry_run: true}]);
		expect(frame).toContain('⏺ Dry run: would write 4 notes  nothing was written');
		expect(frame).toContain('links: 2 links re-spelled in 2 notes, every link that reached it still does');
		expect(frame).toContain('Self-links dropped (1): they would point at the note they are in');
		expect(frame).toContain('L31 [[pass-hat-k-reliability-metric]] → line removed');
		expect(frame).toContain('-- [[pass-hat-k-reliability-metric]]'); // the diff removes the line
		expect(frame).toContain('+## Merged from pass-hat-k-reliability-metric');
		expect(frame).toContain('add write to the command to write it');
	});

	test('move with write: renamed, links re-spelled, records followed, old path deleted', async () => {
		const {api, seen} = answering('move', 27); // note.move (written)
		const app = renderApp({api});
		await run(app, `/move @${Z3} 20_ZETTLEKASTEN/verification-gap-limits-test-time-scaling write`);
		const frame = await app.waitFor('deleted');
		expect(seen).toEqual([{note: Z3, to: '20_ZETTLEKASTEN/verification-gap-limits-test-time-scaling', dry_run: false}]);
		expect(frame).toContain('⏺ Wrote 3 notes  move');
		expect(frame).toContain('✓ index: updated 3 notes');
		expect(frame).toContain('state .vaultengine/baseline-core.json: baseline: moved notes followed');
		expect(frame).toContain('+- [[verification-gap-limits-test-time-scaling]]');
		expect(frame).toContain(`- ${Z3}  20_ZETTLEKASTEN  deleted`);
	});

	test('bad arguments', async () => {
		const app = renderApp();
		await run(app, '/move nowhere');
		await app.waitFor('Usage: /move @note');
		await app.type('/merge @a');
		await app.press(keys.escape, keys.enter);
		await app.waitFor('Usage: /merge @survivor @other');
	});
});

describe('/next and /check baseline', () => {
	test('next: numbered items with why and the command, then the counts by kind', async () => {
		const {api, seen} = answering('next', 29); // vault.next {limit: 3}
		const app = renderApp({api});
		await run(app, '/next');
		const frame = await app.waitFor('By kind (listed / all)');
		expect(seen).toEqual([{limit: 10}]);
		expect(frame).toContain('⏺ Next  107 items, most trusted first');
		expect(frame).toMatch(/1\. write +write \[\[transcript-spotting-heuristics-for-reward-hacking\]\] +trust 97/);
		expect(frame).toContain('why 6 notes link to it and it does not exist');
		expect(frame).toContain('run vault new <VAULT> -f <REPO>/formats/core.yaml --type zettel');
		expect(frame).toContain('… 104 more · /next all');
		expect(flat(frame)).toMatch(/page +6 \/ 10 +zettels in no page that share a tag \(101 zettels in no page\); below the median trust not listed/);
		const kinds = [...frame.split('By kind (listed / all)')[1]!.matchAll(/^ +(refresh|behind|merge|write|extract|page|idea) +\d/gm)].map((m) => m[1]);
		expect(kinds).toEqual(['refresh', 'behind', 'merge', 'write', 'extract', 'page', 'idea']); // the order vault next uses
		expect(tooWide(frame)).toEqual([]);
	});

	test('check baseline: only the new violations count', async () => {
		const seen: unknown[] = [];
		const api = fakeApi({
			check: async (params) => {
				seen.push(params);
				return coreResult(5); // vault.check {baseline: true}
			},
		});
		const app = renderApp({api});
		await run(app, '/check baseline');
		const frame = await app.waitFor('fixed since');
		expect(seen).toEqual([{changed: false, limit: 8, baseline: true}]);
		expect(frame).toContain('✓ no new violations in 180 notes checked');
		expect(frame).toContain('144 violations · 144 still there (not listed) · 0 fixed since');
	});
});
