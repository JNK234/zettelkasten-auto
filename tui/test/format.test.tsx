/**
 * Frame tests for the format commands: /check, /fix, /new, /link, /ingest, /coverage.
 * Results are real engine responses recorded in tests/fixtures/rpc/sample_vault_format.jsonl
 * (request ids are noted next to each use).
 */
import {afterEach, describe, expect, test} from 'vitest';
import {cleanup} from 'ink-testing-library';
import type {
	NoteIngestResult,
	VaultCheckParams,
	VaultCheckResult,
	VaultCoverageResult,
	VaultFixResult,
	WriteResult,
} from '../src/rpc/schema.js';
import {fakeApi, formatResult, keys, renderApp} from './helpers.js';

afterEach(() => {
	cleanup();
});

const EVD = 'EVD - Reward is a deterministic outcome check 1 only if the final database equals the data';
const CLM = 'CLM - Grade the observed end state rather than the trajectory';

/** vault.check as the server answers it: the recorded full check (request 1), cut to params.limit. */
function checkLikeServer(params: VaultCheckParams = {}): VaultCheckResult {
	const full = params.changed ? formatResult<VaultCheckResult>(4) : formatResult<VaultCheckResult>(1);
	const limit = params.limit ?? undefined;
	return {...full, notes: limit ? full.notes.slice(0, limit) : full.notes, limit: limit ?? null};
}

/** The frame with wrapped lines joined: entries are laid out at the terminal width (100 in tests). */
const flat = (frame: string) => frame.replaceAll(/\s*\n\s*/g, ' ');

async function run(app: ReturnType<typeof renderApp>, line: string) {
	await app.waitFor('index fresh');
	await app.type(line);
	await app.press(keys.escape, keys.enter);
}

describe('/check', () => {
	test('summary, the /fix hint, and violations grouped by note with line numbers', async () => {
		const seen: VaultCheckParams[] = [];
		const api = fakeApi({
			check: async (params) => {
				seen.push(params ?? {});
				return checkLikeServer(params);
			},
		});
		const app = renderApp({api});
		await run(app, '/check');
		const frame = await app.waitFor('more notes with violations');
		expect(seen).toEqual([{changed: false, limit: 8}]);
		expect(frame).toContain('⏺ Format check  brain · all 180 notes');
		expect(frame).toContain('✗ 102 violations in 77 notes');
		expect(frame).toContain('59 unresolved-target · 28 missing-heading · 15 missing-key');
		expect(frame).toContain('28 can be repaired automatically: /fix shows the diff first');
		expect(frame).toContain('RLHF  10_SOURCES  (source)');
		expect(frame).toMatch(/L138 +missing-heading +missing heading 'Notes' \(expected at the end of the note\)/);
		expect(frame).toContain('agentic-ai-usage-in-customer-service  20_ZETTLEKASTEN  (zettel)');
		expect(frame).toMatch(/L1 +missing-key +missing metadata 'Created'/);
		expect(frame).toContain('… 69 more notes with violations · /check all lists every note');
	});

	test('/check all and /check changed', async () => {
		const seen: VaultCheckParams[] = [];
		const api = fakeApi({
			check: async (params) => {
				seen.push(params ?? {});
				return checkLikeServer(params);
			},
		});
		const app = renderApp({api});
		await run(app, '/check all');
		let frame = await app.waitFor('✗ 102 violations in 77 notes');
		expect(frame).not.toContain('more notes with violations');
		expect(flat(frame)).toMatch(/EVD - tau2 reward is a product so one zero kills the task +50_MOC\/discourse\/rl-environment-design {2}\(evidence\)/);
		await app.type('/check changed');
		await app.press(keys.escape, keys.enter);
		frame = await app.waitFor('✓ no violations in 0 notes checked');
		expect(frame).toContain('brain · 0 notes changed since the last index');
		expect(seen).toEqual([{changed: false, limit: null}, {changed: true, limit: 8}]);
	});

	test('a changed note and a note it affects', async () => {
		const api = fakeApi({
			check: async () => ({
				...checkLikeServer({changed: true}),
				ok: false,
				scope: {...checkLikeServer({changed: true}).scope, checked: 1, notes: [`x/${EVD}.md`], changed: [`x/${EVD}.md`]},
				summary: {violations: 1, notes: 1, by_code: [{code: 'missing-mirror', count: 1}]},
				notes: [
					{
						path: `x/${CLM}.md`,
						type: 'claim',
						checked: false,
						violations: [{path: `x/${CLM}.md`, line: 11, code: 'missing-mirror', message: `missing \`supported-by:: [[${EVD}]]\``, related: `x/${EVD}.md`}],
					},
				],
			}),
		});
		const app = renderApp({api});
		await run(app, '/check changed');
		const frame = await app.waitFor((f) => flat(f).includes('not changed: affected by a change'));
		expect(frame).toContain('1 can be repaired automatically');
		expect(frame).toMatch(/L11 +missing-mirror +missing `supported-by::/);
	});

	test('without a format the error says how to start the app', async () => {
		const app = renderApp();
		await run(app, '/check');
		await app.waitFor('No format configured. Start vault-tui with --format <name>, e.g. --format brain (error -32003)');
	});

	test('bad arguments', async () => {
		const app = renderApp();
		await run(app, '/check some notes');
		await app.waitFor('Usage: /check [changed] [all | <notes>]');
	});
});

describe('/fix', () => {
	test('previews the diff each repair would apply, and writes nothing', async () => {
		const seen: unknown[] = [];
		const api = fakeApi({
			fix: async (params) => {
				seen.push(params);
				return formatResult<VaultFixResult>(28); // vault.fix {dry_run: true}
			},
		});
		const app = renderApp({api});
		await run(app, '/fix');
		const frame = await app.waitFor('Also changes (25)');
		expect(seen).toEqual([{dry_run: true, only: null}]);
		expect(frame).toContain('⏺ Fix preview: 28 notes would change  nothing written yet');
		expect(frame).toContain('✓ check: no new violations');
		expect(frame).toContain('left after this fix, for you: 59 unresolved-target · 15 missing-key');
		expect(frame).toContain('/fix apply writes these changes');
		expect(frame).toContain('~ RLHF  10_SOURCES');
		expect(frame).toContain("end of note: added 'Notes'");
		expect(frame).toContain('+# Notes');
		expect(frame).toContain('@@ -25,6 +25,15 @@');
		expect(frame).toContain('+## What This Is');
		expect(frame).toContain("before line 32: added 'What This Is', 'Why It Matters', 'Key Details', 'Tensions'");
		expect(frame).toContain('/fix all shows every diff');
	});

	test('/fix apply writes and keeps the index in sync', async () => {
		const seen: unknown[] = [];
		const api = fakeApi({
			fix: async (params) => {
				seen.push(params);
				return formatResult<VaultFixResult>(29); // vault.fix {}
			},
		});
		const app = renderApp({api});
		await run(app, '/fix apply headings');
		const frame = await app.waitFor('⏺ Fixed 28 notes');
		expect(seen).toEqual([{dry_run: false, only: ['missing-heading']}]);
		expect(frame).toContain('✓ index: updated 28 notes');
		expect(frame).toContain('left now, for you: 59 unresolved-target · 15 missing-key');
		expect(frame).toContain('Also fixed (25)');
	});

	test('nothing to fix', async () => {
		const api = fakeApi({fix: async () => formatResult<VaultFixResult>(27)});
		const app = renderApp({api});
		await run(app, '/fix mirrors');
		await app.waitFor('⏺ Nothing to fix automatically');
	});
});

describe('/new', () => {
	test('shows the file the write created, the check and the index', async () => {
		const seen: unknown[] = [];
		const api = fakeApi({
			newNote: async (params) => {
				seen.push(params);
				return formatResult<WriteResult>(17);
			},
		});
		const app = renderApp({api});
		await run(app, '/new zettel "Verifier Hacking Needs a Do-Nothing Baseline" Tags=[[evaluation-methodology]]');
		const frame = await app.waitFor('# References');
		expect(seen).toEqual([
			{type: 'zettel', title: 'Verifier Hacking Needs a Do-Nothing Baseline', values: {Tags: '[[evaluation-methodology]]'}, relations: [], dry_run: false},
		]);
		expect(frame).toContain('⏺ Created verifier-hacking-needs-a-do-nothing-baseline  20_ZETTLEKASTEN');
		expect(frame).toContain('✓ check: no new violations');
		expect(frame).toContain('✓ index: updated 1 note');
		expect(frame).toContain('+ verifier-hacking-needs-a-do-nothing-baseline  20_ZETTLEKASTEN');
		expect(frame).toContain('│ Created: 01-01-2026 10:00');
		expect(frame).toContain('│ ## What This Is');
		expect(frame).toContain('│ A grader that a do-nothing agent passes is paying reward for inaction.');
	});

	test('a refused note: the violations it would add, and the file it would have been', async () => {
		const seen: unknown[] = [];
		const api = fakeApi({
			newNote: async (params) => {
				seen.push(params);
				return formatResult<WriteResult>(19);
			},
		});
		const app = renderApp({api});
		await run(app, '/new claim "Graders must be tested against a known-bad trajectory" Status=#supported informs @"QUE 03 - Grade end state, trajectory or text"');
		const frame = await app.waitFor('The refused change');
		expect(seen).toEqual([
			{
				type: 'claim',
				title: 'Graders must be tested against a known-bad trajectory',
				values: {Status: '#supported'},
				relations: [{relation: 'informs', note: 'QUE 03 - Grade end state, trajectory or text'}],
				dry_run: false,
			},
		]);
		expect(frame).toContain('⏺ Refused: the new note would add 1 violation  nothing was written');
		expect(frame).toContain('Violations it would add');
		expect(frame).toMatch(/L10 +missing-relation +missing relation field `supported-by::` \(required for claim\)/);
		expect(frame).toContain('│ informs:: [[QUE 03 - Grade end state, trajectory or text]]');
	});

	test('usage', async () => {
		const app = renderApp();
		await run(app, '/new zettel');
		await app.waitFor('Usage: /new <type> "<title>"');
	});
});

describe('/link', () => {
	test('both sides: the relation and its mirror, as diffs', async () => {
		const seen: unknown[] = [];
		const api = fakeApi({
			link: async (params) => {
				seen.push(params);
				return formatResult<WriteResult>(12);
			},
		});
		const app = renderApp({api});
		await run(app, `/link @"${EVD}" supports @"${CLM}"`);
		const frame = await app.waitFor('index: updated 2 notes');
		expect(seen).toEqual([{source: EVD, target: CLM, relation: 'supports', dry_run: false}]);
		expect(frame).toContain('⏺ Wrote 2 notes  link');
		expect(frame).toContain('already there, not blocking: 6 unresolved-target');
		const flat = frame.replaceAll(/\n\s*/g, ' '); // long diff lines wrap at 100 columns
		expect(flat).toContain(`line 13: added [[${CLM}]] to \`supports::\``);
		expect(flat).toContain(`line 11: added [[${EVD}]] to \`supported-by::\``);
		expect(flat).toContain(`+supports:: [[CLM - Anchor reward in a deterministic check of the outcome]], [[${CLM}]]`);
		expect(frame).toContain('-supported-by:: [[EVD - AppWorld fails any database change outside an allow-list]]');
		expect(flat).toContain(`The flight example separates what the transcript claims ('Your flight has been booke]], [[${EVD}]]`);
	});

	test('refused: linking evidence to a zettel', async () => {
		const api = fakeApi({link: async () => formatResult<WriteResult>(14)});
		const app = renderApp({api});
		await run(app, `/link @"${EVD}" supports @kl-penalty-in-rlhf`);
		const frame = await app.waitFor('The refused change');
		expect(frame).toContain('⏺ Refused: the link would add 1 violation  nothing was written');
		expect(frame).toMatch(/relation-target-type +`supports::` \[\[kl-penalty-in-rlhf\]\] is a zettel; expected claim/);
		expect(frame.replaceAll(/\n\s*/g, ' ')).toContain('+supported-by:: [[EVD - Reward is a deterministic outcome check 1 only if the final database equals the data]]');
	});

	test('already linked', async () => {
		const api = fakeApi({link: async () => formatResult<WriteResult>(13)});
		const app = renderApp({api});
		await run(app, `/link @"${EVD}" @"${CLM}" as supports`);
		const frame = await app.waitFor('⏺ Nothing to write');
		expect(frame.replaceAll(/\n\s*/g, ' ')).toContain(`already has \`supports:: [[${CLM}]]\``);
	});

	test('usage needs two notes and one relation', async () => {
		const app = renderApp();
		await run(app, '/link @only-one supports');
		await app.waitFor('Usage: /link @note <relation> @note [--dry-run]');
	});
});

describe('/ingest', () => {
	test('a duplicate: the existing note, its scores against the thresholds, and how to open it', async () => {
		const seen: Array<{path?: string | null}> = [];
		const api = fakeApi({
			ingest: async (params) => {
				seen.push(params);
				return formatResult<NoteIngestResult>(21);
			},
		});
		const app = renderApp({api});
		await run(app, '/ingest inbox/deepseek-copy.md');
		const frame = await app.waitFor('open it:');
		expect(seen[0]?.path).toMatch(/^\/.*inbox\/deepseek-copy\.md$/);
		expect(frame).toContain('⏺ Not ingested: deepseek-copy.md duplicates an existing note  nothing was written');
		expect(frame).toContain('≡ DeepSeek R1 - Paper Notes  10_SOURCES');
		expect(frame).toMatch(/content overlap {3}1\.000 █{20} duplicate \(≥ 0\.8\)/);
		expect(frame).toMatch(/title similarity {2}0\.\d{3} .* below 0\.9/);
		expect(frame).toContain('same file         yes    identical SHA-256');
		expect(frame).toContain('open it: /note @"10_SOURCES/DeepSeek R1 - Paper Notes.md"');
	});

	test('a new document: the source note it created', async () => {
		const api = fakeApi({ingest: async () => formatResult<NoteIngestResult>(23)});
		const app = renderApp({api});
		await run(app, '/ingest ~/Downloads/ensembles.html');
		const frame = await app.waitFor('closest existing note');
		expect(frame).toContain('⏺ Created Why Reward Model Ensembles Delay Overoptimization  10_SOURCES');
		expect(frame).toContain('│ Original: ensembles.html');
		const flat = frame.replaceAll(/\s*│\n\s*│\s*/g, ' '); // the box wraps long lines
		expect(flat).toContain('1. [Why Reward Model Ensembles Delay Overoptimization](https://fieldnotes.example.org/posts/ensembles/)');
		expect(frame).toContain('│ Optimizing a policy against a learned reward model works until it does not.');  // wrapped, not cut
		expect(frame).toContain('│ ## The proxy is right on average and wrong at the edges');
		expect(frame).toContain('“Why Reward Model Ensembles Delay Overoptimization”  from ensembles.html · html');
		expect(frame).toContain('not a duplicate');
	});

	test('as <type> and usage', async () => {
		const seen: Array<{type?: string}> = [];
		const api = fakeApi({
			ingest: async (params) => {
				seen.push(params);
				return formatResult<NoteIngestResult>(23);
			},
		});
		const app = renderApp({api});
		await run(app, '/ingest "a file.txt" as source --dry-run');
		await app.waitFor('closest existing note');
		expect(seen[0]).toMatchObject({type: 'source', dry_run: true});
		await app.type('/ingest');
		await app.press(keys.escape, keys.enter);
		await app.waitFor('Usage: /ingest <file.md|.txt|.html> [as <type>] [--dry-run]');
	});
});

describe('/coverage', () => {
	test('sources nothing links to, and notes without a source grouped by why', async () => {
		const seen: unknown[] = [];
		const api = fakeApi({
			coverage: async (params) => {
				seen.push(params);
				return formatResult<VaultCoverageResult>(6);
			},
		});
		const app = renderApp({api});
		await run(app, '/coverage');
		const frame = await app.waitFor('Most cited sources');
		expect(seen).toEqual([{limit: null}]);
		expect(frame).toContain('⏺ Coverage  brain · sources: source · citing: zettel, evidence, topic, comparison');
		expect(frame).toContain('! 28 sources: 23 linked, 5 that nothing links to');
		expect(frame).toContain('! 143 citing notes: 33 cite a source, 110 cite none');
		expect(flat(frame)).toContain('106 cite only notes that do not exist · 0 cite notes that are not sources · 4 cite nothing at all');
		expect(frame).toContain('Sources nothing links to (5)');
		expect(frame).toContain('DeepSeek R1 - Paper Notes  10_SOURCES');
		expect(frame).toContain('Notes that cite only notes that do not exist (106)');
		expect(frame.replaceAll(/\n\s*/g, ' ')).toContain('a-bounty-list-is-a-map-of-what-cannot-be-verified  20_ZETTLEKASTEN  (zettel)  L37 [[Prime Intellect - Environments Hub]]');
		expect(frame).toContain('… 101 more · /coverage all lists every note');
		expect(frame).toContain('Notes that cite nothing at all (4)');
		expect(frame).toContain('q-value-function  20_ZETTLEKASTEN  (zettel)');
		expect(frame).toContain('5  2503.11926 - Monitoring Reasoning Models (CoT Obfuscation)  10_SOURCES');
	});
});

describe('registry and status line', () => {
	test('/help lists the format commands with their usage', async () => {
		const app = renderApp();
		await run(app, '/help');
		const frame = await app.waitFor('Keys');
		for (const usage of [
			'/check [changed] [all | <notes>]',
			'/fix [apply] [mirrors | headings] [all]',
			'/new <type> "<title>"',
			'/link @note <relation> @note [--dry-run]',
			'/ingest <file.md|.txt|.html> [as <type>] [--dry-run]',
			'/coverage [all | <per list>]',
		]) {
			expect(frame).toContain(usage);
		}
	});

	test('the status line names the format', async () => {
		const app = renderApp({formatName: 'brain'});
		await app.waitFor('◆ resolver_vault  ·  format brain');
	});
});
