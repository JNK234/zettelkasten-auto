/**
 * End to end: the real Python server (python -m vaultengine serve) on a copy of
 * the sample vault, driven through the client, the typed API and the full App.
 */
import {cpSync, mkdtempSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {afterAll, beforeAll, describe, expect, test} from 'vitest';
import {cleanup} from 'ink-testing-library';
import {createVaultApi, type VaultApi} from '../src/rpc/api.js';
import {ConnectionClosedError, RpcError} from '../src/rpc/client.js';
import {startVaultServer, type VaultServer} from '../src/rpc/server.js';
import {keys, promptLine, REPO_ROOT, renderApp} from './helpers.js';

/** The error a promise rejects with; fails the test if it resolves. */
async function rejection(promise: Promise<unknown>): Promise<Error> {
	return promise.then(
		() => {
			throw new Error('expected the promise to reject');
		},
		(error: unknown) => error as Error,
	);
}

let workdir: string;
let server: VaultServer;
let api: VaultApi;

beforeAll(() => {
	workdir = mkdtempSync(join(tmpdir(), 'vault-tui-'));
	cpSync(join(REPO_ROOT, 'tests', 'fixtures', 'vault'), join(workdir, 'vault'), {
		recursive: true,
		preserveTimestamps: true,
		filter: (source) => !source.split(/[\\/]/).includes('.vaultengine'), // never reuse an index
	});
	server = startVaultServer({vault: join(workdir, 'vault')});
	api = createVaultApi(server.client);
});

afterAll(() => {
	cleanup();
	server.stop();
	rmSync(workdir, {recursive: true, force: true});
});

describe('against the real vaultengine server', () => {
	test('typed API round trips', async () => {
		const build = await api.build();
		expect(build.parsed).toHaveLength(179);
		const status = await api.status();
		expect(status).toMatchObject({built: true, stale: false, notes: 179, links: 1607});
		const picked = await api.pick({query: 'klpen', limit: 1});
		expect(picked.results[0]?.path).toBe('20_ZETTLEKASTEN/kl-penalty-in-rlhf.md');
		const note = await api.note('20_ZETTLEKASTEN/kl-penalty-in-rlhf.md');
		expect(note.links_out).toHaveLength(10);
		const error = await api.note('nope.md').catch((error_: unknown) => error_);
		expect(error).toBeInstanceOf(RpcError);
		expect((error as RpcError).code).toBe(-32_002);
	});

	test('the App: status line, @ picker and /note, end to end', async () => {
		const app = renderApp({api, vaultPath: join(workdir, 'vault')});
		await app.waitFor('179 notes · 1,607 links  ·  ● index fresh', 15_000);
		await app.type('/note @klpen');
		await app.waitFor('❯ kl-penalty-in-rlhf');
		await app.press(keys.enter);
		await app.waitFor((frame) => promptLine(frame) === '> /note @20_ZETTLEKASTEN/kl-penalty-in-rlhf.md');
		await app.press(keys.enter);
		const frame = await app.waitFor('Links in (1)', 10_000);
		expect(frame).toContain('⏺ kl-penalty-in-rlhf  20_ZETTLEKASTEN/kl-penalty-in-rlhf.md');
		expect(frame).toContain('Links out (10: 6 resolved)');
		expect(frame).toContain('✗ policy-gradient-methods  not in vault');
		expect(frame).toContain('← 1707.06347 - Proximal Policy Optimization Algorithms  20_ZETTLEKASTEN');
	});

	test('when the server dies, requests fail with its exit reason', async () => {
		const closed = new Promise<Error>((resolve) => {
			server.client.events.once('close', resolve);
		});
		server.process.kill('SIGTERM');
		expect((await closed).message).toMatch(/Vault server (was stopped|exited)/);
		await expect(api.status()).rejects.toBeInstanceOf(ConnectionClosedError);
	});
});

describe('server start-up failures are explained', () => {
	test('missing interpreter', async () => {
		const broken = startVaultServer({vault: REPO_ROOT, python: '/nonexistent/python3'});
		const error = await rejection(createVaultApi(broken.client).status());
		expect(error).toBeInstanceOf(ConnectionClosedError);
		expect(error.message).toContain('could not start (/nonexistent/python3)');
	});

	test('server exits with an error: its last stderr line is shown', async () => {
		const broken = startVaultServer({vault: join(REPO_ROOT, 'no-such-vault')});
		const error = await rejection(createVaultApi(broken.client).status());
		expect(error.message).toMatch(/Vault server exited with code 2: error: vault folder not found/);
	});
});
