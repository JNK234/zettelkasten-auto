/**
 * Test helpers. The fake API replays real responses recorded from the Python
 * server (tests/fixtures/rpc/*.jsonl in the repo), so frame tests exercise the
 * actual JSON-RPC contract rather than hand-written shapes.
 */
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {render} from 'ink-testing-library';
import {builtinCommands} from '../src/commands/builtin/index.js';
import {CommandRegistry} from '../src/commands/registry.js';
import {RpcError} from '../src/rpc/client.js';
import type {VaultApi} from '../src/rpc/api.js';
import type {IndexStatusResult, NotesPickResult} from '../src/rpc/schema.js';
import {App, type AppProps} from '../src/ui/App.js';

export const REPO_ROOT = fileURLToPath(new URL('../..', import.meta.url));

type Exchange = {request?: {method?: string; params?: Record<string, unknown>}; response: {result?: unknown} | unknown[] | null};

export function recorded(session: 'resolver_vault' | 'sample_vault'): Map<string, unknown> {
	const results = new Map<string, unknown>();
	const text = readFileSync(`${REPO_ROOT}/tests/fixtures/rpc/${session}.jsonl`, 'utf8');
	for (const line of text.split('\n').filter(Boolean)) {
		const exchange = JSON.parse(line) as Exchange;
		const {request, response} = exchange;
		if (request?.method && response && !Array.isArray(response) && 'result' in response) {
			results.set(`${request.method} ${JSON.stringify(request.params ?? {})}`, response.result);
		}
	}

	return results;
}

export const freshStatus: IndexStatusResult = {
	vault: '/vaults/resolver_vault',
	db: '/vaults/resolver_vault/.vaultengine/index.sqlite',
	built: true,
	last_indexed_at: '2026-01-01T00:00:00+00:00',
	schema_version: 1,
	parser_version: 1,
	resolver_version: 1,
	stale: false,
	pending: {added: [], changed: [], deleted: [], reason: null},
	notes: 6,
	attachments: 2,
	links: 20,
};

export type FakeApi = VaultApi & {calls: string[]};

/** A VaultApi over the recorded resolver_vault session. Override any method. */
export function fakeApi(overrides: Partial<VaultApi> = {}): FakeApi {
	const results = recorded('resolver_vault');
	const calls: string[] = [];
	const lookup = <T,>(method: string, params: Record<string, unknown> = {}): T | undefined =>
		results.get(`${method} ${JSON.stringify(params)}`) as T | undefined;
	const api: VaultApi = {
		build: async () => {
			calls.push('index.build');
			return lookup('index.build')!;
		},
		status: async () => {
			calls.push('index.status');
			return freshStatus;
		},
		stats: async () => {
			calls.push('vault.stats');
			return lookup('vault.stats', {top: 3})!;
		},
		health: async () => {
			calls.push('health.report');
			return lookup('health.report')!;
		},
		pick: async ({query}) => {
			calls.push(`notes.pick ${query}`);
			const empty: NotesPickResult = {query, total: 0, results: []};
			return lookup<NotesPickResult>('notes.pick', {query}) ?? empty;
		},
		note: async (path) => {
			calls.push(`note.get ${path}`);
			const note = lookup('note.get', {path});
			if (!note) {
				throw new RpcError(-32_002, `No indexed note at '${path}'`, {path});
			}

			return note as Awaited<ReturnType<VaultApi['note']>>;
		},
	};
	return Object.assign(api, overrides, {calls});
}

export const keys = {
	up: '\u001B[A',
	down: '\u001B[B',
	right: '\u001B[C',
	left: '\u001B[D',
	enter: '\r',
	tab: '\t',
	escape: '\u001B',
	backspace: '\u007F',
	ctrlU: '\u0015',
};

export const delay = async (ms: number) =>
	new Promise((resolve) => {
		setTimeout(resolve, ms);
	});

export function renderApp(props: Partial<AppProps> = {}) {
	const api = (props.api as FakeApi | undefined) ?? fakeApi();
	const registry = props.registry ?? new CommandRegistry().register(...builtinCommands);
	const instance = render(<App api={api} registry={registry} vaultPath="/vaults/resolver_vault" {...props} />);

	const frame = () => instance.lastFrame() ?? '';

	/** Poll until the frame satisfies `check` (string = contains). */
	const waitFor = async (check: string | ((frame: string) => boolean), timeout = 3000) => {
		const test = typeof check === 'string' ? (text: string) => text.includes(check) : check;
		const started = Date.now();
		while (!test(frame())) {
			if (Date.now() - started > timeout) {
				throw new Error(`Timed out waiting for ${typeof check === 'string' ? JSON.stringify(check) : 'condition'}. Last frame:\n${frame()}`);
			}

			await delay(10);
		}

		return frame();
	};

	/**
	 * Send keystrokes one at a time, letting Ink process each. After a lone ESC,
	 * wait past Ink's 20 ms escape-sequence window; otherwise the next key would
	 * be read as Alt+key, as it would from a real terminal.
	 */
	const press = async (...sequence: string[]) => {
		for (const data of sequence) {
			instance.stdin.write(data);
			await delay(data === keys.escape ? 40 : 15);
		}
	};

	const type = async (text: string) => press(...text);

	return {...instance, api, registry, frame, waitFor, press, type};
}

/** The last prompt box line, e.g. "│ > /note @ppo │" -> "> /note @ppo". */
export function promptLine(frame: string): string {
	const line = frame.split('\n').findLast((row) => row.includes('│ >'));
	return line?.replace(/^\s*│\s?/, '').replace(/\s*│\s*$/, '') ?? '';
}
