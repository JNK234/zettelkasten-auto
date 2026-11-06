/**
 * Typed facade over the JSON-RPC methods. Types are generated from the
 * engine's JSON Schema (src/rpc/schema.ts), so they follow the Python models.
 * UI code and commands depend on this interface only, which keeps them
 * testable with an in-memory fake.
 */
import type {
	HealthReportParams,
	HealthReportResult,
	IndexBuildParams,
	IndexBuildResult,
	IndexStatusResult,
	NoteGetResult,
	NotesPickParams,
	NotesPickResult,
	VaultMethod,
	VaultStatsParams,
	VaultStatsResult,
} from './schema.js';

export type VaultApi = {
	build(params?: IndexBuildParams): Promise<IndexBuildResult>;
	status(): Promise<IndexStatusResult>;
	stats(params?: VaultStatsParams): Promise<VaultStatsResult>;
	health(params?: HealthReportParams): Promise<HealthReportResult>;
	pick(params: NotesPickParams): Promise<NotesPickResult>;
	note(path: string): Promise<NoteGetResult>;
};

export type RpcRequester = {
	request<T>(method: VaultMethod, params?: object): Promise<T>;
};

export function createVaultApi(client: RpcRequester): VaultApi {
	return {
		build: async (params) => client.request('index.build', params),
		status: async () => client.request('index.status'),
		stats: async (params) => client.request('vault.stats', params),
		health: async (params) => client.request('health.report', params),
		pick: async (params) => client.request('notes.pick', params),
		note: async (path) => client.request('note.get', {path}),
	};
}
