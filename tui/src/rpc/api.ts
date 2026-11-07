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
	NoteIngestParams,
	NoteIngestResult,
	NoteLinkParams,
	NoteNewParams,
	NoteSetParams,
	NotesPickParams,
	NotesPickResult,
	VaultCheckParams,
	VaultCheckResult,
	VaultCoverageParams,
	VaultCoverageResult,
	VaultFixParams,
	VaultFixResult,
	VaultMethod,
	VaultStatsParams,
	VaultStatsResult,
	WriteResult,
} from './schema.js';

export type VaultApi = {
	build(params?: IndexBuildParams): Promise<IndexBuildResult>;
	status(): Promise<IndexStatusResult>;
	stats(params?: VaultStatsParams): Promise<VaultStatsResult>;
	health(params?: HealthReportParams): Promise<HealthReportResult>;
	pick(params: NotesPickParams): Promise<NotesPickResult>;
	note(path: string): Promise<NoteGetResult>;
	/** vault.check: every violation of the format (the server needs --format). */
	check(params?: VaultCheckParams): Promise<VaultCheckResult>;
	/** vault.fix: missing mirrors and missing headings; dry_run shows the diff. */
	fix(params?: VaultFixParams): Promise<VaultFixResult>;
	coverage(params?: VaultCoverageParams): Promise<VaultCoverageResult>;
	newNote(params: NoteNewParams): Promise<WriteResult>;
	link(params: NoteLinkParams): Promise<WriteResult>;
	set(params: NoteSetParams): Promise<WriteResult>;
	ingest(params: NoteIngestParams): Promise<NoteIngestResult>;
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
		check: async (params) => client.request('vault.check', params),
		fix: async (params) => client.request('vault.fix', params),
		coverage: async (params) => client.request('vault.coverage', params),
		newNote: async (params) => client.request('note.new', params),
		link: async (params) => client.request('note.link', params),
		set: async (params) => client.request('note.set', params),
		ingest: async (params) => client.request('note.ingest', params),
	};
}
