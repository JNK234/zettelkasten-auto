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
	LinksSuggestParams,
	LinksSuggestResult,
	MentionsUnlinkedParams,
	MentionsUnlinkedResult,
	NoteGetResult,
	NotePreviewParams,
	NotePreviewResult,
	NoteRelatedParams,
	NoteRelatedResult,
	NotesPickParams,
	NotesPickResult,
	NotesSuggestParams,
	NotesSuggestResult,
	SearchQueryParams,
	SearchQueryResult,
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
	search(params: SearchQueryParams): Promise<SearchQueryResult>;
	preview(params: NotePreviewParams): Promise<NotePreviewResult>;
	related(params: NoteRelatedParams): Promise<NoteRelatedResult>;
	suggestLinks(params?: LinksSuggestParams): Promise<LinksSuggestResult>;
	suggestNotes(params?: NotesSuggestParams): Promise<NotesSuggestResult>;
	unlinked(params?: MentionsUnlinkedParams): Promise<MentionsUnlinkedResult>;
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
		search: async (params) => client.request('search.query', params),
		preview: async (params) => client.request('note.preview', params),
		related: async (params) => client.request('note.related', params),
		suggestLinks: async (params) => client.request('links.suggest', params),
		suggestNotes: async (params) => client.request('notes.suggest', params),
		unlinked: async (params) => client.request('mentions.unlinked', params),
	};
}
