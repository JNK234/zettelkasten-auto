/**
 * Typed facade over the JSON-RPC methods. Types are generated from the
 * engine's JSON Schema (src/rpc/schema.ts), so they follow the Python models.
 * UI code and commands depend on this interface only, which keeps them
 * testable with an in-memory fake.
 */
import type {
	FindingsListParams,
	FindingsListResult,
	HealthReportParams,
	HealthReportResult,
	IdeaNewParams,
	IdeaNewResult,
	IdeaSetParams,
	IndexBuildParams,
	IndexBuildResult,
	IndexStatusResult,
	LinksSuggestParams,
	LinksSuggestResult,
	MentionsUnlinkedParams,
	MentionsUnlinkedResult,
	NoteFactsParams,
	NoteFactsResult,
	NoteGetResult,
	NoteIngestParams,
	NoteIngestResult,
	NoteLinkParams,
	NoteMergeParams,
	NoteMoveParams,
	NoteNewParams,
	NotePreviewParams,
	NotePreviewResult,
	NoteRelatedParams,
	NoteRelatedResult,
	NoteSetParams,
	NotesPickParams,
	NotesPickResult,
	NotesSuggestParams,
	NotesSuggestResult,
	PageCheckParams,
	PageCheckResult,
	PageExcludeParams,
	PageNewParams,
	PageNewResult,
	PageRefreshParams,
	RelocationResult,
	SearchQueryParams,
	SearchQueryResult,
	StatefulWriteResult,
	VaultCheckParams,
	VaultCheckResult,
	VaultCoverageParams,
	VaultCoverageResult,
	VaultFixParams,
	VaultFixResult,
	VaultMethod,
	VaultNextParams,
	VaultNextResult,
	VaultPipelineParams,
	VaultPipelineResult,
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
	search(params: SearchQueryParams): Promise<SearchQueryResult>;
	preview(params: NotePreviewParams): Promise<NotePreviewResult>;
	related(params: NoteRelatedParams): Promise<NoteRelatedResult>;
	suggestLinks(params?: LinksSuggestParams): Promise<LinksSuggestResult>;
	suggestNotes(params?: NotesSuggestParams): Promise<NotesSuggestResult>;
	unlinked(params?: MentionsUnlinkedParams): Promise<MentionsUnlinkedResult>;
	/** vault.pipeline: each source's stage and the gaps (format with a pipeline: block). */
	pipeline(params?: VaultPipelineParams): Promise<VaultPipelineResult>;
	noteFacts(params: NoteFactsParams): Promise<NoteFactsResult>;
	pageNew(params: PageNewParams): Promise<PageNewResult>;
	pageRefresh(params: PageRefreshParams): Promise<StatefulWriteResult>;
	pageExclude(params: PageExcludeParams): Promise<StatefulWriteResult>;
	pageCheck(params?: PageCheckParams): Promise<PageCheckResult>;
	findings(params?: FindingsListParams): Promise<FindingsListResult>;
	ideaNew(params: IdeaNewParams): Promise<IdeaNewResult>;
	ideaSet(params: IdeaSetParams): Promise<StatefulWriteResult>;
	move(params: NoteMoveParams): Promise<RelocationResult>;
	merge(params: NoteMergeParams): Promise<RelocationResult>;
	next(params?: VaultNextParams): Promise<VaultNextResult>;
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
		search: async (params) => client.request('search.query', params),
		preview: async (params) => client.request('note.preview', params),
		related: async (params) => client.request('note.related', params),
		suggestLinks: async (params) => client.request('links.suggest', params),
		suggestNotes: async (params) => client.request('notes.suggest', params),
		unlinked: async (params) => client.request('mentions.unlinked', params),
		pipeline: async (params) => client.request('vault.pipeline', params),
		noteFacts: async (params) => client.request('note.facts', params),
		pageNew: async (params) => client.request('page.new', params),
		pageRefresh: async (params) => client.request('page.refresh', params),
		pageExclude: async (params) => client.request('page.exclude', params),
		pageCheck: async (params) => client.request('page.check', params),
		findings: async (params) => client.request('findings.list', params),
		ideaNew: async (params) => client.request('idea.new', params),
		ideaSet: async (params) => client.request('idea.set', params),
		move: async (params) => client.request('note.move', params),
		merge: async (params) => client.request('note.merge', params),
		next: async (params) => client.request('vault.next', params),
	};
}
