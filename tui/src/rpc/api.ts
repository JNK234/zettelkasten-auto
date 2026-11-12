/**
 * Typed facade over the JSON-RPC methods. Types are generated from the
 * engine's JSON Schema (src/rpc/schema.ts), so they follow the Python models.
 * UI code and commands depend on this interface only, which keeps them
 * testable with an in-memory fake.
 */
import type {
	ConfigShowResult,
	EmbedStatusResult,
	ExploreKeptParams,
	ExploreKeptResult,
	ExploreLinkParams,
	ExploreLinkResult,
	ExploreMarkParams,
	ExploreMarkResult,
	ExploreRunParams,
	ExploreRunResult,
	ExploreUnlinkParams,
	ExploreUnlinkResult,
	MapParams,
	MapResult,
	MapSinceParams,
	MapSinceResult,
	NoteDeepenParams,
	NoteDeepenResult,
	NotesExplainParams,
	NotesExplainResult,
	VaultDiscoverParams,
	VaultDiscoverResult,
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
	NoteIngestParams,
	NoteIngestResult,
	NoteLinkParams,
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
	SearchQueryParams,
	SearchQueryResult,
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
	search(params: SearchQueryParams): Promise<SearchQueryResult>;
	preview(params: NotePreviewParams): Promise<NotePreviewResult>;
	related(params: NoteRelatedParams): Promise<NoteRelatedResult>;
	suggestLinks(params?: LinksSuggestParams): Promise<LinksSuggestResult>;
	suggestNotes(params?: NotesSuggestParams): Promise<NotesSuggestResult>;
	unlinked(params?: MentionsUnlinkedParams): Promise<MentionsUnlinkedResult>;
	/** explore.run: a section per lens from a note (or where to start); remembered for mark and linkIt. */
	explore(params?: ExploreRunParams): Promise<ExploreRunResult>;
	/** explore.mark: keep, dismiss or clear results of the last explore run, by ref ("2.3"). */
	mark(params: ExploreMarkParams): Promise<ExploreMarkResult>;
	kept(params?: ExploreKeptParams): Promise<ExploreKeptResult>;
	/** explore.link: a result made a See Also link in both notes; a dry run unless dry_run is false. */
	linkIt(params: ExploreLinkParams): Promise<ExploreLinkResult>;
	unlink(params: ExploreUnlinkParams): Promise<ExploreUnlinkResult>;
	map(params?: MapParams): Promise<MapResult>;
	mapSince(params: MapSinceParams): Promise<MapSinceResult>;
	deepen(params: NoteDeepenParams): Promise<NoteDeepenResult>;
	discover(params?: VaultDiscoverParams): Promise<VaultDiscoverResult>;
	explain(params: NotesExplainParams): Promise<NotesExplainResult>;
	embedStatus(): Promise<EmbedStatusResult>;
	config(): Promise<ConfigShowResult>;
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
		explore: async (params) => client.request('explore.run', params),
		mark: async (params) => client.request('explore.mark', params),
		kept: async (params) => client.request('explore.kept', params),
		linkIt: async (params) => client.request('explore.link', params),
		unlink: async (params) => client.request('explore.unlink', params),
		map: async (params) => client.request('map.get', params),
		mapSince: async (params) => client.request('map.since', params),
		deepen: async (params) => client.request('note.deepen', params),
		discover: async (params) => client.request('vault.discover', params),
		explain: async (params) => client.request('notes.explain', params),
		embedStatus: async () => client.request('embed.status'),
		config: async () => client.request('config.show'),
	};
}
