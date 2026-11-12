"""The JSON-RPC 2.0 API, defined once.

``METHODS`` maps every method name to its params and result models. The
server validates params and results with these models, and ``schema.py``
derives the JSON Schema from them, so the schema cannot drift from the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

JSONRPC_VERSION = "2.0"

# Standard JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# Server-defined error codes (-32000 to -32099)
INDEX_NOT_BUILT = -32001
NOTE_NOT_FOUND = -32002
FORMAT_NOT_CONFIGURED = -32003
WRITE_REJECTED = -32004
WRITE_FAILED = -32005
INVALID_QUERY = -32006
NOT_AVAILABLE = -32007

ERROR_MESSAGES = {
    PARSE_ERROR: "Parse error",
    INVALID_REQUEST: "Invalid Request",
    METHOD_NOT_FOUND: "Method not found",
    INVALID_PARAMS: "Invalid params",
    INTERNAL_ERROR: "Internal error",
    INDEX_NOT_BUILT: "Index not built",
    NOTE_NOT_FOUND: "Note not found",
    FORMAT_NOT_CONFIGURED: "Format not configured",
    WRITE_REJECTED: "Write rejected",
    WRITE_FAILED: "Write failed",
    INVALID_QUERY: "Invalid query",
    NOT_AVAILABLE: "Not available",
}

RequestId = Union[StrictInt, StrictStr, None]
LinkStatus = Literal["resolved", "ambiguous", "unresolved"]
SubpathStatus = Literal["found", "missing", "not_checked"]


class Params(BaseModel):
    """Params are by-name, strictly typed, and unknown names are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----- envelope ------------------------------------------------------------------


class Request(BaseModel):
    """A JSON-RPC 2.0 request. Without "id" it is a notification (no response)."""

    model_config = ConfigDict(extra="forbid", strict=True)
    jsonrpc: Literal["2.0"]
    method: StrictStr
    params: Union[dict[str, Any], list[Any], None] = None
    id: RequestId = None


class ErrorObject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: int
    message: str
    data: Any = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jsonrpc: Literal["2.0"]
    id: RequestId
    error: ErrorObject


# ----- index.build ---------------------------------------------------------------


class IndexBuildParams(Params):
    full: bool = Field(False, description="Re-parse every note and re-resolve every link.")


class IndexBuildResult(Result):
    vault: str
    db: str
    notes: int
    attachments: int
    added: list[str]
    changed: list[str]
    forced: list[str] = Field(description="Unchanged notes re-parsed because of a full run.")
    deleted: list[str]
    attachments_added: list[str]
    attachments_deleted: list[str]
    hashed: int = Field(description="Note files read and hashed.")
    parsed: list[str] = Field(description="added + changed + forced, sorted.")
    full_reason: str | None
    links_checked: int = Field(description="Links re-resolved this run.")
    links_changed_elsewhere: int = Field(description="Of those, links in notes not re-parsed whose result changed.")
    seconds: float


# ----- index.status --------------------------------------------------------------


class IndexStatusParams(Params):
    pass


class PendingChanges(Result):
    added: list[str]
    changed: list[str]
    deleted: list[str]
    reason: str | None = Field(description="Set when a version or setting change forces a rebuild.")


class IndexStatusResult(Result):
    vault: str
    db: str
    built: bool
    last_indexed_at: str | None
    schema_version: int | None
    parser_version: int | None
    resolver_version: int | None
    stale: bool | None = Field(description="True if index.build would change anything; null if not built.")
    pending: PendingChanges | None
    notes: int
    attachments: int
    links: int


# ----- vault.stats ---------------------------------------------------------------


class VaultStatsParams(Params):
    top: int = Field(10, ge=1, le=1000, description="Rows in each top-N list.")


class LinkCounts(Result):
    total: int
    wikilink: int
    markdown: int
    embeds: int
    resolved: int
    unresolved: int
    ambiguous: int
    unresolved_targets: int


class TagCount(Result):
    tag: str
    notes: int


class TagStats(Result):
    uses: int
    distinct: int
    top: list[TagCount]


class PropertyCount(Result):
    key: str
    notes: int


class PropertyStats(Result):
    distinct_keys: int
    top: list[PropertyCount]


class FolderCount(Result):
    folder: str
    notes: int


class LinkedNote(Result):
    path: str
    backlinks: int
    linking_notes: int


class WantedNote(Result):
    target: str
    links: int
    linking_notes: int


class VaultStatsResult(Result):
    notes: int
    attachments: int
    notes_with_frontmatter: int
    frontmatter_errors: int
    links: LinkCounts
    note_edges: int = Field(description="Distinct resolved note-to-note connections.")
    tags: TagStats
    properties: PropertyStats
    aliases: int
    headings: int
    blocks: int
    folders: list[FolderCount]
    most_linked: list[LinkedNote]
    most_wanted: list[WantedNote] = Field(description="Unresolved targets by number of linking notes.")


# ----- health.report -------------------------------------------------------------


class HealthReportParams(Params):
    limit: int | None = Field(None, ge=1, description="Max items per category (null = all). Summary counts are never limited.")


class LinkRef(Result):
    source: str
    line: int
    raw: str


class UnresolvedTarget(Result):
    target: str
    links: list[LinkRef]


class AmbiguousLink(LinkRef):
    candidates: list[str]


class MissingSubpath(LinkRef):
    target_path: str
    kind: Literal["heading", "block"]
    subpath: str


class DuplicateName(Result):
    name: str
    paths: list[str]


class HealthSummary(Result):
    unresolved_links: int
    unresolved_targets: int
    ambiguous_links: int
    missing_subpaths: int
    orphans: int
    dead_ends: int
    duplicate_basenames: int
    empty_notes: int


class HealthReportResult(Result):
    summary: HealthSummary
    unresolved_links: list[UnresolvedTarget]
    ambiguous_links: list[AmbiguousLink]
    missing_subpaths: list[MissingSubpath]
    orphans: list[str] = Field(description="Notes no other note links to.")
    dead_ends: list[str] = Field(description="Notes with no resolved links to another note.")
    duplicate_basenames: list[DuplicateName]
    empty_notes: list[str]
    limit: int | None


# ----- notes.pick ----------------------------------------------------------------


class NotesPickParams(Params):
    query: str = Field(min_length=1, max_length=200, description="Space-separated tokens, each matched as a subsequence.")
    limit: int = Field(20, ge=1, le=200)


class PickedNote(Result):
    path: str
    title: str
    matched_field: Literal["title", "alias"]
    matched_text: str
    score: int
    positions: list[int] = Field(description="Indexes of matched characters in matched_text.")


class NotesPickResult(Result):
    query: str
    total: int = Field(description="Matching notes before the limit was applied.")
    results: list[PickedNote]


# ----- note.get ------------------------------------------------------------------


class NoteGetParams(Params):
    path: str = Field(min_length=1, description="Vault-relative path, e.g. from notes.pick.")


class TagItem(Result):
    name: str
    source: Literal["frontmatter", "inline"]
    line: int | None


class HeadingItem(Result):
    level: int
    text: str
    line: int


class BlockItem(Result):
    id: str
    line: int


class OutgoingLink(Result):
    line: int
    raw: str
    syntax: Literal["wikilink", "markdown"]
    is_embed: bool
    target: str = Field(description="As written; empty for same-note links.")
    heading_path: list[str]
    block_id: str | None
    display: str | None
    status: LinkStatus
    target_path: str | None
    matched_by: Literal["self", "path", "name", "alias"] | None
    candidates: list[str] = Field(description="Matching files when ambiguous.")
    subpath_status: SubpathStatus | None


class IncomingLink(Result):
    source: str
    line: int
    raw: str
    is_embed: bool
    heading_path: list[str]
    block_id: str | None
    subpath_status: SubpathStatus | None


class NoteGetResult(Result):
    path: str
    title: str
    aliases: list[str]
    has_frontmatter: bool
    frontmatter_error: str | None
    properties: dict[str, Any]
    tags: list[TagItem]
    headings: list[HeadingItem]
    blocks: list[BlockItem]
    body_chars: int
    line_count: int
    links_out: list[OutgoingLink]
    links_in: list[IncomingLink] = Field(description="Resolved links from other notes (self-links excluded).")


# ----- formats: shared pieces -------------------------------------------------------

ViolationCode = Literal[
    "missing-heading", "heading-order", "missing-key", "missing-value", "bad-value", "missing-relation",
    "unexpected-relation", "unknown-field", "bad-relation", "unresolved-target", "ambiguous-target",
    "relation-target-type", "missing-mirror",
]
WriteStatus = Literal["written", "refused", "dry-run", "unchanged"]
FixKind = Literal["missing-mirror", "missing-heading"]


class FormatRef(Result):
    name: str
    path: str


class Violation(Result):
    path: str
    line: int = Field(description="Where it is, or for something missing, where it belongs.")
    code: ViolationCode
    message: str
    related: str | None = Field(description="Another note involved: for missing-mirror, the note whose field implies it.")


class CodeCount(Result):
    code: str
    count: int


class WrittenFile(Result):
    path: str
    action: Literal["create", "modify"]
    changes: list[str] = Field(description="What was done, for people: 'line 11: added [[X]] to `supported-by::`'.")
    diff: str = Field(description="Unified diff of the file (against /dev/null for a new file).")
    content: str | None = Field(description="The whole new file, for action 'create'; null otherwise.")


class IndexSync(Result):
    status: Literal["updated", "no-index", "needs-rebuild"]
    synced: list[str] = Field(description="Written notes now up to date in the index.")
    pending: list[str] = Field(description="Written notes left 'changed' for vault.check changed=true.")
    reason: str | None = None


class WriteResult(Result):
    operation: Literal["new", "link", "set", "fix", "ingest", "link-it", "unlink"]
    status: WriteStatus = Field(description="written; refused (would add violations, nothing written); dry-run; unchanged.")
    dry_run: bool
    written: bool
    files: list[WrittenFile]
    already: list[str] = Field(description="Parts of the request that were already so, e.g. an existing mirror.")
    new_violations: list[Violation] = Field(description="Why a write was refused: violations it would add.")
    tolerated: list[Violation] = Field(description="Violations already in the notes written; they do not block a write.")
    index: IndexSync | None = Field(description="Null unless written.")
    recovered: list[str] = Field(description="Files restored from an interrupted earlier write.")


class WriteParams(Params):
    dry_run: bool = Field(False, description="Check and return the diff; write nothing.")


# ----- vault.check ----------------------------------------------------------------


class VaultCheckParams(Params):
    changed: bool = Field(False, description="Only notes added or changed since the last index.build, plus violations they cause elsewhere.")
    limit: int | None = Field(None, ge=1, description="Max notes listed (null = all). The summary is never limited.")


class CheckScope(Result):
    mode: Literal["all", "changed"]
    checked: int = Field(description="Notes checked.")
    notes: list[str] | None = Field(description="The notes checked, for mode 'changed'.")
    added: list[str]
    changed: list[str]
    deleted: list[str]
    last_indexed_at: str | None
    reason: str | None = Field(description="Why a 'changed' check looked at every note (the index needs a rebuild).")


class CheckSummary(Result):
    violations: int
    notes: int
    by_code: list[CodeCount]


class CheckedNote(Result):
    path: str
    type: str | None
    checked: bool = Field(description="False: not changed itself, but affected by a changed or deleted note.")
    violations: list[Violation]


class VaultCheckResult(Result):
    format: FormatRef
    scope: CheckScope
    ok: bool
    summary: CheckSummary
    notes: list[CheckedNote] = Field(description="Notes with violations, by path; violations by line.")
    limit: int | None


# ----- vault.fix ---------------------------------------------------------------------


class VaultFixParams(WriteParams):
    only: list[FixKind] | None = Field(None, min_length=1, description="Repair only these kinds (default: both).")


class SkippedFix(Result):
    path: str
    line: int
    code: ViolationCode
    message: str
    related: str | None
    reason: str = Field(description="Why this one is not repaired automatically.")


class VaultFixResult(WriteResult):
    skipped: list[SkippedFix]
    remaining: list[CodeCount] = Field(description="All violations left once the fix is applied.")


# ----- note.new / note.link / note.set --------------------------------------------------


class RelationRef(Params):
    relation: str = Field(min_length=1, description="A relation of the format, e.g. supports.")
    note: str = Field(min_length=1, description="Vault path or note name.")


class NoteNewParams(WriteParams):
    type: str = Field(min_length=1, description="A note type of the format, e.g. zettel.")
    title: str = Field(min_length=1)
    folder: str | None = Field(None, description="Default: the type's match folder.")
    name: str | None = Field(None, description="File name without .md; default: from the title per the type's new.name.")
    values: dict[str, str] = Field(default_factory=dict, description="Metadata values; {now} works for date keys.")
    relations: list[RelationRef] = Field(default_factory=list, description="Relations to existing notes; their mirrors are written there.")
    sections: dict[str, str] = Field(default_factory=dict, description="Text under required headings.")


class NoteLinkParams(WriteParams):
    source: str = Field(min_length=1, description="Note A (path or name): gets `relation:: [[B]]`.")
    target: str = Field(min_length=1, description="Note B: gets the mirror if the relation has an inverse.")
    relation: str = Field(min_length=1)


class NoteSetParams(WriteParams):
    note: str = Field(min_length=1)
    key: str = Field(min_length=1, description="A metadata key of the note's type.")
    value: str = Field(description="As written in the note, e.g. '#supported'; {now} works for date keys.")


# ----- note.ingest --------------------------------------------------------------------------


class NoteIngestParams(WriteParams):
    path: str | None = Field(None, min_length=1, description="A .md, .txt or .html file the server can read.")
    content: str | None = Field(None, description="The document itself, instead of path; needs filename.")
    filename: str | None = Field(None, min_length=1, description="Original file name (decides the kind); with content.")
    type: str = Field("source", min_length=1, description="Note type with an ingest: block.")
    title: str | None = None
    folder: str | None = None
    values: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_source(self) -> "NoteIngestParams":
        if (self.path is None) == (self.content is None):
            raise ValueError("give exactly one of path or content")
        if self.content is not None and self.filename is None:
            raise ValueError("content needs a filename (its extension decides markdown, text or html)")
        return self


class IngestDocument(Result):
    file: str
    kind: Literal["markdown", "text", "html"]
    title: str
    sha256: str
    url: str | None
    warnings: list[str]


class DuplicateCandidate(Result):
    path: str
    duplicate: bool
    reasons: list[str]
    title_similarity: float
    titles_compared: list[str] = Field(description="[the document's title, the note's closest title]")
    content_overlap: float | None = Field(description="Null when a text is too short to compare.")
    shared_shingles: int
    shingles_compared: int
    same_hash: bool


class NoteIngestResult(WriteResult):
    status: Literal["written", "refused", "dry-run", "unchanged", "duplicate"] = Field(
        description="duplicate: an existing note matches; nothing written (see duplicates)."
    )
    document: IngestDocument
    duplicates: list[DuplicateCandidate]
    candidates: list[DuplicateCandidate] = Field(description="The closest existing notes, duplicates first (up to 5).")


# ----- vault.coverage ---------------------------------------------------------------------


class VaultCoverageParams(Params):
    limit: int | None = Field(None, ge=1, description="Max items per list (null = all). Counts are never limited.")


class CoverageSources(Result):
    total: int
    linked: int
    unlinked: list[str] = Field(description="Sources no other note links to.")


class CoverageReasons(Result):
    missing: int
    not_a_source: int
    none: int


class CoverageCitation(Result):
    line: int
    raw: str
    role: str | None
    status: LinkStatus
    target_path: str | None
    target_type: str | None


class UncitedNote(Result):
    path: str
    type: str | None
    reason: Literal["missing", "not-a-source", "none"]
    citations: list[CoverageCitation]


class CoverageCiting(Result):
    total: int
    cite_a_source: int
    without_source: int
    by_reason: CoverageReasons
    notes: list[UncitedNote]


class MostCited(Result):
    path: str
    citing_notes: int


class VaultCoverageResult(Result):
    format: FormatRef
    source_types: list[str]
    citing_types: list[str]
    citation_roles: list[str]
    sources: CoverageSources
    citing: CoverageCiting
    most_cited: list[MostCited]
    limit: int | None


# ----- search.query ---------------------------------------------------------------


class SearchQueryParams(Params):
    query: str = Field(
        min_length=1, max_length=2000,
        description='The query language of `vault search` (README: "Search and queries"). A syntax error is error -32006, '
                    "with data.start/data.end (offsets in the query) and data.hint.",
    )
    limit: int = Field(10, ge=1, le=500, description="Hits returned; total counts all matches.")
    fuzzy: bool = Field(True, description="Let words match close spellings.")
    facets: int = Field(8, ge=0, le=200, description="Rows of tag and folder counts (0 = all).")
    snippet_width: int = Field(100, ge=20, le=400, description="Longest snippet, in characters.")


class Mark(Result):
    start: int = Field(description="Offset of the first marked character in the text it belongs to.")
    end: int
    kind: Literal["exact", "fuzzy"] = Field(description="fuzzy: a close spelling of a query word.")


class FieldCount(Result):
    field: Literal["title", "aliases", "headings", "tags", "body"]
    count: int = Field(description="Occurrences (for a phrase or NEAR pair: its matches).")


class ClauseMatch(Result):
    clause: str = Field(description="The clause as written in the query, e.g. reward, \"reward hacking\", a NEAR/3 b.")
    exact: bool
    term: str | None = Field(description="For a word: the indexed word that matched (a close spelling when not exact).")
    edits: int
    fields: list[FieldCount]


class Snippet(Result):
    field: Literal["body", "headings", "aliases", "tags"]
    text: str
    marks: list[Mark]
    truncated_start: bool
    truncated_end: bool


class SearchHit(Result):
    path: str
    title: str
    score: float
    exact: int = Field(description="Words, phrases and NEAR pairs matched exactly; the first sort key.")
    matches: list[ClauseMatch] = Field(description="Empty when the note matched only the filters.")
    title_marks: list[Mark]
    snippet: Snippet | None


class Variant(Result):
    term: str
    edits: int
    notes: int


class SearchClause(Result):
    label: str
    kind: Literal["word", "phrase", "near"]
    required: bool
    notes: int = Field(description="Notes that match it exactly.")
    variants: list[Variant] = Field(description="Close spellings found in the index.")


class Count(Result):
    name: str
    notes: int


class Facets(Result):
    tags: list[Count] = Field(description="Tags of all matching notes; a nested tag also counts for its parents.")
    folders: list[Count] = Field(description="Folders of all matching notes, every ancestor; \"\" is the vault root.")


class SearchQueryResult(Result):
    query: str
    tree: str = Field(description="The parsed query as an s-expression.")
    notes: int = Field(description="Notes in the vault.")
    total: int = Field(description="Matching notes before the limit.")
    order: Literal["exact_then_score", "score", "path"]
    warnings: list[str]
    clauses: list[SearchClause] = Field(description="Words, phrases and NEAR pairs that rank (not negated).")
    hits: list[SearchHit]
    facets: Facets


# ----- note.preview --------------------------------------------------------------


class NotePreviewParams(Params):
    path: str = Field(min_length=1, description="Vault-relative path, e.g. from search.query.")
    query: str | None = Field(None, max_length=2000, description="Mark this query's matches, as search.query counts them.")


class PreviewLine(Result):
    number: int = Field(description="1-based line number in the file.")
    text: str
    marks: list[Mark]


class NotePreviewResult(Result):
    path: str
    title: str
    query: str | None
    matched: bool = Field(description="Whether the note matches the query at all.")
    marks: int = Field(description="Marked spans in the whole note.")
    first_match_line: int | None
    lines: list[PreviewLine] = Field(description="Every line of the file; matches marked in the body and headings.")


# ----- note.related --------------------------------------------------------------


class NoteRelatedParams(Params):
    note: str = Field(min_length=1, description="Note name, path or alias as in a link, or the target of links to a missing note.")
    limit: int = Field(10, ge=1, le=500)
    unlinked: bool = Field(False, description="Only notes with no link to or from the note.")


class RelatedSeed(Result):
    label: str
    path: str | None = Field(description="null for a missing note (a link target without a note).")
    linkers: list[str] = Field(description="For a missing note: the notes that link to it.")


class RelatedHit(Result):
    path: str
    title: str
    score: float = Field(description="The weighted sum of the scaled signals (engine.related.weights; default 0.7 text + 0.1 graph + 0.1 tags + 0.05 co-citation + 0.05 coupling); see `weights`.")
    graph: float
    text: float = Field(description="Centered cosine of the two notes' embeddings, over the best one (0 to 1).")
    tags: float
    cocitation: float = Field(description="Notes that link to both, scaled (0 unless engine.related.weights.cocitation > 0).")
    coupling: float = Field(description="What both link to, scaled (0 unless engine.related.weights.coupling > 0).")
    linked: bool
    hops: int | None
    reason: str = Field(description='E.g. "2 hops via kl-penalty-in-rlhf · shares #rl · similar text (kl, ppo)"; the words are'
                        " those in both notes that add most to their embedding cosine.")


class TextSignal(Result):
    """The state of the text (embedding) signal for this answer."""

    available: bool = Field(description="False when the embedding model cannot be loaded; the other signals still rank.")
    model_id: str | None = Field(description="The embedding model behind the text signal.")
    unavailable: str | None = Field(description="Why the text signal is off (e.g. the model folder is missing).")
    uncached: int = Field(description="Notes without a cached vector, embedded in memory for this answer (`vault embed` caches them).")


class NoteRelatedResult(Result):
    seed: RelatedSeed
    compared: int
    hits: list[RelatedHit]
    text_signal: TextSignal
    weights: dict[str, float] = Field(description="The weights used: graph, cocitation, coupling, tags, text (engine.related.weights, rescaled to sum to 1 without the text signal).")


# ----- links.suggest -------------------------------------------------------------


class LinksSuggestParams(Params):
    limit: int = Field(10, ge=1, le=500)
    note: str | None = Field(None, min_length=1, description="Only pairs with this note.")
    path: str | None = Field(None, min_length=1, description="Only pairs whose paths both contain this text.")


class LinkSuggestion(Result):
    a: str
    b: str
    score: float = Field(description="(0.4 shared + 0.3 cited + 0.2 tags + 0.3 text) / 1.2; without the text signal, the rescaled `weights`.")
    shared: float
    cited: float
    tags: float = Field(description="Weighted Jaccard of the two notes' #tags and tag links (Tags: [[topic]]), as in note.related.")
    text: float = Field(description="Centered cosine of the two notes' embeddings (0 if negative).")
    shared_with: list[str]
    cited_by: list[str]
    shared_tags: list[str] = Field(description="The #tags and [[tag links]] both notes have, rarest first.")
    terms: list[str] = Field(description="Words in both notes that add most to their embedding cosine.")
    reason: str


class LinksSuggestResult(Result):
    note: str | None
    path: str | None
    links: list[LinkSuggestion] = Field(description="Never a pair that is already linked, or a note with itself.")
    text_signal: TextSignal
    weights: dict[str, float] = Field(description="The weights used: shared, cited, tags, text (rescaled to sum to 1 without the text signal).")


# ----- notes.suggest -------------------------------------------------------------


class NotesSuggestParams(Params):
    limit: int = Field(10, ge=1, le=500)
    min_links: int = Field(2, ge=1, description="Only missing notes linked from at least this many notes.")


class MissingNote(Result):
    name: str
    linking_notes: int
    links: int
    mentions: int = Field(description="Other notes that contain the name without linking it.")
    folders: list[Count]
    about: list[str]


class NameCount(Result):
    name: str
    linking_notes: int


class ProbablyExisting(Result):
    name: str
    linking_notes: int
    path: str


class NotesSuggestResult(Result):
    total_missing: int
    notes: list[MissingNote]
    same_idea: list[list[NameCount]] = Field(description="Overlapping names: the same idea twice, or a topic and a narrower one.")
    existing: list[ProbablyExisting]


# ----- mentions.unlinked ---------------------------------------------------------


class MentionsUnlinkedParams(Params):
    note: str | None = Field(None, min_length=1, description="Only mentions of this note; omit for every note.")
    min_words: int = Field(1, ge=1, description="Ignore titles and aliases shorter than this many words.")
    limit: int | None = Field(20, ge=1, description="Mentioned notes returned (null = all); totals are never limited.")
    per_note: int = Field(20, ge=1, le=10000, description="Mentions returned per mentioned note.")


class UnlinkedMention(Result):
    source: str = Field(description="The note with the mention; it does not link to the target.")
    line: int
    start: int = Field(description="Offset in the source note's text.")
    end: int
    before: str
    text: str
    after: str


class MentionedNote(Result):
    path: str
    title: str
    names: list[str] = Field(description="Titles and aliases that matched.")
    total: int
    notes: int
    mentions: list[UnlinkedMention]


class MentionsUnlinkedResult(Result):
    note: str | None
    total: int
    notes: int = Field(description="Notes with at least one unlinked mention.")
    targets: list[MentionedNote] = Field(description="By number of mentioning notes.")


# ----- explore: explore.run, explore.mark, explore.kept, explore.link, explore.unlink ----------------------------------

ExploreLensName = Literal["neighbourhood", "cited", "foundations", "backbone", "cluster", "hidden", "role", "bridge", "gap",
                          "surprise", "unwritten", "wander", "topicless"]


class ExploreRunParams(Params):
    note: str | None = Field(None, min_length=1, description="A note (name, path or alias) or a missing note; null: where to start.")
    lenses: list[ExploreLensName] | None = Field(None, min_length=1, description="Null: engine.explore.lenses (or start_lenses without a note).")
    per_lens: int | None = Field(None, ge=3, le=5, description="Results per section; null: engine.explore.per_lens.")
    fresh: bool = Field(False, description="Leave out what was seen or dismissed before from this start.")
    record: bool = Field(True, description="Remember the run: its results become refs for explore.mark/link, and count as seen.")
    refresh: bool = Field(False, description="Recompute the stored vault-wide numbers now.")


class ExploreItem(Result):
    ref: str = Field(description="'2.3': section 2, result 3 of this run; what explore.mark and explore.link take.")
    lens: ExploreLensName
    kind: Literal["note", "pair", "link", "missing", "community"] = Field(
        description="note: a note; pair: two notes not linked (a gap's joiners); link: a link that is there (surprise); "
                    "missing: a missing note; community: a community or a pair of them.")
    target: str = Field(description="What it is about, as remembered: a path, 'a ↔ b', '[[name]]', 'community: …'.")
    title: str
    evidence: str = Field(description="One line: why the lens found it, with the numbers it rests on.")
    notes: list[str] = Field(description="The notes it is about (vault paths).")
    link: list[str] | None = Field(description="The two notes explore.link would link; null when it is not two notes to link.")
    start_at: str | None = Field(description="Without a start: where explore would start for it (pass it as note).")
    mark: Literal["keep", "dismiss"] | None = Field(description="Your earlier mark for this result from this start.")


class ExploreSection(Result):
    lens: ExploreLensName
    kind: Literal["deepen", "discover", "map"]
    title: str
    what: str
    budget: int = Field(description="Results this section may show (3 to 5; deepen and discover get the same total).")
    found: int = Field(description="Candidates the lens found.")
    hidden: int = Field(description="Left out: seen or dismissed before (fresh).")
    duplicates: int = Field(description="Left out: near-duplicates of a result shown.")
    kept: int = Field(description="Your keeps of this lens's results, over every start.")
    dismissed: int
    keep_score: float = Field(description="(kept + 1) / (kept + dismissed + 2): the sections are ordered by it.")
    applies: bool = Field(description="False: the lens does not apply to this start (e.g. a missing note) and takes no space.")
    note: str | None = Field(description="Why there are no results, when there are none.")
    results: list[ExploreItem]


class ExploreNumbers(Result):
    source: Literal["current", "reused"] = Field(description="current: computed for the vault as it is; reused: the vault-wide "
                                                            "distributions are from a few notes ago (see computed_at).")
    computed_at: str | None = None
    changed_since: int | None = None
    threshold: int | None = None


class ExploreRunResult(Result):
    start: str | None = Field(description="The start's path, '[[missing note]]', or null (proposals).")
    start_kind: Literal["note", "missing"] | None
    fresh: bool
    duplicate_cutoff: float | None = Field(description="Similarity at or above which two notes are near-duplicates.")
    similarity: Literal["embedding", "key terms"]
    memory: str = Field(description="The memory file (marks, seen, links made, map snapshots).")
    numbers: ExploreNumbers
    text_signal: TextSignal
    sections: list[ExploreSection] = Field(description="Ordered by keep_score, ties in the configured order.")


class ExploreMarkParams(Params):
    verdict: Literal["keep", "dismiss", "clear"]
    refs: list[str] = Field(min_length=1, description="Refs of the last explore.run ('2.3').")


class MarkedItem(Result):
    ref: str
    lens: ExploreLensName
    target: str
    title: str
    verdict: Literal["keep", "dismiss", "clear"]
    changed: bool = Field(description="False for a clear of a result that was not marked.")


class KeepRate(Result):
    lens: ExploreLensName
    kept: int
    dismissed: int


class ExploreMarkResult(Result):
    marked: list[MarkedItem]
    keep_rates: list[KeepRate] = Field(description="For the lenses marked here, over every start.")


class ExploreKeptParams(Params):
    lens: ExploreLensName | None = None


class KeptItem(Result):
    start: str | None
    lens: ExploreLensName
    target: str
    kind: Literal["note", "pair", "link", "missing", "community"]
    title: str
    evidence: str
    link: list[str] | None
    at: str


class ExploreKeptResult(Result):
    memory: str
    kept: list[KeptItem] = Field(description="Oldest first.")


class ExploreLinkParams(Params):
    ref: str | None = Field(None, description="A result of the last explore.run; or give a and b.")
    a: str | None = None
    b: str | None = None
    reason: str | None = Field(None, description="The one-line reason; default: the result's lens and evidence.")
    dry_run: bool = Field(True, description="True (the default): return the diff and write nothing.")

    @model_validator(mode="after")
    def _one_way(self) -> "ExploreLinkParams":
        if (self.ref is None) == (self.a is None or self.b is None):
            raise ValueError("give ref, or a and b")
        return self


class ExploreLinkResult(WriteResult):
    ref: str | None
    lens: ExploreLensName | None
    notes: list[str] = Field(description="The two notes, as given or from the result.")
    reason: str
    kept: bool = Field(description="The result counted as a keep (a link made from a result).")


class ExploreUnlinkParams(Params):
    a: str = Field(min_length=1)
    b: str = Field(min_length=1)
    dry_run: bool = Field(True, description="True (the default): return the diff and write nothing.")


class ExploreUnlinkResult(WriteResult):
    recorded: bool = Field(description="link-it made this link: its section goes too when nothing else is left in it.")


# ----- map.get, map.since ---------------------------------------------------------------------------------------------


class MapParams(Params):
    limit: int = Field(10, ge=0, description="Items per list (0 = all).")
    meaning: bool | None = Field(None, description="Add meaning edges for the communities (null: engine.explore.meaning).")
    record: bool = Field(False, description="Also save a snapshot in the memory (for map.since).")


class TagCountItem(Result):
    tag: str
    notes: int


class MapPhrase(Result):
    phrase: str
    in_community: int
    in_vault: int
    score: float


class MapTopicPage(Result):
    path: str
    reaches: int


class MapCommunity(Result):
    number: int
    name: str
    notes: list[str]
    phrases: list[MapPhrase]
    hubs: list[str]
    topic_pages: list[MapTopicPage]


class MapBetweenness(Result):
    exact: bool
    sources: int
    visits: int
    budget: int


class MapBridge(Result):
    path: str
    betweenness: float
    betweenness_percentile: float
    constraint: float | None
    constraint_percentile: float | None
    community: int
    spans: list[int]


class MapHub(Result):
    path: str
    score: float


class MapAuthority(Result):
    target: str
    missing: bool
    score: float


class MeaningEdge(Result):
    a: str
    b: str
    cosine: float


class MovedNote(Result):
    note: str
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    from_: int = Field(alias="from")
    to: int


class PlacedNote(Result):
    note: str
    community: int


class MeaningChanges(Result):
    stranded_before: list[str]
    stranded_after: list[str]
    placed: list[PlacedNote]
    moved: list[MovedNote]
    match: dict[int, int | None]
    split: dict[int, list[int]]
    joined: dict[int, list[int]]
    communities_before: int
    communities_after: int


class MapMeaning(Result):
    edges: list[MeaningEdge]
    cutoff: float | None
    percentile: float
    unavailable: str | None
    changes: MeaningChanges | None


class MapRecorded(Result):
    at: str
    bytes: int
    memory: str
    communities: int
    gaps: int
    basis: str
    missing: int


class MapResult(Result):
    percentile: float
    topic_pages: str = Field(description="How topic pages were decided (the format's types), or why there are none.")
    notes: int
    missing_notes: int
    tags: int
    left_out_tags: list[TagCountItem]
    nodes: int
    edges: int
    communities: list[MapCommunity]
    notes_in_communities: int
    alone: int
    modularity: float
    seed: int
    resolution: float
    tag_rule: str
    meaning_edges: int
    meaning_cutoff: float | None
    meaning_unavailable: str | None
    betweenness: MapBetweenness
    bridges: list[MapBridge]
    hubs: list[MapHub]
    authorities: list[MapAuthority]
    meaning: MapMeaning | None = None
    recorded: MapRecorded | None = None


class MapSinceParams(Params):
    since: str = Field(min_length=1, description="A date (2026-11-10) or a time (2026-11-10T14:30).")


class SinceGap(Result):
    communities: list[int]
    names: list[str]
    links_then: float
    expected_then: float
    links_now: float
    expected_now: float
    added: list[list[str]] = Field(description="Links between the two sides made since (pairs of paths).")
    status: Literal["open", "closed", "gone"]
    why: str


class SinceCommunity(Result):
    community: int
    name: str
    notes: int
    best_match: int | None = None
    overlap: float


class SinceWritten(Result):
    name: str
    linked_from: int
    note: str


class SinceTopicPage(Result):
    community_then: int
    name: str
    community: int
    topic_pages: list[str]


class MapSinceResult(Result):
    since: str
    snapshot: str = Field(description="When the snapshot compared with was recorded.")
    chosen: str = Field(description="How it was chosen.")
    notes: list[int] = Field(description="[then, now]")
    links: list[int]
    communities: list[int]
    formed: list[SinceCommunity]
    gone: list[SinceCommunity]
    gaps: list[SinceGap]
    gap_basis: str | None
    written: list[SinceWritten]
    topic_pages: list[SinceTopicPage]


# ----- note.deepen ----------------------------------------------------------------------------------------------------

DeepenLensName = Literal["neighbourhood", "cited", "foundations", "backbone", "cluster"]


class NoteDeepenParams(Params):
    note: str = Field(min_length=1)
    lenses: list[DeepenLensName] | None = Field(None, min_length=1)
    limit: int = Field(8, ge=0, description="Notes per lens (0 = all that stand out).")


class DeepenHit(Result):
    path: str
    score: float
    linked: bool
    reason: str


class DeepenNeighbourhood(Result):
    cutoff: float
    percentile: float
    compared: int
    kept: int
    text_unavailable: str | None
    uncached: int
    hits: list[DeepenHit]


class EvidenceName(Result):
    name: str
    weight: float


class DeepenPairNote(Result):
    path: str
    value: float
    percentile: float
    evidence: list[EvidenceName]


class DeepenPairs(Result):
    cutoff: float
    percentile: float
    pairs_in_vault: int
    exact: bool
    found: int
    kept: int
    notes: list[DeepenPairNote]
    below_cutoff: list[DeepenPairNote]


class DeepenBackbone(Result):
    community: int | None
    chain: list[str]
    steps: list[str] = []
    note_on_chain: int | None
    steps_off: int | None
    joins_at: str | None
    length: int | None = None
    community_notes: int | None = None


class ClusterPhrase(Result):
    phrase: str
    in_community: int
    in_vault: int


class ClusterNeighbour(Result):
    community: int
    name: str
    weight: float
    through: list[str]


class DeepenCluster(Result):
    community: int | None
    name: str | None = None
    notes: int | None = None
    phrases: list[ClusterPhrase] = []
    hubs: list[str] = []
    topic_pages: list[MapTopicPage] = []
    weight_inside: float | None = None
    inside_percentile: float | None = None
    weight: float
    outside_share: float | None = None
    participation: float
    neighbours: list[ClusterNeighbour] = []


class DeepenLenses(Result):
    neighbourhood: DeepenNeighbourhood | None = None
    cited: DeepenPairs | None = None
    foundations: DeepenPairs | None = None
    backbone: DeepenBackbone | None = None
    cluster: DeepenCluster | None = None


class NoteDeepenResult(Result):
    note: str
    percentile: float
    lenses: DeepenLenses


# ----- vault.discover -------------------------------------------------------------------------------------------------

DiscoverLensName = Literal["hidden", "role", "bridge", "gap", "surprise", "unwritten", "wander"]


class VaultDiscoverParams(Params):
    note: str | None = Field(None, min_length=1, description="Null: the lenses over the whole vault (gap, surprise, unwritten).")
    lenses: list[DiscoverLensName] | None = Field(None, min_length=1)
    limit: int = Field(3, ge=0, description="Results per lens (0 = all that stand out).")
    steps: int = Field(8, ge=1, le=200, description="wander: steps to take.")
    seed: int | None = Field(None, description="wander and sampling; null: engine.explore.seed.")


class Middle(Result):
    middle: str
    kind: str
    notes: int
    weight: float


class HiddenResult(Result):
    note: str
    score: float
    percentile: float
    middles: list[Middle]
    reason: str


class HiddenLens(Result):
    cutoff: float
    pairs_in_vault: int
    exact: bool
    sources: int
    found: int
    kept: int
    results: list[HiddenResult]


class RoleResult(Result):
    note: str
    community: int
    distance: float
    percentile: float
    features: dict[str, list[float]] = Field(description="feature -> [this note's, that note's] raw value.")
    reason: str


class RoleLens(Result):
    cutoff: float
    pairs_in_vault: int
    exact: bool
    found: int
    kept: int
    results: list[RoleResult]


class CommunityWeight(Result):
    community: int
    weight: float


class Broker(Result):
    communities: list[int]
    share: float
    pairs: int
    reason: str


class BridgeLens(Result):
    betweenness: float
    betweenness_percentile: float
    constraint: float | None
    constraint_percentile: float | None
    community: int
    reaches: list[CommunityWeight]
    exact: bool
    sources: int
    brokers: list[Broker]


class Unavailable(Result):
    unavailable: str | None


class GapJoin(Result):
    a: str
    b: str
    score: float
    middles: list[str]


class GapResult(Result):
    communities: list[int]
    names: list[str]
    closeness: float
    closeness_percentile: float
    links: int
    expected: float
    ratio: float
    join: list[GapJoin]
    reason: str


class GapFound(Result):
    cutoff: float
    pairs: int
    found: int
    results: list[GapResult]


class GapLens(Result):
    with_embeddings: GapFound | Unavailable
    without_embeddings: GapFound | Unavailable


class SurpriseResult(Result):
    a: str
    b: str
    lines: list[int]
    similarity: float
    percentile: float
    communities: list[int]
    ratio: float | None
    reason: str


class SurpriseFound(Result):
    cutoff: float
    linked_pairs: int
    found: int
    results: list[SurpriseResult]


class SurpriseLens(Result):
    with_embeddings: SurpriseFound | Unavailable
    without_embeddings: SurpriseFound | Unavailable


class UnwrittenResult(Result):
    names: list[str]
    linking_notes: int
    communities: dict[int, int] = Field(description="community -> notes in it that link the missing note.")
    spread: int
    reason: str


class UnwrittenFound(Result):
    cutoff: float
    groups: int
    found: int
    by: str
    merge_cutoff: float | None
    merged: int
    results: list[UnwrittenResult]


class UnwrittenLens(Result):
    with_embeddings: UnwrittenFound | Unavailable
    without_embeddings: UnwrittenFound | Unavailable


class WanderHop(Result):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    from_: str = Field(alias="from")
    to: str
    lines_out: list[int]
    lines_back: list[int]
    community: int
    new_community: bool
    revisit: bool
    reason: str


class WanderLens(Result):
    seed: int
    steps: int
    walk: list[str]
    hops: list[WanderHop]
    communities: int


class DiscoverLenses(Result):
    hidden: HiddenLens | None = None
    role: RoleLens | None = None
    bridge: BridgeLens | None = None
    gap: GapLens | None = None
    surprise: SurpriseLens | None = None
    unwritten: UnwrittenLens | None = None
    wander: WanderLens | None = None


class VaultDiscoverResult(Result):
    note: str | None
    percentile: float
    seed: int
    text_signal: TextSignal
    lenses: DiscoverLenses


# ----- notes.explain --------------------------------------------------------------------------------------------------


class NotesExplainParams(Params):
    a: str = Field(min_length=1)
    b: str = Field(min_length=1)
    words: int = Field(8, ge=0, le=100, description="Words listed behind the text score.")


class CitingNote(Result):
    note: str
    weight: float
    links_to: int


class SharedReference(Result):
    target: str
    missing: bool
    weight: float
    linked_from: int


class WeightedTag(Result):
    tag: str
    weight: float


class LeftOutTag(Result):
    tag: str
    notes: int


class SharedWord(Result):
    word: str
    share: float


class DirectEvidence(Result):
    a_to_b: list[int] = Field(description="Lines of a with a link to b.")
    b_to_a: list[int]
    a_to_b_tag_lines: list[int]
    b_to_a_tag_lines: list[int]


class CocitationEvidence(Result):
    both: list[CitingNote]
    citing_a: int
    citing_b: int
    total_a: float
    total_b: float


class CouplingEvidence(Result):
    both: list[SharedReference]
    references_a: int
    references_b: int
    total_a: float
    total_b: float


class TagsEvidence(Result):
    shared: list[WeightedTag]
    only_a: list[WeightedTag]
    only_b: list[WeightedTag]
    shared_weight: float
    union_weight: float
    left_out: list[LeftOutTag]
    notes: int
    common_over: float


class TextEvidence(Result):
    cosine: float
    model_id: str
    words: list[SharedWord] | None = None
    shared_words: int | None = None
    shared_total: float | None = None
    one_note_total: float | None = None
    baseline: float | None = None
    matches_vectors: bool | None = None


class NoEvidence(Result):
    pass


class ExplainSignal(Result):
    name: Literal["direct", "cocitation", "coupling", "tags", "text"]
    value: float
    weight: float
    contribution: float = Field(description="weight x value; the score is their sum.")
    available: bool
    unavailable: str | None
    evidence: DirectEvidence | CocitationEvidence | CouplingEvidence | TagsEvidence | TextEvidence | NoEvidence


class ExplainWeights(Result):
    direct: float
    cocitation: float
    coupling: float
    tags: float
    text: float


class NotesExplainResult(Result):
    configured_weights: ExplainWeights
    a: str
    b: str
    score: float
    signals: list[ExplainSignal]
    model_id: str | None
    uncached: int


# ----- embed.status, config.show --------------------------------------------------------------------------------------


class EmbedStatusParams(Params):
    pass


class OtherModel(Result):
    model_id: str
    vectors: int
    dim: int | None


class EmbedStatusResult(Result):
    model_id: str | None
    notes: int
    with_vector: int
    missing: int
    missing_paths: list[str]
    dim: int | None
    other_models: list[OtherModel]
    cache_vectors: int
    cache_bytes: int
    provider: str
    model_path: str | None
    model_found: bool


class ConfigShowParams(Params):
    pass


class SettingRow(Result):
    key: str
    value: str | None
    source: str = Field(description="default, file, env NAME or flag --x.")
    env: str = Field(description="The environment variable that sets it.")


class ConfigShowResult(Result):
    config_file: str | None
    config_found_by: str
    precedence: list[str]
    settings: list[SettingRow]


# ----- registry ------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    name: str
    summary: str
    params: type[Params]
    result: type[Result]


METHODS: dict[str, MethodSpec] = {
    spec.name: spec
    for spec in (
        MethodSpec("index.build", "Create or incrementally update the index.", IndexBuildParams, IndexBuildResult),
        MethodSpec("index.status", "Whether the index exists and is up to date.", IndexStatusParams, IndexStatusResult),
        MethodSpec("vault.stats", "Summary numbers for the indexed vault.", VaultStatsParams, VaultStatsResult),
        MethodSpec("health.report", "Link and structure problems.", HealthReportParams, HealthReportResult),
        MethodSpec("notes.pick", "Fuzzy-find notes by title and alias.", NotesPickParams, NotesPickResult),
        MethodSpec("note.get", "Properties, tags, headings, blocks and links in/out of one note.", NoteGetParams, NoteGetResult),
        MethodSpec("vault.check", "Every violation of the format, per note, with line numbers.", VaultCheckParams, VaultCheckResult),
        MethodSpec("vault.fix", "Repair missing mirrors and missing required headings (safe write path).", VaultFixParams, VaultFixResult),
        MethodSpec("vault.coverage", "Sources nothing links to, and notes that cite no source.", VaultCoverageParams, VaultCoverageResult),
        MethodSpec("note.new", "Create a note of a type; refused if it would add a violation.", NoteNewParams, WriteResult),
        MethodSpec("note.link", "Relate note A to note B and write the mirror in B.", NoteLinkParams, WriteResult),
        MethodSpec("note.set", "Set a metadata value of a note.", NoteSetParams, WriteResult),
        MethodSpec("note.ingest", "Turn a document into a note; nothing is written if it duplicates one.", NoteIngestParams, NoteIngestResult),
        MethodSpec("search.query", "Search with text and structure; ranked hits with snippets, tag and folder counts.",
                   SearchQueryParams, SearchQueryResult),
        MethodSpec("note.preview", "A note's lines with a query's matches marked.", NotePreviewParams, NotePreviewResult),
        MethodSpec("note.related", "Notes related to a note by links, text and tags, each with a reason.",
                   NoteRelatedParams, NoteRelatedResult),
        MethodSpec("links.suggest", "Pairs of notes that probably belong linked, with the evidence.", LinksSuggestParams, LinksSuggestResult),
        MethodSpec("notes.suggest", "Missing notes worth writing, overlapping names, links meant for existing notes.",
                   NotesSuggestParams, NotesSuggestResult),
        MethodSpec("mentions.unlinked", "Plain-text mentions of note titles and aliases that are not links yet.",
                   MentionsUnlinkedParams, MentionsUnlinkedResult),
        MethodSpec("explore.run", "Explore from a note (or where to start): a section per lens, each result with its lens and "
                   "evidence; remembered for explore.mark and explore.link.", ExploreRunParams, ExploreRunResult),
        MethodSpec("explore.mark", "Keep, dismiss or clear results of the last explore.run (the memory, never the notes).",
                   ExploreMarkParams, ExploreMarkResult),
        MethodSpec("explore.kept", "What was kept from explore, with its lens and evidence.", ExploreKeptParams, ExploreKeptResult),
        MethodSpec("explore.link", "A result (or two notes) made a link under See Also in both notes; dry run unless dry_run=false.",
                   ExploreLinkParams, ExploreLinkResult),
        MethodSpec("explore.unlink", "Undo explore.link: remove the See Also lines linking the two notes; dry run unless dry_run=false.",
                   ExploreUnlinkParams, ExploreUnlinkResult),
        MethodSpec("map.get", "The map: communities, topic pages, bridges, hubs and authorities; record=true saves a snapshot.",
                   MapParams, MapResult),
        MethodSpec("map.since", "What changed since a recorded map: gaps closed, communities formed or gone, missing notes written.",
                   MapSinceParams, MapSinceResult),
        MethodSpec("note.deepen", "Five lenses on one note: neighbourhood, cited together, same foundations, backbone, cluster.",
                   NoteDeepenParams, NoteDeepenResult),
        MethodSpec("vault.discover", "Discover lenses: hidden links, same roles, bridges, gaps, surprises, unwritten notes, a wander.",
                   VaultDiscoverParams, VaultDiscoverResult),
        MethodSpec("notes.explain", "How two notes relate, signal by signal, with the evidence.", NotesExplainParams, NotesExplainResult),
        MethodSpec("embed.status", "How many notes have a cached vector for the configured model.", EmbedStatusParams, EmbedStatusResult),
        MethodSpec("config.show", "Every engine setting, its value and where it came from.", ConfigShowParams, ConfigShowResult),
    )
}
