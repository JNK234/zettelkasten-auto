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
    operation: Literal["new", "link", "set", "fix", "ingest"]
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
    score: float = Field(description="0.5 graph + 0.3 text + 0.2 tags.")
    graph: float
    text: float
    tags: float
    linked: bool
    hops: int | None
    reason: str = Field(description='E.g. "2 hops via kl-penalty-in-rlhf · shares #rl · similar text (kl, ppo)".')


class NoteRelatedResult(Result):
    seed: RelatedSeed
    compared: int
    hits: list[RelatedHit]


# ----- links.suggest -------------------------------------------------------------


class LinksSuggestParams(Params):
    limit: int = Field(10, ge=1, le=500)
    note: str | None = Field(None, min_length=1, description="Only pairs with this note.")
    path: str | None = Field(None, min_length=1, description="Only pairs whose paths both contain this text.")


class LinkSuggestion(Result):
    a: str
    b: str
    score: float = Field(description="0.4 shared + 0.3 cited + 0.3 text.")
    shared: float
    cited: float
    text: float
    shared_with: list[str]
    cited_by: list[str]
    terms: list[str]
    reason: str


class LinksSuggestResult(Result):
    note: str | None
    path: str | None
    links: list[LinkSuggestion] = Field(description="Never a pair that is already linked, or a note with itself.")


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
    )
}
