"""The JSON-RPC 2.0 API, defined once.

``METHODS`` maps every method name to its params and result models. The
server validates params and results with these models, and ``schema.py``
derives the JSON Schema from them, so the schema cannot drift from the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

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

ERROR_MESSAGES = {
    PARSE_ERROR: "Parse error",
    INVALID_REQUEST: "Invalid Request",
    METHOD_NOT_FOUND: "Method not found",
    INVALID_PARAMS: "Invalid params",
    INTERNAL_ERROR: "Internal error",
    INDEX_NOT_BUILT: "Index not built",
    NOTE_NOT_FOUND: "Note not found",
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
    )
}
