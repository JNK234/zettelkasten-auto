"""Data model for a parsed vault note.

All positions refer to the note text after newline normalisation (CRLF/CR -> LF,
leading BOM removed). Line numbers are 1-based and count frontmatter lines, so
they match what an editor shows for the file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

LinkSyntax = Literal["wikilink", "markdown"]
TagSource = Literal["frontmatter", "inline"]


@dataclass(frozen=True)
class Link:
    """An internal link or embed, in either wikilink or markdown syntax.

    ``target`` is the note/file part as written (may be empty for same-note
    links such as ``[[#Heading]]``). Resolution to an actual file is not done here.
    """

    raw: str
    syntax: LinkSyntax
    target: str
    heading_path: tuple[str, ...] = ()  # [[Note#A#B]] -> ("A", "B")
    block_id: str | None = None  # [[Note#^abc]] -> "abc"
    display: str | None = None  # alias text after "|" or markdown link text
    is_embed: bool = False
    line: int = 0
    start: int = 0
    end: int = 0

    @property
    def is_same_note(self) -> bool:
        return self.target == ""


@dataclass(frozen=True)
class ExternalLink:
    """A markdown link whose destination has a URL scheme (https:, mailto:, ...)."""

    raw: str
    url: str
    display: str
    is_embed: bool = False
    line: int = 0


@dataclass(frozen=True)
class Tag:
    """A tag without its leading '#'. Nested tags keep their '/' separators."""

    name: str
    source: TagSource
    line: int | None = None  # None for frontmatter tags


@dataclass(frozen=True)
class Heading:
    level: int
    text: str
    line: int


@dataclass(frozen=True)
class BlockId:
    id: str
    line: int


@dataclass(frozen=True)
class HeaderField:
    """A ``Key: value`` line in the block at the very top of the body.

    Many vaults keep metadata there instead of in YAML frontmatter::

        Created: 24-09-2026 16:29
        Status: #baby
        Tags: [[reward]] [[evaluation-methodology]]

    ``value`` is the stripped text after the colon. ``links`` and ``tags`` are the
    links/inline tags found inside the value (the same objects as in
    ``ParsedNote.links`` / ``ParsedNote.tags``).
    """

    key: str
    value: str
    line: int
    value_start: int = 0
    value_end: int = 0
    links: tuple[Link, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class InlineField:
    """A Dataview-style ``key:: value`` line anywhere in the body.

    Used for typed relations, e.g. ``supports:: [[Claim A]], [[Claim B]]``.
    Nothing here knows which keys mean what; that is the formats layer's job.
    """

    key: str
    value: str
    line: int
    value_start: int = 0
    value_end: int = 0
    links: tuple[Link, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass
class ParsedNote:
    path: str | None = None
    frontmatter: dict[str, Any] = field(default_factory=dict)
    frontmatter_error: str | None = None
    has_frontmatter: bool = False
    body_start_line: int = 1
    last_line: int = 0  # number of the last line (a final newline does not start a new one)
    aliases: list[str] = field(default_factory=list)
    tags: list[Tag] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)  # non-embed internal links
    embeds: list[Link] = field(default_factory=list)
    external_links: list[ExternalLink] = field(default_factory=list)
    headings: list[Heading] = field(default_factory=list)
    block_ids: list[BlockId] = field(default_factory=list)
    header_fields: list[HeaderField] = field(default_factory=list)
    inline_fields: list[InlineField] = field(default_factory=list)

    @property
    def tag_names(self) -> list[str]:
        """Distinct tag names in first-seen order (frontmatter first).

        Obsidian treats tags case-insensitively; the first spelling wins here.
        """
        seen: dict[str, str] = {}
        for tag in self.tags:
            seen.setdefault(tag.name.lower(), tag.name)
        return list(seen.values())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tag_names"] = self.tag_names
        return data
