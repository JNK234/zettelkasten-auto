"""Parse a single Obsidian note into structured data.

Pipeline:
1. normalise newlines and strip a BOM;
2. split off YAML frontmatter (tags and aliases come from there too);
3. mask frontmatter, fenced/inline code and %% comments (see ``mask.py``);
4. scan the masked text for headings, block IDs, wikilinks/embeds, markdown
   links and inline tags. Because masking keeps offsets stable, every match
   maps straight back to a line in the original file.

Nothing here resolves links to files. That belongs to the vault index.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote

from vaultengine.frontmatter import frontmatter_aliases, frontmatter_tags, split_frontmatter
from vaultengine.mask import mask_code, mask_region
from vaultengine.model import BlockId, ExternalLink, Heading, Link, ParsedNote, Tag

# Bump whenever parse output can change for the same input; the SQLite index
# re-parses every note when the stored version differs.
PARSER_VERSION = 1

_WIKILINK = re.compile(r"(?<!\\)(?P<bang>!?)\[\[(?P<inner>[^\[\]\n]+?)\]\]")

_MARKDOWN_LINK = re.compile(
    r"""
    (?<![\\!])(?P<bang>!?)(?<!\\)
    \[(?P<text>(?:[^\[\]\n\\]|\\.|\[[^\[\]\n]*\])*)\]
    \(
      [ \t]*
      (?:<(?P<angle>[^<>\n]*)>|(?P<bare>(?:[^\s()\\]|\\.|\((?:[^\s()\\]|\\.)*\))+))
      (?:[ \t]+(?:"[^"\n]*"|'[^'\n]*'|\([^()\n]*\)))?
      [ \t]*
    \)
    """,
    re.VERBOSE,
)

_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
_HEADING = re.compile(r"^ {0,3}(?P<hashes>#{1,6})(?=[ \t]|$)(?P<rest>.*)$", re.MULTILINE)
_CLOSING_HASHES = re.compile(r"(?:^|[ \t]+)#+[ \t]*$")
_BLOCK_ID = re.compile(r"(?:^|[ \t])\^(?P<id>[A-Za-z0-9-]+)[ \t]*$", re.MULTILINE)
_TAG_START = re.compile(r"(?<!\S)#")


class _LineIndex:
    def __init__(self, text: str) -> None:
        self._starts = [0] + [match.end() for match in re.finditer("\n", text)]

    def line_of(self, offset: int) -> int:
        return bisect.bisect_right(self._starts, offset)


def normalize_text(text: str) -> str:
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def is_valid_tag(name: str) -> bool:
    """Obsidian requires at least one non-numeric character (#1984 is not a tag)."""
    return bool(name) and not name.isdigit()


def _is_tag_char(char: str) -> bool:
    if char in "_-/\u200d":
        return True
    category = unicodedata.category(char)
    # Letters, numbers, combining marks, and symbols such as emoji.
    return category[0] in "LNM" or category == "So"


def _split_subpath(subpath: str) -> tuple[tuple[str, ...], str | None]:
    subpath = subpath.strip()
    if subpath.startswith("^"):
        return (), (subpath[1:].strip() or None)
    return tuple(part.strip() for part in subpath.split("#") if part.strip()), None


def _parse_wikilink_inner(inner: str) -> tuple[str, tuple[str, ...], str | None, str | None] | None:
    target_part, has_pipe, display = inner.partition("|")
    if has_pipe and target_part.endswith("\\"):
        target_part = target_part[:-1]  # "[[Note\|Alias]]" inside a table
    path, has_hash, subpath = target_part.partition("#")
    heading_path, block_id = _split_subpath(subpath) if has_hash else ((), None)
    path = path.strip()
    if not path and not heading_path and block_id is None:
        return None
    display = display.strip() if has_pipe else None
    return path, heading_path, block_id, (display or None)


def _extract_wikilinks(masked: str, lines: _LineIndex) -> list[Link]:
    found = []
    for match in _WIKILINK.finditer(masked):
        parsed = _parse_wikilink_inner(match["inner"])
        if parsed is None:
            continue
        target, heading_path, block_id, display = parsed
        found.append(
            Link(
                raw=match.group(),
                syntax="wikilink",
                target=target,
                heading_path=heading_path,
                block_id=block_id,
                display=display,
                is_embed=bool(match["bang"]),
                line=lines.line_of(match.start()),
                start=match.start(),
                end=match.end(),
            )
        )
    return found


def _extract_markdown_links(masked: str, lines: _LineIndex) -> tuple[list[Link], list[ExternalLink], list[tuple[int, int]]]:
    internal: list[Link] = []
    external: list[ExternalLink] = []
    destinations: list[tuple[int, int]] = []
    for match in _MARKDOWN_LINK.finditer(masked):
        group = "angle" if match["angle"] is not None else "bare"
        destination = match[group].strip()
        destinations.append((match.start(group), match.end(group)))
        if not destination:
            continue
        is_embed = bool(match["bang"])
        line = lines.line_of(match.start())
        if _URL_SCHEME.match(destination):
            external.append(ExternalLink(match.group(), destination, match["text"], is_embed, line))
            continue
        path, has_hash, subpath = unquote(destination).partition("#")
        heading_path, block_id = _split_subpath(subpath) if has_hash else ((), None)
        if not path.strip() and not heading_path and block_id is None:
            continue
        internal.append(
            Link(
                raw=match.group(),
                syntax="markdown",
                target=path.strip(),
                heading_path=heading_path,
                block_id=block_id,
                display=match["text"],
                is_embed=is_embed,
                line=line,
                start=match.start(),
                end=match.end(),
            )
        )
    return internal, external, destinations


def _extract_tags(masked: str, lines: _LineIndex) -> list[Tag]:
    tags = []
    for match in _TAG_START.finditer(masked):
        end = match.end()
        while end < len(masked) and _is_tag_char(masked[end]):
            end += 1
        name = masked[match.end() : end]
        if is_valid_tag(name):
            tags.append(Tag(name=name, source="inline", line=lines.line_of(match.start())))
    return tags


def _extract_headings(original: str, masked: str, lines: _LineIndex) -> list[Heading]:
    headings = []
    for match in _HEADING.finditer(masked):
        # Detect on masked text (skips code); read the text from the original
        # so inline code inside a heading is preserved.
        rest = original[match.start("rest") : match.end("rest")]
        text = _CLOSING_HASHES.sub("", rest).strip()
        if text:
            headings.append(Heading(len(match["hashes"]), text, lines.line_of(match.start())))
    return headings


def _extract_block_ids(masked: str, lines: _LineIndex) -> list[BlockId]:
    return [BlockId(m["id"], lines.line_of(m.start("id"))) for m in _BLOCK_ID.finditer(masked)]


def parse_note(text: str, path: str | None = None) -> ParsedNote:
    text = normalize_text(text)
    lines = _LineIndex(text)
    block = split_frontmatter(text)

    note = ParsedNote(
        path=path,
        frontmatter=block.data,
        frontmatter_error=block.error,
        has_frontmatter=block.present,
        body_start_line=block.body_start_line,
        aliases=frontmatter_aliases(block.data),
    )
    note.tags.extend(
        Tag(name=name, source="frontmatter") for name in frontmatter_tags(block.data) if is_valid_tag(name)
    )

    masked = mask_code(mask_region(text, 0, block.end_offset))
    note.headings = _extract_headings(text, masked, lines)
    note.block_ids = _extract_block_ids(masked, lines)

    wikilinks = _extract_wikilinks(masked, lines)
    without_wikilinks = masked
    for link in wikilinks:
        without_wikilinks = mask_region(without_wikilinks, link.start, link.end)
    markdown_links, note.external_links, destinations = _extract_markdown_links(without_wikilinks, lines)

    all_links = sorted(wikilinks + markdown_links, key=lambda link: link.start)
    note.links = [link for link in all_links if not link.is_embed]
    note.embeds = [link for link in all_links if link.is_embed]

    # '#' inside link targets or URLs is a subpath/fragment, never a tag.
    tag_scan = without_wikilinks
    for start, end in destinations:
        tag_scan = mask_region(tag_scan, start, end)
    note.tags.extend(_extract_tags(tag_scan, lines))
    return note


def parse_file(path: str | Path, vault_root: str | Path | None = None) -> ParsedNote:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    shown = file_path.relative_to(vault_root).as_posix() if vault_root else file_path.as_posix()
    return parse_note(text, path=shown)
