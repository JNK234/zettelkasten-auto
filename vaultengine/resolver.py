"""Resolve parsed links to vault files.

Rules (as specified):
1. A target containing '/' is an explicit path and only matches that path.
2. Otherwise the target matches files by basename, case-insensitively.
3. Frontmatter aliases count as names (can be switched off, see below).
4. More than one distinct matching file -> "ambiguous"; nothing is picked.
5. No matching file -> "unresolved"; the link is kept.
6. #Heading and #^block subpaths are checked against the target note.
7. Non-markdown files (images, PDFs, ...) resolve like any other file.

Assumptions where the rules are silent (also listed in the README of the change):
* Paths: a leading '/' means vault root; './' and '../' are relative to the
  linking note's folder; any other path is tried from the vault root first,
  then relative to the linking note's folder. A path that matches nothing is
  unresolved and does NOT fall back to a basename match.
* Path and basename matching are case-insensitive and NFC-normalised; the
  '.md' extension may be written or omitted.
* A basename match and an alias match on two different files is ambiguous
  (both are "matches"). Both on the same file is a normal resolution.
* Headings are compared after removing # ^ | : % [ ] \\ , collapsing spaces
  and ignoring case. Obsidian's API says it strips "special characters and
  shrinks consecutive spaces" but does not list which characters.
* A nested heading link [[Note#A#B]] needs a heading B somewhere inside the
  section of a heading A (at any deeper level).
* Block IDs are compared case-insensitively.
* Subpaths on non-markdown targets (e.g. file.pdf#page=3) are not checked.

Note: in Obsidian itself, a plain [[alias]] link is, as far as I know, not
resolved through aliases (aliases feed autocomplete, which writes
[[Real Name|alias]]). ``resolve_aliases=False`` gives that behaviour.
"""

from __future__ import annotations

import posixpath
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, Sequence

from vaultengine.model import Heading, Link
from vaultengine.vault import Vault, is_markdown

Status = Literal["resolved", "ambiguous", "unresolved"]
MatchedBy = Literal["self", "path", "name", "alias"]
SubpathStatus = Literal["found", "missing", "not_checked"]

# Bump whenever resolution can change for the same inputs; the SQLite index
# re-resolves every link when the stored version differs.
RESOLVER_VERSION = 1

_HEADING_SPECIALS = re.compile(r"[#^|:%\[\]\\]")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class ResolvedLink:
    source: str
    link: Link
    status: Status
    target_path: str | None = None
    candidates: tuple[str, ...] = ()
    matched_by: MatchedBy | None = None
    subpath_status: SubpathStatus | None = None  # None when the link has no subpath

    @property
    def has_subpath(self) -> bool:
        return bool(self.link.heading_path) or self.link.block_id is not None


def normalize_heading(text: str) -> str:
    text = _HEADING_SPECIALS.sub(" ", unicodedata.normalize("NFC", text))
    return _SPACES.sub(" ", text).strip().lower()


def heading_path_exists(headings: Sequence[Heading], path: Sequence[str]) -> bool:
    normalized = [normalize_heading(heading.text) for heading in headings]

    def search(start: int, end: int, remaining: Sequence[str]) -> bool:
        if not remaining:
            return True
        wanted = normalize_heading(remaining[0])
        for index in range(start, end):
            if normalized[index] != wanted:
                continue
            section_end = index + 1
            while section_end < end and headings[section_end].level > headings[index].level:
                section_end += 1
            if search(index + 1, section_end, remaining[1:]):
                return True
        return False

    return bool(path) and search(0, len(headings), list(path))


def _path_candidates(vault: Vault, source: str, target: str) -> list[str]:
    folder = posixpath.dirname(source)
    if target.startswith("/"):
        attempts = [target.lstrip("/")]
    elif target.startswith(("./", "../")):
        attempts = [posixpath.normpath(posixpath.join(folder, target))]
    else:
        attempts = [posixpath.normpath(target), posixpath.normpath(posixpath.join(folder, target))]
    for attempt in attempts:
        if attempt.startswith("../") or attempt == "..":
            continue  # escapes the vault
        found = vault.files_at_path(attempt)
        if found:
            return found
    return []


def _unique(paths: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(paths))


class Resolver:
    def __init__(self, vault: Vault, resolve_aliases: bool = True) -> None:
        self.vault = vault
        self.resolve_aliases = resolve_aliases

    def resolve(self, source: str, link: Link) -> ResolvedLink:
        if link.is_same_note:
            return self._with_subpath(ResolvedLink(source, link, "resolved", source, (source,), "self"))

        target = link.target.strip()
        if "/" in target:
            candidates, matched_by = _unique(_path_candidates(self.vault, source, target)), "path"
        else:
            by_name = self.vault.files_named(target)
            by_alias = self.vault.notes_with_alias(target) if self.resolve_aliases else []
            candidates = _unique(by_name + by_alias)
            matched_by = "name" if len(candidates) == 1 and candidates[0] in by_name else "alias"

        if not candidates:
            result = ResolvedLink(source, link, "unresolved")
        elif len(candidates) > 1:
            result = ResolvedLink(source, link, "ambiguous", None, tuple(sorted(candidates)))
        else:
            result = ResolvedLink(source, link, "resolved", candidates[0], tuple(candidates), matched_by)
        return self._with_subpath(result)

    def _with_subpath(self, result: ResolvedLink) -> ResolvedLink:
        if not result.has_subpath:
            return result
        status: SubpathStatus = "not_checked"
        target = result.target_path
        if result.status == "resolved" and target is not None and is_markdown(target):
            note = self.vault.notes[target]
            link = result.link
            if link.block_id is not None:
                wanted = link.block_id.lower()
                found = any(block.id.lower() == wanted for block in note.block_ids)
            else:
                found = heading_path_exists(note.headings, link.heading_path)
            status = "found" if found else "missing"
        return ResolvedLink(
            result.source, result.link, result.status, result.target_path,
            result.candidates, result.matched_by, status,
        )
