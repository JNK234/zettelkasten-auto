"""`vault move` and `vault merge`: a note changes place or name, or two notes become one, and every
link in the vault keeps reaching what it reached.

Links are rewritten the way the resolver (``vaultengine.resolver``, Obsidian's rules) resolves them:

* every link, embed, heading link (``[[n#H]]``) and block link (``[[n#^b]]``) that resolved to a
  moved note must resolve to it at its new path afterwards, and a subpath that was found must still
  be found; wikilinks and markdown links, in the body and in YAML frontmatter;
* a link is rewritten only when it no longer would: its display text (``|alias``), subpath and
  embed mark are kept, and its style too. A bare name stays a bare name when that name reaches only
  the note (Obsidian's "shortest path when possible"), else it becomes the vault path; a path stays a
  path (relative ones relative to the linking note); a markdown link keeps ``%20`` or ``<…>``;
* links to *other* notes that a new name would make ambiguous are rewritten to their path, so they
  keep their target too; links that matched no note and match the new name now reach it (reported);
* a link that reached a merged note through one of its aliases gets the alias as display text
  (``[[survivor|alias]]``), so what the reader sees does not change;
* every candidate is checked with the resolver against the vault as it will be. If no spelling
  reaches the right note, nothing is written.

``vault merge A B``: A survives and keeps its text; B's text (its frontmatter as a YAML code block,
engine block markers removed) is appended between ``<!-- vault:merged -->`` markers under
``## Merged from B`` for you to tidy; B is deleted and links to it reach A. B may also be a note that
does not exist (a name links point at): then those links are rewritten to A. When neither exists,
links to B are rewritten to A's name.

What the engine remembers follows: baselines (``.vaultengine/baseline-*.json``), page records
(``.vaultengine/pages.json``: a page's record, zettels and exclusions), and the finding blocks of
ideas (their note links are rewritten, and their id recomputed when the notes it is derived from moved).
The write itself goes through ``run_write``: journaled, atomic, refused if it would add a violation.
"""

from __future__ import annotations

import datetime as dt
import json
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from vaultengine.formats.edit import Lines
from vaultengine.formats.typed import TypedVault
from vaultengine.formats.write import (
    FileChange, NoteMissing, WriteError, WritePlan, _linkers, _name_keys, after_vault, find_note, read_text,
)
from vaultengine.model import Link, field_kinds
from vaultengine.parser import normalize_text, parse_note
from vaultengine.resolver import ResolvedLink, Resolver
from vaultengine.vault import is_markdown, link_name, lookup_key

_MARKER_LINE = re.compile(r"^[ \t]*<!-- /?vault:[\w-]+ -->[ \t]*$")
_FM_LINK = re.compile(r"\[\[([^\[\]\n]+)\]\]")
_MD_LINK = re.compile(r"^(!?\[.*\]\()(.*)\)$", re.DOTALL)
_ID_LINE = re.compile(r"^(- id: )([a-z]+-[0-9a-f]{10})[ \t]*$")
_METHOD_LINE = re.compile(r"^- method: ([a-z-]+)")
_NOTES_LINE = re.compile(r"^- notes: (.*)$")


class MoveError(WriteError):
    """The move or merge cannot be done as asked (or a link could not be kept). Nothing was written."""


@dataclass
class Expectation:
    """What a link must resolve to after the write."""

    target: str | None = None  # a path: must resolve to it
    subpath_found: bool = False  # its #heading / #^block was found before: must still be
    name: str | None = None  # a missing name: the link must be written with this target (it matches no note)
    alias: str | None = None  # reached through this alias of a merged note: keep it as display text


@dataclass
class SelfLink:
    """A link a merge would have turned into a link to the note it is in, and what was done with it."""

    path: str
    line: int
    link: str  # as it was written
    action: str  # "line removed" | "unlinked" | "same-note link" | "removed"
    result: str  # what replaces it ("" when removed)

    def __str__(self) -> str:
        shown = {"line removed": "line removed", "removed": "removed"}.get(self.action, self.result)
        return f"{self.path}:{self.line}  {self.link} -> {shown}"


@dataclass
class Report:
    rewritten: list[str] = field(default_factory=list)  # "path:line  old -> new"
    now_resolving: list[str] = field(default_factory=list)  # links that matched no note and reach the moved one now
    self_links: list[SelfLink] = field(default_factory=list)  # links that would point at the note they are in: dropped


# ----- spelling a link ---------------------------------------------------------------------------


def _wikilink(raw: str, target: str, display: str | None) -> str:
    embed = raw.startswith("!")
    inner = raw[3:-2] if embed else raw[2:-2]
    left, bar, right = inner.partition("|")
    _, hash_, sub = left.partition("#")
    if display is not None and not bar:
        bar, right = "|", display
    return ("!" if embed else "") + "[[" + target + hash_ + sub + bar + right + "]]"


def _markdown(raw: str, path: str) -> str:
    match = _MD_LINK.match(raw)
    head, inner = match[1], match[2]
    angle = inner.startswith("<")
    if angle:
        close = inner.index(">")
        dest, rest = inner[1:close], inner[close + 1:]
    else:
        dest, space, title = inner.partition(" ")
        rest = (space + title) if space else ""
    _, hash_, sub = dest.partition("#")
    encoded = "%" in dest
    if encoded or (" " in path and not angle):
        path = quote(path, safe="/-_.~()!$&'*+,;=:@")
    new = path + hash_ + sub
    return head + ("<" + new + ">" if angle else new) + rest + ")"


def _spellings(raw: str, syntax: str, written: str, source: str, target: str, before_vault) -> list[str]:
    """Ways to write a link to ``target`` from ``source``, in the link's own style, best first.
    ``written``: the link's target as parsed (markdown ones decoded, with their .md)."""
    md = written.lower().endswith(".md")
    stem = target[:-3] if is_markdown(target) else target
    full = target if md or not is_markdown(target) else stem
    folder = posixpath.dirname(source)
    relative = posixpath.relpath(full, folder or ".")
    if syntax == "markdown":
        if written.startswith("/"):
            dest = "/" + full
        elif written.startswith(("./", "../")) or not before_vault.files_at_path(posixpath.normpath(written)):
            dest = "./" + relative if written.startswith("./") and not relative.startswith("../") else relative
        else:
            dest = full
        return list(dict.fromkeys([_markdown(raw, dest), _markdown(raw, target)]))
    if written.startswith(("./", "../")):
        rel = relative if relative.startswith("../") else "./" + relative
        return list(dict.fromkeys([_wikilink(raw, rel, None), _wikilink(raw, full, None)]))
    if "/" in written:
        return [_wikilink(raw, ("/" if written.startswith("/") else "") + full, None)]
    name = (link_name(target) + (".md" if md else "")) if is_markdown(target) else posixpath.basename(target)
    return list(dict.fromkeys([_wikilink(raw, name, None), _wikilink(raw, full, None)]))


def _parse_link(raw: str) -> Link:
    note = parse_note(raw + "\n")
    links = sorted(note.links + note.embeds, key=lambda l: l.start)
    if not links:
        raise MoveError(f"{raw!r} does not parse as a link")
    return links[0]


def _meets(edge: ResolvedLink, want: Expectation) -> bool:
    if want.name is not None:
        return edge.status == "unresolved" and lookup_key(edge.link.target.strip()) == lookup_key(want.name)
    if want.target is None:
        return True
    if edge.status != "resolved" or edge.target_path != want.target:
        return False
    return not want.subpath_found or edge.subpath_status in (None, "found")


# ----- links a merge would point at the note they are in ----------------------------------------

_ONLY_LINK = re.compile(r"^[ \t]*(?:>[ \t]*)*(?:[-*+][ \t]+(?:\[[ xX]\][ \t]+)?|\d+[.)][ \t]+)?$")
_SEPARATOR = re.compile(r"(?:[ \t]*[,;][ \t]*|[ \t]+)$")


def _visible(link: Link) -> str:
    """What a reader sees of a link once it is no link: its display text, else its target as written."""
    if link.display is not None:
        return link.display
    return link.target.strip()


def _same_note(link: Link) -> str:
    """The link rewritten to point inside the note it is in (``[[#Heading]]``, ``[text](#heading)``)."""
    if link.syntax == "markdown":
        match = _MD_LINK.match(link.raw)
        inner = match[2]
        if inner.startswith("<"):
            dest, rest = inner[1:inner.index(">")], inner[inner.index(">") + 1:]
        else:
            dest, space, title = inner.partition(" ")
            rest = (space + title) if space else ""
        return match[1] + "#" + dest.partition("#")[2] + rest + ")"
    embed = link.raw.startswith("!")
    inner = link.raw[3:-2] if embed else link.raw[2:-2]
    left, bar, right = inner.partition("|")
    return ("!" if embed else "") + "[[#" + left.partition("#")[2] + bar + right + "]]"


def _drop_self_link(content: str, start: int, end: int, link: Link, field_kind: str | None) -> tuple[str, str, int, int]:
    """How to drop a link (at ``content[start:end]``) that would point at the note it is in:
    (action, replacement, start, end) — the span may grow to take a separator with it.

    * a line holding only the link (after a list, task or quote marker) goes;
    * a link with a heading or block becomes a same-note link (``[[#Heading]]``): the reader still jumps there;
    * in a field value (``Tags: …``, ``supports:: …``) the link goes with its separator; an inline field left
      empty goes as a line;
    * an embed goes (what it showed is in the note now); any other link becomes its visible text.
    """
    before, after = content[:start], content[end:]
    if link.heading_path or link.block_id is not None:
        return "same-note link", _same_note(link), start, end
    alone = _ONLY_LINK.match(before) and not after.strip()
    if field_kind == "inline" and not after.strip() and re.search(r"::[ \t]*$", before):
        alone = True
    if alone:
        return "line removed", "", 0, len(content)
    if field_kind is not None or link.is_embed:
        gap = _SEPARATOR.search(before)
        if gap and gap.start() > 0 and not before[:gap.start()].endswith(":"):
            return "removed", "", gap.start(), end
        trailing = re.match(r"[ \t]*[,;]?[ \t]*", after)
        return "removed", "", start, end + (trailing.end() if trailing else 0)
    return "unlinked", _visible(link), start, end


# ----- the relocation --------------------------------------------------------------------------


class Relocation:
    def __init__(self, typed: TypedVault, path_map: dict[str, str], base: dict[str, str | None],
                 name_map: dict[str, str] | None = None, merged_into: dict[str, str] | None = None) -> None:
        """``path_map``: notes that move (old -> new path; a merged note -> its survivor). ``base``: contents
        before links are rewritten ({path: text, None to delete}). ``name_map``: missing names merged
        (key -> survivor path, or survivor name when the survivor does not exist either). ``merged_into``:
        survivor -> the note merged into it (its links are appended after the survivor's own)."""
        self.typed = typed
        self.path_map = path_map
        self.base = base
        self.name_map = name_map or {}
        self.merged_into = merged_into or {}
        self.report = Report()
        self.before = typed.vault
        self.before_resolver = Resolver(self.before)
        self.after = after_vault(typed, [FileChange(p, None, t) for p, t in base.items()])
        self.resolver = Resolver(self.after.vault)
        self.root = Path(typed.vault.root) if typed.vault.root is not None else None

    def _text(self, path: str) -> str:
        if path in self.base and self.base[path] is not None:
            return self.base[path]
        if self.root is None:
            raise MoveError(f"{path}: the vault has no folder to read from")
        return read_text(self.root, path)

    def _expect(self, edge: ResolvedLink) -> Expectation:
        if edge.status == "resolved" and edge.target_path is not None:
            # A merged note's aliases go with it: a link that reached it through one keeps the alias as display text.
            merged_away = edge.target_path in self.merged_into.values()
            return Expectation(self.path_map.get(edge.target_path, edge.target_path), edge.subpath_status == "found",
                               alias=edge.link.target.strip() if edge.matched_by == "alias" and merged_away else None)
        if edge.status == "unresolved":
            key = lookup_key(edge.link.target.strip())
            if key in self.name_map:
                survivor = self.name_map[key]
                if survivor in self.after.vault.notes:
                    return Expectation(survivor)
                return Expectation(name=survivor)
        return Expectation()

    def _old_path(self, after_path: str) -> str:
        return next((o for o, n in self.path_map.items() if n == after_path and o not in self.merged_into.values()), after_path)

    def _before_edges(self, after_path: str) -> list[ResolvedLink]:
        old = next((o for o, n in self.path_map.items() if n == after_path and o not in self.merged_into.values()), after_path)
        edges = list(self.typed.graph.outgoing(old)) if old in self.before.notes else []
        if after_path in self.merged_into:
            edges += list(self.typed.graph.outgoing(self.merged_into[after_path]))
        return edges

    def notes_to_check(self) -> list[str]:
        keys: set[str] = set(self.name_map)
        for old, new in self.path_map.items():
            keys |= _name_keys(self.before, old) | _name_keys(self.after.vault, new)
        found: set[str] = set(self.path_map.values()) | set(self.merged_into)
        for old in self.path_map:
            found |= set(self.typed.graph.backlink_sources(old))
        found |= _linkers(self.before, keys) | _linkers(self.after.vault, keys)
        found |= self._frontmatter_linkers(keys)
        return sorted({self.path_map.get(p, p) for p in found if self.path_map.get(p, p) in self.after.vault.notes})

    def _frontmatter_linkers(self, keys: set[str]) -> set[str]:
        vault = self.before
        found = set()
        if hasattr(vault, "conn"):
            rows = vault.conn.execute("SELECT f.path, n.frontmatter_block FROM notes n JOIN files f ON f.id = n.file_id "
                                      "WHERE n.frontmatter_block LIKE '%[[%'")
            candidates = {row["path"]: row["frontmatter_block"] for row in rows if row["path"] not in vault.stale_paths}
            for path in vault._fresh:
                candidates[path] = None
        else:
            candidates = {p: None for p in vault.notes if vault.notes[p].has_frontmatter}
        for path, block in candidates.items():
            if block is None:
                if not vault.notes[path].has_frontmatter:
                    continue
                try:
                    block = read_text(self.root, path)
                except (OSError, WriteError, TypeError):
                    continue
            for target in _FM_LINK.findall(block):
                target = target.split("|")[0].split("#")[0].strip()
                if lookup_key(target) in keys or lookup_key(posixpath.basename(target)) in keys:
                    found.add(path)
                    break
        return found

    def fix(self, after_path: str, text: str) -> str:
        """``text`` (the note's content at ``after_path``) with its links rewritten to meet their expectations."""
        normalized = normalize_text(text)
        note = parse_note(normalized, path=after_path)
        links = sorted(note.links + note.embeds, key=lambda l: l.start)
        before = self._before_edges(after_path)
        if len(before) != len(links):
            raise MoveError(f"{after_path}: cannot match its {len(links)} links with the {len(before)} it had")
        line_starts = [0]
        for index, char in enumerate(normalized):
            if char == "\n":
                line_starts.append(index + 1)
        lines = Lines(text)
        split = normalized.split("\n")
        kinds = field_kinds(note)
        edits: dict[int, list[tuple[int, int, str]]] = {}
        for link, edge in zip(links, before):
            want = self._expect(edge)
            column = link.start - line_starts[link.line - 1]
            if self._becomes_self_link(after_path, edge, want, link):
                self._drop(after_path, link.line, split[link.line - 1], column, link, kinds.get(link.start), edits, lines)
                continue
            now = self.resolver.resolve(after_path, link)
            if want.target is None and want.name is None:
                if edge.status != "resolved" and now.status == "resolved" and now.target_path in self.path_map.values():
                    self.report.now_resolving.append(f"{after_path}:{link.line}  {link.raw} -> {now.target_path}")
                continue
            if _meets(now, want) and want.alias is None:
                continue
            new = self._respell(after_path, link, edge, want)
            edits.setdefault(link.line, []).append((column, column + len(link.raw), new))
            self.report.rewritten.append(f"{after_path}:{link.line}  {link.raw} -> {new}")
        edits.update({line: edits.get(line, []) + extra for line, extra in self._fix_frontmatter(after_path, normalized, lines).items()})
        for line, changes in edits.items():
            if line in lines.drop:
                continue
            content = lines.text(line)
            for start, end, new in sorted(changes, reverse=True):
                if content[start:end] != split[line - 1][start:end]:
                    raise MoveError(f"{after_path}:{line}: the link is not where it was parsed; nothing was written")
                content = content[:start] + new + content[end:]
            lines.replace[line] = content
        return lines.render()

    def _becomes_self_link(self, after_path: str, edge: ResolvedLink, want: Expectation, link: Link) -> bool:
        """A merge made this link point at the note it is in (it pointed elsewhere: at the other note, or
        from the other note's text at the survivor). Self-links a note already had are left alone."""
        if want.target != after_path or after_path not in self.merged_into or link.is_same_note:
            return False
        return not (edge.status == "resolved" and edge.target_path == edge.source)

    def _drop(self, after_path: str, number: int, content: str, column: int, link: Link, kind: str | None,
              edits: dict[int, list[tuple[int, int, str]]], lines: Lines) -> None:
        action, result, start, end = _drop_self_link(content, column, column + len(link.raw), link, kind)
        if action == "line removed":
            lines.drop.add(number)
        else:
            edits.setdefault(number, []).append((start, end, result))
        self.report.self_links.append(SelfLink(after_path, number, link.raw, action, result))

    def _respell(self, after_path: str, link: Link, edge: ResolvedLink, want: Expectation) -> str:
        if want.name is not None:
            return _wikilink(link.raw, want.name, None) if link.syntax == "wikilink" else _markdown(link.raw, want.name + ".md")
        for candidate in _spellings(link.raw, link.syntax, link.target.strip(), after_path, want.target, self.before):
            if want.alias is not None and link.syntax == "wikilink":
                candidate = _wikilink(candidate, _wikilink_target(candidate), want.alias)
            if _meets(self.resolver.resolve(after_path, _parse_link(candidate)), Expectation(want.target, want.subpath_found)):
                return candidate
        raise MoveError(f"{after_path}:{link.line}: no way to write {link.raw} so it still reaches {want.target}; "
                        "nothing was written")

    def _fix_frontmatter(self, after_path: str, normalized: str, lines: Lines) -> dict[int, list[tuple[int, int, str]]]:
        if not normalized.startswith("---"):
            return {}
        note = parse_note(normalized)
        if not note.has_frontmatter:
            return {}
        old_path = (after_path if after_path in self.merged_into  # a survivor's frontmatter is its own
                    else next((o for o, n in self.path_map.items() if n == after_path), after_path))
        edits: dict[int, list[tuple[int, int, str]]] = {}
        for number, line in enumerate(normalized.split("\n")[1:note.body_start_line - 2], start=2):
            for match in _FM_LINK.finditer(line):
                raw = match.group(0)
                link = _parse_link(raw)
                edge = self.before_resolver.resolve(old_path, link)
                want = self._expect(edge)
                if self._becomes_self_link(after_path, edge, want, link):
                    self._drop_in_yaml(after_path, number, line, match, link, edits, lines)
                    continue
                if want.target is None and want.name is None:
                    continue
                if _meets(self.resolver.resolve(after_path, link), want) and want.alias is None:
                    continue
                new = self._respell(after_path, link, edge, want)
                edits.setdefault(number, []).append((match.start(), match.end(), new))
                self.report.rewritten.append(f"{after_path}:{number}  {raw} -> {new} (frontmatter)")
        return edits


    def _drop_in_yaml(self, after_path: str, number: int, line: str, match: re.Match, link: Link,
                      edits: dict[int, list[tuple[int, int, str]]], lines: Lines) -> None:
        """In frontmatter: a list item holding only the link goes; elsewhere the link becomes its visible text."""
        item = re.match(r"^[ \t]*-[ \t]*([\"']?)(.*?)\1[ \t]*$", line)
        if item and item[2] == match.group(0):
            lines.drop.add(number)
            self.report.self_links.append(SelfLink(after_path, number, link.raw, "line removed", ""))
            return
        text = _visible(link)
        edits.setdefault(number, []).append((match.start(), match.end(), text))
        self.report.self_links.append(SelfLink(after_path, number, link.raw, "unlinked", text))


def _wikilink_target(raw: str) -> str:
    inner = raw[3:-2] if raw.startswith("!") else raw[2:-2]
    return inner.partition("|")[0].partition("#")[0]


# ----- what the engine remembers -----------------------------------------------------------------


def _follow_state(typed: TypedVault, path_map: dict[str, str],
                  rewritten: dict[str, tuple[str, str]] | None = None) -> list[FileChange]:
    """Baselines and page records, with the moved notes followed. ``rewritten``: notes whose only change is
    the engine's link rewriting ({new path: (text before, text after)}): a page record that held the text
    before holds the text after, so the rewrite does not look like your edit."""
    from vaultengine.formats.check import baseline_follow
    from vaultengine.formats.pages import sha256_text

    root = Path(typed.vault.root) if typed.vault.root is not None else None
    rewritten = rewritten or {}
    if root is None or not (path_map or rewritten):
        return []
    followed_hash = {sha256_text(old): sha256_text(new) for old, new in rewritten.values()}
    changes = []
    engine = root / ".vaultengine"
    for file in sorted(engine.glob("baseline-*.json")) if engine.exists() else []:
        old = file.read_text(encoding="utf-8")
        new = baseline_follow(old, path_map)
        if new is not None and new != old:
            changes.append(FileChange(file.relative_to(root).as_posix(), old, new, ["baseline: moved notes followed"]))
    pages = engine / "pages.json"
    if pages.exists():
        old = pages.read_text(encoding="utf-8")
        try:
            data = json.loads(old)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("pages"), dict):
            records = {}
            for page, record in data["pages"].items():
                zettels: dict[str, str] = {}
                for zettel, digest in record.get("zettels", {}).items():
                    new_path = path_map.get(zettel, zettel)
                    if new_path in rewritten and zettel == new_path or (zettel in path_map and path_map[zettel] in rewritten):
                        digest = followed_hash.get(digest, digest)
                    if new_path not in zettels or zettel == new_path:
                        zettels[new_path] = digest if zettel == new_path else zettels.get(new_path, digest)
                excluded = sorted({path_map.get(z, z) for z in record.get("excluded", [])} - set(zettels))
                records[path_map.get(page, page)] = {**record, "zettels": dict(sorted(zettels.items())), "excluded": excluded}
            data["pages"] = records
            new = json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
            if new != old:
                changes.append(FileChange(".vaultengine/pages.json", old, new, ["page records: moved notes followed"]))
    return changes


def _follow_finding_ids(typed: TypedVault, relocation: Relocation, path: str, text: str) -> str:
    """An idea's finding block with its id recomputed from its notes as they are after the write."""
    from vaultengine.findings import PREFIX, finding_id

    ideas = typed.spec.ideas
    if ideas is None or typed.spec.type_for(path) is None or typed.spec.type_for(path).name != ideas.type:
        return text
    lines = Lines(text)
    method = notes_line = id_line = None
    inside = False
    for number in range(1, lines.real + 1):
        content = lines.text(number)
        if content.strip() == f"<!-- vault:{ideas.block} -->":
            inside = True
        elif content.strip() == f"<!-- /vault:{ideas.block} -->":
            break
        elif inside:
            if _METHOD_LINE.match(content):
                method = _METHOD_LINE.match(content)[1]
            elif _NOTES_LINE.match(content):
                notes_line = content
            elif _ID_LINE.match(content):
                id_line = number
    if method not in PREFIX or notes_line is None or id_line is None:
        return text
    keys = []
    for raw in re.findall(r"!?\[\[[^\[\]]+\]\]", notes_line):
        edge = relocation.resolver.resolve(path, _parse_link(raw))
        keys.append(edge.target_path if edge.status == "resolved" else edge.link.target.strip())
    if len({lookup_key(k) for k in keys}) < len(keys):
        return text  # the finding's notes became one note: it was resolved; its id stays as it was
    match = _ID_LINE.match(lines.text(id_line))
    new_id = finding_id(method, keys)
    if new_id == match[2]:
        return text
    lines.replace[id_line] = match[1] + new_id
    relocation.report.rewritten.append(f"{path}:{id_line}  finding id {match[2]} -> {new_id}")
    return lines.render()


# ----- move and merge ------------------------------------------------------------------------------


def _destination(typed: TypedVault, source: str, dest: str) -> str:
    dest = dest.strip().replace("\\", "/")
    root = Path(typed.vault.root) if typed.vault.root is not None else None
    folder_given = dest.endswith("/") or (root is not None and (root / dest).is_dir()) or dest in ("", ".")
    if folder_given:
        folder = dest.rstrip("/")
        folder = "" if folder in ("", ".") else folder
        new = f"{folder}/{posixpath.basename(source)}" if folder else posixpath.basename(source)
    elif "/" in dest:
        new = dest if is_markdown(dest) else dest + ".md"
    else:
        folder = posixpath.dirname(source)
        name = dest if is_markdown(dest) else dest + ".md"
        new = f"{folder}/{name}" if folder else name
    if new.startswith("/") or ".." in new.split("/"):
        raise MoveError(f"{dest!r} is not a place in the vault for a note")
    new = posixpath.normpath(new)
    if new.startswith("../") or any(part.startswith(".") for part in new.split("/")):
        raise MoveError(f"{dest!r} is not a place in the vault for a note")
    return new


def _finish(typed: TypedVault, relocation: Relocation, operation: str, summary: list[str],
            deleted: list[str], created: dict[str, str], path_map: dict[str, str]) -> WritePlan:
    files: list[FileChange] = []
    rewritten: dict[str, tuple[str, str]] = {}  # notes the engine only re-spelled links in
    for path in relocation.notes_to_check():
        text = relocation._text(path)
        new = _follow_finding_ids(typed, relocation, path, relocation.fix(path, text))
        if path in created:
            files.append(FileChange(path, None, new, [f"new path of {created[path]}"]))
            rewritten[path] = (text, new)
        else:
            old = read_text(relocation.root, path)
            if new != old:
                what = "links rewritten" if path not in relocation.merged_into else "merged"
                files.append(FileChange(path, old, new, [what]))
                if path not in relocation.merged_into:
                    rewritten[path] = (old, new)
    for path in deleted:
        files.append(FileChange(path, read_text(relocation.root, path), None, ["deleted"]))
    notes = summary + ([f"{len(relocation.report.now_resolving)} links that matched no note now reach it"]
                       if relocation.report.now_resolving else [])
    plan = WritePlan(operation, files, notes, _follow_state(typed, path_map, rewritten), dict(path_map))
    plan.report = relocation.report  # type: ignore[attr-defined]
    return plan


def plan_move(typed: TypedVault, note_ref: str, dest: str) -> WritePlan:
    source = find_note(typed, note_ref)
    new = _destination(typed, source, dest)
    if lookup_key(new) == lookup_key(source):
        raise MoveError(f"{source} is already there")
    if typed.vault.files_at_path(new):
        raise MoveError(f"{new} already exists")
    text = read_text(Path(typed.vault.root), source)
    relocation = Relocation(typed, {source: new}, {source: None, new: text})
    return _finish(typed, relocation, "move", [f"{source} -> {new}"], [source], {new: source}, {source: new})


def plan_merge(typed: TypedVault, survivor_ref: str, other_ref: str, now: dt.datetime | None = None) -> WritePlan:
    now = now or dt.datetime.now()
    survivor = _find_or_name(typed, survivor_ref)
    other = _find_or_name(typed, other_ref)
    if lookup_key(survivor) == lookup_key(other):
        raise MoveError("a note cannot be merged into itself")
    survivor_exists, other_exists = survivor in typed.vault.notes, other in typed.vault.notes
    if other_exists and not survivor_exists:
        raise MoveError(f"{survivor!r} does not exist and {other} does: to give {other} that name, "
                        f"use `vault move … {other_ref!r} {survivor_ref!r}`")
    if not other_exists:
        key = lookup_key(other)
        if not _linkers(typed.vault, {key}):
            raise MoveError(f"no note and no link named {other!r}")
        target = survivor if survivor_exists else survivor
        relocation = Relocation(typed, {}, {}, {key: target})
        what = f"links to [[{other}]] (no note) -> " + (survivor if survivor_exists else f"[[{survivor}]] (no note either)")
        return _finish(typed, relocation, "merge", [what], [], {}, {})
    root = Path(typed.vault.root)
    survivor_text, other_text = read_text(root, survivor), read_text(root, other)
    merged = survivor_text + _merged_section(survivor_text, other_text, link_name(other))
    relocation = Relocation(typed, {other: survivor}, {other: None, survivor: merged}, merged_into={survivor: other})
    return _finish(typed, relocation, "merge", [f"{other} merged into {survivor}"], [other], {}, {other: survivor})


def _find_or_name(typed: TypedVault, ref: str) -> str:
    """A note's path, or the name as given when no note has it (a name links point at)."""
    try:
        return find_note(typed, ref)
    except NoteMissing as exc:
        if "matches" in str(exc):
            raise
        name = ref.strip()
        return name[:-3] if is_markdown(name) and "/" not in name else name


def _merged_section(survivor_text: str, other_text: str, other_name: str) -> str:
    newline = "\r\n" if "\r\n" in survivor_text else "\n"
    normalized = normalize_text(other_text)
    note = parse_note(normalized)
    body_lines = normalized.split("\n")
    frontmatter: list[str] = []
    if note.has_frontmatter:
        frontmatter = ["```yaml", *body_lines[1:note.body_start_line - 2], "```", ""]
        body_lines = body_lines[note.body_start_line - 1:]
    body = [line for line in body_lines if not _MARKER_LINE.match(line)]
    while body and not body[-1].strip():
        body.pop()
    lead = "" if survivor_text.endswith(("\n", "\r")) or not survivor_text else newline
    section = ["", "<!-- vault:merged -->", f"## Merged from {other_name}", "", *frontmatter, *body, "<!-- /vault:merged -->"]
    return lead + newline.join(section) + newline
