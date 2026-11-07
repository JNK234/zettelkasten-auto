"""Apply a format (see ``spec.py``) to a loaded vault.

``TypedVault`` answers, per note:
* its type: the first type in the format whose ``match`` fits the path;
* its metadata: declared keys read from YAML frontmatter, the ``Key: value``
  header block at the top of the note, or either (frontmatter wins);
* its relations: every link inside a ``name:: [[a]], [[b]]`` field whose name is a
  declared relation, plus edges implied by mirror pairs: if A ``supports`` B, B is
  ``supported-by`` A whether or not B says so (``origin == "mirror"``);
* a role for every link: the relation name for links in a relation field, the
  key's ``link_role`` for links in a metadata value, otherwise the role of the
  innermost enclosing section that has one (type roles before global roles);
* issues: where the note departs from its type.

Issue codes:
  missing-heading, heading-order         the type's headings
  missing-key, missing-value, bad-value  the type's metadata
  missing-relation, unexpected-relation  the type's relations block
  unknown-field                          a ``key::`` field that is not a declared relation
  bad-relation                           a relation value with text that is not a link
  unresolved-target, ambiguous-target    a relation link that matches no note / several files
  relation-target-type                   a relation pointing at a note of the wrong type
  missing-mirror                         A supports B but B lacks ``supported-by:: [[A]]``
                                         (reported on B, the note to edit; ``related`` is A)

Every issue has a line. For something missing, it is the line where it belongs:
the heading a missing heading should precede (or the last line), the relation
field a missing mirror should join, the header block for a missing key.
Frontmatter values are reported at line 1.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from vaultengine.formats.spec import FormatSpec, MetadataKey, NoteType, relation_key
from vaultengine.graph import LinkGraph
from vaultengine.model import HeaderField, Heading, Link, ParsedNote
from vaultengine.resolver import ResolvedLink
from vaultengine.vault import Vault, link_name, lookup_key

ItemKind = Literal["tag", "link", "text"]
_FRONTMATTER_LINK = re.compile(r"\[\[([^\[\]|#^]*)(?:[#^|][^\[\]]*)?\]\]")
_TOKEN = re.compile(r"[^\s,]+")
_INTEGER = re.compile(r"[+-]?\d+")


@dataclass(frozen=True)
class Issue:
    path: str
    code: str
    message: str
    line: int
    related: str | None = None  # another note involved (for missing-mirror: the note whose field implies it)
    detail: dict[str, Any] | None = field(default=None, compare=False)  # machine-readable facts, used by `vault fix`

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "line": self.line, "code": self.code, "message": self.message, "related": self.related}


@dataclass(frozen=True)
class HeadingGap:
    """Required headings missing at one place: before ``before`` (a present heading), or at the end."""

    names: tuple[str, ...]
    levels: tuple[int, ...]  # level to write each one at
    before: Heading | None
    fixable: bool  # False when a name is a /regex/ (no literal text to write)


@dataclass(frozen=True)
class RelationAnchor:
    """Where a relation field for a note goes.

    ``append``: add to the end of the existing field on ``line``; ``after``: a new
    field line after ``line`` (the last relation field, or the relations heading).
    """

    kind: Literal["append", "after"]
    line: int
    empty_value: bool = False  # for append: the field has no value yet


@dataclass(frozen=True)
class MetaItem:
    kind: ItemKind
    value: str  # tag name without '#', link target as written, or text


@dataclass(frozen=True)
class MetadataValue:
    key: str  # as written in the note
    source: Literal["inline", "frontmatter"]
    raw: Any  # the text after "Key:", or the YAML value
    line: int | None  # None for frontmatter
    items: tuple[MetaItem, ...]

    @property
    def values(self) -> list[str]:
        return [item.value for item in self.items]

    @property
    def is_empty(self) -> bool:
        return not self.items


@dataclass(frozen=True)
class RelationEdge:
    source: str  # the note the relation belongs to
    relation: str
    target: str  # as written ([[target]]), or the linking note's name for mirrors
    target_path: str | None
    status: str  # resolved | ambiguous | unresolved
    line: int | None  # None for mirror edges
    origin: Literal["declared", "mirror"] = "declared"
    mirror_of: str | None = None  # for mirrors: the note whose field implies this edge

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoledLink:
    edge: ResolvedLink
    role: str | None
    section: str | None  # innermost heading above the link, None above the first heading
    via: Literal["field", "metadata", "section"] | None

    def to_dict(self) -> dict[str, Any]:
        link = self.edge.link
        return {
            "line": link.line, "raw": link.raw, "target": link.target, "target_path": self.edge.target_path,
            "status": self.edge.status, "role": self.role, "section": self.section, "via": self.via,
        }


def _section_chain(headings: list[Heading], line: int) -> list[Heading]:
    """Headings enclosing ``line``, innermost first. A link on a heading line belongs to that heading."""
    stack: list[Heading] = []
    for heading in headings:
        if heading.line > line:
            break
        while stack and stack[-1].level >= heading.level:
            stack.pop()
        stack.append(heading)
    return stack[::-1]


class TypedVault:
    def __init__(self, spec: FormatSpec, vault: Vault, graph: LinkGraph | None = None) -> None:
        self.spec = spec
        self.vault = vault
        self.graph = graph or LinkGraph(vault)
        self._edges: dict[tuple[str, int], ResolvedLink] = {
            (edge.source, edge.link.start): edge for edge in self.graph.edges
        }
        self._types: dict[str, NoteType | None] = {path: spec.type_for(path) for path in vault.notes}
        self._heading_levels = self._usual_heading_levels()
        self._issues: dict[str, list[Issue]] = defaultdict(list)
        self._declared: dict[str, list[RelationEdge]] = defaultdict(list)
        self._mirrors: dict[str, list[RelationEdge]] = defaultdict(list)
        self._metadata: dict[str, dict[str, MetadataValue]] = {}
        for path in sorted(vault.notes):
            self._read_note(path)
        self._link_mirrors()

    # ----- lookups ---------------------------------------------------------------

    def type_of(self, path: str) -> str | None:
        note_type = self._types.get(path)
        return note_type.name if note_type else None

    def notes_of_type(self, name: str) -> list[str]:
        return sorted(path for path, note_type in self._types.items() if note_type and note_type.name == name)

    @property
    def untyped(self) -> list[str]:
        return sorted(path for path, note_type in self._types.items() if note_type is None)

    def metadata(self, path: str) -> dict[str, MetadataValue]:
        """Declared metadata keys found in the note, by the key name used in the format."""
        return dict(self._metadata.get(path, {}))

    def relations(self, path: str, mirrors: bool = True) -> list[RelationEdge]:
        """Relations of ``path``: the ones it declares, then the ones implied by mirror pairs."""
        return list(self._declared.get(path, [])) + (list(self._mirrors.get(path, [])) if mirrors else [])

    def issues(self, path: str | None = None) -> list[Issue]:
        """Issues of one note, or of all notes, ordered by path and line."""
        paths = [path] if path is not None else sorted(self._issues)
        return [issue for key in paths for issue in sorted(self._issues.get(key, []), key=lambda i: i.line)]

    def links(self, path: str) -> list[RoledLink]:
        """Every link and embed of the note, in order, with its role."""
        note = self.vault.notes[path]
        note_type = self._types.get(path)
        spans: list[tuple[int, int, str | None, str]] = []
        for field in note.inline_fields:
            relation = self.spec.relation(field.key)
            spans.append((field.value_start, field.value_end, relation.name if relation else relation_key(field.key), "field"))
        if note_type is not None and note_type.metadata_source != "frontmatter":
            declared = {lookup_key(key.name): key for key in note_type.metadata}
            for header in note.header_fields:
                key = declared.get(lookup_key(header.key))
                spans.append((header.value_start, header.value_end, key.link_role if key else None, "metadata"))
        roles = (note_type.link_roles if note_type else ()) + self.spec.link_roles

        result = []
        for link in sorted(note.links + note.embeds, key=lambda item: item.start):
            edge = self._edges[(path, link.start)]
            chain = _section_chain(note.headings, link.line)
            section = chain[0].text if chain else None
            span = next((s for s in spans if s[0] <= link.start and link.end <= s[1]), None)
            if span is not None:
                result.append(RoledLink(edge, span[2], section, span[3]))
                continue
            role = next((name for heading in chain for name, pattern in roles if pattern.matches(heading.text)), None)
            result.append(RoledLink(edge, role, section, "section" if role else None))
        return result

    # ----- reading one note ------------------------------------------------------------

    def _issue(self, path: str, code: str, message: str, line: int, related: str | None = None,
               detail: dict[str, Any] | None = None) -> None:
        self._issues[path].append(Issue(path, code, message, line, related, detail))

    # ----- placement (shared by the checks and `vault fix`) ----------------------------------

    def _usual_heading_levels(self) -> dict[tuple[str, int], int]:
        """(type, index of required heading) -> the level that heading has in most notes of the type."""
        counts: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
        for path, note_type in self._types.items():
            if note_type is None:
                continue
            for index, pattern in enumerate(note_type.headings):
                heading = next((h for h in self.vault.notes[path].headings if pattern.matches(h.text)), None)
                if heading is not None:
                    counts[(note_type.name, index)][heading.level] += 1
        return {key: counter.most_common(1)[0][0] for key, counter in counts.items()}

    def usual_heading_level(self, type_name: str, index: int, default: int = 2) -> int:
        """The level the type's index-th required heading has in most notes of the type."""
        return self._heading_levels.get((type_name, index), default)

    def heading_gaps(self, path: str) -> list[HeadingGap]:
        """The type's missing headings, grouped by where they belong.

        A missing heading goes before the next required heading that is present;
        if none follows, at the end of the note. Its level is the one the heading
        has in most notes of the type, else the level of its neighbours, else 2.
        """
        note_type = self._types.get(path)
        if note_type is None:
            return []
        note = self.vault.notes[path]
        present = [next((h for h in note.headings if p.matches(h.text)), None) for p in note_type.headings]
        groups: dict[int | None, list[int]] = {}
        for index, heading in enumerate(present):
            if heading is None:
                following = next((present[j] for j in range(index + 1, len(present)) if present[j] is not None), None)
                groups.setdefault(following.line if following else None, []).append(index)
        gaps = []
        for before_line, indexes in groups.items():
            before = next((h for h in note.headings if h.line == before_line), None)
            levels = []
            for index in indexes:
                previous = next((present[j] for j in range(index - 1, -1, -1) if present[j] is not None), None)
                neighbour = previous or before
                levels.append(self._heading_levels.get((note_type.name, index), neighbour.level if neighbour else 2))
            patterns = [note_type.headings[index] for index in indexes]
            gaps.append(HeadingGap(
                tuple(p.pattern for p in patterns), tuple(levels), before,
                fixable=all(not p.is_regex for p in patterns),
            ))
        return gaps

    def relation_anchor(self, path: str, relation: str) -> RelationAnchor | None:
        """Where a ``relation::`` field goes in this note, or None if there is no obvious place.

        The existing field of that relation (its last one); else a new line after the
        note's last relation field; else right after the type's ``relations_heading``.
        """
        note = self.vault.notes[path]
        fields = [f for f in note.inline_fields if self.spec.relation(f.key) is not None]
        same = [f for f in fields if self.spec.relation(f.key).name == relation]
        if same:
            return RelationAnchor("append", same[-1].line, empty_value=not same[-1].value)
        if fields:
            return RelationAnchor("after", fields[-1].line)
        note_type = self._types.get(path)
        if note_type is not None and note_type.relations_heading is not None:
            heading = next((h for h in note.headings if note_type.relations_heading.matches(h.text)), None)
            if heading is not None:
                return RelationAnchor("after", heading.line)
        return None

    def _header_line(self, note: ParsedNote) -> int:
        return note.header_fields[0].line if note.header_fields else note.body_start_line

    def _read_note(self, path: str) -> None:
        note = self.vault.notes[path]
        note_type = self._types[path]
        self._read_relations(path, note, note_type)
        if note_type is None:
            return
        self._check_headings(path, note, note_type)
        self._metadata[path] = {}
        for key in note_type.metadata:
            value = self._find_metadata(note, key, note_type.metadata_source)
            if value is not None:
                self._metadata[path][key.name] = value
            self._check_metadata(path, key, value)
        declared = {edge.relation for edge in self._declared.get(path, [])}
        for name in note_type.relations_required:
            if name not in declared:
                anchor = self.relation_anchor(path, name)
                self._issue(
                    path, "missing-relation", f"missing relation field `{name}::` (required for {note_type.name})",
                    anchor.line if anchor else self._header_line(note),
                )

    def _check_headings(self, path: str, note: ParsedNote, note_type: NoteType) -> None:
        found: list[tuple[str, Heading]] = []
        for pattern in note_type.headings:
            heading = next((h for h in note.headings if pattern.matches(h.text)), None)
            if heading is not None:
                found.append((pattern.pattern, heading))
        for gap in self.heading_gaps(path):
            label = "heading" if len(gap.names) == 1 else "headings"
            names = ", ".join(repr(name) for name in gap.names)
            where = f"before {gap.before.text!r}" if gap.before else "at the end of the note"
            self._issue(
                path, "missing-heading", f"missing {label} {names} (expected {where})",
                gap.before.line if gap.before else max(note.last_line, 1),
                detail={"headings": list(gap.names), "levels": list(gap.levels),
                        "before_line": gap.before.line if gap.before else None, "fixable": gap.fixable},
            )
        if note_type.ordered_headings:
            for (name_a, a), (name_b, b) in zip(found, found[1:]):
                if b.line < a.line:
                    expected = ", ".join(p.pattern for p in note_type.headings)
                    self._issue(
                        path, "heading-order",
                        f"heading {name_b!r} (line {b.line}) comes before {name_a!r} (line {a.line}); expected order: {expected}",
                        b.line,
                    )
                    break

    def _read_relations(self, path: str, note: ParsedNote, note_type: NoteType | None) -> None:
        for field in note.inline_fields:
            relation = self.spec.relation(field.key)
            if relation is None:
                declared = ", ".join(sorted(self.spec.relations)) or "none"
                self._issue(path, "unknown-field", f"`{field.key}::` is not a declared relation (declared: {declared})", field.line)
                continue
            if note_type is not None and not note_type.allows_relation(relation.name):
                allowed = ", ".join(note_type.relations_required + note_type.relations_optional) or "none"
                self._issue(
                    path, "unexpected-relation",
                    f"`{field.key}::` is not a relation of {note_type.name} (allowed: {allowed})", field.line,
                )
            leftover = _outside_links(field.value, field.value_start, field.links)
            if leftover:
                self._issue(path, "bad-relation", f"`{field.key}::` has text that is not a link: {' '.join(leftover)!r}", field.line)
            for link in field.links:
                edge = self._edges[(path, link.start)]
                self._declared[path].append(
                    RelationEdge(path, relation.name, link.target, edge.target_path, edge.status, field.line)
                )
                if edge.status == "unresolved":
                    self._issue(path, "unresolved-target", f"`{field.key}::` {link.raw} matches no note", field.line,
                                detail={"target": link.target})
                elif edge.status == "ambiguous":
                    self._issue(
                        path, "ambiguous-target",
                        f"`{field.key}::` {link.raw} matches {len(edge.candidates)} files: {', '.join(edge.candidates)}",
                        field.line, detail={"target": link.target},
                    )

    # ----- metadata -------------------------------------------------------------------

    def _find_metadata(self, note: ParsedNote, key: MetadataKey, source: str) -> MetadataValue | None:
        if source in ("frontmatter", "either"):
            for name, raw in note.frontmatter.items():
                if lookup_key(name) == lookup_key(key.name):
                    return MetadataValue(name, "frontmatter", raw, None, _frontmatter_items(raw, key))
        if source in ("inline", "either"):
            for header in note.header_fields:
                if lookup_key(header.key) == lookup_key(key.name):
                    return MetadataValue(header.key, "inline", header.value, header.line, _inline_items(header, key))
        return None

    def _check_metadata(self, path: str, key: MetadataKey, value: MetadataValue | None) -> None:
        if value is None:
            if key.required:
                self._issue(path, "missing-key", f"missing metadata {key.name!r}", self._header_line(self.vault.notes[path]))
            return
        line = value.line or 1  # frontmatter starts on line 1
        if value.is_empty:
            if key.required and not key.allow_empty:
                self._issue(path, "missing-value", f"{key.name!r} is empty", line)
            return
        kind = key.kinds[0]
        if kind in ("datetime", "date"):
            if not _valid_date(value.raw, kind, key.date_format or ""):
                self._issue(path, "bad-value", f"{key.name}: {_shown(value.raw)} is not a {kind} like {key.date_format!r}", line)
            return
        if kind == "integer":
            raw = value.raw
            ok = (isinstance(raw, int) and not isinstance(raw, bool)) or (isinstance(raw, str) and _INTEGER.fullmatch(raw.strip()))
            if not ok:
                self._issue(path, "bad-value", f"{key.name}: {_shown(raw)} is not a whole number", line)
            return
        problems = []
        for item in value.items:
            problem = _item_problem(item, key)
            if problem:
                problems.append(problem)
        if len(value.items) > 1 and not key.multiple:
            problems.append(f"expected one value, got {len(value.items)}")
        for problem in problems:
            self._issue(path, "bad-value", f"{key.name}: {problem}", line)

    # ----- mirrors ---------------------------------------------------------------------

    def _link_mirrors(self) -> None:
        declared = {
            (edge.source, edge.relation, edge.target_path)
            for edges in self._declared.values() for edge in edges if edge.target_path
        }
        implied: set[tuple[str, str, str]] = set()
        for source in sorted(self._declared):
            for edge in self._declared[source]:
                relation = self.spec.relations[edge.relation]
                target = edge.target_path
                if edge.status != "resolved" or target is None or target == source:
                    continue
                if relation.targets and target in self.vault.notes:
                    target_type = self.type_of(target)
                    if target_type not in relation.targets:
                        self._issue(
                            source, "relation-target-type",
                            f"`{edge.relation}::` [[{edge.target}]] is a {target_type or 'note of no type'}; "
                            f"expected {' or '.join(relation.targets)}",
                            edge.line,
                        )
                        continue  # the link is wrong; asking the target to mirror it would be noise
                if relation.inverse is None or target not in self.vault.notes:
                    continue
                mirror = (target, relation.inverse, source)
                if mirror in declared or mirror in implied:
                    continue
                implied.add(mirror)
                name = self.link_text(source)
                self._mirrors[target].append(
                    RelationEdge(target, relation.inverse, name, source, "resolved", None, "mirror", source)
                )
                anchor = self.relation_anchor(target, relation.inverse)
                self._issue(
                    target, "missing-mirror",
                    f"missing `{relation.inverse}:: [[{name}]]` (mirror of `{edge.relation}::` in {source}:{edge.line})",
                    anchor.line if anchor else self._header_line(self.vault.notes[target]),
                    related=source,
                    detail={"relation": relation.inverse, "link": name, "source": source, "source_line": edge.line},
                )

    def link_text(self, path: str) -> str:
        """What to write inside [[...]] to reach ``path``: its name, or its path if the name is taken."""
        name = link_name(path)
        return name if len(self.vault.files_named(name)) == 1 else path[:-3] if path.lower().endswith(".md") else path

    # ----- reports ---------------------------------------------------------------------

    def note_report(self, path: str) -> dict[str, Any]:
        metadata = [
            {"key": name, "written_as": value.key, "source": value.source, "line": value.line,
             "raw": value.raw, "values": value.values}
            for name, value in self._metadata.get(path, {}).items()
        ]
        return {
            "path": path,
            "type": self.type_of(path),
            "metadata": metadata,
            "relations": [edge.to_dict() for edge in self.relations(path)],
            "links": [link.to_dict() for link in self.links(path)],
            "issues": [issue.to_dict() for issue in self.issues(path)],
        }

    def report(self, examples: int = 3, only_type: str | None = None) -> dict[str, Any]:
        """Summary of the whole vault; with ``only_type``, types and issues of that type only."""
        issues = self.issues()
        if only_type is not None:
            members = set(self.notes_of_type(only_type))
            issues = [issue for issue in issues if issue.path in members]
        with_issues = {issue.path for issue in self.issues()}
        types = []
        for note_type in self.spec.types:
            if only_type is not None and note_type.name != only_type:
                continue
            paths = self.notes_of_type(note_type.name)
            clean = [path for path in paths if path not in with_issues]
            types.append({
                "name": note_type.name,
                "description": note_type.description,
                "match": " or ".join(m.describe() for m in note_type.match),
                "notes": len(paths),
                "clean": len(clean),
                "issues": sum(len(self._issues.get(path, [])) for path in paths),
                "examples": (clean + [p for p in paths if p not in clean])[:examples],
            })

        by_code: dict[str, list[Issue]] = defaultdict(list)
        for issue in issues:
            by_code[issue.code].append(issue)

        relation_rows = []
        declared_all = [edge for edges in self._declared.values() for edge in edges]
        mirrors_all = [edge for edges in self._mirrors.values() for edge in edges]
        for name, relation in sorted(self.spec.relations.items()):
            mine = [edge for edge in declared_all if edge.relation == name]
            status = Counter(edge.status for edge in mine)
            relation_rows.append({
                "name": name,
                "inverse": relation.inverse,
                "targets": list(relation.targets),
                "declared": len(mine),
                "resolved": status["resolved"],
                "unresolved": status["unresolved"],
                "ambiguous": status["ambiguous"],
                "implied_by_mirror": sum(edge.relation == name for edge in mirrors_all),
            })

        roles: Counter[str | None] = Counter()
        for path in self.vault.notes:
            roles.update(link.role for link in self.links(path))
        return {
            "format": {"name": self.spec.name, "path": self.spec.path},
            "only_type": only_type,
            "notes": len(self.vault.notes),
            "typed": len(self.vault.notes) - len(self.untyped),
            "untyped": self.untyped,
            "types": types,
            "issues": {
                "total": len(issues),
                "notes": len({issue.path for issue in issues}),
                "by_code": [
                    {"code": code, "count": len(items), "notes": len({i.path for i in items})}
                    for code, items in sorted(by_code.items(), key=lambda kv: (-len(kv[1]), kv[0]))
                ],
                "items": [issue.to_dict() for issue in issues],
            },
            "relations": relation_rows,
            "link_roles": [
                {"role": role, "links": count}
                for role, count in sorted(roles.items(), key=lambda kv: (-kv[1], kv[0] or ""))
                if role is not None
            ],
            "links_without_role": roles[None],
        }


# ----- value helpers -----------------------------------------------------------------


def _outside_links(value: str, value_start: int, links: tuple[Link, ...]) -> list[str]:
    chars = list(value)
    for link in links:
        for index in range(max(link.start - value_start, 0), min(link.end - value_start, len(chars))):
            chars[index] = " "
    return _TOKEN.findall("".join(chars))


def _inline_items(header: HeaderField, key: MetadataKey) -> tuple[MetaItem, ...]:
    if not header.value:
        return ()
    if key.kinds[0] in ("text", "datetime", "date", "integer"):
        return (MetaItem("text", header.value),)
    items: list[tuple[int, MetaItem]] = [
        (link.start - header.value_start, MetaItem("link", link.target)) for link in header.links
    ]
    chars = list(header.value)
    for link in header.links:
        for index in range(max(link.start - header.value_start, 0), min(link.end - header.value_start, len(chars))):
            chars[index] = " "
    tag_keys = {lookup_key(tag) for tag in header.tags}
    for match in _TOKEN.finditer("".join(chars)):
        token = match.group()
        if token.startswith("#") and lookup_key(token[1:]) in tag_keys:
            items.append((match.start(), MetaItem("tag", token[1:])))
        else:
            items.append((match.start(), MetaItem("text", token)))
    return tuple(item for _, item in sorted(items, key=lambda pair: pair[0]))


def _frontmatter_items(raw: Any, key: MetadataKey) -> tuple[MetaItem, ...]:
    """YAML values: lists and scalars. Properties name tags without '#' and links as "[[x]]"."""
    if raw is None or raw == "" or raw == []:
        return ()
    if key.kinds[0] in ("text", "datetime", "date", "integer"):
        return (MetaItem("text", raw if isinstance(raw, str) else str(raw)),)
    values = raw if isinstance(raw, list) else [raw]
    items = []
    for value in values:
        if isinstance(value, (dict, list)) or value is None:
            items.append(MetaItem("text", _shown(value)))
            continue
        text = str(value)
        links = list(_FRONTMATTER_LINK.finditer(text))
        items.extend(MetaItem("link", match[1].strip()) for match in links)
        rest = _FRONTMATTER_LINK.sub(" ", text)
        for token in _TOKEN.findall(rest) if "tags" in key.kinds else ([rest.strip()] if rest.strip() else []):
            if "tags" in key.kinds:
                items.append(MetaItem("tag", token.lstrip("#")))
            else:
                items.append(MetaItem("text", token))
    return tuple(items)


def _item_problem(item: MetaItem, key: MetadataKey) -> str | None:
    kinds = key.kinds
    if kinds[0] == "text":
        return None if key.allows(item.value) else f"{item.value!r} is not an allowed value ({', '.join(key.values or ())})"
    if item.kind == "text":
        wanted = " or ".join({"tags": "a #tag", "links": "a [[link]]"}[kind] for kind in kinds)
        return f"{item.value!r} is not {wanted}"
    if item.kind == "tag" and "tags" not in kinds:
        return f"#{item.value} is a tag; expected a [[link]]"
    if item.kind == "link" and "links" not in kinds:
        return f"[[{item.value}]] is a link; expected a #tag"
    if not key.allows(item.value):
        shown = f"#{item.value}" if item.kind == "tag" else f"[[{item.value}]]"
        return f"{shown} is not an allowed value ({', '.join(key.values or ())})"
    return None


def _valid_date(raw: Any, kind: str, date_format: str) -> bool:
    if isinstance(raw, dt.datetime):
        return True
    if isinstance(raw, dt.date):
        return kind == "date"
    if not isinstance(raw, str):
        return False
    try:
        dt.datetime.strptime(raw.strip(), date_format)
    except ValueError:
        return False
    return True


def _shown(value: Any) -> str:
    return repr(value) if isinstance(value, str) else repr(str(value))
