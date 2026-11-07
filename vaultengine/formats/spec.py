"""Load and validate a format file: one vault's note conventions, written in YAML.

Nothing about any particular vault is hard-coded in the engine; everything a
format knows comes from its file (see ``formats/brain.yaml`` for a full example
and docs/formats.md for the reference). Shape::

    format: 1                      # version of this file layout
    name: brain
    metadata_source: inline        # inline | frontmatter | either (default: either)
    relations:                     # typed relation fields, written  name:: [[a]], [[b]]
      supports: {inverse: supported-by, targets: [claim]}
      from: {targets: [source]}    # no inverse: one-way
    link_roles:                    # role: [section headings]
      tension: [Tensions]
    types:
      claim:
        match: {folder: 50_MOC/discourse, name: "CLM - *"}
        headings: [Relations, Scope]
        metadata:
          Status: {required: true, kind: tags, values: [supported, contested]}
        relations: {required: [informs], optional: [refines]}

Validation is strict: unknown keys, wrong value types and references to
undeclared types or relations are errors that name the place in the file, so a
typo cannot silently switch a check off.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from vaultengine.resolver import normalize_heading
from vaultengine.vault import lookup_key

FORMAT_VERSION = 1
KINDS = ("text", "tags", "links", "datetime", "date", "integer")
SCALAR_KINDS = ("text", "datetime", "date", "integer")
METADATA_SOURCES = ("inline", "frontmatter", "either")
DEFAULT_DATE_FORMATS = {"datetime": "%Y-%m-%dT%H:%M", "date": "%Y-%m-%d"}
_NAME = re.compile(r"^[^\W_][\w\-]*$")


class FormatError(ValueError):
    """The format file is not valid. The message names the file and the place in it."""


# ----- pieces ------------------------------------------------------------------


@dataclass(frozen=True)
class SectionPattern:
    """Matches a heading. Plain text compares like heading links do (case, spacing and
    # ^ | : % [ ] \\ are ignored); ``/regex/`` is searched case-insensitively."""

    pattern: str
    _regex: re.Pattern[str] | None = field(default=None, compare=False, repr=False)
    _normalized: str = field(default="", compare=False, repr=False)

    @classmethod
    def parse(cls, pattern: str, where: str) -> "SectionPattern":
        if len(pattern) > 2 and pattern.startswith("/") and pattern.endswith("/"):
            try:
                return cls(pattern, re.compile(pattern[1:-1], re.IGNORECASE))
            except re.error as exc:
                raise FormatError(f"{where}: invalid regular expression {pattern!r}: {exc}") from None
        normalized = normalize_heading(pattern)
        if not normalized:
            raise FormatError(f"{where}: empty heading name")
        return cls(pattern, None, normalized)

    @property
    def is_regex(self) -> bool:
        return self._regex is not None

    def matches(self, heading: str) -> bool:
        if self._regex is not None:
            return self._regex.search(heading) is not None
        return normalize_heading(heading) == self._normalized


@dataclass(frozen=True)
class Match:
    """A note matches when it is inside ``folder`` (at any depth) and its file name
    without ``.md`` matches the glob ``name``. Either may be left out; both compare
    case-insensitively."""

    folder: str | None = None
    name: str | None = None

    def matches(self, path: str) -> bool:
        key = lookup_key(path)
        if self.folder is not None:
            folder = lookup_key(self.folder).strip("/")
            if folder and not key.startswith(folder + "/"):
                return False
        if self.name is not None:
            base = key.rsplit("/", 1)[-1]
            base = base[:-3] if base.endswith(".md") else base
            if not fnmatch.fnmatchcase(base, lookup_key(self.name)):
                return False
        return True

    def describe(self) -> str:
        parts = []
        if self.folder is not None:
            parts.append(f"in {self.folder}/")
        if self.name is not None:
            parts.append(f"named {self.name!r}")
        return " and ".join(parts)


@dataclass(frozen=True)
class MetadataKey:
    name: str
    required: bool = False
    allow_empty: bool = False
    kinds: tuple[str, ...] = ("text",)
    values: tuple[str, ...] | None = None
    multiple: bool = False
    date_format: str | None = None
    link_role: str | None = None
    description: str | None = None

    def allows(self, value: str) -> bool:
        return self.values is None or lookup_key(value) in {lookup_key(v) for v in self.values}


@dataclass(frozen=True)
class Relation:
    name: str
    inverse: str | None = None
    targets: tuple[str, ...] = ()  # note types allowed at the other end; empty = any
    description: str | None = None


@dataclass(frozen=True)
class NewNote:
    """How `vault new` writes a note of a type.

    ``name``: file name without .md; ``{title}`` is the title with characters file
    names and links cannot hold removed, ``{slug}`` is it lower-cased with runs of
    other characters turned into '-'. ``defaults``: metadata values used when none is
    given; ``{now}`` becomes the current time in the key's date format.
    """

    name: str = "{title}"
    max_name_length: int | None = None
    title_heading: bool = True
    spaced_metadata: bool = False  # a blank line between metadata lines
    defaults: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class IngestSpec:
    """How `vault ingest --as <type>` lays out a converted document.

    ``body_heading``: the required heading the document goes under;
    ``references_heading``: where "1. [title](url)" goes; ``provenance``: metadata
    keys for the original file name, its SHA-256 and the time of ingestion.
    """

    body_heading: str
    references_heading: str | None = None
    file_key: str | None = None
    hash_key: str | None = None
    date_key: str | None = None


@dataclass(frozen=True)
class Coverage:
    """What `vault coverage` reports on: notes of ``sources`` types nothing links to, and
    notes of ``citing`` types without a link to a source. Links in ``citation_roles``
    count as citations even when their target is missing (reported as such)."""

    sources: tuple[str, ...] = ()
    citing: tuple[str, ...] = ()
    citation_roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class NoteType:
    name: str
    match: tuple[Match, ...]
    description: str | None = None
    headings: tuple[SectionPattern, ...] = ()
    ordered_headings: bool = False
    metadata: tuple[MetadataKey, ...] = ()
    metadata_source: str = "either"
    relations_constrained: bool = False  # True when the type has a `relations:` block
    relations_required: tuple[str, ...] = ()
    relations_optional: tuple[str, ...] = ()
    link_roles: tuple[tuple[str, SectionPattern], ...] = ()
    relations_heading: SectionPattern | None = None  # where relation fields go (used by `vault fix`)
    new: NewNote = NewNote()
    title_section: SectionPattern | None = None  # first link text under it is also a title (duplicates)
    ingest: IngestSpec | None = None

    def matches(self, path: str) -> bool:
        return any(m.matches(path) for m in self.match)

    def allows_relation(self, name: str) -> bool:
        return not self.relations_constrained or name in self.relations_required or name in self.relations_optional


@dataclass(frozen=True)
class FormatSpec:
    name: str
    types: tuple[NoteType, ...]
    relations: dict[str, Relation]
    link_roles: tuple[tuple[str, SectionPattern], ...] = ()
    description: str | None = None
    path: str | None = None
    coverage: Coverage | None = None

    def type_named(self, name: str) -> NoteType | None:
        return next((t for t in self.types if t.name == name), None)

    def type_for(self, path: str) -> NoteType | None:
        """The first type (in file order) whose ``match`` fits the path."""
        return next((t for t in self.types if t.matches(path)), None)

    def relation(self, key: str) -> Relation | None:
        """The relation a field key names (case-insensitive, spaces count as '-')."""
        return self.relations.get(relation_key(key))


def relation_key(key: str) -> str:
    return re.sub(r"[\s_]+", "-", lookup_key(key))


# ----- loading -----------------------------------------------------------------


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys (plain YAML keeps the last one silently)."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            mark = key_node.start_mark
            raise FormatError(f"line {mark.line + 1}: duplicate key {key!r}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def resolve_format_path(name_or_path: str | Path, search: Iterable[Path] = ()) -> Path:
    """A path to a file, or a bare name looked up as ``<dir>/<name>.yaml`` in ``search``."""
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") or candidate.exists() or "/" in str(name_or_path):
        return candidate
    search = list(dict.fromkeys(Path(d).resolve() for d in search))
    for directory in search:
        for suffix in (".yaml", ".yml"):
            found = Path(directory) / f"{name_or_path}{suffix}"
            if found.exists():
                return found
    looked = ", ".join(str(Path(d)) for d in search) or "nowhere"
    raise FormatError(f"no format named {str(name_or_path)!r} (looked in {looked})")


def load_format(path: str | Path) -> FormatSpec:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FormatError(f"{path}: cannot read format file: {exc.strerror or exc}") from None
    try:
        return parse_format(text, source=str(path))
    except FormatError as exc:
        message = str(exc)
        raise FormatError(message if message.startswith(str(path)) else f"{path}: {message}") from None


def parse_format(text: str, source: str = "<format>") -> FormatSpec:
    try:
        data = yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise FormatError(f"invalid YAML: {exc}".replace("\n", " ")) from None
    return _Parser(source).spec(data)


class _Parser:
    def __init__(self, source: str) -> None:
        self.source = source

    # --- generic helpers

    def _mapping(self, value: Any, where: str, allowed: Iterable[str], required: Iterable[str] = ()) -> dict:
        if not isinstance(value, dict):
            raise FormatError(f"{where}: expected a mapping, got {_type_name(value)}")
        allowed = tuple(allowed)
        for key in value:
            if not isinstance(key, str) or key not in allowed:
                raise FormatError(f"{where}: unknown key {key!r} (allowed: {', '.join(allowed)})")
        for key in required:
            if key not in value:
                raise FormatError(f"{where}: missing required key {key!r}")
        return value

    def _str(self, value: Any, where: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise FormatError(f"{where}: expected a non-empty string, got {_type_name(value)}")
        return value.strip()

    def _bool(self, value: Any, where: str) -> bool:
        if not isinstance(value, bool):
            raise FormatError(f"{where}: expected true or false, got {_type_name(value)}")
        return value

    def _str_list(self, value: Any, where: str, allow_scalar: bool = True) -> tuple[str, ...]:
        if allow_scalar and isinstance(value, (str, int, float)) and not isinstance(value, bool):
            value = [value]
        if not isinstance(value, list):
            raise FormatError(f"{where}: expected a list, got {_type_name(value)}")
        items = []
        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, (str, int, float)) or not str(item).strip():
                raise FormatError(f"{where}[{index}]: expected a non-empty string, got {_type_name(item)}")
            items.append(str(item).strip())
        return tuple(items)

    def _ident(self, value: Any, where: str) -> str:
        name = self._str(value, where)
        if not _NAME.match(name):
            raise FormatError(f"{where}: {name!r} is not a valid name (letters, digits, '-' and '_')")
        return name

    def _choice(self, value: Any, where: str, choices: tuple[str, ...]) -> str:
        value = self._str(value, where)
        if value not in choices:
            raise FormatError(f"{where}: {value!r} is not one of: {', '.join(choices)}")
        return value

    # --- sections

    def spec(self, data: Any) -> FormatSpec:
        if data is None:
            raise FormatError("the file is empty")
        data = self._mapping(
            data, "top level",
            ("format", "name", "description", "metadata_source", "relations", "link_roles", "types", "coverage"),
            required=("format", "types"),
        )
        if data["format"] != FORMAT_VERSION:
            raise FormatError(f"format: this engine reads format {FORMAT_VERSION} files, got {data['format']!r}")
        source = self._choice(data.get("metadata_source", "either"), "metadata_source", METADATA_SOURCES)
        relations = self._relations(data.get("relations") or {})
        roles = self._link_roles(data.get("link_roles") or {}, "link_roles")
        types_data = data["types"]
        if not isinstance(types_data, dict) or not types_data:
            raise FormatError(f"types: expected a mapping with at least one type, got {_type_name(types_data)}")
        types = tuple(self._type(name, body, source, relations) for name, body in types_data.items())

        type_names = {t.name for t in types}
        for relation in relations.values():
            for target in relation.targets:
                if target not in type_names:
                    raise FormatError(f"relations.{relation.name}.targets: unknown type {target!r}")
        name = self._str(data["name"], "name") if "name" in data else Path(self.source).stem
        description = self._str(data["description"], "description") if "description" in data else None
        coverage = self._coverage(data["coverage"], type_names, relations, roles, types) if "coverage" in data else None
        return FormatSpec(name, types, relations, roles, description, self.source, coverage)

    def _coverage(self, data: Any, type_names: set[str], relations: dict[str, Relation],
                  roles: tuple, types: tuple[NoteType, ...]) -> Coverage:
        where = "coverage"
        data = self._mapping(data, where, ("sources", "citing", "citation_roles"), required=("sources", "citing"))
        known_roles = {role for role, _ in roles} | set(relations) | {
            role for t in types for role, _ in t.link_roles} | {k.link_role for t in types for k in t.metadata if k.link_role}
        found = {}
        for key in ("sources", "citing"):
            names = self._str_list(data[key], f"{where}.{key}")
            for index, name in enumerate(names):
                if name not in type_names:
                    raise FormatError(f"{where}.{key}[{index}]: unknown type {name!r}")
            found[key] = names
        cite_roles = self._str_list(data.get("citation_roles", []), f"{where}.citation_roles")
        for index, role in enumerate(cite_roles):
            if role not in known_roles:
                raise FormatError(f"{where}.citation_roles[{index}]: {role!r} is not a link role or relation of this format")
        return Coverage(found["sources"], found["citing"], cite_roles)

    def _relations(self, data: Any) -> dict[str, Relation]:
        data = self._mapping(data, "relations", data.keys() if isinstance(data, dict) else ())
        found: dict[str, Relation] = {}
        inverses: dict[str, tuple[str, str]] = {}  # inverse key -> (declared by, where)
        for raw_name, body in data.items():
            where = f"relations.{raw_name}"
            name = relation_key(self._ident(raw_name, where))
            if name in found:
                raise FormatError(f"{where}: relation {name!r} is declared twice")
            body = self._mapping(body if body is not None else {}, where, ("inverse", "targets", "description"))
            inverse = relation_key(self._ident(body["inverse"], f"{where}.inverse")) if "inverse" in body else None
            targets = self._str_list(body.get("targets", []), f"{where}.targets")
            description = self._str(body["description"], f"{where}.description") if "description" in body else None
            found[name] = Relation(name, inverse, targets, description)
            if inverse is not None:
                if inverse in inverses and inverse != name:
                    raise FormatError(f"{where}.inverse: {inverse!r} is already the inverse of {inverses[inverse][0]!r}")
                inverses[inverse] = (name, where)
        # An inverse is a relation too. If both sides are declared they must agree.
        for inverse, (owner, where) in inverses.items():
            declared = found.get(inverse)
            if declared is None:
                found[inverse] = Relation(inverse, owner)
            elif inverse != owner and declared.inverse not in (None, owner):
                raise FormatError(
                    f"{where}.inverse: {owner!r} names {inverse!r} as its inverse, "
                    f"but {inverse!r} names {declared.inverse!r}"
                )
            elif declared.inverse is None:
                found[inverse] = Relation(declared.name, owner, declared.targets, declared.description)
        return found

    def _link_roles(self, data: Any, where: str) -> tuple[tuple[str, SectionPattern], ...]:
        data = self._mapping(data, where, data.keys() if isinstance(data, dict) else ())
        roles = []
        for role, sections in data.items():
            role_where = f"{where}.{role}"
            role = self._ident(role, role_where)
            for index, pattern in enumerate(self._str_list(sections, role_where)):
                roles.append((role, SectionPattern.parse(pattern, f"{role_where}[{index}]")))
        return tuple(roles)

    def _type(self, name: Any, body: Any, default_source: str, relations: dict[str, Relation]) -> NoteType:
        where = f"types.{name}"
        name = self._ident(name, where)
        body = self._mapping(
            body, where,
            ("description", "match", "headings", "ordered_headings", "metadata", "metadata_source", "relations",
             "relations_heading", "link_roles", "new", "title_section", "ingest"),
            required=("match",),
        )
        matches = body["match"]
        matches = matches if isinstance(matches, list) else [matches]
        if not matches:
            raise FormatError(f"{where}.match: expected at least one rule")
        match = tuple(self._match(item, f"{where}.match" + (f"[{i}]" if len(matches) > 1 else "")) for i, item in enumerate(matches))
        headings = tuple(
            SectionPattern.parse(text, f"{where}.headings[{i}]")
            for i, text in enumerate(self._str_list(body.get("headings", []), f"{where}.headings"))
        )
        source = self._choice(body.get("metadata_source", default_source), f"{where}.metadata_source", METADATA_SOURCES)
        metadata = self._metadata(body.get("metadata") or {}, f"{where}.metadata")

        constrained = "relations" in body
        required: tuple[str, ...] = ()
        optional: tuple[str, ...] = ()
        if constrained:
            rel = self._mapping(body["relations"] or {}, f"{where}.relations", ("required", "optional"))
            required = self._relation_names(rel.get("required", []), f"{where}.relations.required", relations)
            optional = self._relation_names(rel.get("optional", []), f"{where}.relations.optional", relations)
        description = self._str(body["description"], f"{where}.description") if "description" in body else None
        relations_heading = (
            SectionPattern.parse(self._str(body["relations_heading"], f"{where}.relations_heading"), f"{where}.relations_heading")
            if "relations_heading" in body else None
        )
        return NoteType(
            name=name,
            match=match,
            description=description,
            headings=headings,
            ordered_headings=self._bool(body.get("ordered_headings", False), f"{where}.ordered_headings"),
            metadata=metadata,
            metadata_source=source,
            relations_constrained=constrained,
            relations_required=required,
            relations_optional=optional,
            link_roles=self._link_roles(body.get("link_roles") or {}, f"{where}.link_roles"),
            relations_heading=relations_heading,
            new=self._new(body.get("new"), f"{where}.new", match, metadata),
            title_section=(
                SectionPattern.parse(self._str(body["title_section"], f"{where}.title_section"), f"{where}.title_section")
                if "title_section" in body else None
            ),
            ingest=self._ingest(body.get("ingest"), f"{where}.ingest", headings, metadata),
        )

    def _ingest(self, data: Any, where: str, headings: tuple[SectionPattern, ...],
                metadata: tuple[MetadataKey, ...]) -> IngestSpec | None:
        if data is None:
            return None
        data = self._mapping(data, where, ("body_heading", "references_heading", "provenance"), required=("body_heading",))
        names = {h.pattern for h in headings if not h.is_regex}
        found = {}
        for key in ("body_heading", "references_heading"):
            if key in data:
                value = self._str(data[key], f"{where}.{key}")
                if value not in names:
                    raise FormatError(f"{where}.{key}: {value!r} is not one of the type's headings")
                found[key] = value
        provenance = self._mapping(data.get("provenance") or {}, f"{where}.provenance", ("file", "hash", "date"))
        declared = {k.name: k for k in metadata}
        keys = {}
        for role, key_name in provenance.items():
            key_name = self._str(key_name, f"{where}.provenance.{role}")
            if key_name not in declared:
                raise FormatError(f"{where}.provenance.{role}: {key_name!r} is not a metadata key of the type")
            if role == "date" and declared[key_name].kinds[0] not in ("datetime", "date"):
                raise FormatError(f"{where}.provenance.date: {key_name!r} must be a datetime or date key")
            keys[role] = key_name
        return IngestSpec(found["body_heading"], found.get("references_heading"), keys.get("file"), keys.get("hash"), keys.get("date"))

    def _new(self, data: Any, where: str, match: tuple[Match, ...], metadata: tuple[MetadataKey, ...]) -> NewNote:
        default_name = "{title}"
        if len(match) == 1 and match[0].name and match[0].name.count("*") == 1 and not set("?[") & set(match[0].name):
            default_name = match[0].name.replace("*", "{title}")
        if data is None:
            return NewNote(name=default_name)
        data = self._mapping(data, where, ("name", "max_name_length", "title_heading", "spaced_metadata", "defaults"))
        name = self._str(data.get("name", default_name), f"{where}.name")
        unknown = set(re.findall(r"{(\w*)}", name)) - {"title", "slug"}
        if unknown or "{" in re.sub(r"{(title|slug)}", "", name):
            raise FormatError(f"{where}.name: only {{title}} and {{slug}} can be used in a name, got {name!r}")
        max_length = data.get("max_name_length")
        if max_length is not None and (isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 10):
            raise FormatError(f"{where}.max_name_length: expected a whole number of at least 10, got {max_length!r}")
        defaults_data = self._mapping(data.get("defaults") or {}, f"{where}.defaults", [k.name for k in metadata])
        defaults = []
        for key, value in defaults_data.items():
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise FormatError(f"{where}.defaults.{key}: expected a value written as text, got {_type_name(value)}")
            defaults.append((key, str(value)))
        return NewNote(
            name=name,
            max_name_length=max_length,
            title_heading=self._bool(data.get("title_heading", True), f"{where}.title_heading"),
            spaced_metadata=self._bool(data.get("spaced_metadata", False), f"{where}.spaced_metadata"),
            defaults=tuple(defaults),
        )

    def _match(self, data: Any, where: str) -> Match:
        data = self._mapping(data, where, ("folder", "name"))
        if not data:
            raise FormatError(f"{where}: give a folder, a name pattern, or both")
        folder = self._str(data["folder"], f"{where}.folder").strip("/") if "folder" in data else None
        name = self._str(data["name"], f"{where}.name") if "name" in data else None
        return Match(folder, name)

    def _relation_names(self, value: Any, where: str, relations: dict[str, Relation]) -> tuple[str, ...]:
        names = []
        for index, raw in enumerate(self._str_list(value, where)):
            key = relation_key(raw)
            if key not in relations:
                raise FormatError(f"{where}[{index}]: unknown relation {raw!r} (declare it under relations:)")
            names.append(key)
        return tuple(names)

    def _metadata(self, data: Any, where: str) -> tuple[MetadataKey, ...]:
        data = self._mapping(data, where, data.keys() if isinstance(data, dict) else ())
        keys = []
        seen: set[str] = set()
        for name, body in data.items():
            key_where = f"{where}.{name}"
            name = self._str(name, key_where)
            if lookup_key(name) in seen:
                raise FormatError(f"{key_where}: key declared twice (keys are case-insensitive)")
            seen.add(lookup_key(name))
            body = self._mapping(
                body if body is not None else {}, key_where,
                ("required", "allow_empty", "kind", "values", "multiple", "format", "link_role", "description"),
            )
            kinds = self._str_list(body.get("kind", "text"), f"{key_where}.kind")
            for kind in kinds:
                if kind not in KINDS:
                    raise FormatError(f"{key_where}.kind: unknown kind {kind!r} (expected one of: {', '.join(KINDS)})")
            if len(kinds) > 1 and any(kind in SCALAR_KINDS for kind in kinds):
                raise FormatError(f"{key_where}.kind: only 'tags' and 'links' can be combined")
            date_format = None
            if "format" in body:
                if kinds[0] not in ("datetime", "date"):
                    raise FormatError(f"{key_where}.format: only datetime and date keys take a format")
                date_format = self._str(body["format"], f"{key_where}.format")
            elif kinds[0] in DEFAULT_DATE_FORMATS:
                date_format = DEFAULT_DATE_FORMATS[kinds[0]]
            scalar = kinds[0] in SCALAR_KINDS
            multiple = self._bool(body.get("multiple", not scalar), f"{key_where}.multiple")
            if multiple and scalar:
                raise FormatError(f"{key_where}.multiple: a {kinds[0]} key holds one value")
            values = self._str_list(body["values"], f"{key_where}.values") if "values" in body else None
            if values is not None and kinds[0] in ("datetime", "date", "integer"):
                raise FormatError(f"{key_where}.values: not supported for {kinds[0]} keys")
            keys.append(
                MetadataKey(
                    name=name,
                    required=self._bool(body.get("required", False), f"{key_where}.required"),
                    allow_empty=self._bool(body.get("allow_empty", False), f"{key_where}.allow_empty"),
                    kinds=kinds,
                    values=values,
                    multiple=multiple,
                    date_format=date_format,
                    link_role=self._ident(body["link_role"], f"{key_where}.link_role") if "link_role" in body else None,
                    description=self._str(body["description"], f"{key_where}.description") if "description" in body else None,
                )
            )
        return tuple(keys)


def _type_name(value: Any) -> str:
    if value is None:
        return "nothing"
    return {dict: "a mapping", list: "a list", str: "a string", bool: "a boolean"}.get(type(value), type(value).__name__)
