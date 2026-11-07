"""Line edits that leave every other byte of a note alone.

A note is split into lines with their own terminators; edits append to a line,
insert lines before/after one, or replace one line's text. Untouched lines keep
their bytes (CRLF, BOM, trailing spaces). Line numbers are the parser's (1-based,
after the BOM), so positions from ``ParsedNote`` can be used directly.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import yaml

from vaultengine.formats.spec import MetadataKey
from vaultengine.formats.typed import TypedVault
from vaultengine.parser import _INLINE_FIELD

_SPLIT = re.compile(r"(\r\n|\r|\n)")


class Lines:
    def __init__(self, raw: str) -> None:
        self.bom = "\ufeff" if raw.startswith("\ufeff") else ""
        parts = _SPLIT.split(raw[len(self.bom):])
        self.content = parts[0::2]
        self.ends = parts[1::2] + [""]
        self.newline = Counter(self.ends[:-1]).most_common(1)[0][0] if len(self.ends) > 1 else "\n"
        # "a\nb\n" splits into ["a", "b", ""]: the last, empty piece is not a line.
        self.real = len(self.content) - (1 if len(self.content) > 1 and self.content[-1] == "" else 0)
        if self.real == 1 and self.content[0] == "":
            self.real = 0
        self.append: dict[int, list[str]] = defaultdict(list)
        self.replace: dict[int, str] = {}
        self.before: dict[int, list[str]] = defaultdict(list)
        self.after: dict[int, list[str]] = defaultdict(list)
        self.at_end: list[str] = []

    def text(self, line: int) -> str:
        return self.content[line - 1]

    def render(self) -> str:
        out = [self.bom]
        last = self.real
        for index in range(1, last + 1):
            content, end = self.content[index - 1], self.ends[index - 1]
            nl = end or self.newline
            out.extend(line + nl for line in self.before.get(index, []))
            if index in self.replace:
                content = self.replace[index]
            if index in self.append:
                stripped = content.rstrip(" \t")
                content = stripped + "".join(self.append[index]) + content[len(stripped):]
            inserted_after = self.after.get(index, [])
            trailing = self.at_end if index == last else []
            if (inserted_after or trailing) and not end:
                end = self.newline  # the last line gains a newline so something can follow it
            out.append(content + end)
            out.extend(line + nl for line in inserted_after)
            out.extend(line + self.newline for line in trailing)
        if last == 0:
            out.extend(line + self.newline for line in self.at_end)
        return "".join(out)


def add_relation(typed: TypedVault, lines: Lines, path: str, relation: str, links: list[str]) -> str:
    """Add ``relation:: [[link]], ...`` to a note. Returns what was done, for people.

    Where: the existing field of that relation; else a new line after the last
    relation field (same list/quote prefix); else under the type's relations heading;
    else after the metadata lines at the top; else before the first line.
    """
    shown = ", ".join(f"[[{link}]]" for link in links)
    anchor = typed.relation_anchor(path, relation)
    if anchor is not None and anchor.kind == "append":
        lines.append[anchor.line].append((" " if anchor.empty_value else ", ") + shown)
        return f"line {anchor.line}: added {shown} to `{relation}::`"
    if anchor is not None:
        match = _INLINE_FIELD.match(lines.text(anchor.line))
        prefix = match["prefix"] if match else ""
        lines.after[anchor.line].append(f"{prefix}{relation}:: {shown}")
        return f"after line {anchor.line}: new field `{relation}:: {shown}`"
    note = typed.vault.notes[path]
    if note.header_fields:
        line = note.header_fields[-1].line
        lines.after[line].append(f"{relation}:: {shown}")
        return f"after line {line}: new field `{relation}:: {shown}`"
    if lines.real >= note.body_start_line:
        lines.before[note.body_start_line].append(f"{relation}:: {shown}")
        return f"before line {note.body_start_line}: new field `{relation}:: {shown}`"
    lines.at_end.append(f"{relation}:: {shown}")
    return f"end of note: new field `{relation}:: {shown}`"


def set_header_value(typed: TypedVault, lines: Lines, path: str, key: MetadataKey, value: str) -> str:
    """Set ``Key: value`` in the header block: replace the existing line's value, or
    add the line where the type's key order puts it. Returns what was done."""
    note = typed.vault.notes[path]
    note_type = typed.spec.type_for(path)
    wanted = key.name.lower()
    existing = next((f for f in note.header_fields if f.key.lower() == wanted), None)
    if existing is not None:
        text = lines.text(existing.line)
        key_end = text.index(":")  # keys cannot contain ':'
        lines.replace[existing.line] = f"{text[:key_end + 1]} {value}".rstrip() if value else text[:key_end + 1]
        return f"line {existing.line}: {existing.key}: {value}"

    order = [k.name.lower() for k in note_type.metadata] if note_type else [wanted]
    position = order.index(wanted) if wanted in order else len(order)
    present = [(order.index(f.key.lower()) if f.key.lower() in order else len(order), f) for f in note.header_fields]
    spaced = len(note.header_fields) > 1 and note.header_fields[1].line - note.header_fields[0].line > 1
    new_line = f"{key.name}: {value}".rstrip()
    earlier = [f for rank, f in present if rank < position]
    if earlier:
        anchor = earlier[-1].line
        lines.after[anchor].extend(["", new_line] if spaced else [new_line])
        return f"after line {anchor}: {new_line}"
    if note.header_fields:
        anchor = note.header_fields[0].line
        lines.before[anchor].extend([new_line, ""] if spaced else [new_line])
        return f"before line {anchor}: {new_line}"
    start = note.body_start_line
    if lines.real >= start:
        block = [new_line] if not lines.text(start).strip() else [new_line, ""]
        lines.before[start].extend(block)
        return f"before line {start}: {new_line}"
    lines.at_end.append(new_line)
    return f"end of note: {new_line}"


class EditRefused(ValueError):
    """The edit cannot be made without touching text it should not touch."""


def set_frontmatter_value(typed: TypedVault, lines: Lines, path: str, key: MetadataKey, value: str) -> str:
    """Set a property in YAML frontmatter, one line only.

    An existing ``key: value`` on one line is replaced; a missing key is added before
    the closing ``---``. A value spread over several lines (a block list, a folded
    string) is refused rather than rewritten. The value is written as a YAML string,
    quoted when needed (``"[[link]]"``).
    """
    note = typed.vault.notes[path]
    if not note.has_frontmatter:
        raise EditRefused(f"{path} has no frontmatter")
    closing = note.body_start_line - 1  # the line of the closing ---
    dumped = yaml.safe_dump({key.name: value}, allow_unicode=True, default_flow_style=False, width=10_000).rstrip("\n")
    if "\n" in dumped:
        raise EditRefused(f"the value for {key.name!r} would take several lines of YAML")
    item = dumped.split(":", 1)[1]
    wanted = key.name.lower()
    for line in range(2, closing):
        text = lines.text(line)
        match = re.match(r"^([^\s#:][^:]*?)\s*:(?=\s|$)", text)
        if not match or match[1].strip().strip("'\"").lower() != wanted:
            continue
        following = lines.text(line + 1) if line + 1 < closing else ""
        rest = text[match.end():].strip()
        if rest in ("", "|", ">", "|-", ">-") or (following[:1] in (" ", "\t", "-") and following.strip()):
            raise EditRefused(f"{key.name!r} spans several lines in the frontmatter of {path}; edit it by hand")
        lines.replace[line] = f"{text[:match.end()]}{item}"
        return f"line {line}: {text[:match.end()]}{item}"
    lines.before[closing].append(f"{key.name}:{item}")
    return f"before line {closing}: {key.name}:{item}"
