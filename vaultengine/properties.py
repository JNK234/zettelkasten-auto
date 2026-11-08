"""Note properties beyond frontmatter, and how property values are compared.

A property can come from three places (the ``source`` column of ``properties``):

* ``frontmatter`` - the YAML block at the top (Obsidian's properties);
* ``inline``      - a Dataview-style field on its own line, ``key:: value``
                    (also as a list item, ``- key:: value``), anywhere in the body;
* ``header``      - a ``Key: value`` line at the very top of the body, where this
                    project's note format keeps ``Created:``, ``Tags:`` and ``Status:``.
                    Only the leading run of such lines (and blank lines) counts: the
                    first line of anything else ends the header. The key is one word;
                    the value may be empty (``Tags:`` alone still sets ``tags``).

Text inside code and comments is never a field. Keys match case-insensitively.
Stats, health and the note view keep reporting frontmatter properties only.

Values
------
``items(value)`` is what ``=`` compares against: each list element, or for a
string the whole string plus every ``#tag`` and ``[[link]]`` in it, normalised
by ``normalize_item`` (lowercase, no leading ``#``, no ``[[ ]]`` or ``|alias``).
``:`` is a case-insensitive substring test on the text of the value.
``<``, ``<=``, ``>``, ``>=`` compare numbers with numbers and dates with dates.

Dates are ISO (``2026-09-24``, ``2026-09-24T19:20``, ``2026-09-24 19:20``) or the
day-first format of this project's template (``24-09-2026``, ``24-09-2026 19:20``).
A query date may be a year, a month or a day and stands for that whole period.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

# Bump when extraction can change for the same note (part of search.SEARCH_VERSION).
PROPERTIES_VERSION = 1

_INLINE_FIELD = re.compile(r"^[ \t]*(?:>[ \t]*)*(?:[-*+][ \t]+)?(?P<key>[^\s:\[\]()#>][^:\[\]()\n]{0,59}?)[ \t]*::[ \t]*(?P<value>.*?)[ \t]*$", re.M)
_HEADER_FIELD = re.compile(r"(?P<key>[A-Za-z][A-Za-z0-9_-]{0,39}):(?:[ \t]+(?P<value>.*?))?[ \t]*$")
_TAG_IN_VALUE = re.compile(r"(?<![^\s,])#([^\s,#\[\]]+)")
_LINK_IN_VALUE = re.compile(r"\[\[([^\[\]|#]+)(?:[#|][^\[\]]*)?\]\]")


@dataclass(frozen=True)
class Field:
    key: str
    value: Any
    source: str  # "inline" | "header"
    key_span: tuple[int, int] = (0, 0)  # offsets of "key:" / "key::" in the note text


def body_fields(text: str, masked: str) -> list[Field]:
    """Inline and header fields of a note.

    Found in ``masked`` (the note with frontmatter, code and comments blanked, same
    length as ``text``); values are read from ``text`` so nothing inside them is lost.
    """
    found: list[Field] = []
    offset = 0
    for line in masked.split("\n"):
        if line.strip():
            match = _HEADER_FIELD.fullmatch(line)
            if match is None:
                break
            # The key comes from the masked line; the value is the rest of the original line.
            value = text[offset + match.end("key") + 1 : offset + len(line)].strip()
            found.append(Field(match["key"], value, "header", (offset + match.start("key"), offset + match.end("key") + 1)))
        offset += len(line) + 1
    for match in _INLINE_FIELD.finditer(masked):
        line_end = masked.find("\n", match.start("value"))
        value = text[match.start("value") : len(text) if line_end == -1 else line_end].strip()
        colons = masked.index("::", match.end("key"))
        found.append(Field(match["key"].strip(), value, "inline", (match.start("key"), colons + 2)))
    return found


# ----- values ------------------------------------------------------------------


def normalize_item(text: str) -> str:
    text = text.strip()
    link = _LINK_IN_VALUE.fullmatch(text)
    if link:
        text = link.group(1)
    return text.lstrip("#").strip().casefold()


def value_texts(value_json: str) -> list[str]:
    value = json.loads(value_json)
    if isinstance(value, list):
        return [json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v) for v in value if v is not None]
    if value is None:
        return []
    if isinstance(value, bool):
        return ["true" if value else "false"]
    return [json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)]


def items(value_json: str) -> set[str]:
    found = set()
    for text in value_texts(value_json):
        found.add(normalize_item(text))
        found.update(normalize_item(tag) for tag in _TAG_IN_VALUE.findall(text))
        found.update(normalize_item(link) for link in _LINK_IN_VALUE.findall(text))
    return found


def number(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None


# ----- dates -------------------------------------------------------------------

_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?)?(?:Z|[+-]\d{2}:?\d{2})?")
_DAY_FIRST = re.compile(r"(\d{2})-(\d{2})-(\d{4})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?")
_QUERY_DATE = re.compile(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?")


def parse_date(text: str) -> datetime | None:
    """A date or date-time written in a property value, or None."""
    text = text.strip()
    for pattern, order in ((_ISO, (0, 1, 2)), (_DAY_FIRST, (2, 1, 0))):
        match = pattern.fullmatch(text)
        if match:
            parts = match.groups()
            y, m, d = (int(parts[i]) for i in order)
            hh, mm, ss = (int(p) if p else 0 for p in parts[3:6])
            try:
                return datetime(y, m, d, hh, mm, ss)
            except ValueError:
                return None
    return None


@dataclass(frozen=True)
class Period:
    """[start, end) of a year, month or day written in a query."""

    start: datetime
    end: datetime
    text: str


def parse_period(text: str) -> Period | None:
    match = _QUERY_DATE.fullmatch(text.strip())
    if not match:
        return None
    year, month, day = match.groups()
    try:
        if day:
            start = datetime(int(year), int(month), int(day))
            return Period(start, start + timedelta(days=1), text)
        if month:
            start = datetime(int(year), int(month), 1)
            end = datetime(int(year) + (int(month) == 12), int(month) % 12 + 1, 1)
            return Period(start, end, text)
        return Period(datetime(int(year), 1, 1), datetime(int(year) + 1, 1, 1), text)
    except ValueError:
        return None


def compare(op: str, value: datetime | float, target: Period | float) -> bool:
    """``value op target``; a period compares by its bounds (``>`` = after it ends)."""
    if isinstance(target, Period):
        if op == ">":
            return value >= target.end
        if op == ">=":
            return value >= target.start
        if op == "<":
            return value < target.start
        if op == "<=":
            return value < target.end
        return target.start <= value < target.end  # "="
    if op == ">":
        return value > target
    if op == ">=":
        return value >= target
    if op == "<":
        return value < target
    if op == "<=":
        return value <= target
    return value == target


def as_date(value: date | datetime | str) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    return parse_date(value)
