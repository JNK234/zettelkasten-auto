"""YAML frontmatter handling, following Obsidian's rules.

Obsidian only treats a block as frontmatter when the very first line of the file
is ``---`` and a later line is ``---``. Anything else (for example a ``---``
horizontal rule further down) is body text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import yaml

_DELIM = re.compile(r"---[ \t]*")
# libyaml-backed loader when available (~8x faster, same results); pure Python otherwise.
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


@dataclass(frozen=True)
class FrontmatterBlock:
    data: dict[str, Any]
    error: str | None
    present: bool
    end_offset: int  # offset of the first body character
    body_start_line: int  # 1-based line number of the first body line


def split_frontmatter(text: str) -> FrontmatterBlock:
    """Locate and parse the frontmatter block at the top of ``text``.

    Invalid YAML still counts as a frontmatter block (Obsidian hides it from
    the body and reports a properties error), but yields no data.
    """
    lines = text.split("\n")
    if not lines or not _DELIM.fullmatch(lines[0]):
        return FrontmatterBlock({}, None, False, 0, 1)

    for index in range(1, len(lines)):
        if _DELIM.fullmatch(lines[index]):
            yaml_text = "\n".join(lines[1:index])
            end_offset = sum(len(line) + 1 for line in lines[: index + 1])
            data, error = _load_yaml(yaml_text)
            return FrontmatterBlock(data, error, True, min(end_offset, len(text)), index + 2)

    # No closing delimiter: Obsidian renders the opening '---' as a rule.
    return FrontmatterBlock({}, None, False, 0, 1)


def _load_yaml(yaml_text: str) -> tuple[dict[str, Any], str | None]:
    try:
        loaded = yaml.load(yaml_text, Loader=_YAML_LOADER)
    except yaml.YAMLError as exc:
        return {}, f"invalid YAML: {exc}".replace("\n", " ")
    if loaded is None:
        return {}, None
    if not isinstance(loaded, dict):
        return {}, f"frontmatter is a {type(loaded).__name__}, not a mapping"
    return {str(key): value for key, value in loaded.items()}, None


def _get_key(data: dict[str, Any], name: str) -> Any:
    """Case-insensitive key lookup (Obsidian matches property names that way)."""
    for key, value in data.items():
        if key.lower() == name:
            return value
    return None


def _string_items(value: Any, split_pattern: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = re.split(split_pattern, value)
    elif isinstance(value, (list, tuple)):
        items = [str(item) for item in value if item is not None and not isinstance(item, (dict, list))]
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        items = [str(value)]
    else:
        return []
    return [item.strip() for item in items if item and item.strip()]


def frontmatter_tags(data: dict[str, Any]) -> list[str]:
    """Tags from the ``tags`` property: a list, or a string split on commas/whitespace.

    A leading '#' is stripped. Validity (e.g. purely numeric names) is checked
    by the caller so inline and frontmatter tags share one rule.
    """
    return [item.lstrip("#") for item in _string_items(_get_key(data, "tags"), r"[,\s]+")]


def frontmatter_aliases(data: dict[str, Any]) -> list[str]:
    """Aliases from the ``aliases`` property: a list, or a string split on commas."""
    return _string_items(_get_key(data, "aliases"), r",")
