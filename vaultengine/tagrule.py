"""Which links are tags: ``Tags: [[reward]] [[rl-environment-design]]`` tags a note with two topics.

A link is a *tag link* when it sits in a metadata field whose role is ``tag``:

* with a format (``engine.format`` or ``--format``): a header field (``Key: value`` at the top of
  the note) that the note's type declares with ``link_role: tag`` (``formats/brain.yaml``:
  ``Tags: {kind: links, link_role: tag}`` in every type). The type comes from the path alone,
  and the index records the field each link is in (``links.field_kind``/``field_key``), so
  no note is read;
* without a format: a field named ``tags`` (any case), header or ``tags::`` inline, the
  convention Obsidian uses for its own tags property.

What a tag link changes (``graphrank.py``, ``related.py``, ``relations.py``): in the pair signals
it is a tag, like ``#tag``: shared tag links count under shared tags, not as coupling (both
link to it) or co-citation (a note links to both). It is still a link: it makes two notes
"linked" (the direct signal, ``related``'s linked/not linked) and it is an edge of the graph
that ``related`` walks and ``vault map`` clusters, where a tag is a node its notes attach to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from vaultengine.vault import lookup_key


class TagRule:
    def __init__(self, description: str, keys_for: Callable[[str], frozenset[str]], sources: frozenset[str]) -> None:
        self.description = description  # for output: where the rule comes from
        self._keys_for = keys_for  # note path -> lookup keys of its tag fields
        self._sources = sources  # field sources that count ("header", "inline")

    def is_tag(self, path: str, field_source: str | None, field_key: str | None) -> bool:
        if field_key is None or field_source not in self._sources:
            return False
        return lookup_key(field_key) in self._keys_for(path)


def default_rule() -> TagRule:
    keys = frozenset({"tags"})
    return TagRule("fields named 'tags' (no format configured)", lambda path: keys, frozenset({"header", "inline"}))


def format_rule(spec) -> TagRule:
    """Tag fields as the format declares them, per note type."""
    by_type: dict[str | None, frozenset[str]] = {}

    def keys_for(path: str) -> frozenset[str]:
        note_type = spec.type_for(path)
        name = note_type.name if note_type else None
        if name not in by_type:
            if note_type is None or note_type.metadata_source == "frontmatter":
                by_type[name] = frozenset()
            else:
                by_type[name] = frozenset(lookup_key(k.name) for k in note_type.metadata if k.link_role == "tag")
        return by_type[name]

    fields = sorted({k.name for t in spec.types for k in t.metadata if k.link_role == "tag"})
    return TagRule(f"format {spec.name} ({', '.join(fields) or 'no tag fields'})", keys_for, frozenset({"header"}))


def rule_for_format(name_or_path: str | Path | None) -> TagRule:
    """The rule for a configured format (a name or a file), or the default without one. Raises FormatError."""
    if not name_or_path:
        return default_rule()
    from vaultengine.formats import BUNDLED_FORMATS_DIR, load_format, resolve_format_path

    return format_rule(load_format(resolve_format_path(name_or_path, [Path("formats"), BUNDLED_FORMATS_DIR])))
