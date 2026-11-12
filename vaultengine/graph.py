"""Vault-wide link graph: outgoing links, backlinks, and Obsidian-style summaries.

Every link and embed in every note becomes one ``ResolvedLink`` edge, so
wikilinks on a ``Tags:`` line are edges like any other link.
"""

from __future__ import annotations

from collections import defaultdict

from vaultengine.model import field_kinds
from vaultengine.resolver import ResolvedLink, Resolver
from vaultengine.vault import Vault


class LinkGraph:
    def __init__(self, vault: Vault, resolve_aliases: bool = True) -> None:
        self.vault = vault
        self._field_kinds: dict[str, dict[int, str]] = {}
        resolver = Resolver(vault, resolve_aliases=resolve_aliases)
        self.edges: list[ResolvedLink] = []
        self._outgoing: dict[str, list[ResolvedLink]] = defaultdict(list)
        self._incoming: dict[str, list[ResolvedLink]] = defaultdict(list)
        for source in sorted(vault.notes):
            note = vault.notes[source]
            for link in sorted(note.links + note.embeds, key=lambda item: item.start):
                edge = resolver.resolve(source, link)
                self.edges.append(edge)
                self._outgoing[source].append(edge)
                if edge.status == "resolved" and edge.target_path != source:
                    self._incoming[edge.target_path].append(edge)

    def outgoing(self, path: str) -> list[ResolvedLink]:
        return list(self._outgoing.get(path, []))

    def _kinds(self, path: str) -> dict[int, str]:
        if path not in self._field_kinds:
            self._field_kinds[path] = field_kinds(self.vault.notes[path])
        return self._field_kinds[path]

    def outgoing_fields(self, path: str) -> list[tuple[str | None, str, str | None]]:
        """(target path, status, field kind) of each link of ``path``, in order. Field kind:
        'header', 'inline' or None (see ``model.field_kinds``)."""
        kinds = self._kinds(path) if path in self.vault.notes else {}
        return [(e.target_path, e.status, kinds.get(e.link.start)) for e in self._outgoing.get(path, [])]

    def incoming_fields(self, path: str) -> list[tuple[str, str | None]]:
        """(source, field kind) of each resolved link to ``path`` from another note."""
        return [(e.source, self._kinds(e.source).get(e.link.start)) for e in self._incoming.get(path, [])]

    def backlinks(self, path: str) -> list[ResolvedLink]:
        """Resolved links pointing at ``path`` from other files (self-links excluded)."""
        return list(self._incoming.get(path, []))

    def backlink_sources(self, path: str) -> list[str]:
        return sorted({edge.source for edge in self.backlinks(path)})

    @property
    def resolved_links(self) -> dict[str, dict[str, int]]:
        """{source: {target path: count}}, the shape of Obsidian's metadataCache.resolvedLinks.

        Same-note links ([[#Heading]]) are left out.
        """
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for edge in self.edges:
            if edge.status == "resolved" and not edge.link.is_same_note:
                counts[edge.source][edge.target_path] += 1
        return {source: dict(targets) for source, targets in counts.items()}

    @property
    def unresolved_links(self) -> dict[str, dict[str, int]]:
        """{source: {link target as written: count}}, like metadataCache.unresolvedLinks."""
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for edge in self.edges:
            if edge.status == "unresolved":
                counts[edge.source][edge.link.target] += 1
        return {source: dict(targets) for source, targets in counts.items()}

    def ambiguous(self) -> list[ResolvedLink]:
        return [edge for edge in self.edges if edge.status == "ambiguous"]

    def missing_subpaths(self) -> list[ResolvedLink]:
        """Links whose note resolved but whose #heading or #^block does not exist."""
        return [edge for edge in self.edges if edge.subpath_status == "missing"]
