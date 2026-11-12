"""`vault pipeline` and `vault note`: where each source is, and what the engine knows
about one note, worked out from the links on every run. Nothing here is stored.

Per the format's ``pipeline:`` block (types playing three roles):

* a note of a ``sources`` type is **ingested**;
* **extracted** once a note of an ``extract`` type (a zettel) links to it;
* **compiled** once a note of a ``compile`` type (a page) links to one of those.

A note of an ``extract`` type that rests on a source note is itself **extracted**,
and **compiled** once a page links to it; one that rests on no source note has no
stage. "Links to" is ``TypedVault.note_links``, the same definition as the
``min_links`` rule: resolved links and embeds anywhere in the body, not links in
metadata values such as ``Tags:``, not links that match no note. So a URL, or a
[[link]] to a source note that is not in the vault, moves nothing. Notes in ignored
folders have no type and play no role.

The gaps: sources nothing has been extracted from (stage ingested), and extract
notes no compile note links to. For extract notes that rest on no source note, the
reason comes from `vault coverage` (``citation_gap``): they cite notes that are
missing from the vault, notes that are not sources, or nothing; ``url-only`` is
added for notes whose only citations are URLs.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Any

from vaultengine.formats.coverage import citation_dict, citation_gap
from vaultengine.formats.typed import TypedVault

STAGES = ("ingested", "extracted", "compiled")


class PipelineNotConfigured(ValueError):
    """The format has no ``pipeline:`` block."""


class Pipeline:
    def __init__(self, typed: TypedVault) -> None:
        config = typed.spec.pipeline
        if config is None:
            raise PipelineNotConfigured(f"format {typed.spec.name!r} has no `pipeline:` block")
        self.typed = typed
        self.source_types = set(config.sources)
        self.extract_types = set(config.extract)
        self.compile_types = set(config.compile)
        self._lists: dict[str, list[str]] | None = None

    def _by_role(self) -> dict[str, list[str]]:
        """Every note of each role: typing every note, so only worked out when a question needs all of them."""
        if self._lists is None:
            lists: dict[str, list[str]] = {"sources": [], "extracts": [], "compiles": []}
            for path in sorted(self.typed.vault.notes):
                note_type = self.typed.type_of(path)
                if note_type in self.source_types:
                    lists["sources"].append(path)
                elif note_type in self.extract_types:
                    lists["extracts"].append(path)
                elif note_type in self.compile_types:
                    lists["compiles"].append(path)
            self._lists = lists
        return self._lists

    @property
    def sources(self) -> list[str]:
        return self._by_role()["sources"]

    @property
    def extracts(self) -> list[str]:
        return self._by_role()["extracts"]

    @property
    def compiles(self) -> list[str]:
        return self._by_role()["compiles"]

    # ----- one hop at a time -------------------------------------------------------------

    def _from(self, path: str, types: set[str]) -> list[str]:
        return [p for p in self.typed.linked_from(path) if self.typed.type_of(p) in types]

    def _to(self, path: str, types: set[str]) -> list[str]:
        return [p for p in self.typed.note_links(path) if self.typed.type_of(p) in types]

    def extracted_from(self, source: str) -> list[str]:
        """Extract notes (zettels) that link to ``source``."""
        return sorted(self._from(source, self.extract_types))

    def compiled_in(self, source: str) -> list[dict[str, Any]]:
        """Compile notes (pages) that link to a zettel extracted from ``source``, with those zettels."""
        via: dict[str, list[str]] = {}
        for zettel in self.extracted_from(source):
            for page in self._from(zettel, self.compile_types):
                via.setdefault(page, []).append(zettel)
        return [{"path": page, "via": sorted(zettels)} for page, zettels in sorted(via.items())]

    def stage(self, path: str) -> str | None:
        note_type = self.typed.type_of(path)
        if note_type in self.source_types:
            return "compiled" if self.compiled_in(path) else "extracted" if self.extracted_from(path) else "ingested"
        if note_type in self.extract_types:
            if not self._to(path, self.source_types):
                return None  # extracted from nothing: it rests on no source note (a too-few-links violation)
            return "compiled" if self._from(path, self.compile_types) else "extracted"
        return None

    def sources_under(self, path: str) -> list[dict[str, Any]]:
        """Source notes ``path`` rests on: linked directly (``direct``), and/or through the
        extract notes it links to (``via`` lists them)."""
        direct = set(self._to(path, self.source_types))
        found: dict[str, list[str]] = {source: [] for source in direct}
        for zettel in self._to(path, self.extract_types):
            for source in self._to(zettel, self.source_types):
                if source != path:
                    found.setdefault(source, []).append(zettel)
        return [{"path": p, "direct": p in direct, "via": sorted(v)} for p, v in sorted(found.items())]

    def unsourced_reason(self, path: str) -> tuple[str, list[dict[str, Any]]]:
        coverage = self.typed.spec.coverage
        roles = set(coverage.citation_roles) if coverage else set()
        reason, shown = citation_gap(self.typed, path, roles)
        citations = [citation_dict(self.typed, link) for link in shown]
        if reason == "none" and roles:
            note = self.typed.vault.notes[path]
            urls = [e for e in note.external_links if self.typed.section_role(path, e.line) in roles]
            if urls:
                reason = "url-only"
                citations = [{"line": e.line, "raw": e.raw, "url": e.url} for e in urls]
        return reason, citations

    def trust(self, path: str) -> dict[str, Any]:
        """How much the vault relies on the note, from use (``todo.Trust``); worked out, never stored."""
        from vaultengine.formats.todo import Trust

        if not hasattr(self, "_trust"):
            self._trust = Trust(self.typed, self)
        return self._trust.of(path)

    # ----- reports -------------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        typed = self.typed
        if hasattr(typed.graph, "prefetch"):
            typed.graph.prefetch()  # index-backed: every link row in one query, not one query per note
        rows = []
        for source in self.sources:
            zettels = self.extracted_from(source)
            pages = self.compiled_in(source)
            stage = "compiled" if pages else "extracted" if zettels else "ingested"
            rows.append({"path": source, "stage": stage, "zettels": zettels, "pages": pages})
        order = {stage: index for index, stage in enumerate(reversed(STAGES))}
        rows.sort(key=lambda row: (order[row["stage"]], -len(row["zettels"]), row["path"]))
        stages = Counter(row["stage"] for row in rows)

        in_no_page = [z for z in self.extracts if not self._from(z, self.compile_types)]
        unsourced = []
        bare = [z for z in self.extracts if not self._to(z, self.source_types)]
        if hasattr(typed.vault.notes, "prefetch"):
            typed.vault.notes.prefetch(bare)  # their citations are read from the notes
        for zettel in self.extracts:
            if not self._to(zettel, self.source_types):
                reason, citations = self.unsourced_reason(zettel)
                unsourced.append({"path": zettel, "reason": reason, "citations": citations})
        by_reason = Counter(item["reason"] for item in unsourced)
        return {
            "format": {"name": typed.spec.name, "path": typed.spec.path},
            "roles": {"sources": sorted(self.source_types), "extract": sorted(self.extract_types),
                      "compile": sorted(self.compile_types)},
            "stages": {stage: stages[stage] for stage in STAGES},
            "sources": rows,
            "extract": {
                "total": len(self.extracts),
                "in_a_compile_note": len(self.extracts) - len(in_no_page),
                "resting_on_a_source": len(self.extracts) - len(unsourced),
                "unsourced": {"total": len(unsourced),
                              "by_reason": {r: by_reason[r] for r in ("missing", "not-a-source", "url-only", "none")},
                              "notes": unsourced},
            },
            "compile": {"total": len(self.compiles),
                        "using_an_extract_note": sum(1 for p in self.compiles if self._to(p, self.extract_types))},
            "gaps": {
                "not_extracted": [row["path"] for row in rows if row["stage"] == "ingested"],
                "not_compiled": in_no_page,
            },
            "ignored": len(typed.ignored),
        }

    def note(self, path: str) -> dict[str, Any]:
        typed = self.typed
        note_type = typed.type_of(path)

        def described(paths: list[str]) -> list[dict[str, Any]]:
            return [{"path": p, "type": typed.type_of(p), "ignored": typed.is_ignored(p)} for p in paths]

        unresolved = [link for link in typed.links(path) if link.via != "metadata" and link.edge.status != "resolved"]
        if note_type in self.source_types:
            zettels_using = [{"path": z, "direct": True, "via": []} for z in self.extracted_from(path)]
            pages_using = [{**page, "direct": path in self.typed.note_links(page["path"])} for page in self.compiled_in(path)]
        else:
            zettels_using = [{"path": z, "direct": True, "via": []} for z in sorted(self._from(path, self.extract_types))]
            pages_using = [{"path": p, "direct": True, "via": []} for p in sorted(self._from(path, self.compile_types))]
        modified = None
        if typed.vault.root is not None:
            try:
                stamp = (Path(typed.vault.root) / path).stat().st_mtime
                modified = dt.datetime.fromtimestamp(stamp).astimezone().isoformat(timespec="seconds")
            except OSError:
                pass
        return {
            "format": {"name": typed.spec.name, "path": typed.spec.path},
            "path": path,
            "type": note_type,
            "ignored": typed.is_ignored(path),
            "stage": self.stage(path),
            "modified": modified,
            "rests_on": {
                "sources": [] if note_type in self.source_types else self.sources_under(path),
                "extract": [{"path": z, "direct": True, "via": []} for z in self._to(path, self.extract_types)],
            },
            "used_by": {"extract": zettels_using, "compile": pages_using},
            "linked_from": described(typed.linked_from(path)),
            "links_to": described(typed.note_links(path)),
            "links_to_nothing": [{"line": l.edge.link.line, "raw": l.edge.link.raw, "status": l.edge.status}
                                 for l in unresolved],
            "violations": [issue.to_dict() for issue in typed.issues(path)],
            "trust": self.trust(path),
            "roles": {"sources": sorted(self.source_types), "extract": sorted(self.extract_types),
                      "compile": sorted(self.compile_types)},
        }
