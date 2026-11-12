"""Findings: what the engine discovered in the vault, each with a stable id.

Worked out from the index on every run (nothing is stored, nothing is written), with
``suggest.py`` and ``connect.py``:

* ``missing-note``: a note that does not exist, linked from at least as many notes as the 90th
  percentile of all missing notes (never fewer than 2);
* ``same-idea``: two missing notes whose names say the same thing (one name is the other plus
  at most two words, ``Suggester.same_idea``);
* ``existing-name``: a missing note whose name is probably an existing note's
  (``Suggester.close_to_existing``), linked from at least the 75th percentile of missing notes (and 2);
* ``link``: two notes that are not linked but probably belong linked, with structural evidence
  (shared neighbours or cited together), scoring at least the 99th percentile of such pairs (and
  ``engine.findings.link_floor``), at most ``engine.findings.link_limit`` of them, best first (``LinkSuggester``).

Thresholds are relative to the vault: nearest-rank percentiles of what it has, with a floor, so they
move with the vault and are the same on every run of the same vault (``Findings.thresholds`` says
what they came to). They never depend on how one finding ranks against another.

The id is derived from what the finding is, its method and the notes it is about, not from
its evidence or rank: ``<prefix>-<first 10 hex of sha256(method, sorted note keys)>``. The same
finding has the same id on every run, and edits elsewhere (or new evidence for it) do not change
it. Notes are compared by lookup key (case and Unicode form do not matter); an existing note by its
path, a missing one by its link name.

With a format, some notes take no part (``skip_for``): notes in its ignored folders, ideas (an
idea links the notes of the finding it came from; it must not turn into findings of its own), and
the index and log notes `vault page` keeps. They are not subjects of findings, and links from them
do not count as linking notes. Links in metadata values (``Tags: [[x]]``) do not count either
(``counts_for``): under a format they file a note rather than link it, so a tag nobody has written a
note for is not a missing note, and sharing a tag is not a shared neighbour.

Scores and thresholds are relative (a neighbour shared by many notes weighs less, ``connect.py``; a
percentile moves with the vault), so an edit can move a borderline finding across its threshold. Its
id never changes; whether it is listed can.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from vaultengine.connect import LinkSuggester
from vaultengine.suggest import Suggester, name_words
from vaultengine.vault import link_name, lookup_key

# Thresholds are relative to the vault: a nearest-rank percentile of what the vault has, never below a floor.
MISSING_PERCENTILE, MISSING_FLOOR = 90, 2  # linking notes of a missing note, among all missing notes
EXISTING_PERCENTILE, EXISTING_FLOOR = 75, 2  # linking notes of a name that is probably an existing note
LINK_PERCENTILE = 99  # score of a link, among pairs with structural evidence (floor and cap: engine.findings)
METHODS = {
    "missing-note": "a note many notes link to that does not exist",
    "same-idea": "two names for the same idea",
    "existing-name": "a link name that is probably an existing note",
    "link": "two notes that probably belong linked",
}
PREFIX = {"missing-note": "missing", "same-idea": "same", "existing-name": "name", "link": "link"}


def _notes(count: int) -> str:
    return f"{count} note" if count == 1 else f"{count} notes"


def skip_for(spec) -> Callable[[str], bool]:
    """Notes that take no part in findings under this format (see the module doc)."""
    types = set()
    if spec.ideas is not None:
        types.add(spec.ideas.type)
    if spec.pages is not None:
        for path in (spec.pages.index, spec.pages.log):
            note_type = spec.type_for(path)
            if note_type is not None and note_type.name not in spec.pipeline.compile:
                types.add(note_type.name)

    def skip(path: str) -> bool:
        if spec.is_ignored(path):
            return True
        note_type = spec.type_for(path)
        return note_type is not None and note_type.name in types

    return skip


def counts_for(spec) -> Callable[[str, str | None], bool]:
    """Which links count as evidence under this format: not links from notes that take no part
    (``skip_for``), and not links in metadata values (``Tags: [[x]]`` files a note, it does not link it;
    the same rule as ``TypedVault.note_links``)."""
    skip = skip_for(spec)
    known: dict[str, tuple[bool, bool]] = {}  # path -> (skipped, header links are metadata): once per note, not per link

    def counts(source: str, field_kind: str | None) -> bool:
        if source not in known:
            note_type = spec.type_for(source)
            known[source] = (skip(source), note_type is not None and note_type.metadata_source != "frontmatter")
        skipped, metadata = known[source]
        return not skipped and not (field_kind == "header" and metadata)

    return counts


def finding_id(method: str, keys: list[str]) -> str:
    digest = hashlib.sha256("\0".join([method, *sorted(lookup_key(k) for k in keys)]).encode("utf-8")).hexdigest()
    return f"{PREFIX[method]}-{digest[:10]}"


@dataclass
class Finding:
    id: str
    method: str
    title: str
    notes: list[dict[str, Any]]  # {"name", "path" (None for a missing note), "linking_notes"}
    evidence: list[str]  # lines, for people
    score: float  # for ordering within a method
    data: dict[str, Any] = field(default_factory=dict)  # machine-readable evidence

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "about": METHODS[self.method]}


class Findings:
    def __init__(self, conn: sqlite3.Connection, ignored: Callable[[str], bool] | None = None,
                 counts: Callable[[str, str | None], bool] | None = None, engine=None, settings=None) -> None:
        """``engine``: the ``Relatedness`` link findings score with (its embedding provider and tag rule);
        default: one built on first use from the configured settings. ``settings``: ``FindingsSettings``
        (the link floor and cap); default: their defaults."""
        from vaultengine.settings import FindingsSettings

        self.conn = conn
        self.settings = settings or FindingsSettings()
        self.ignored = ignored or (lambda path: False)
        self.counts = counts
        self.suggester = Suggester(conn, engine, counts=counts)
        self.paths = self.suggester.paths  # file id -> path (without building the link graph or embeddings)
        self.thresholds: dict[str, dict[str, Any]] = {}
        self._linker_counts: list[int] | None = None

    # ----- helpers ----------------------------------------------------------------------------

    def _linkers(self, key: str) -> list[str]:
        """Notes (not ignored) that link to the missing note ``key``."""
        return sorted(p for p in (self.paths[f] for f in self.suggester.linkers.get(key, ())) if not self.ignored(p))

    def _threshold(self, name: str, percentile: int, floor: int) -> int:
        """At least the nearest-rank ``percentile`` of the linking-note counts of every missing note, and ``floor``."""
        if self._linker_counts is None:
            self._linker_counts = sorted(c for c in (len(self._linkers(k)) for k in self.suggester.linkers) if c)
        counts = self._linker_counts
        value = counts[max(1, -(-len(counts) * percentile // 100)) - 1] if counts else floor
        threshold = max(floor, value)
        self.thresholds[name] = {"linking_notes": threshold, "percentile": percentile, "floor": floor,
                                 "percentile_value": value}
        return threshold

    def _missing(self, key: str, linkers: list[str]) -> dict[str, Any]:
        return {"name": self.suggester.name(key), "path": None, "linking_notes": len(linkers)}

    # ----- the methods ------------------------------------------------------------------------

    def missing_notes(self) -> list[Finding]:
        found = []
        minimum = self._threshold("missing-note", MISSING_PERCENTILE, MISSING_FLOOR)
        for key in self.suggester.linkers:
            linkers = self._linkers(key)
            if len(linkers) < minimum:
                continue
            name = self.suggester.name(key)
            folders: dict[str, int] = {}
            for path in linkers:
                folder = path.rsplit("/", 1)[0] if "/" in path else ""
                folders[folder] = folders.get(folder, 0) + 1
            top = sorted(folders.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            found.append(Finding(
                finding_id("missing-note", [name]), "missing-note", f"[[{name}]] does not exist; {len(linkers)} notes link to it",
                [self._missing(key, linkers)],
                [f"linked from {len(linkers)} notes: " + ", ".join(link_name(p) for p in linkers[:5])
                 + (f" +{len(linkers) - 5}" if len(linkers) > 5 else ""),
                 "mostly in " + ", ".join(f"{folder or '(root)'}/ ({n})" for folder, n in top)],
                float(len(linkers)),
                {"key": key, "linking_notes": linkers},
            ))
        return found

    def same_ideas(self) -> list[Finding]:
        found = []
        for group in self.suggester.same_idea():
            core_name = group[0][0]
            core_key = lookup_key(core_name)
            core_linkers = self._linkers(core_key)
            for name, _ in group[1:]:
                key = lookup_key(name)
                linkers = self._linkers(key)
                if not core_linkers or not linkers:
                    continue
                extra = sorted(name_words(name) - name_words(core_name))
                found.append(Finding(
                    finding_id("same-idea", [core_name, name]), "same-idea",
                    f"[[{core_name}]] and [[{name}]] may be one idea under two names",
                    [self._missing(core_key, core_linkers), self._missing(key, linkers)],
                    [f"neither note exists; [[{core_name}]] is linked from {_notes(len(core_linkers))}, "
                     f"[[{name}]] from {len(linkers)}",
                     f"[[{name}]] is [[{core_name}]] plus: {', '.join(extra)}" if extra
                     else "the same words, spelled differently"],
                    float(len(core_linkers) + len(linkers)),
                    {"linking_notes": {core_name: core_linkers, name: linkers}, "extra_words": extra},
                ))
        return found

    def existing_names(self) -> list[Finding]:
        found = []
        minimum = self._threshold("existing-name", EXISTING_PERCENTILE, EXISTING_FLOOR)
        for name, _, existing in self.suggester.close_to_existing(min_links=1):
            linkers = self._linkers(lookup_key(name))
            if self.ignored(existing) or len(linkers) < minimum:
                continue
            found.append(Finding(
                finding_id("existing-name", [name, existing]), "existing-name",
                f"[[{name}]] is probably [[{link_name(existing)}]]",
                [self._missing(lookup_key(name), linkers), {"name": link_name(existing), "path": existing, "linking_notes": None}],
                [f"[[{name}]] does not exist; {_notes(len(linkers))} link to it: "
                 + ", ".join(link_name(p) for p in linkers[:4]) + (f" +{len(linkers) - 4}" if len(linkers) > 4 else ""),
                 f"its words are all in the name of {existing}"],
                float(len(linkers)),
                {"linking_notes": linkers, "existing": existing},
            ))
        return found

    def links(self) -> list[Finding]:
        suggester = LinkSuggester(self.conn, self.suggester.engine, self.counts)
        found = []
        floor, cap = self.settings.link_floor, self.settings.link_limit
        ranked = suggester.suggestions(limit=cap, min_score=floor, structural_percentile=LINK_PERCENTILE)
        self.thresholds["link"] = {"score": round(suggester.threshold or floor, 4), "percentile": LINK_PERCENTILE,
                                   "floor": floor, "limit": cap, "capped": len(ranked) >= cap}
        for s in ranked:
            if not (s.shared_with or s.cited_by) or self.ignored(s.a) or self.ignored(s.b):
                continue
            a, b = link_name(s.a), link_name(s.b)
            evidence = []
            if s.shared_with:
                evidence.append("both link to: " + ", ".join(s.shared_with[:4]) + (f" +{len(s.shared_with) - 4}" if len(s.shared_with) > 4 else ""))
            if s.cited_by:
                evidence.append("cited together by: " + ", ".join(s.cited_by[:4]) + (f" +{len(s.cited_by) - 4}" if len(s.cited_by) > 4 else ""))
            if s.terms:
                evidence.append("similar text: " + ", ".join(s.terms))
            if s.shared_tags:
                evidence.append("shared tags: " + ", ".join(s.shared_tags[:4]) + (f" +{len(s.shared_tags) - 4}" if len(s.shared_tags) > 4 else ""))
            evidence.append(f"score {s.score:.2f} (shared neighbours {s.shared:.2f}, cited together {s.cited:.2f}, "
                            f"tags {s.tags:.2f}, text {s.text:.2f})")
            found.append(Finding(
                finding_id("link", [s.a, s.b]), "link", f"[[{a}]] and [[{b}]] are not linked but probably belong together",
                [{"name": a, "path": s.a, "linking_notes": None}, {"name": b, "path": s.b, "linking_notes": None}],
                evidence, s.score,
                {"score": s.score, "shared": s.shared, "cited": s.cited, "tags": s.tags, "text": s.text,
                 "shared_with": s.shared_with, "cited_by": s.cited_by, "shared_tags": s.shared_tags, "terms": s.terms},
            ))
        return found

    def all(self, methods: list[str] | None = None) -> list[Finding]:
        run = {"missing-note": self.missing_notes, "same-idea": self.same_ideas,
               "existing-name": self.existing_names, "link": self.links}
        found = []
        for method in methods or list(run):
            items = run[method]()
            items.sort(key=lambda f: (-f.score, f.id))
            found.extend(items)
        return found

    def get(self, finding_id_: str) -> Finding | None:
        method = next((m for m, prefix in PREFIX.items() if finding_id_.startswith(prefix + "-")), None)
        if method is None:
            return None
        return next((f for f in self.all([method]) if f.id == finding_id_), None)
