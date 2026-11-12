"""`vault next`: one queue of what needs doing, worked out from the vault on every run.

Each item has a kind, what it is about, why, and the exact command that resolves it (``<…>`` marks
the one thing only you can supply: what a new zettel says, where a source lives). Kinds:

* ``refresh``  a page whose zettels changed, or were linked or dropped, since its last refresh, or whose
               index line is out of date (from the engine's page record)   -> vault page refresh
* ``behind``   a zettel a page is behind on (two or more of its tags or links in common)
               -> vault page exclude (or link it in the page's text and refresh)
* ``merge``    two names for one idea (`vault findings`: same-idea, existing-name)  -> vault merge
* ``write``    a missing note many notes link to (`vault findings`: missing-note)    -> vault new
* ``extract``  a source no zettel links to                                          -> vault new (a zettel)
* ``page``     zettels in no page that share a tag, enough for a page               -> vault page new
* ``idea``     an open idea nothing links to                                        -> vault idea set … dropped

Ranking. Every item is ranked by the trust of what it is about: how much the vault relies on it,
worked out from use and never stored. For a note: log2(1 + the notes that link to it) + the pages that
cite it (for a source, also the pages that cite its zettels) + the ideas that cite it, not counting the
index and log notes the engine keeps; shown as the share of the vault's notes trusted less (a
percentile, `vault note --json`). A missing note's trust comes from the
notes that link to it, the same way. Ties: by kind (in the order above), then by name. So the queue is
the same on every run of the same vault.

Cutoffs are relative: findings use their percentile thresholds; sources no zettel links to, and
clusters of zettels in no page, are listed when their trust is at least the median of their kind (the
rest are counted, not listed). Pages and ideas are listed whenever they need something.
"""

from __future__ import annotations

import hashlib
import math
import shlex
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from vaultengine.formats.pages import Pages, shared_tags
from vaultengine.formats.pipeline import Pipeline
from vaultengine.formats.typed import TypedVault
from vaultengine.vault import link_name, lookup_key

KINDS = ("refresh", "behind", "merge", "write", "extract", "page", "idea")
ABOUT = {
    "refresh": "pages out of date with their zettels",
    "behind": "zettels a page is behind on",
    "merge": "two names for one idea",
    "write": "missing notes many notes link to",
    "extract": "sources no zettel links to",
    "page": "zettels in no page that share a tag",
    "idea": "open ideas nothing builds on",
}


class Trust:
    """How much the vault relies on each note, from use: links in, pages and ideas citing it."""

    def __init__(self, typed: TypedVault, pipeline: Pipeline) -> None:
        if hasattr(typed.graph, "prefetch"):
            typed.graph.prefetch()
        self.typed = typed
        self.pipeline = pipeline
        ideas = typed.spec.ideas.type if typed.spec.ideas else None
        pages_spec = typed.spec.pages
        engine = {lookup_key(pages_spec.index), lookup_key(pages_spec.log)} if pages_spec else set()
        self.facts: dict[str, dict[str, Any]] = {}
        for path in typed.vault.notes:
            if typed.is_ignored(path) or lookup_key(path) in engine:
                continue
            # The index and log notes the engine keeps link every page: not use, so not counted.
            linkers = [p for p in typed.linked_from(path) if lookup_key(p) not in engine and not typed.is_ignored(p)]
            pages = {p for p in linkers if typed.type_of(p) in pipeline.compile_types}
            if typed.type_of(path) in pipeline.source_types:
                pages |= {page["path"] for page in pipeline.compiled_in(path)}
            citing_ideas = [p for p in linkers if ideas and typed.type_of(p) == ideas]
            self.facts[path] = {"links_in": len(linkers), "pages": len(pages), "ideas": len(citing_ideas),
                                "raw": self.raw(len(linkers), len(pages), len(citing_ideas))}
        self._sorted = sorted(f["raw"] for f in self.facts.values())

    @staticmethod
    def raw(links_in: int, pages: int, ideas: int) -> float:
        return round(math.log2(1 + links_in) + pages + ideas, 6)

    def percentile(self, raw: float) -> int:
        """The share of the vault's notes (in %) that are trusted less: 0 for what nothing uses."""
        if not self._sorted:
            return 0
        return bisect_left(self._sorted, raw) * 100 // len(self._sorted)

    def of(self, path: str) -> dict[str, Any]:
        facts = self.facts.get(path, {"links_in": 0, "pages": 0, "ideas": 0, "raw": 0.0})
        return {**facts, "percentile": self.percentile(facts["raw"])}


@dataclass
class Item:
    kind: str
    subject: str  # a note path, or a missing note's name
    title: str
    why: str
    command: str
    trust: int  # percentile
    raw: float
    key: str = ""  # tie-breaker after kind and subject
    id: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _q(text: str) -> str:
    return shlex.quote(text)


class Queue:
    def __init__(self, typed: TypedVault, vault_arg: str, format_arg: str, findings=None) -> None:
        self.typed = typed
        self.vault_arg = vault_arg
        self.format_arg = format_arg
        self.pipeline = Pipeline(typed)
        self.pages = Pages(typed) if typed.spec.pages else None
        self.trust = Trust(typed, self.pipeline)
        self.findings = findings
        self.counts: dict[str, dict[str, int]] = {}

    def _cmd(self, *parts: str) -> str:
        return "vault " + " ".join(parts)

    def _item(self, kind: str, subject: str, title: str, why: str, command: str, raw: float, key: str = "") -> Item:
        digest = hashlib.sha256("\0".join([kind, lookup_key(subject), key]).encode("utf-8")).hexdigest()[:8]
        return Item(kind, subject, title, why, command, self.trust.percentile(raw), raw, key, f"{kind}-{digest}")

    # ----- the kinds --------------------------------------------------------------------------

    def refresh_and_behind(self) -> list[Item]:
        if self.pages is None:
            return []
        managed = [p for p in self.pages.pages() if self.pages.managed(p)]
        self.typed.prefetch(managed)  # the pages in a few batches, not one at a time (their zettels need no parsing here)
        items = []
        for page in managed:
            reasons = self.pages.record_staleness(page)
            if reasons:
                items.append(self._item(
                    "refresh", page, f"refresh {link_name(page)}", "; ".join(reasons),
                    self._cmd("page", "refresh", _q(self.vault_arg), _q(page), "-f", _q(self.format_arg), "--write"),
                    self.trust.of(page)["raw"]))
        for page, rows in self.pages.behind_all(managed).items():
            for row in rows:
                zettel = row["path"]
                common = ", ".join(row["tags"] + row["links"])
                items.append(self._item(
                    "behind", zettel, f"{link_name(page)} is behind on {link_name(zettel)}",
                    f"in common with the page: {common}; if it belongs, link it in the page's text and refresh, "
                    "else exclude it",
                    self._cmd("page", "exclude", _q(self.vault_arg), _q(page), _q(zettel), "-f", _q(self.format_arg), "--write"),
                    self.trust.of(zettel)["raw"], key=page))
        return items

    def merges_and_wanted(self) -> list[Item]:
        if self.findings is None:
            return []
        items = []
        for finding in self.findings.all(["same-idea", "existing-name", "missing-note"]):
            notes = finding.notes
            if finding.method == "missing-note":
                name = notes[0]["name"]
                linkers = finding.data["linking_notes"]
                items.append(self._item(
                    "write", name, f"write [[{name}]]", f"{len(linkers)} notes link to it and it does not exist",
                    self._write_command(name, linkers), self._raw_of_linkers(linkers)))
                continue
            survivor, other = (notes[1], notes[0]) if finding.method == "existing-name" else (notes[0], notes[1])
            survivor_ref = survivor["path"] or survivor["name"]
            raw = (self.trust.of(survivor["path"])["raw"] if survivor["path"]
                   else self._raw_of_linkers(finding.data["linking_notes"].get(survivor["name"], [])))
            why = finding.evidence[0]
            items.append(self._item(
                "merge", other["name"], f"merge [[{other['name']}]] into [[{survivor['name']}]]", why,
                self._cmd("merge", _q(self.vault_arg), _q(survivor_ref), _q(other["name"]), "-f", _q(self.format_arg), "--write"),
                raw, key=finding.id))
        return items

    def _raw_of_linkers(self, linkers: list[str]) -> float:
        ideas = self.typed.spec.ideas.type if self.typed.spec.ideas else None
        pages = sum(1 for p in linkers if self.typed.type_of(p) in self.pipeline.compile_types)
        idea_count = sum(1 for p in linkers if ideas and self.typed.type_of(p) == ideas)
        return Trust.raw(len(linkers), pages, idea_count)

    def _write_command(self, name: str, linkers: list[str]) -> str:
        """A zettel (or, when most links to it are citations, a source) with the name links use."""
        typed = self.typed
        coverage = typed.spec.coverage
        roles = set(coverage.citation_roles) if coverage else set()
        citing = total = 0
        for linker in linkers:
            for link in typed.links(linker):
                if lookup_key(link.edge.link.target.strip()) == lookup_key(name):
                    total += 1
                    citing += link.role in roles
        as_source = total and citing * 2 > total
        tags_key = self.pages.config.tags if self.pages else "Tags"
        tags = shared_tags(typed, tags_key, [p for p in linkers if p in typed.vault.notes]) or "<tags>"
        parts = ["new", _q(self.vault_arg), "-f", _q(self.format_arg)]
        if as_source:
            source_type = sorted(self.pipeline.source_types)[0]
            parts += ["--type", source_type, "--name", _q(name), "--title", _q(name), "--set", _q(f"Tags={tags}"),
                      "--set", _q("Source=<where the original is>")]
        else:
            zettel_type = sorted(self.pipeline.extract_types)[0]
            sources = Counter(s for p in linkers for s in typed.note_links(p) if typed.type_of(s) in self.pipeline.source_types)
            best = sorted(sources, key=lambda s: (-sources[s], s))[0] if sources else None
            body = f"From [[{link_name(best)}]]." if best else "From [[<a source note>]]."
            parts += ["--type", zettel_type, "--name", _q(name), "--title", _q(name), "--set", _q(f"Tags={tags}"),
                      "--body", _q(body)]
        return self._cmd(*parts)

    def extract(self) -> list[Item]:
        typed = self.typed
        unextracted = [s for s in self.pipeline.sources if not self.pipeline.extracted_from(s)]
        ranked = sorted(unextracted, key=lambda s: (-self.trust.of(s)["raw"], s))
        cut = _median_cut([self.trust.of(s)["raw"] for s in ranked])
        shown = [s for s in ranked if self.trust.of(s)["raw"] >= cut]
        self.counts["extract"] = {"listed": len(shown), "all": len(unextracted)}
        tags_key = self.pages.config.tags if self.pages else "Tags"
        zettel_type = sorted(self.pipeline.extract_types)[0]
        items = []
        for source in shown:
            tags = shared_tags(typed, tags_key, [source]) or "<tags>"
            facts = self.trust.of(source)
            items.append(self._item(
                "extract", source, f"extract a zettel from {link_name(source)}",
                f"no zettel links to it; {facts['links_in']} notes link to it",
                self._cmd("new", _q(self.vault_arg), "-f", _q(self.format_arg), "--type", zettel_type,
                          "--title", _q("<what the zettel says>"), "--set", _q(f"Tags={tags}"),
                          "--body", _q(f"From [[{link_name(source)}]].")),
                facts["raw"]))
        return items

    def page_clusters(self) -> list[Item]:
        if self.pages is None:
            return []
        typed = self.typed
        in_no_page = [z for z in self.pipeline.extracts if not self.pipeline._from(z, self.pipeline.compile_types)]
        if hasattr(typed.vault.notes, "prefetch"):
            typed.vault.notes.prefetch(in_no_page)  # their tags, in batches rather than one note at a time
        page_type = typed.spec.type_named(sorted(self.pipeline.compile_types)[0])
        minimum = max([m.count for m in page_type.min_links if m.type in self.pipeline.extract_types] or [1])
        counts: Counter[str] = Counter()
        names: dict[str, str] = {}
        tags_of = {}
        for zettel in in_no_page:
            tags_of[zettel] = [(k, n) for k, n, _ in self.pages.tags(zettel)]
            for key, name in tags_of[zettel]:
                counts[key] += 1
                names.setdefault(key, name)
        groups: dict[str, list[str]] = {}
        for zettel in in_no_page:
            keys = [k for k, _ in tags_of[zettel]]
            if keys:
                best = max(keys, key=lambda k: (counts[k], -keys.index(k)))
                groups.setdefault(best, []).append(zettel)
        clusters = {k: v for k, v in groups.items() if len(v) >= minimum}
        raws = {k: sum(self.trust.of(z)["raw"] for z in v) / len(v) for k, v in clusters.items()}
        cut = _median_cut(list(raws.values()))
        items = []
        shown = [k for k in clusters if raws[k] >= cut]
        self.counts["page"] = {"listed": len(shown), "all": len(clusters), "zettels_in_no_page": len(in_no_page)}
        for key in shown:
            zettels = sorted(clusters[key], key=lambda z: (-self.trust.of(z)["raw"], z))
            chosen = zettels[:25]
            more = f" (the 25 most trusted of {len(zettels)})" if len(zettels) > 25 else ""
            links = " ".join(f"[[{typed.link_text(z)}]]" for z in chosen)
            items.append(self._item(
                "page", names[key], f"a page for {len(zettels)} zettels tagged {names[key]}",
                f"{len(zettels)} zettels in no page share the tag {names[key]}{more}",
                self._cmd("page", "new", _q(self.vault_arg), _q(names[key]), "--from", _q(links), "-f", _q(self.format_arg),
                          "--write"),
                raws[key]))
        return items

    def open_ideas(self) -> list[Item]:
        typed = self.typed
        config = typed.spec.ideas
        if config is None:
            return []
        items = []
        for idea in typed.notes_of_type(config.type):
            status = typed.metadata(idea).get(config.status)
            if status is None or [v.lower() for v in status.values] != ["open"]:
                continue
            builders = [p for p in typed.linked_from(idea) if typed.type_of(p) not in (config.type, None)
                        and not self._engine_note(p)]
            if builders:
                continue
            items.append(self._item(
                "idea", idea, f"open idea {link_name(idea)}", "open, and no note links to it: write on it, or drop it",
                self._cmd("idea", "set", _q(self.vault_arg), _q(idea), "dropped", "-f", _q(self.format_arg), "--write"),
                self.trust.of(idea)["raw"]))
        return items

    def _engine_note(self, path: str) -> bool:
        pages = self.typed.spec.pages
        return pages is not None and lookup_key(path) in {lookup_key(pages.index), lookup_key(pages.log)}

    # ----- the queue ----------------------------------------------------------------------------

    def items(self) -> list[Item]:
        items = self.refresh_and_behind() + self.merges_and_wanted() + self.extract() + self.page_clusters() + self.open_ideas()
        order = {kind: i for i, kind in enumerate(KINDS)}
        items.sort(key=lambda it: (-it.trust, -it.raw, order[it.kind], lookup_key(it.subject), it.key))
        for kind in KINDS:
            self.counts.setdefault(kind, {"listed": sum(1 for it in items if it.kind == kind)})
            self.counts[kind].setdefault("all", self.counts[kind]["listed"])
        return items


def _median_cut(values: list[float]) -> float:
    """The lower median: at least half the values are at or above it."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]
