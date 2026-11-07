"""Notes related to a note (or to a missing note), from links, tags and text, each with a reason.

Three signals, each scaled to 0..1 by its largest value among the candidates:

* graph - personalized PageRank from the note over the link graph, missing
          notes included (``graphrank.py``), divided by each note's degree so
          hubs do not come first for everything;
* text  - "more like this": the note's 30 most distinctive terms (tf of the
          term, weighted by field as in search, sublinear: 1 + ln tf, times
          idf = ln(N / df); only terms of 2+ characters in 2 to N/2 notes) are run
          as a query, each weighted by its share, and every note is scored with the
          same BM25F as ``vault search``. The tags field is left out (tags are
          the third signal). Words of template headings (a heading
          written the same way in 3 or more notes, like "Related zettels") do not
          count: they say which template a note uses, not what it is about;
* tags  - weighted Jaccard of the two notes' tag sets (nested tags count their
          parents too), each tag weighted by ln(N / notes with the tag). Tags on
          more than a third of the notes (status markers like #baby) are left out:
          sharing them says nothing.

score = 0.5 * graph + 0.3 * text + 0.2 * tags.

Each result says why, in terms that can be checked by opening the notes:
"links here", "2 hops via kl-penalty-in-rlhf", "shares #ai-safety",
"similar text (kl, divergence, ppo)". A reason is only given when it is a fact
worth reading: hops up to 3, tags that are on at most a third of the notes,
text when its signal is at least ``TEXT_REASON`` of the best match.
Results that have no link to or from the note are marked "not linked".

A missing note (a link target with no note) can be the seed too: its graph node
is the missing note itself, its text is what the notes linking to it are about,
and its tags are those most of them share.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from vaultengine.graphrank import Graph, build_graph, hop_paths, personalized_pagerank
from vaultengine.querylang import QueryError
from vaultengine.search import FIELD_WEIGHTS, _bm25f, _Evaluator, _Searcher, idf
from vaultengine.tokenize import tokenize
from vaultengine.vault import link_name, lookup_key

WEIGHTS = {"graph": 0.5, "text": 0.3, "tags": 0.2}
PROFILE_TERMS = 30
TEXT_REASON = 0.25
COMMON_TAG = 1 / 3  # tags on more of the notes than this are not given as a reason


@dataclass
class Seed:
    label: str  # note title, or the missing target as written
    path: str | None  # None for a missing note
    node: int
    terms: dict[str, float]  # term -> weight, max 1
    tags: set[str]
    linkers: list[str] = field(default_factory=list)  # for a missing note: notes that link to it


@dataclass
class RelatedHit:
    path: str
    title: str
    score: float
    graph: float
    text: float
    tags: float
    linked: bool  # a resolved link between the two notes, either way
    hops: int | None
    reason: str
    shared_tags: list[str] = field(default_factory=list)
    shared_terms: list[str] = field(default_factory=list)


@dataclass
class RelatedResult:
    seed: Seed
    candidates: int  # notes with any signal
    hits: list[RelatedHit]
    compared: int = 0  # notes considered (all but the seed)

    def to_dict(self) -> dict:
        return asdict(self)


class Relatedness:
    """Everything ``related`` needs, loaded once; call ``seed`` then ``related``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.graph: Graph = build_graph(conn)
        self.searcher = _Searcher(conn, fuzzy=False)
        self.n_notes = self.searcher.n_notes
        self.paths = {row[0]: row[1] for row in conn.execute("SELECT id, path FROM files WHERE kind = 'note'")}
        self._profiles: dict[int, dict[str, float]] = {}
        self.df = dict(conn.execute("SELECT term, df FROM terms"))
        self.note_tags: dict[int, set[str]] = defaultdict(set)
        for file_id, name_key in conn.execute("SELECT file_id, name_key FROM tags"):
            parts = name_key.split("/")
            self.note_tags[file_id].update("/".join(parts[:depth]) for depth in range(1, len(parts) + 1))
        tag_notes = Counter(tag for tags in self.note_tags.values() for tag in tags)
        self.tag_idf = {tag: math.log(self.n_notes / n) for tag, n in tag_notes.items()}
        self.tag_notes = tag_notes
        for tags in self.note_tags.values():  # tags on most notes carry no signal
            tags -= {tag for tag in tags if tag_notes[tag] > COMMON_TAG * self.n_notes}
        headings = defaultdict(set)
        for file_id, text in conn.execute("SELECT file_id, text FROM headings"):
            headings[file_id].add(text)
        template_count = Counter(lookup_key(text) for texts in headings.values() for text in texts)
        self.template_terms: dict[int, Counter[str]] = {
            file_id: Counter(term for text in texts if template_count[lookup_key(text)] >= 3 for term in tokenize(text))
            for file_id, texts in headings.items()
        }

    # -- seeds --

    def term_profile(self, file_id: int) -> dict[str, float]:
        if file_id not in self._profiles:
            self._profiles[file_id] = self._term_profile(file_id)
        return self._profiles[file_id]

    def all_profiles(self) -> dict[int, dict[str, float]]:
        """Term profiles of every note, read from the index in one pass (and cached)."""
        missing = set(self.paths) - set(self._profiles)
        if missing:
            weighted: dict[int, Counter[str]] = defaultdict(Counter)
            for file_id, term, name, tf in self.conn.execute("SELECT file_id, term, field, tf FROM postings WHERE field != 'tags'"):
                if file_id in missing:
                    self._add(weighted[file_id], file_id, term, name, tf)
            for file_id in missing:
                self._profiles[file_id] = self._score(weighted.get(file_id, Counter()))
        return {f: self._profiles[f] for f in self.paths}

    def _add(self, weighted: Counter[str], file_id: int, term: str, name: str, tf: int) -> None:
        if name == "headings":
            tf -= self.template_terms.get(file_id, Counter())[term]
        if tf > 0:
            weighted[term] += FIELD_WEIGHTS[name] * tf

    def _term_profile(self, file_id: int) -> dict[str, float]:
        weighted: Counter[str] = Counter()
        rows = self.conn.execute("SELECT term, field, tf FROM postings WHERE file_id = ? AND field != 'tags'", (file_id,))
        for term, name, tf in rows:  # tags have their own signal
            self._add(weighted, file_id, term, name, tf)
        return self._score(weighted)

    def _score(self, weighted: Counter[str]) -> dict[str, float]:
        scores = {}
        for term, tf in weighted.items():
            df = self.df.get(term, 0)
            if 2 <= df <= self.n_notes / 2 and len(term) >= 2 and any(char.isalpha() for char in term):
                scores[term] = (1 + math.log(tf)) * math.log(self.n_notes / df)
        return scores

    @staticmethod
    def _top(scores: dict[str, float]) -> dict[str, float]:
        top = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:PROFILE_TERMS]
        best = top[0][1] if top else 1.0
        return {term: weight / best for term, weight in top}

    def seed(self, ref: str) -> Seed:
        """The seed for a note reference (name, path or alias, like a link) or a missing link target."""
        evaluator = _Evaluator(self.conn, self.searcher)
        file_id = evaluator.resolve(ref, (0, len(ref)))
        if file_id is not None and file_id in self.paths:
            path = self.paths[file_id]
            return Seed(link_name(path), path, self.graph.by_file[file_id], self._top(self.term_profile(file_id)),
                        set(self.note_tags.get(file_id, set())))
        node = self.graph.by_missing.get(lookup_key(ref))
        if node is None:
            raise QueryError(f'no note or link target called "{ref}"', None, None,
                             "give a note name, path or alias, or the target of a link to a missing note")
        linker_nodes = sorted(other for other in self.graph.adjacency[node] if not self.graph.is_missing(other))
        linkers = [self.graph.note_ids[n] for n in linker_nodes]
        combined: Counter[str] = Counter()
        for linker in linkers:
            for term, weight in self._top(self.term_profile(linker)).items():
                combined[term] += weight
        tag_counts = Counter(tag for linker in linkers for tag in self.note_tags.get(linker, set()))
        shared = {tag for tag, n in tag_counts.items() if n >= 2 and n >= 0.3 * len(linkers)}
        return Seed(self.graph.labels[node], None, node, self._top(dict(combined)), shared,
                    [self.paths[linker] for linker in linkers])

    # -- scores --

    def text_scores(self, seed: Seed) -> dict[int, tuple[float, list[str]]]:
        """note id -> (score, terms that contributed most)."""
        if not seed.terms:
            return {}
        per_note: dict[int, dict[str, dict[str, tuple[int, int]]]] = defaultdict(lambda: defaultdict(dict))
        for file_id, term, name, tf, length in self.searcher.rows(seed.terms):
            per_note[file_id][term][name] = (tf, length)
        scores = {}
        for file_id, terms in per_note.items():
            contributions = {}
            for term, fields in terms.items():
                _, score, _ = _bm25f(fields, self.searcher.avg_lengths, idf(self.n_notes, self.df.get(term, 0)))
                contributions[term] = seed.terms[term] * score
            ranked = sorted(contributions, key=lambda t: -contributions[t])
            scores[file_id] = (sum(contributions.values()), ranked[:3])
        return scores

    def tag_score(self, a: set[str], b: set[str]) -> float:
        union = sum(self.tag_idf.get(t, 0.0) for t in a | b)
        return sum(self.tag_idf.get(t, 0.0) for t in a & b) / union if union else 0.0

    def related(self, seed: Seed, limit: int | None = 10, unlinked_only: bool = False) -> RelatedResult:
        graph = self.graph
        seed_file = graph.note_ids[seed.node]
        rank = personalized_pagerank(graph.adjacency, seed.node)
        graph_raw = {graph.note_ids[n]: r / graph.degree(n) for n, r in rank.items()
                     if n != seed.node and not graph.is_missing(n) and graph.degree(n)}
        text_raw = self.text_scores(seed)
        text_raw.pop(seed_file, None)
        tags_raw = {f: self.tag_score(seed.tags, tags) for f, tags in self.note_tags.items() if f != seed_file}
        tags_raw = {f: v for f, v in tags_raw.items() if v > 0}

        def scaled(values: dict[int, float]) -> dict[int, float]:
            top = max(values.values(), default=0.0)
            return {k: v / top for k, v in values.items()} if top > 0 else {}

        g = scaled({k: v for k, v in graph_raw.items() if v > 0})
        x = scaled({k: v[0] for k, v in text_raw.items()})
        t = scaled(tags_raw)
        paths = hop_paths(graph, seed.node)
        hits = []
        for file_id in set(g) | set(x) | set(t):
            node = graph.by_file[file_id]
            linked = (seed.node, node) in graph.direct or (node, seed.node) in graph.direct
            if graph.is_missing(seed.node):  # a note linking to the missing note
                linked = seed.node in graph.adjacency[node]
            if unlinked_only and linked:
                continue
            path = self.paths[file_id]
            shared_tags = sorted(
                (tag for tag in seed.tags & self.note_tags.get(file_id, set())
                 if self.tag_notes[tag] <= COMMON_TAG * self.n_notes and not any(o.startswith(tag + "/") for o in seed.tags & self.note_tags[file_id])),
                key=lambda tag: (-self.tag_idf[tag], tag),
            )
            shared_terms = text_raw.get(file_id, (0, []))[1]
            hit = RelatedHit(
                path=path, title=link_name(path),
                score=WEIGHTS["graph"] * g.get(file_id, 0) + WEIGHTS["text"] * x.get(file_id, 0) + WEIGHTS["tags"] * t.get(file_id, 0),
                graph=g.get(file_id, 0.0), text=x.get(file_id, 0.0), tags=t.get(file_id, 0.0),
                linked=linked, hops=paths[node].hops if node in paths else None, reason="",
                shared_tags=shared_tags[:3], shared_terms=shared_terms if x.get(file_id, 0) >= TEXT_REASON else [],
            )
            hit.reason = self.reason(seed, node, hit, paths)
            hits.append(hit)
        hits.sort(key=lambda h: (-h.score, h.path))
        compared = self.n_notes - (0 if seed_file is None else 1)
        return RelatedResult(seed, len(hits), hits if limit is None else hits[:limit], compared)

    def reason(self, seed: Seed, node: int, hit: RelatedHit, paths) -> str:
        graph = self.graph
        parts = []
        out_link = (seed.node, node) in graph.direct
        in_link = (node, seed.node) in graph.direct
        if out_link and in_link:
            parts.append("linked both ways")
        elif out_link:
            parts.append("linked from it")
        elif in_link:
            parts.append("links to it")
        elif node in paths and paths[node].hops <= 3:
            path = paths[node]
            if graph.is_missing(seed.node) and path.hops == 1:
                parts.append("links to it")
            else:
                more = f" +{len(path.via) - 1}" if len(path.via) > 1 else ""
                via = f" via {self._node_name(path.via[0])}{more}" if path.via else ""
                parts.append(f"{path.hops} hops{via}")
        if hit.shared_tags:
            parts.append("shares " + ", ".join("#" + tag for tag in hit.shared_tags))
        if hit.shared_terms:
            parts.append("similar text (" + ", ".join(hit.shared_terms) + ")")
        return " · ".join(parts)

    def _node_name(self, node: int) -> str:
        name = self.graph.labels[node] if self.graph.is_missing(node) else link_name(self.graph.labels[node])
        if len(name) > 40:
            name = name[:39].rstrip() + "…"
        return f"{name} (no note yet)" if self.graph.is_missing(node) else name


def related(conn: sqlite3.Connection, ref: str, limit: int | None = 10, unlinked_only: bool = False) -> RelatedResult:
    engine = Relatedness(conn)
    return engine.related(engine.seed(ref), limit=limit, unlinked_only=unlinked_only)
