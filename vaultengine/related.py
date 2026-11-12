"""Notes related to a note (or to a missing note), from links, tags and text, each with a reason.

Three signals, each scaled to 0..1 by its largest value among the candidates:

* graph - personalized PageRank from the note over the link graph, missing
          notes included (``graphrank.py``), divided by each note's degree so
          hubs do not come first for everything;
* text  - the cosine of the two notes' embeddings (``embedding/``: the configured
          provider, vectors cached in the index), centered: the mean vector of the
          vault's notes is subtracted from both first. Without that, a static model
          puts every pair of notes in one vault at a cosine of 0.5 to 0.9 (on the
          sample vault: median 0.65, lowest 5% under 0.49), and every note would look
          somewhat similar to every other; centered, the median pair is about 0 and only
          notes closer than the vault's average get a text signal (negatives count as 0);
* tags  - weighted Jaccard of the two notes' tag sets (nested tags count their
          parents too), each tag weighted by ln(N / notes with the tag). Tags on
          more than a third of the notes (status markers like #baby) are left out:
          sharing them says nothing. A note's tags are its #tags and what its tag
          links point to (``Tags: [[reward]]`` is the tag [[reward]]; ``tagrule.py``).

score = 0.5 * graph + 0.3 * text + 0.2 * tags, the default of ``engine.related.weights``. Two more
signals can be weighted in there: cocitation and coupling (``graphrank.Graph``: notes that link to
both, what both link to), each scaled the same way; ``vault experiment`` measures which mix
recovers held-out links best.

When the embedding model cannot be loaded (missing or broken files), only the text
signal is switched off: the result says so and why, and the other weights are rescaled
to sum to 1 (5/7 graph, 2/7 tags), which keeps the same order the two would give anyway.
Vectors not cached in the index are computed in memory for the run and reported, never
written (``vault embed`` caches them).

Each result says why, in terms that can be checked by opening the notes:
"links here", "2 hops via kl-penalty-in-rlhf", "shares #ai-safety",
"similar text (kl, divergence, ppo)". A reason is only given when it is a fact
worth reading: hops up to 3, tags that are on at most a third of the notes,
text when its signal is at least ``TEXT_REASON`` of the best match. The words in
"similar text (...)" are words in both notes that add the most to their cosine,
read off the embedding (``embedding/explain.py``); with a static model the reason
needs at least one such word (two near-empty notes are not "similar text"), a model
that cannot name words gives a bare "similar text". Results
that have no link to or from the note are marked "not linked".

A missing note (a link target with no note) can be the seed too: its graph node
is the missing note itself, its vector is the mean of the vectors of the notes
linking to it, and its tags are those most of them share.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vaultengine.graphrank import Graph, build_graph, hop_paths
from vaultengine.querylang import QueryError
from vaultengine.search import _Evaluator, _Searcher
from vaultengine.vault import link_name, lookup_key

if TYPE_CHECKING:
    import numpy as np

    from vaultengine.embedding import EmbeddingProvider
    from vaultengine.embedding.explain import WordBag
    from vaultengine.embedding.store import NoteVectors

WEIGHTS = {"graph": 0.5, "cocitation": 0.0, "coupling": 0.0, "tags": 0.2, "text": 0.3}  # engine.related.weights
# engine.related.unlinked_weights: for what is not linked yet (related --unlinked, discover, explore); measured by
# `vault experiment` on hidden links (docs/experiments/2026-11-10.md)
UNLINKED_WEIGHTS = {"graph": 0.1, "cocitation": 0.05, "coupling": 0.05, "tags": 0.1, "text": 0.7}
SIGNALS = ("graph", "cocitation", "coupling", "tags", "text")
TEXT_REASON = 0.25
COMMON_TAG = 1 / 3  # tags on more of the notes than this are not given as a reason


@dataclass
class Seed:
    label: str  # note title, or the missing target as written
    path: str | None  # None for a missing note
    node: int
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
    shared_terms: list[str] = field(default_factory=list)  # words behind "similar text (...)"
    cosine: float = 0.0  # centered cosine of the two embeddings (text is this over the best one)
    cocitation: float = 0.0
    coupling: float = 0.0
    cited_by: list[str] = field(default_factory=list)  # notes linking both (when cocitation counts)
    both_link: list[str] = field(default_factory=list)  # what both link to (when coupling counts)


@dataclass
class RelatedResult:
    seed: Seed
    candidates: int  # notes with any signal
    hits: list[RelatedHit]
    compared: int = 0  # notes considered (all but the seed)
    model_id: str | None = None  # the embedding model behind the text signal
    text_unavailable: str | None = None  # why there is no text signal (the model cannot be loaded)
    uncached: int = 0  # notes embedded in memory for this run: no cached vector (`vault embed` caches them)
    weights: dict[str, float] = field(default_factory=lambda: dict(WEIGHTS))  # the weights used

    def to_dict(self) -> dict:
        return asdict(self)


def tag_label(tag: str) -> str:
    """How a tag is shown: #name for a #tag, [[name]] for a tag link."""
    return tag if tag.startswith("[[") else "#" + tag


def note_tag_sets(hashtags, graph: Graph, n_notes: int) -> tuple[dict[int, set[str]], dict[str, float], Counter[str]]:
    """(note -> its tags, common ones left out; tag -> ln(N / notes with it); tag -> notes with it).

    ``hashtags``: (file id, name key) of every #tag. A note's tags are its #tags with their parents and the
    targets of its tag links (``[[topic]]``); tags on more than ``COMMON_TAG`` of the notes are left out.
    """
    note_tags: dict[int, set[str]] = defaultdict(set)
    for file_id, name_key in hashtags:
        parts = name_key.split("/")
        note_tags[file_id].update("/".join(parts[:depth]) for depth in range(1, len(parts) + 1))
    for node, targets in graph.tag_links.items():  # tag links: [[topic]]
        note_tags[graph.note_ids[node]].update(graph.tag_name(t) for t in targets)
    tag_notes = Counter(tag for tags in note_tags.values() for tag in tags)
    tag_idf = {tag: math.log(n_notes / n) for tag, n in tag_notes.items()}
    for tags in note_tags.values():  # tags on most notes carry no signal
        tags -= {tag for tag in tags if tag_notes[tag] > COMMON_TAG * n_notes}
    return note_tags, tag_idf, tag_notes


def tag_similarity(a: set[str], b: set[str], tag_idf: dict[str, float]) -> float:
    """Weighted Jaccard of two tag sets, each tag weighing its idf."""
    union = sum(tag_idf.get(t, 0.0) for t in a | b)
    return sum(tag_idf.get(t, 0.0) for t in a & b) / union if union else 0.0


def link_signals(graph: Graph, seed: int, want: set[str]) -> dict[str, dict[int, float]]:
    """The link signals from a graph node to every note (file id -> raw value, positive only).

    graph: personalized PageRank divided by degree; cocitation and coupling: ``Graph.cocitation``/``coupling``.
    """
    out: dict[str, dict[int, float]] = {}
    if "graph" in want:
        rank = graph.walk().personalized_pagerank(seed)
        out["graph"] = {graph.note_ids[n]: r / graph.degree(n) for n, r in rank.items()
                        if n != seed and not graph.is_missing(n) and graph.degree(n) and r > 0}
    if "cocitation" in want:
        candidates = {b for c in graph.incoming[seed] for b in graph.outgoing[c]}
        out["cocitation"] = {graph.note_ids[b]: v for b in candidates if b != seed and not graph.is_missing(b)
                             for v in [graph.cocitation(seed, b)[0]] if v > 0}
    if "coupling" in want:
        candidates = {b for z in graph.outgoing[seed] for b in graph.incoming[z]}
        out["coupling"] = {graph.note_ids[b]: v for b in candidates if b != seed and not graph.is_missing(b)
                           for v in [graph.coupling(seed, b)[0]] if v > 0}
    return out


def scaled(values: dict[int, float]) -> dict[int, float]:
    """Each value over the largest (0..1); nothing when no value is positive."""
    top = max(values.values(), default=0.0)
    return {k: v / top for k, v in values.items()} if top > 0 else {}


def combine(raw: dict[str, dict[int, float]], weights: dict[str, float]) -> dict[int, float]:
    """score = the sum over signals of weight x (raw / its largest value): how ``related`` ranks."""
    total: dict[int, float] = defaultdict(float)
    for name, values in raw.items():
        w = weights.get(name, 0.0)
        if w:
            for k, v in scaled(values).items():
                total[k] += w * v
    return dict(total)


def active_weights(weights: dict[str, float], unavailable: set[str]) -> dict[str, float]:
    """The weights of the signals that are available, rescaled to sum to 1 (0 for the others)."""
    total = sum(w for k, w in weights.items() if k not in unavailable)
    return {k: (0.0 if k in unavailable or not total else w / total) for k, w in weights.items()}


def configured_provider() -> "EmbeddingProvider":
    """The provider from the engine settings (config/config.yaml, env), for callers that did not pass one."""
    from vaultengine.settings import load_settings

    return load_settings().provider()


class TextSimilarity:
    """Centered cosines between note embeddings, and the words behind them. Loads vectors on first use."""

    def __init__(self, conn: sqlite3.Connection, root: str | Path | None, provider: Any = None) -> None:
        self.conn = conn
        if root is None:
            row = conn.execute("SELECT value FROM meta WHERE key = 'vault_root'").fetchone()
            root = row[0] if row else "."
        self.root = Path(root)
        self._provider = provider
        self.unavailable: str | None = None  # why there is no text signal, once loading failed
        self._vectors: NoteVectors | None = None
        self._centered: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._bags: dict[int, WordBag] = {}
        self._texts: dict[str, str] = {}  # note texts read while computing vectors in memory

    @property
    def provider(self) -> "EmbeddingProvider":
        if self._provider is None:
            self._provider = configured_provider()
        elif callable(self._provider) and not hasattr(self._provider, "embed"):
            self._provider = self._provider()
        return self._provider

    def available(self) -> bool:
        """Load the model and the vectors; False (with ``unavailable`` saying why) if the model cannot be loaded."""
        from vaultengine.embedding import EmbeddingError
        from vaultengine.settings import SettingsError

        if self._vectors is not None:
            return True
        if self.unavailable is not None:
            return False
        try:
            self.vectors
        except (EmbeddingError, SettingsError) as exc:
            self.unavailable = str(exc)
            return False
        return True

    @property
    def uncached(self) -> list[str]:
        """Notes embedded in memory for this run because the index has no vector for them."""
        return self._vectors.computed if self._vectors is not None else []

    @property
    def explains(self) -> bool:
        """Whether the provider can name the words behind a similarity (static models)."""
        from vaultengine.embedding import StaticEmbeddingProvider

        return self.available() and isinstance(self.provider, StaticEmbeddingProvider)

    @property
    def vectors(self) -> "NoteVectors":
        if self._vectors is None:
            import numpy as np

            from vaultengine.embedding.store import note_vectors

            self._vectors = note_vectors(self.conn, self.root, self.provider, self._texts)
            matrix = self._vectors.matrix
            used = np.abs(matrix).sum(axis=1) > 0  # a note with no known words has no direction
            self._mean = matrix[used].mean(axis=0) if used.any() else np.zeros(matrix.shape[1], np.float32)
            centered = matrix - self._mean
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            norms[~used] = np.inf  # -> a zero row: no similarity with anything
            norms[norms == 0] = np.inf
            self._centered = (centered / norms).astype(np.float32)
        return self._vectors

    @property
    def mean(self) -> "np.ndarray":
        self.vectors
        return self._mean

    @property
    def centered(self) -> "np.ndarray":
        """Row i: the centered, unit-length vector of note ``vectors.file_ids[i]`` (zeros if it has none)."""
        self.vectors
        return self._centered

    def center(self, vector: "np.ndarray") -> "np.ndarray | None":
        import numpy as np

        if vector is None or not vector.any():
            return None
        u = vector - self.mean
        norm = np.linalg.norm(u)
        return u / norm if norm > 0 else None

    def unit_mean(self, file_ids: list[int]) -> "np.ndarray | None":
        """The normalised mean of these notes' vectors (the vector of a missing note)."""
        import numpy as np

        rows = [v for v in (self.vectors.get(f) for f in file_ids) if v is not None and v.any()]
        if not rows:
            return None
        mean = np.mean(rows, axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else None

    def cosines(self, vector: "np.ndarray | None") -> dict[int, float]:
        """file id -> centered cosine with ``vector`` (a raw unit vector), for every note with a vector."""
        u = self.center(vector)
        if u is None:
            return {}
        values = self.centered @ u.astype(self.centered.dtype)
        return {f: float(v) for f, v in zip(self.vectors.file_ids, values)}

    # -- the words behind a similarity (static providers only) --

    def bag(self, file_id: int, path: str) -> "WordBag | None":
        from vaultengine.embedding.explain import text_bag
        from vaultengine.embedding.store import note_text

        if not self.explains:
            return None
        if file_id not in self._bags:
            text = self._texts.get(path) or note_text(self.root, path)
            if text is None:
                return None
            self._bags[file_id] = text_bag(self.provider, text)
        return self._bags[file_id]

    def combined_bag(self, notes: list[tuple[int, str]]) -> "WordBag | None":
        from vaultengine.embedding.explain import combine

        bags = [b for b in (self.bag(f, p) for f, p in notes) if b is not None]
        return combine(bags, self.vectors.matrix.shape[1]) if bags else None

    def words(self, a: "WordBag | None", b: "WordBag | None", count: int = 3) -> list[str]:
        from vaultengine.embedding.explain import shared_words

        if a is None or b is None:
            return []
        return shared_words(a, b, self.mean, count)


class Relatedness:
    """Everything ``related`` needs, loaded once; call ``seed`` then ``related``.

    ``provider``: the embedding provider (or a function returning it), default: the configured one.
    ``root``: the vault folder, default: the one the index was built from.
    Vectors are loaded on the first text score (missing ones embedded in memory, never written).
    """

    def __init__(self, conn: sqlite3.Connection, provider: Any = None, root: str | Path | None = None,
                 tag_rule=None, weights: dict[str, float] | None = None) -> None:
        self.conn = conn
        self.graph: Graph = build_graph(conn, tag_rule)
        self.searcher = _Searcher(conn, fuzzy=False)
        self.n_notes = self.searcher.n_notes
        self.paths = {row[0]: row[1] for row in conn.execute("SELECT id, path FROM files WHERE kind = 'note'")}
        self.text = TextSimilarity(conn, root, provider)
        self.weights = {**WEIGHTS, **(weights or {})}
        self.note_tags, self.tag_idf, self.tag_notes = note_tag_sets(
            conn.execute("SELECT file_id, name_key FROM tags").fetchall(), self.graph, self.n_notes)

    # -- seeds --

    def seed(self, ref: str) -> Seed:
        """The seed for a note reference (name, path or alias, like a link) or a missing link target."""
        evaluator = _Evaluator(self.conn, self.searcher)
        file_id = evaluator.resolve(ref, (0, len(ref)))
        if file_id is not None and file_id in self.paths:
            path = self.paths[file_id]
            return Seed(link_name(path), path, self.graph.by_file[file_id], set(self.note_tags.get(file_id, set())))
        node = self.graph.by_missing.get(lookup_key(ref))
        if node is None:
            raise QueryError(f'no note or link target called "{ref}"', None, None,
                             "give a note name, path or alias, or the target of a link to a missing note")
        linker_nodes = sorted(other for other in self.graph.adjacency[node] if not self.graph.is_missing(other))
        linkers = [self.graph.note_ids[n] for n in linker_nodes]
        tag_counts = Counter(tag for linker in linkers for tag in self.note_tags.get(linker, set()))
        shared = {tag for tag, n in tag_counts.items() if n >= 2 and n >= 0.3 * len(linkers)}
        return Seed(self.graph.labels[node], None, node, shared, [self.paths[linker] for linker in linkers])

    def _linker_ids(self, seed: Seed) -> list[int]:
        return [f for f in (self.graph.note_ids[n] for n in self.graph.adjacency[seed.node]) if f in self.paths]

    def seed_vector(self, seed: Seed) -> "np.ndarray | None":
        """The seed's embedding (unit length): the note's own, or the mean of the notes linking to a missing note."""
        if seed.path is not None:
            return self.text.vectors.get(self.graph.note_ids[seed.node])
        return self.text.unit_mean(self._linker_ids(seed))

    # -- scores --

    def text_scores(self, seed: Seed) -> dict[int, float]:
        """note id -> centered cosine with the seed, for the notes where it is positive."""
        seed_file = self.graph.note_ids[seed.node]
        return {f: c for f, c in self.text.cosines(self.seed_vector(seed)).items() if c > 0 and f != seed_file}

    def seed_words(self, seed: Seed):
        if seed.path is not None:
            return self.text.bag(self.graph.note_ids[seed.node], seed.path)
        return self.text.combined_bag([(f, self.paths[f]) for f in self._linker_ids(seed)])

    def tag_score(self, a: set[str], b: set[str]) -> float:
        return tag_similarity(a, b, self.tag_idf)

    def describe(self, seed: Seed, hits: list[RelatedHit], words: bool = True, paths=None) -> None:
        """Fill in each hit's reason (and, with ``words``, the words behind "similar text": that reads the notes)."""
        graph = self.graph
        paths = paths if paths is not None else hop_paths(graph, seed.node)
        seed_bag = self.seed_words(seed) if words and any(h.text >= TEXT_REASON for h in hits) else None
        by_path = {path: file_id for file_id, path in self.paths.items()}
        for hit in hits:  # reasons only for what is returned: the words mean reading the notes
            if hit.text >= TEXT_REASON and seed_bag is not None:
                hit.shared_terms = self.text.words(seed_bag, self.text.bag(by_path[hit.path], hit.path))
            hit.reason = self.reason(seed, graph.by_file[by_path[hit.path]], hit, paths)

    def related(self, seed: Seed, limit: int | None = 10, unlinked_only: bool = False, words: bool = True,
                weights: dict[str, float] | None = None, reasons: bool = True) -> RelatedResult:
        """The notes related to ``seed``, best first; ``words``: name the words behind "similar text" (reads the notes).
        ``weights``: this call's mix instead of the engine's (e.g. ``UNLINKED_WEIGHTS`` for unlinked notes);
        ``reasons``: False leaves the reasons empty (``describe`` fills them in for the hits that are shown)."""
        graph = self.graph
        seed_file = graph.note_ids[seed.node]
        mix = {**WEIGHTS, **weights} if weights is not None else self.weights
        text_on = self.text.available() if mix["text"] else False
        weights = active_weights(mix, set() if text_on else {"text"})
        raw = link_signals(graph, seed.node, {k for k in ("graph", "cocitation", "coupling") if weights[k]})
        text_raw = self.text_scores(seed) if text_on else {}
        raw["text"] = text_raw
        tags_raw = {f: self.tag_score(seed.tags, tags) for f, tags in self.note_tags.items() if f != seed_file}
        raw["tags"] = {f: v for f, v in tags_raw.items() if v > 0} if weights["tags"] else {}
        parts = {name: scaled(values) for name, values in raw.items()}
        score = combine(raw, weights)
        g, x, t = parts.get("graph", {}), parts["text"], parts["tags"]
        co, cp = parts.get("cocitation", {}), parts.get("coupling", {})
        paths = hop_paths(graph, seed.node)
        hits = []
        for file_id in score:
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
            hit = RelatedHit(
                path=path, title=link_name(path), score=score[file_id],
                graph=g.get(file_id, 0.0), text=x.get(file_id, 0.0), tags=t.get(file_id, 0.0),
                linked=linked, hops=paths[node].hops if node in paths else None, reason="",
                shared_tags=shared_tags[:3], cosine=text_raw.get(file_id, 0.0),
                cocitation=co.get(file_id, 0.0), coupling=cp.get(file_id, 0.0),
            )
            if hit.cocitation:
                hit.cited_by = [self._node_name(c) for c, _ in graph.cocitation(seed.node, node)[1][:2]]
            if hit.coupling:
                hit.both_link = [self._node_name(z) for z, _ in graph.coupling(seed.node, node)[1][:2]]
            hits.append(hit)
        hits.sort(key=lambda h: (-h.score, h.path))
        shown = hits if limit is None else hits[:limit]
        if reasons:
            self.describe(seed, shown, words, paths)
        compared = self.n_notes - (0 if seed_file is None else 1)
        model_id = self.text.vectors.model_id if text_on else None
        return RelatedResult(seed, len(hits), shown, compared, model_id, self.text.unavailable,
                             len(self.text.uncached), weights)

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
            parts.append("shares " + ", ".join(tag_label(tag) for tag in hit.shared_tags))
        if hit.cited_by:
            parts.append("cited together by " + ", ".join(hit.cited_by))
        if hit.both_link:
            parts.append("both link " + ", ".join(hit.both_link))
        if hit.text >= TEXT_REASON and hit.shared_terms:
            parts.append("similar text (" + ", ".join(hit.shared_terms) + ")")
        elif hit.text >= TEXT_REASON and not self.text.explains:  # a model that cannot name words
            parts.append("similar text")
        return " · ".join(parts)

    def _node_name(self, node: int) -> str:
        name = self.graph.labels[node] if self.graph.is_missing(node) else link_name(self.graph.labels[node])
        if len(name) > 40:
            name = name[:39].rstrip() + "…"
        return f"{name} (no note yet)" if self.graph.is_missing(node) else name


def related(conn: sqlite3.Connection, ref: str, limit: int | None = 10, unlinked_only: bool = False,
            provider: Any = None, root: str | Path | None = None) -> RelatedResult:
    engine = Relatedness(conn, provider, root)
    return engine.related(engine.seed(ref), limit=limit, unlinked_only=unlinked_only)
