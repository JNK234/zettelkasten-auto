"""How two notes relate, signal by signal, with the evidence for each (`vault explain`).

Five signals, each a number from 0 to 1 and each with what it rests on:

* direct      - 1 if either note links to the other (resolved links and embeds, tag links
                included: ``Tags: [[b]]`` in a links a to b), else 0. Evidence: the links
                each way, with their lines, tag links marked.
* cocitation  - the notes that link to both (``graphrank.Graph.cocitation``): a weighted
                cosine, each citing note weighing 1 / ln(1 + the nodes it links to).
* coupling    - what both link to, missing notes included (``Graph.coupling``): a weighted
                cosine, each reference weighing 1 / ln(1 + the notes linking to it).
* tags        - weighted Jaccard of the two tag sets, each tag weighing ln(N / notes with
                it), nested tags counting their parents, tags on more than a third of the
                notes left out (the same as ``related.py``). Tags are #tags and the targets of
                tag links (``tagrule.py``), which therefore never count as coupling.
* text        - the centered cosine of the two embeddings (``related.py``), 0 if negative,
                split into the words behind it (``embedding/explain.py``).

All five come from one graph: ``graphrank.build_graph`` (the undirected link graph that
``related`` walks, with its link directions), plus the tags and vectors ``Relatedness``
already loads.

score = sum of weight x value, with the weights of ``engine.explore.weights`` rescaled to sum
to 1 over the signals that are available. Without the embedding model (missing or broken),
the text signal is unavailable, says why, and the other four share its weight.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vaultengine.querylang import QueryError
from vaultengine.related import COMMON_TAG, Relatedness, active_weights
from vaultengine.vault import link_name

SIGNALS = ("direct", "cocitation", "coupling", "tags", "text")
LABELS = {"direct": "direct links", "cocitation": "co-citation", "coupling": "coupling", "tags": "shared tags", "text": "text"}


def default_weights() -> dict[str, float]:
    from vaultengine.settings import ExploreWeights

    return ExploreWeights().model_dump()


@dataclass
class Signal:
    name: str
    value: float  # 0..1
    weight: float  # the weight used (rescaled; 0 when unavailable)
    contribution: float  # weight * value
    available: bool = True
    unavailable: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class PairRelation:
    a: str  # note paths
    b: str
    score: float
    signals: list[Signal]
    model_id: str | None = None
    uncached: int = 0  # notes embedded in memory for this run (no cached vector)

    def signal(self, name: str) -> Signal:
        return next(s for s in self.signals if s.name == name)

    def to_dict(self) -> dict:
        return asdict(self)


class Relations:
    """Load once (graph, tags, vectors on first use), then ``explain(a, b)`` any pair."""

    def __init__(self, conn: sqlite3.Connection, provider: Any = None, root: str | Path | None = None,
                 weights: dict[str, float] | None = None, engine: Relatedness | None = None, tag_rule=None) -> None:
        self.conn = conn
        self.engine = engine or Relatedness(conn, provider, root, tag_rule)
        self.graph = self.engine.graph
        self.weights = dict(weights or default_weights())
        self.raw_tags: dict[int, set[str]] = {}  # every tag of a note, parents included (common ones too)
        for file_id, name_key in conn.execute("SELECT file_id, name_key FROM tags"):
            parts = name_key.split("/")
            self.raw_tags.setdefault(file_id, set()).update("/".join(parts[:d]) for d in range(1, len(parts) + 1))
        for node, targets in self.graph.tag_links.items():
            self.raw_tags.setdefault(self.graph.note_ids[node], set()).update(self.graph.tag_name(t) for t in targets)

    def node(self, ref: str) -> int:
        """The graph node of a note reference (name, path or alias). Raises QueryError."""
        seed = self.engine.seed(ref)
        if seed.path is None:
            raise QueryError(f'"{ref}" is not a note yet (only links to it exist)', None, None,
                             "explain compares two notes; `vault related` also takes a missing note")
        return seed.node

    def explain(self, a_ref: str, b_ref: str, words: int = 8) -> PairRelation:
        a, b = self.node(a_ref), self.node(b_ref)
        if a == b:
            raise QueryError(f'"{a_ref}" and "{b_ref}" are the same note', None, None, "give two different notes")
        text_on = self.engine.text.available()
        weights = active_weights(self.weights, set() if text_on else {"text"})
        signals = [self._direct(a, b), self._cocitation(a, b), self._coupling(a, b), self._tags(a, b)]
        signals.append(self._text(a, b, words) if text_on else
                       Signal("text", 0.0, 0.0, 0.0, False, self.engine.text.unavailable))
        for signal in signals:
            signal.weight = weights[signal.name]
            signal.contribution = signal.weight * signal.value
        return PairRelation(
            self.graph.labels[a], self.graph.labels[b], sum(s.contribution for s in signals), signals,
            self.engine.text.vectors.model_id if text_on else None, len(self.engine.text.uncached),
        )

    # -- the signals --

    def _name(self, node: int) -> str:
        return f"{self.graph.labels[node]} (no note yet)" if self.graph.is_missing(node) else link_name(self.graph.labels[node])

    def _direct(self, a: int, b: int) -> Signal:
        g = self.graph
        ab, ba = g.lines.get((a, b), []), g.lines.get((b, a), [])
        tagged = lambda s, t: sorted(line for x, y, line in g.tag_lines if (x, y) == (s, t))
        return Signal("direct", 1.0 if ab or ba else 0.0, 0.0, 0.0, evidence={
            "a_to_b": sorted(ab), "b_to_a": sorted(ba), "a_to_b_tag_lines": tagged(a, b), "b_to_a_tag_lines": tagged(b, a)})

    def _cocitation(self, a: int, b: int) -> Signal:
        g = self.graph
        value, both = g.cocitation(a, b)
        return Signal("cocitation", value, 0.0, 0.0, evidence={
            "both": [{"note": g.labels[c], "weight": w, "links_to": len(g.outgoing[c])} for c, w in both],
            "citing_a": len(g.incoming[a]), "citing_b": len(g.incoming[b]),
            "total_a": sum(g.citer_weight(c) for c in g.incoming[a]),
            "total_b": sum(g.citer_weight(c) for c in g.incoming[b]),
        })

    def _coupling(self, a: int, b: int) -> Signal:
        g = self.graph
        value, both = g.coupling(a, b)
        return Signal("coupling", value, 0.0, 0.0, evidence={
            "both": [{"target": g.labels[z], "missing": g.is_missing(z), "weight": w, "linked_from": len(g.incoming[z])}
                     for z, w in both],
            "references_a": len(g.outgoing[a]), "references_b": len(g.outgoing[b]),
            "total_a": sum(g.reference_weight(z) for z in g.outgoing[a]),
            "total_b": sum(g.reference_weight(z) for z in g.outgoing[b]),
        })

    def _tags(self, a: int, b: int) -> Signal:
        e = self.engine
        fa, fb = self.graph.note_ids[a], self.graph.note_ids[b]
        ta, tb = e.note_tags.get(fa, set()), e.note_tags.get(fb, set())  # common tags already left out
        weight = lambda tags: sorted(({"tag": t, "weight": e.tag_idf[t]} for t in tags), key=lambda x: (-x["weight"], x["tag"]))
        left_out = sorted((self.raw_tags.get(fa, set()) | self.raw_tags.get(fb, set())) - ta - tb)
        return Signal("tags", e.tag_score(ta, tb), 0.0, 0.0, evidence={
            "shared": weight(ta & tb), "only_a": weight(ta - tb), "only_b": weight(tb - ta),
            "shared_weight": sum(e.tag_idf[t] for t in ta & tb), "union_weight": sum(e.tag_idf[t] for t in ta | tb),
            "left_out": [{"tag": t, "notes": e.tag_notes[t]} for t in left_out], "notes": e.n_notes,
            "common_over": COMMON_TAG,
        })

    def _text(self, a: int, b: int, words: int) -> Signal:
        from vaultengine.embedding.explain import shown, split

        text = self.engine.text
        fa, fb = self.graph.note_ids[a], self.graph.note_ids[b]
        cosines = text.cosines(text.vectors.get(fa))
        cosine = cosines.get(fb, 0.0) if cosines else 0.0
        evidence: dict[str, Any] = {"cosine": cosine, "model_id": text.vectors.model_id}
        bag_a, bag_b = text.bag(fa, self.graph.labels[a]), text.bag(fb, self.graph.labels[b])
        parts = split(bag_a, bag_b, text.mean) if bag_a is not None and bag_b is not None else None
        if parts is not None:
            shared = sorted(parts.shared, key=lambda w: (-parts.credits[w], w))
            evidence.update(
                words=[{"word": w, "share": parts.credits[w]} for w in shared if shown(w)][:words],
                shared_words=len(shared),
                shared_total=sum(parts.credits[w] for w in shared),
                one_note_total=sum(c for w, c in parts.credits.items() if w not in parts.shared),
                baseline=parts.baseline,
                # The words are read from the notes now; the vectors are the indexed ones.
                matches_vectors=bool(abs(parts.total() - cosine) < 1e-4),
            )
        return Signal("text", max(0.0, cosine), 0.0, 0.0, evidence=evidence)
