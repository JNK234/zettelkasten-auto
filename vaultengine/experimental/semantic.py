"""EXPERIMENTAL: suggest notes by text similarity. Advisory only; nothing here gates a write.

Three ways to rank candidate notes against a query text:

* ``lexical``: TF-IDF cosine over words (pure Python, idf from the whole vault);
* ``semantic``: cosine of WordLlama embeddings (``pip install wordllama``; a 16 MB static
  embedding model that ships inside the wheel, so it runs offline);
* ``fused``: reciprocal-rank fusion of the two (score = Σ 1/(60 + rank)).

Measured on the sample vault (``python scripts/semantic_eval.py``), lexical ranking
with an idf over the whole vault was the best of the three on every task: a note's
source (33/33 first), the claim an evidence note supports (16/26 first, 24/26 in the
top 3), and a paraphrase of a paper already in the vault (7/8 first). WordLlama did
not beat it (26/33, 11/26, 6/8), also not with paragraph-level matching or fused.
Only rankings were usable: WordLlama similarities between unrelated long notes are
around 0.6, so no fixed threshold separates "same work" from "same field", and the
duplicate gate of ``vault ingest`` does not use embeddings. Everything defaults to
lexical; ``semantic`` and ``fused`` stay available to re-run the comparison on a
larger vault or with a stronger model.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from vaultengine.formats.ingest import note_body
from vaultengine.formats.similarity import words
from vaultengine.formats.typed import TypedVault

Method = Literal["lexical", "semantic", "fused"]
RRF_K = 60
_LINK = re.compile(r"!?\[\[[^\]]*\]\]")
_MODEL = None


class SemanticUnavailable(RuntimeError):
    """WordLlama is not installed (pip install wordllama, or the [semantic] extra)."""


def load_model():
    """WordLlama from the files inside its wheel, never from the network."""
    global _MODEL
    if _MODEL is None:
        try:
            import wordllama
            from wordllama import WordLlama
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SemanticUnavailable("semantic ranking needs WordLlama: pip install wordllama") from exc
        logging.getLogger("wordllama").setLevel(logging.WARNING)
        # The wheel ships weights/ and tokenizers/; pointing the cache there avoids a download.
        _MODEL = WordLlama.load(cache_dir=Path(wordllama.__file__).parent, disable_download=True)
    return _MODEL


def semantic_available() -> bool:
    try:
        load_model()
    except SemanticUnavailable:
        return False
    return True


def note_text(typed: TypedVault, path: str) -> str:
    """The note's body without metadata lines and without [[links]] (a link to the answer would be cheating)."""
    text = (typed.vault.root / path).read_text(encoding="utf-8", errors="replace")
    return _LINK.sub(" ", note_body(text))


class _Tfidf:
    def __init__(self, corpus: Iterable[str]) -> None:
        documents = [Counter(words(text)) for text in corpus]
        self.size = len(documents)
        df: Counter[str] = Counter()
        for counts in documents:
            df.update(counts.keys())
        self.idf = {w: math.log((1 + self.size) / (1 + n)) + 1 for w, n in df.items()}

    def vector(self, text: str) -> dict[str, float]:
        counts = Counter(words(text))
        vec = {w: (1 + math.log(c)) * self.idf.get(w, math.log(1 + self.size) + 1) for w, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {w: v / norm for w, v in vec.items()}

    @staticmethod
    def cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(w, 0.0) for w, v in a.items())


@dataclass(frozen=True)
class Suggestion:
    path: str
    score: float  # of the method used (RRF score for fused)
    lexical: float | None
    semantic: float | None
    rank: int

    def to_dict(self) -> dict:
        return {"path": self.path, "rank": self.rank, "score": round(self.score, 4),
                "lexical": None if self.lexical is None else round(self.lexical, 4),
                "semantic": None if self.semantic is None else round(self.semantic, 4)}


class Ranker:
    """Ranks candidates for query texts. Build once per vault; texts are cached by path."""

    def __init__(self, typed: TypedVault) -> None:
        self.typed = typed
        self._texts: dict[str, str] = {}
        self._tfidf: _Tfidf | None = None
        self._vectors: dict[str, dict[str, float]] = {}
        self._embeddings: dict[str, object] = {}

    def text(self, path: str) -> str:
        if path not in self._texts:
            self._texts[path] = note_text(self.typed, path)
        return self._texts[path]

    def _lexical(self) -> _Tfidf:
        if self._tfidf is None:
            self._tfidf = _Tfidf(self.text(p) for p in sorted(self.typed.vault.notes))
        return self._tfidf

    def _embed(self, key: str, text: str):
        if key not in self._embeddings:
            self._embeddings[key] = load_model().embed([text], norm=True)[0]
        return self._embeddings[key]

    def rank(self, query: str, candidates: list[str], method: Method, limit: int = 3, query_key: str | None = None) -> list[Suggestion]:
        if not candidates:
            return []
        lexical = semantic = None
        if method in ("lexical", "fused"):
            tfidf = self._lexical()
            q = tfidf.vector(query)
            for path in candidates:
                if path not in self._vectors:
                    self._vectors[path] = tfidf.vector(self.text(path))
            lexical = {p: tfidf.cosine(q, self._vectors[p]) for p in candidates}
        if method in ("semantic", "fused"):
            q = self._embed(query_key or f"query:{hash(query)}", query)
            semantic = {p: float(q @ self._embed(p, self.text(p))) for p in candidates}
        if method == "lexical":
            scores = lexical
        elif method == "semantic":
            scores = semantic
        else:
            def ranks(table):
                return {p: i for i, p in enumerate(sorted(table, key=lambda p: (-table[p], p)), start=1)}

            lr, sr = ranks(lexical), ranks(semantic)
            scores = {p: 1 / (RRF_K + lr[p]) + 1 / (RRF_K + sr[p]) for p in candidates}
        order = sorted(candidates, key=lambda p: (-scores[p], p))[:limit]
        return [
            Suggestion(p, scores[p], lexical[p] if lexical else None, semantic[p] if semantic else None, i)
            for i, p in enumerate(order, start=1)
        ]


# ----- the three uses ------------------------------------------------------------------------------


def _of_types(typed: TypedVault, types: Iterable[str]) -> list[str]:
    wanted = set(types)
    return sorted(p for p in typed.vault.notes if typed.type_of(p) in wanted)


def suggest_sources(ranker: Ranker, path: str, method: Method = "lexical", limit: int = 3) -> list[Suggestion]:
    """Sources this note probably draws on, that it does not link to yet (for `vault coverage`)."""
    typed = ranker.typed
    if typed.spec.coverage is None:
        raise ValueError(f"format {typed.spec.name!r} has no coverage: block (it says which types are sources)")
    linked = {l.edge.target_path for l in typed.links(path) if l.edge.target_path}
    candidates = [p for p in _of_types(typed, typed.spec.coverage.sources) if p != path and p not in linked]
    return ranker.rank(ranker.text(path), candidates, method, limit, query_key=path)


def suggest_targets(ranker: Ranker, path: str, relation: str, method: Method = "lexical", limit: int = 3) -> list[Suggestion]:
    """Notes this note might have `relation` to (for `vault link`): notes of the relation's target
    types, minus itself and the ones it already relates to that way."""
    typed = ranker.typed
    rel = typed.spec.relation(relation)
    if rel is None:
        raise ValueError(f"{relation!r} is not a relation of format {typed.spec.name!r}")
    pool = _of_types(typed, rel.targets) if rel.targets else sorted(typed.vault.notes)
    already = {e.target_path for e in typed.relations(path) if e.relation == rel.name}
    candidates = [p for p in pool if p != path and p not in already]
    return ranker.rank(ranker.text(path), candidates, method, limit, query_key=path)


def related_to_text(ranker: Ranker, text: str, type_name: str, method: Method = "lexical", limit: int = 3) -> list[Suggestion]:
    """Existing notes of a type that a new document resembles (for `vault ingest`): catches a
    paraphrase of something already in the vault, which title and content overlap cannot."""
    candidates = _of_types(ranker.typed, [type_name])
    return ranker.rank(_LINK.sub(" ", text), candidates, method, limit)


# ----- evaluation on the vault itself ---------------------------------------------------------------


def evaluate(typed: TypedVault, methods: Iterable[Method] = ("lexical", "semantic", "fused")) -> dict:
    """Leave-the-link-out accuracy on what the vault already says:
    A) for notes that cite a source, is a cited source ranked first / in the top 3?
    B) for evidence that supports claims, is a supported claim ranked first / in the top 3?"""
    ranker = Ranker(typed)
    source_types = set(typed.spec.coverage.sources) if typed.spec.coverage else set()
    citing_types = set(typed.spec.coverage.citing) if typed.spec.coverage else set()
    tasks = {"A_source_of_note": [], "B_claim_of_evidence": []}
    for path in sorted(typed.vault.notes):
        kind = typed.type_of(path)
        if kind in citing_types:
            cited = {l.edge.target_path for l in typed.links(path)
                     if l.edge.status == "resolved" and typed.type_of(l.edge.target_path) in source_types}
            if cited:
                pool = [p for p in sorted(typed.vault.notes) if typed.type_of(p) in source_types and p != path]
                tasks["A_source_of_note"].append((path, cited, pool))
        supported = {e.target_path for e in typed.relations(path, mirrors=False) if e.relation == "supports" and e.target_path}
        if supported:
            rel = typed.spec.relation("supports")
            pool = [p for p in sorted(typed.vault.notes) if typed.type_of(p) in set(rel.targets)]
            tasks["B_claim_of_evidence"].append((path, supported, pool))
    results = {}
    for task, items in tasks.items():
        results[task] = {"queries": len(items), "candidates": len(items[0][2]) if items else 0}
        for method in methods:
            top1 = top3 = 0
            for path, truth, pool in items:
                ranked = [s.path for s in ranker.rank(ranker.text(path), pool, method, 3, query_key=path)]
                top1 += bool(ranked) and ranked[0] in truth
                top3 += bool(set(ranked) & truth)
            results[task][method] = {"top1": top1, "top3": top3}
    return results


def evaluate_paraphrases(typed: TypedVault, methods: Iterable[Method] = ("lexical", "semantic", "fused")) -> dict:
    """C) Descriptions of eight sample papers in other words (``paraphrases.py``): is the paper's
    note ranked first among the sources? The duplicate gate scores all of them near 0."""
    from vaultengine.experimental.paraphrases import PARAPHRASES

    ranker = Ranker(typed)
    present = {path: text for path, text in PARAPHRASES.items() if path in typed.vault.notes}
    result = {"queries": len(present)}
    for method in methods:
        top1 = top3 = 0
        for truth, text in present.items():
            ranked = [s.path for s in related_to_text(ranker, text, typed.type_of(truth), method, limit=3)]
            top1 += bool(ranked) and ranked[0] == truth
            top3 += truth in ranked
        result[method] = {"top1": top1, "top3": top3}
    return result
