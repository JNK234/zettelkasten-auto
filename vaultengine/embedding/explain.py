"""Which words make two texts similar, read off a static embedding. No counting of shared terms.

With a static model a text's raw vector is the mean of its token vectors, so it is a sum
over words: ``raw = sum_w alpha_w * E_w``, where ``E_w`` is the sum of the word's token
vectors and ``alpha_w`` is its count over the text's token count. Similarity is the centered
cosine (see ``related.py``): with ``a_hat = raw_a / |raw_a|`` and ``u_a = (a_hat - mean) / Z_a``
(``Z_a = |a_hat - mean|``),

    u_a . u_b = sum_w alpha_w * (E_w . u_b) / (|raw_a| * Z_a)  -  (mean . u_b) / Z_a

Averaging that over the two sides, every word gets a credit and the credits add up exactly:

    cosine = sum over all words of credit(w)  -  (K_a + K_b) / 2
    credit(w) = (alpha_a(w) * (E_w . u_b) / (|raw_a| * Z_a) + alpha_b(w) * (E_w . u_a) / (|raw_b| * Z_b)) / 2

(a word missing from one text has alpha 0 there; K_a = (mean . u_b) / Z_a is the part that comes
from subtracting the vault's mean, the same for every word). Words in both texts with the largest
positive credit are the ones named in "similar text (...)".

A ``WordBag`` is any such weighted sum of words; ``combine`` builds the bag of a missing
note (the mean of the unit vectors of the notes that link to it) from theirs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vaultengine.embedding.provider import StaticEmbeddingProvider

MIN_WORD = 2


@dataclass
class WordBag:
    alpha: dict[str, float] = field(default_factory=dict)  # word -> weight in the raw vector
    vectors: dict[str, np.ndarray] = field(default_factory=dict)  # word -> E_w (sum of its token vectors)

    def raw(self, dim: int) -> np.ndarray:
        total = np.zeros(dim, dtype=np.float64)
        for word, weight in self.alpha.items():
            total += weight * self.vectors[word]
        return total


def text_bag(provider: StaticEmbeddingProvider, text: str) -> WordBag:
    words = provider.words(text)
    n = sum(len(ids) for _, ids in words)
    bag = WordBag()
    if not n:
        return bag
    pieces: dict[str, list[int]] = {}
    for word, ids in words:
        bag.alpha[word] = bag.alpha.get(word, 0.0) + 1.0 / n
        pieces.setdefault(word, ids)
    for word, ids in pieces.items():
        bag.vectors[word] = provider.token_vectors(ids).sum(axis=0, dtype=np.float64)
    return bag


def combine(bags: list[WordBag], dim: int) -> WordBag:
    """The bag of the mean of the bags' unit vectors."""
    out = WordBag()
    usable = [(bag, np.linalg.norm(bag.raw(dim))) for bag in bags]
    usable = [(bag, norm) for bag, norm in usable if norm > 0]
    for bag, norm in usable:
        for word, weight in bag.alpha.items():
            out.alpha[word] = out.alpha.get(word, 0.0) + weight / (norm * len(usable))
            out.vectors.setdefault(word, bag.vectors[word])
    return out


def _centered(bag: WordBag, mean: np.ndarray) -> tuple[np.ndarray, float] | None:
    raw = bag.raw(len(mean))
    norm = np.linalg.norm(raw)
    if norm == 0:
        return None
    u = raw / norm - mean
    z = np.linalg.norm(u)
    return (u / z, norm * z) if z > 0 else None


@dataclass
class Split:
    """A centered cosine split into word credits (see the module docstring); ``total()`` is the cosine."""

    credits: dict[str, float]  # every word of either text -> its credit
    shared: set[str]  # words in both texts
    baseline: float  # -(K_a + K_b) / 2: from subtracting the vault's mean vector

    def total(self) -> float:
        return sum(self.credits.values()) + self.baseline


def split(a: WordBag, b: WordBag, mean: np.ndarray) -> Split | None:
    """The cosine of a and b after subtracting ``mean``, word by word; None if either has no vector."""
    ca, cb = _centered(a, mean), _centered(b, mean)
    if ca is None or cb is None:
        return None
    (ua, scale_a), (ub, scale_b) = ca, cb
    credits: dict[str, float] = {}
    for word, alpha in a.alpha.items():
        credits[word] = credits.get(word, 0.0) + alpha * float(a.vectors[word] @ ub) / scale_a / 2
    for word, alpha in b.alpha.items():
        credits[word] = credits.get(word, 0.0) + alpha * float(b.vectors[word] @ ua) / scale_b / 2
    norm_a, norm_b = np.linalg.norm(a.raw(len(mean))), np.linalg.norm(b.raw(len(mean)))
    baseline = -(float(mean @ ub) * norm_a / scale_a + float(mean @ ua) * norm_b / scale_b) / 2
    return Split(credits, set(a.alpha) & set(b.alpha), baseline)


def shown(word: str) -> bool:
    """Words worth naming to a person: letters only, 2+ of them (not 7b, β or punctuation)."""
    return len(word) >= MIN_WORD and word.isalpha()


def shared_words(a: WordBag, b: WordBag, mean: np.ndarray, count: int = 3) -> list[str]:
    """Words in both texts with the largest positive credit, largest first (the credits of ``split``,
    computed for the shared words only: this runs for every suggested pair)."""
    ca, cb = _centered(a, mean), _centered(b, mean)
    if ca is None or cb is None:
        return []
    (ua, scale_a), (ub, scale_b) = ca, cb
    words = [w for w in a.alpha.keys() & b.alpha.keys() if shown(w)]
    if not words:
        return []
    e = np.stack([a.vectors[w] for w in words])
    alpha_a = np.array([a.alpha[w] for w in words])
    alpha_b = np.array([b.alpha[w] for w in words])
    credit = dict(zip(words, ((alpha_a * (e @ ub) / scale_a + alpha_b * (e @ ua) / scale_b) / 2).tolist()))
    return sorted((w for w, c in credit.items() if c > 0), key=lambda w: (-credit[w], w))[:count]
