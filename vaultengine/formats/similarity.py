"""Duplicate scores: title similarity and content overlap. Both are in [0, 1].

Words: text is Unicode-decomposed, accents dropped, lower-cased, and split into runs
of letters and digits ("R1-Zero-Like" -> r1 zero like). Markdown and wikilink syntax
and URLs are removed first for content (link text is kept).

Title similarity: 2·M / (|A| + |B|) over the two word sequences, where M is the
number of words matched in order (difflib's longest-matching-blocks). Before
matching, a word of 5+ letters that is not in the other title is replaced by its
closest word there if they are 85% alike, so a typo ("Trainng") still matches.
Word-level, not letter-level: one different word in a short title costs a lot
("… Long Tasks" vs "… Short Tasks" is 0.857, while letter-level similarity gives
0.92 and would call two different papers the same).

Content overlap: the share of the shorter text's 5-word sequences (shingles) that
also occur in the other text: |A ∩ B| / min(|A|, |B|). Word order within five words
matters, paragraph order and formatting do not, and a text fully contained in a
longer one scores 1.0. Texts with fewer than 40 shingles (about 45 words) are too
short to compare and get no score.

A title similarity >= 0.9 or a content overlap >= 0.8 makes a duplicate.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

TITLE_THRESHOLD = 0.9
OVERLAP_THRESHOLD = 0.8
SHINGLE_WORDS = 5
MIN_SHINGLES = 40
_EPSILON = 1e-9

_WORD = re.compile(r"[a-z0-9]+")
_WIKILINK = re.compile(r"!?\[\[([^\]|]*?)(?:\|([^\]]*))?\]\]")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"\b(?:https?|ftp)://\S+")


def words(text: str) -> list[str]:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return _WORD.findall(ascii_text)


def title_similarity(a: str, b: str) -> float:
    first, second = words(a), words(b)
    if not first or not second:
        return 0.0
    long_words = sorted({w for w in first if len(w) >= 5})
    known = set(first)
    aligned = []
    for word in second:
        if word not in known and len(word) >= 5:
            close = difflib.get_close_matches(word, long_words, n=1, cutoff=0.85)
            word = close[0] if close else word
        aligned.append(word)
    matcher = difflib.SequenceMatcher(None, first, aligned, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return 2 * matched / (len(first) + len(aligned))


def content_words(text: str) -> list[str]:
    text = _WIKILINK.sub(lambda m: f" {m[2] or m[1]} ", text)
    text = _MD_LINK.sub(lambda m: f" {m[1]} ", text)
    return words(_URL.sub(" ", text))


def shingles(text: str, size: int = SHINGLE_WORDS) -> frozenset[tuple[str, ...]]:
    found = content_words(text)
    return frozenset(tuple(found[i:i + size]) for i in range(len(found) - size + 1))


@dataclass(frozen=True)
class Overlap:
    score: float | None  # None: a text is too short to compare
    shared: int
    smaller: int  # shingles in the shorter text


def content_overlap(a: frozenset, b: frozenset, minimum: int = MIN_SHINGLES) -> Overlap:
    smaller = min(len(a), len(b))
    if smaller < minimum:
        return Overlap(None, 0, smaller)
    shared = len(a & b) if len(a) <= len(b) else len(b & a)
    return Overlap(shared / smaller, shared, smaller)


def is_title_duplicate(score: float) -> bool:
    return score >= TITLE_THRESHOLD - _EPSILON


def is_content_duplicate(score: float | None) -> bool:
    return score is not None and score >= OVERLAP_THRESHOLD - _EPSILON
