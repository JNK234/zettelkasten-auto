"""Fuzzy matching for note pickers.

Each whitespace-separated query token must appear in the candidate as a
subsequence (characters in order, gaps allowed), like Obsidian's quick
switcher: "alignmnt fakng" matches "Alignment Faking". Tokens may match in any
order. Scoring is fzf-style and deterministic:

* every matched character scores, more at word starts and camelCase humps;
* consecutive matches earn a bonus, gaps cost a little per skipped character;
* an exact match, a prefix match, or the whole query as a substring starting at
  a word boundary rank higher ("ppo" in "Notes on PPO" yes, "pg" in "upgrade" no,
  so typing initials such as "pg" still prefers "Policy Gradient").

Matching is case-insensitive. Positions index into the candidate text, so a UI
can highlight them.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

_SEPARATORS = frozenset(" -_/.()[]{}:,;'\"!?#|+&")

_MATCH = 16
_BOUNDARY = 10
_FIRST_CHAR = 8
_CONSECUTIVE = 12
_GAP_START = 3  # a gap of g >= 1 skipped characters costs _GAP_START + (g - 1)
_EXACT = 1000
_PREFIX = 150
_SUBSTRING = 80
_NEG = -(10**9)


@dataclass(frozen=True)
class FuzzyMatch:
    score: int
    positions: tuple[int, ...]


def _fold(text: str) -> str:
    """Lower-case while keeping exactly one output character per input character."""
    return "".join(lowered if len(lowered := char.lower()) == 1 else char for char in text)


def _boundary_bonus(text: str, index: int) -> int:
    if index == 0:
        return _BOUNDARY + _FIRST_CHAR
    previous, current = text[index - 1], text[index]
    if previous in _SEPARATORS or (previous.islower() and current.isupper()):
        return _BOUNDARY
    return 0


def _match_token(token: str, folded: str, bonuses: list[int]) -> tuple[int, list[int]] | None:
    if not _is_subsequence(token, folded):
        return None
    n, m = len(token), len(folded)
    best = [[_NEG] * m for _ in range(n)]
    back = [[-1] * m for _ in range(n)]
    for j in range(m):
        if folded[j] == token[0]:
            best[0][j] = _MATCH + bonuses[j]
    for i in range(1, n):
        running, running_at = _NEG, -1  # max over k <= j-2 of best[i-1][k] + k
        for j in range(m):
            if j >= 2 and best[i - 1][j - 2] != _NEG and best[i - 1][j - 2] + (j - 2) > running:
                running, running_at = best[i - 1][j - 2] + (j - 2), j - 2
            if folded[j] != token[i]:
                continue
            options = []
            if j >= 1 and best[i - 1][j - 1] != _NEG:
                options.append((best[i - 1][j - 1] + _CONSECUTIVE, j - 1))
            if running_at >= 0:
                # from k: best[i-1][k] - (_GAP_START + g - 1) with gap g = j - k - 1
                options.append((running - j - _GAP_START + 2, running_at))
            if options:
                score, previous = max(options)
                best[i][j] = score + _MATCH + bonuses[j]
                back[i][j] = previous
    end = max(range(m), key=lambda j: (best[n - 1][j], -j))
    if best[n - 1][end] == _NEG:
        return None
    positions = [end]
    for i in range(n - 1, 0, -1):
        positions.append(back[i][positions[-1]])
    return best[n - 1][end], positions[::-1]


def _occurrences(text: str, part: str) -> list[int]:
    starts, start = [], text.find(part)
    while start >= 0:
        starts.append(start)
        start = text.find(part, start + 1)
    return starts


@dataclass(frozen=True)
class Prepared:
    """A candidate pre-processed once, so repeated queries only pay for matching."""

    text: str
    folded: str
    bonuses: tuple[int, ...]


def prepare(text: str) -> Prepared:
    original = unicodedata.normalize("NFC", text)
    return Prepared(original, _fold(original), tuple(_boundary_bonus(original, i) for i in range(len(original))))


def _is_subsequence(token: str, folded: str) -> bool:
    position = -1
    for char in token:
        position = folded.find(char, position + 1)
        if position < 0:
            return False
    return True


def fuzzy_match(query: str, text: str | Prepared) -> FuzzyMatch | None:
    folded_query = " ".join(_fold(unicodedata.normalize("NFC", query)).split())
    if not folded_query:
        return None
    tokens = folded_query.split(" ")
    candidate = text if isinstance(text, Prepared) else None
    folded = candidate.folded if candidate else _fold(unicodedata.normalize("NFC", text))
    if not all(_is_subsequence(token, folded) for token in tokens):
        return None  # cheap rejection: most candidates stop here
    if candidate is None:
        candidate = prepare(text)
    bonuses = list(candidate.bonuses)

    total, positions = 0, set()
    for token in tokens:
        matched = _match_token(token, folded, bonuses)
        if matched is None:
            return None
        total += matched[0]
        positions.update(matched[1])

    if folded == folded_query:
        total += _EXACT
    elif folded.startswith(folded_query):
        total += _PREFIX
    elif any(bonuses[start] for start in _occurrences(folded, folded_query)):
        total += _SUBSTRING
    return FuzzyMatch(total, tuple(sorted(positions)))
