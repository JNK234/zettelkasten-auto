"""Near-miss spellings of a search word among the indexed terms.

Distance is optimal string alignment (restricted Damerau-Levenshtein): an
insertion, deletion, substitution or swap of two adjacent characters each cost
one edit, so ``zettelkatsen`` is one edit from ``zettelkasten``.

How many edits a query word may be off by depends on its length in characters:
0 for words of 1-3 characters, 1 for 4-7, 2 for 8 or more. Words containing a
digit are never matched fuzzily: ``2503.11926`` and ``2503.11927`` are different
papers, not a typo of each other.

``close_terms`` walks the sorted vocabulary as if it were a trie. It keeps one
dynamic-programming row per character of the current prefix, reuses the rows
shared with the previous term, computes only the band of cells that can still
be within budget, and skips every term under a prefix that is already over
budget with one binary search. The result is the same as checking every term
(tested), at a fraction of the cost.
"""

from __future__ import annotations

import bisect
from typing import Sequence

_FAR = 1 << 20


def edit_budget(word: str) -> int:
    if any(char.isdigit() for char in word):
        return 0
    length = len(word)
    return 0 if length < 4 else 1 if length <= 7 else 2


def osa_distance(a: str, b: str) -> int:
    """Reference implementation (full table)."""
    rows = [list(range(len(b) + 1))]
    for i in range(1, len(a) + 1):
        row = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            value = min(rows[i - 1][j] + 1, row[j - 1] + 1, rows[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                value = min(value, rows[i - 2][j - 2] + 1)
            row[j] = value
        rows.append(row)
    return rows[-1][-1]


def close_terms(word: str, budget: int, vocabulary: Sequence[str]) -> dict[str, int]:
    """{term: distance} for the terms of sorted ``vocabulary`` within ``budget`` edits of ``word``."""
    m = len(word)
    found: dict[str, int] = {}
    # rows[d] holds distances between word[:j] and the first d characters of the current prefix.
    rows = [[j if j <= budget else _FAR for j in range(m + 1)]]
    prefix = ""
    index, count = 0, len(vocabulary)
    while index < count:
        term = vocabulary[index]
        shared = 0
        limit = min(len(prefix), len(term))
        while shared < limit and prefix[shared] == term[shared]:
            shared += 1
        del rows[shared + 1 :]
        prefix = term[:shared]
        pruned = False
        for d in range(shared + 1, len(term) + 1):
            char = term[d - 1]
            above = rows[d - 1]
            above2 = rows[d - 2] if d >= 2 else None
            row = [_FAR] * (m + 1)
            if d <= budget:
                row[0] = d
            best = row[0]
            for j in range(max(1, d - budget), min(m, d + budget) + 1):
                value = above[j - 1] + (word[j - 1] != char)
                if above[j] + 1 < value:
                    value = above[j] + 1
                if row[j - 1] + 1 < value:
                    value = row[j - 1] + 1
                if d > 1 and j > 1 and char == word[j - 2] and term[d - 2] == word[j - 1] and above2[j - 2] + 1 < value:
                    value = above2[j - 2] + 1
                row[j] = value
                if value < best:
                    best = value
            rows.append(row)
            prefix = term[:d]
            if best > budget:
                # Every term starting with term[:d] is over budget too: jump past them.
                index = bisect.bisect_left(vocabulary, term[: d - 1] + chr(ord(char) + 1), index + 1)
                pruned = True
                break
        if pruned:
            continue
        distance = rows[len(term)][m]
        if distance <= budget:
            found[term] = distance
        index += 1
    return found
