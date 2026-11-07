"""Split text into search terms on Unicode word boundaries (UAX #29).

``tokenize(text)`` returns the lowercased words of ``text`` in order:
1. NFC-normalise (macOS file names are NFD, typed text usually NFC);
2. segment with the UAX #29 word-boundary rules WB3c-WB13b;
3. keep the segments that contain a letter or a digit (drop spaces,
   punctuation, emoji and bare ``_`` runs);
4. remove format characters such as soft hyphens (``co\\u00adop`` -> ``coop``)
   and lowercase.

No stemming and no stop words. Numbers are kept, and UAX #29 keeps
``2503.11926``, ``3,000``, ``v1.2``, ``don't``, ``e.g`` and ``snake_case``
together as single words. Hyphens and slashes split: ``gpt-4o`` -> ``gpt``, ``4o``.
Ideographic and Hiragana text comes out as one character per term, as in
UAX #29 (no dictionary segmentation).

The rules are written as one pattern over the ``Word_Break`` property classes
of the ``regex`` module, not with its ``\\b``. ``regex``'s ``\\b`` attaches a
leading apostrophe or soft hyphen to the next word (``'active'`` -> ``'active``).
Lines are segmented one at a time because a newline is always a boundary
(WB3a/b), and are further split at characters that are always boundaries
(dashes, arrows, ...). Pure-ASCII pieces, the common case, take an equivalent
stdlib ``re`` pattern that runs several times faster. Both paths are fuzz-tested against an
independent UAX #29 implementation (``uniseg``) in the tests.
"""

from __future__ import annotations

import re
import unicodedata

import regex

# Bump when the terms produced for the same text can change. The index re-reads
# every note when the stored value differs (see index.SEARCH_VERSION).
TOKENIZER_VERSION = 1

# Word_Break property classes (UAX #29, table 3). AHLetter = ALetter | Hebrew_Letter.
_E = r"\p{WB=Extend}\p{WB=Format}\p{WB=ZWJ}"  # absorbed into the preceding character (WB4)
_A = r"\p{WB=ALetter}\p{WB=Hebrew_Letter}"
_H = r"\p{WB=Hebrew_Letter}"
_N = r"\p{WB=Numeric}"
_K = r"\p{WB=Katakana}"
_X = r"\p{WB=ExtendNumLet}"
_ML = r"\p{WB=MidLetter}\p{WB=MidNumLet}\p{WB=Single_Quote}"  # MidLetter | MidNumLetQ
_MN = r"\p{WB=MidNum}\p{WB=MidNumLet}\p{WB=Single_Quote}"  # MidNum | MidNumLetQ
_SQ = r"\p{WB=Single_Quote}"
_DQ = r"\p{WB=Double_Quote}"

# A word is a sequence of runs. (X E*)+ is written X[XE]*, which is the same
# language and much faster. All quantifiers are possessive, so nothing backtracks.
#   letters: WB5, with WB6/7 joins (a'b, e.g) and WB7b/c (Hebrew " Hebrew)
_L_RUN = rf"[{_A}][{_A}{_E}]*+(?:[{_ML}][{_E}]*+[{_A}][{_A}{_E}]*+|(?<=[{_H}][{_E}]*)[{_DQ}][{_E}]*+[{_H}][{_A}{_E}]*+)*+"
#   digits: WB8, with WB11/12 joins (3,000  1.5)
_N_RUN = rf"[{_N}][{_N}{_E}]*+(?:[{_MN}][{_E}]*+[{_N}][{_N}{_E}]*+)*+"
_K_RUN = rf"[{_K}][{_K}{_E}]*+"  # WB13
_X_RUN = rf"[{_X}][{_X}{_E}]*+"  # WB13a/b join ExtendNumLet with everything except Mid*
_AN = rf"(?:{_L_RUN}|{_N_RUN})++"  # letters and digits join both ways (WB9/10)
# Katakana never touches letters/digits directly, only through ExtendNumLet.
_CORE = rf"(?:{_X_RUN}(?:{_AN}|{_K_RUN})?+|{_AN}|{_K_RUN})(?:{_X_RUN}(?:{_AN}|{_K_RUN})?+)*+"
# WB3c: ZWJ x Extended_Pictographic, after anything.
_ZWJ_EMOJI = rf"(?:(?<=\u200d)\p{{Extended_Pictographic}}[{_E}]*+)*+"
# WB7a: Hebrew_Letter x Single_Quote. Nothing can join after that quote, so it ends the word.
_TAIL = rf"(?:(?<=[{_H}][{_E}]*)[{_SQ}][{_E}]*+)?{_ZWJ_EMOJI}"
# Letters/digits with no word-joining class (Han, Hiragana, Thai, ...) are one-character segments (WB999).
_OTHER = rf"[[\p{{L}}\p{{N}}]--[{_A}{_N}{_K}{_X}{_E}{_ML}{_MN}{_DQ}]][{_E}]*+{_ZWJ_EMOJI}"

_WORD = regex.compile(rf"{_CORE}{_TAIL}|{_OTHER}", regex.V1)
_X_ONLY = regex.compile(rf"[{_X}{_E}]+", regex.V1)
_FORMAT = regex.compile(r"\p{WB=Format}+", regex.V1)

# The same rules restricted to ASCII: letters, digits, '_'; MidLetter/MidNumLetQ
# = : . '   MidNum/MidNumLetQ = , ; . '   (no Extend, Format, Katakana or Hebrew).
_ASCII_L = r"[A-Za-z]++(?:[:.'][A-Za-z]++)*+"
_ASCII_N = r"[0-9]++(?:[,;.'][0-9]++)*+"
_ASCII_AN = rf"(?:{_ASCII_L}|{_ASCII_N})++"
_ASCII_WORD = re.compile(rf"(?:_++(?:{_ASCII_AN})?+|{_ASCII_AN})(?:_++(?:{_ASCII_AN})?+)*+")


# Characters that are always a boundary on both sides and never part of a word:
# Word_Break=Other, and not a letter, digit or pictograph (those can start a
# word or join after a ZWJ). No rule joins across them, and characters absorbed
# after them (WB4) cannot start a word. Splitting at them gives the same words
# and lets the ASCII parts of a line (around an em dash, say) take the fast path.
_SEPARATORS = regex.compile(r"[[\p{WB=Other}]--[\p{L}\p{N}\p{Extended_Pictographic}]]+", regex.V1)


def word_segments(text: str) -> list[str]:
    """UAX #29 word segments of ``text`` that contain a letter or digit, unnormalised."""
    found: list[str] = []
    for line in text.split("\n"):
        for part in (line,) if line.isascii() else _SEPARATORS.split(line):
            if part.isascii():
                found.extend(word for word in _ASCII_WORD.findall(part) if word.strip("_"))
            else:
                found.extend(word for word in _WORD.findall(part) if word[0].isalnum() or not _X_ONLY.fullmatch(word))
    return found


def word_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets in ``text`` of the same segments ``word_segments`` returns.

    Slower than ``word_segments``; used to place snippets, not for indexing.
    """
    spans: list[tuple[int, int]] = []
    line_start = 0
    for line in text.split("\n"):
        if line.isascii():
            parts = [(0, len(line))]
        else:
            parts, position = [], 0
            for separator in _SEPARATORS.finditer(line):
                parts.append((position, separator.start()))
                position = separator.end()
            parts.append((position, len(line)))
        for part_start, part_end in parts:
            part = line[part_start:part_end]
            offset = line_start + part_start
            if part.isascii():
                spans.extend((offset + m.start(), offset + m.end()) for m in _ASCII_WORD.finditer(part) if m.group().strip("_"))
            else:
                spans.extend(
                    (offset + m.start(), offset + m.end())
                    for m in _WORD.finditer(part)
                    if m.group()[0].isalnum() or not _X_ONLY.fullmatch(m.group())
                )
        line_start += len(line) + 1
    return spans


def normalize_term(word: str) -> str:
    """A segment as a search term: format characters removed, lowercased (text already NFC)."""
    if not word.isascii():
        word = _FORMAT.sub("", word)
    return word.lower()


def tokenize(text: str) -> list[str]:
    """Lowercased search terms of ``text``, in order (see module docstring)."""
    # normalize_term inlined: this runs once per word of every note.
    return [
        word.lower() if word.isascii() else _FORMAT.sub("", word).lower()
        for word in word_segments(unicodedata.normalize("NFC", text))
    ]
