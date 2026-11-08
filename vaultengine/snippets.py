"""Show where a search hit matched: the title and a short piece of the note with the matches marked.

Snippets are made from the note file as it is now (read again for the hits that
are shown), tokenised exactly as the index does, with character offsets. So the
marks are the same words the ranking counted, and a snippet never points into
text that has since changed.

Which text is shown: the stretch of body text (within ``WINDOW`` positions)
that contains the most different query clauses, then the most exact matches,
then the one where those clauses are closest together, then the most matches,
then the earliest. A note that matched only in a heading, alias or tag shows
that heading, alias or tag instead. A title match is marked in the title.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from vaultengine.parser import normalize_text, parse_note_with_layout
from vaultengine.search import VALUE_GAP, Hit, Near, Phrase, Word, field_segments, parse_query
from vaultengine.tokenize import normalize_term, word_spans
from vaultengine.vault import link_name

WINDOW = 24  # positions a body snippet is chosen within
_SPACE = re.compile(r"\s+")


@dataclass
class Piece:
    text: str
    mark: str | None = None  # None, "exact" or "fuzzy"


@dataclass
class Snippet:
    field: str  # "body", "headings", "aliases" or "tags"
    pieces: list[Piece]
    truncated_start: bool = False
    truncated_end: bool = False

    @property
    def text(self) -> str:
        return "".join(piece.text for piece in self.pieces)

    def to_dict(self) -> dict:
        marks, position = [], 0
        for piece in self.pieces:
            if piece.mark:
                marks.append([position, position + len(piece.text), piece.mark])
            position += len(piece.text)
        return {"field": self.field, "text": self.text, "marks": marks,
                "truncated_start": self.truncated_start, "truncated_end": self.truncated_end}


@dataclass
class HitContext:
    title: list[Piece]
    snippet: Snippet | None


@dataclass
class _Token:
    term: str
    position: int
    start: int  # offsets in the value text (title, alias, tag) or in the note text (heading, body)
    end: int
    value: int  # index of the value within the field


@dataclass
class _Mark:
    first: int  # token indexes, inclusive
    last: int
    clause: str
    kind: str  # "exact" | "fuzzy"


def _tokens(values: list[tuple[str, int | None]]) -> list[_Token]:
    tokens, position = [], 0
    for index, (text, offset) in enumerate(values):
        base = offset or 0
        spans = word_spans(text)
        for start, end in spans:
            tokens.append(_Token(normalize_term(text[start:end]), position, base + start, base + end, index))
            position += 1
        if spans:  # as search.postings: no gap after a value without words
            position += VALUE_GAP
    return tokens


def _marks(tokens: list[_Token], hit: Hit, clauses) -> list[_Mark]:
    marks = []
    for clause in clauses:
        match = hit.matches.get(clause.label)
        if match is None:
            continue
        kind = "exact" if match.exact else "fuzzy"
        if isinstance(clause, Word):
            marks += [_Mark(i, i, clause.label, kind) for i, t in enumerate(tokens) if t.term == match.term]
        elif isinstance(clause, Phrase):
            n = len(clause.terms)
            for i in range(len(tokens) - n + 1):
                if all(tokens[i + k].term == clause.terms[k] and tokens[i + k].position == tokens[i].position + k for k in range(n)):
                    marks.append(_Mark(i, i + n - 1, clause.label, kind))
        elif isinstance(clause, Near):
            for i, a in enumerate(tokens):
                if a.term not in (clause.left, clause.right):
                    continue
                other = clause.right if a.term == clause.left else clause.left
                if any(b.term == other and j != i and abs(b.position - a.position) <= clause.distance for j, b in enumerate(tokens)):
                    marks.append(_Mark(i, i, clause.label, kind))
    return sorted(marks, key=lambda m: (m.first, m.last))


def _pieces(text: str, start: int, end: int, tokens: list[_Token], marks: list[_Mark]) -> list[Piece]:
    """text[start:end] as pieces, marked spans separate, whitespace collapsed."""
    spans = []
    for mark in marks:
        a, b = tokens[mark.first].start, tokens[mark.last].end
        if a >= start and b <= end:
            if spans and a < spans[-1][1]:  # overlapping marks (e.g. a phrase and one of its words)
                prev = spans[-1]
                spans[-1] = (prev[0], max(prev[1], b), "exact" if "exact" in (prev[2], mark.kind) else "fuzzy")
            else:
                spans.append((a, b, mark.kind))
    pieces, position = [], start
    for a, b, kind in spans:
        if a > position:
            pieces.append(Piece(_SPACE.sub(" ", text[position:a])))
        pieces.append(Piece(_SPACE.sub(" ", text[a:b]), kind))
        position = b
    if position < end:
        pieces.append(Piece(_SPACE.sub(" ", text[position:end])))
    return pieces


def _body_snippet(text: str, tokens: list[_Token], marks: list[_Mark], width: int) -> Snippet:
    best_key, best = None, []
    for index, first in enumerate(marks):
        inside = [m for m in marks[index:] if tokens[m.last].position - tokens[first.first].position < WINDOW]
        covered = {m.clause for m in inside}
        # The tightest stretch that still covers those clauses (one mark per clause).
        spread = max(tokens[min((m for m in inside if m.clause == c), key=lambda m: m.first).last].position for c in covered)
        spread -= tokens[first.first].position
        key = (len(covered), sum(m.kind == "exact" for m in inside), -spread, len(inside), -first.first)
        if best_key is None or key > best_key:
            best_key, best = key, inside
    lo, hi = best[0].first, max(m.last for m in best)
    value = tokens[lo].value
    while hi > lo and tokens[hi].end - tokens[lo].start > width:  # keep the first match when the cluster is too wide
        hi -= 1
    # Grow by whole words, alternating right and left, while the text fits.
    grew = True
    while grew:
        grew = False
        for step in (1, -1):
            i = hi + 1 if step == 1 else lo - 1
            if 0 <= i < len(tokens) and tokens[i].value == value:
                a, b = (tokens[lo].start, tokens[i].end) if step == 1 else (tokens[i].start, tokens[hi].end)
                if b - a <= width:
                    lo, hi = (lo, i) if step == 1 else (i, hi)
                    grew = True
    start, end = tokens[lo].start, tokens[hi].end
    return Snippet(
        "body",
        _pieces(text, start, end, tokens, [m for m in marks if m.first >= lo and m.last <= hi]),
        truncated_start=lo > 0 and tokens[lo - 1].value == value,
        truncated_end=hi + 1 < len(tokens) and tokens[hi + 1].value == value,
    )


def hit_context(root: str | Path, hit: Hit, query: str, width: int = 100) -> HitContext:
    clauses = parse_query(query)
    title_text = link_name(hit.path)
    title_tokens = _tokens([(title_text, None)])
    title = _pieces(title_text, 0, len(title_text), title_tokens, _marks(title_tokens, hit, clauses))
    try:
        raw = (Path(root) / hit.path).read_bytes()
    except OSError:
        return HitContext(title, None)
    text = unicodedata.normalize("NFC", normalize_text(raw.decode("utf-8", errors="replace")))
    note, layout = parse_note_with_layout(text, hit.path)
    segments = field_segments(hit.path, note, layout)

    tokens = _tokens(segments["body"])
    marks = _marks(tokens, hit, clauses)
    if marks:
        return HitContext(title, _body_snippet(text, tokens, marks, width))
    for name in ("headings", "aliases", "tags"):
        values = segments[name]
        tokens = _tokens(values)
        marks = _marks(tokens, hit, clauses)
        if not marks:
            continue
        value = tokens[marks[0].first].value
        chosen = [m for m in marks if tokens[m.first].value == value]
        if name == "headings":
            heading, span = note.headings[value], layout.headings[value]
            # The heading text starts with the space after the '#'s, which collapses to one.
            pieces = [Piece("#" * heading.level)] + _pieces(text, span.text_start, span.text_end, tokens, chosen)
        else:
            value_text = values[value][0]  # alias and tag offsets are within the value itself
            prefix = "alias: " if name == "aliases" else "#"
            pieces = [Piece(prefix)] + _pieces(value_text, 0, len(value_text), tokens, chosen)
        return HitContext(title, Snippet(name, [p for p in pieces if p.text]))
    return HitContext(title, None)


@dataclass
class PreviewLine:
    number: int  # 1-based, as in an editor
    text: str
    marks: list[tuple[int, int, str]] = field(default_factory=list)  # (start, end, "exact" | "fuzzy") in the line


def preview(root: str | Path, path: str, hit: Hit | None, query: str | None) -> list[PreviewLine] | None:
    """The note's lines with every match of the query marked, as the ranking counted them (None: file gone).

    Matches are marked in the body and headings (the text a reader sees); frontmatter
    values and tags are shown but not marked.
    """
    try:
        raw = (Path(root) / path).read_bytes()
    except OSError:
        return None
    text = unicodedata.normalize("NFC", normalize_text(raw.decode("utf-8", errors="replace")))
    spans: list[tuple[int, int, str]] = []
    if hit is not None and query:
        clauses = parse_query(query)
        note, layout = parse_note_with_layout(text, path)
        segments = field_segments(path, note, layout)
        for name in ("body", "headings"):
            tokens = _tokens(segments[name])
            spans += [(tokens[m.first].start, tokens[m.last].end, m.kind) for m in _marks(tokens, hit, clauses)]
    spans.sort()
    merged: list[tuple[int, int, str]] = []
    for start, end, kind in spans:  # a phrase and one of its words overlap: one mark, exact wins
        if merged and start < merged[-1][1]:
            a, b, k = merged[-1]
            merged[-1] = (a, max(b, end), "exact" if "exact" in (k, kind) else "fuzzy")
        else:
            merged.append((start, end, kind))
    lines, offset = [], 0
    for number, line in enumerate(text.split("\n"), 1):
        end_of_line = offset + len(line)
        marks = [(max(a, offset) - offset, min(b, end_of_line) - offset, kind)
                 for a, b, kind in merged if a < end_of_line and b > offset]
        lines.append(PreviewLine(number, line, [m for m in marks if m[1] > m[0]]))
        offset = end_of_line + 1
    return lines
