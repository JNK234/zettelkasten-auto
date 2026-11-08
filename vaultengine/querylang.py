"""The vault query language: lexer, syntax tree and parser.

    query      := or
    or         := and ("OR" and)*
    and        := unary (["AND"] unary | "NEAR/n" word)*
    unary      := ("-" | "NOT") unary | primary
    primary    := "(" or ")" | "\"phrase\"" | word | field | "[" property "]"
    field      := tag:T | path:P | links-to:R | linked-from:R | within:N of:R
                | has:X | missing:X | modified:D
    property   := key | key (":" | "=" | "<" | "<=" | ">" | ">=") value
    R          := [[Note]] | "Note" | Note
    D          := [op] date | date..date        date = YYYY | YYYY-MM | YYYY-MM-DD

Precedence: NOT binds tightest, then AND (also implicit, by writing things
next to each other), then OR. ``OR``, ``AND``, ``NOT`` and ``NEAR/n`` are
operators only in capitals; in lowercase they are words.

In a sequence of things written next to each other, every part must match,
except the bare words: of those, at least one must match (they rank the results,
as in ``vault search``). ``AND`` makes a word required too (``(and a b)`` in the
tree; spelling variants still count). A word written
after ``-`` or ``NOT`` is matched exactly, without spelling variants.

``parse(query)`` returns the syntax tree (``None`` for an empty query) or raises
``QueryError`` with the exact span of the problem; ``QueryError.render(query)``
shows it with a caret line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Union

from vaultengine.properties import Period, number, parse_period
from vaultengine.spelling import osa_distance
import unicodedata

from vaultengine.tokenize import normalize_term, tokenize, word_spans

MAX_NEAR = 99  # positions of different headings/aliases are 100 apart (search.VALUE_GAP)
MAX_HOPS = 10
FIELDS = ("tag", "path", "links-to", "linked-from", "within", "of", "has", "missing", "modified")
HAS_BUILTINS = ("tags", "aliases", "headings", "links", "backlinks", "unresolved", "frontmatter", "properties")
COMPARISONS = (">=", "<=", ">", "<")


class QueryError(ValueError):
    def __init__(self, message: str, start: int | None = None, end: int | None = None, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.start = start
        self.end = end if end is not None else (start + 1 if start is not None else None)
        self.hint = hint

    def render(self, query: str) -> str:
        """The message, the query, a caret line under the problem, and the hint."""
        lines = [f"error: {self.message}"]
        if self.start is not None:
            lines.append("  " + query)
            lines.append("  " + " " * self.start + "^" * max(1, self.end - self.start))
        if self.hint:
            lines.append(f"  {self.hint}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"message": self.message, "start": self.start, "end": self.end, "hint": self.hint}


# ----- syntax tree ---------------------------------------------------------------

Span = tuple[int, int]


def _q(text: str) -> str:
    return '"' + text.replace('"', '\\"') + '"'


@dataclass(frozen=True)
class Word:
    term: str
    exact: bool = False  # quoted (or under negation): exact spelling only
    span: Span = field(default=(0, 0), compare=False)

    @property
    def label(self) -> str:
        return f'"{self.term}"' if self.exact else self.term

    @property
    def required(self) -> bool:
        return self.exact

    def sexpr(self) -> str:
        return f"(exact {self.term})" if self.exact else self.term


@dataclass(frozen=True)
class Phrase:
    terms: tuple[str, ...]
    span: Span = field(default=(0, 0), compare=False)
    required = True

    @property
    def label(self) -> str:
        return '"' + " ".join(self.terms) + '"'

    def sexpr(self) -> str:
        return f"(phrase {_q(' '.join(self.terms))})"


@dataclass(frozen=True)
class Near:
    left: str
    right: str
    distance: int
    span: Span = field(default=(0, 0), compare=False)
    required = True

    @property
    def label(self) -> str:
        return f"{self.left} NEAR/{self.distance} {self.right}"

    def sexpr(self) -> str:
        return f"(near {self.left} {self.right} {self.distance})"


@dataclass(frozen=True)
class And:
    required: tuple["Node", ...]
    any: tuple[Word, ...]  # bare words: at least one must match
    span: Span = field(default=(0, 0), compare=False)

    def sexpr(self) -> str:
        parts = [n.sexpr() for n in self.required]
        if self.any:
            parts.append("(any " + " ".join(w.sexpr() for w in self.any) + ")")
        return "(and " + " ".join(parts) + ")"


@dataclass(frozen=True)
class Or:
    items: tuple["Node", ...]
    span: Span = field(default=(0, 0), compare=False)

    def sexpr(self) -> str:
        return "(or " + " ".join(n.sexpr() for n in self.items) + ")"


@dataclass(frozen=True)
class Not:
    item: "Node"
    span: Span = field(default=(0, 0), compare=False)

    def sexpr(self) -> str:
        return f"(not {self.item.sexpr()})"


@dataclass(frozen=True)
class Tag:
    name: str  # without '#'; also matches nested tags below it
    span: Span = field(default=(0, 0), compare=False)

    def sexpr(self) -> str:
        return f"(tag {self.name})"


@dataclass(frozen=True)
class Path:
    text: str  # case-insensitive substring of the note's path
    span: Span = field(default=(0, 0), compare=False)

    def sexpr(self) -> str:
        return f"(path {_q(self.text)})"


@dataclass(frozen=True)
class Prop:
    key: str
    op: str | None = None  # None: the note has the property
    value: str | None = None
    number: float | None = field(default=None, compare=False)  # the value read as a number
    period: Union[Period, tuple[Period, Period], None] = field(default=None, compare=False)  # ... or as a date
    span: Span = field(default=(0, 0), compare=False)

    def sexpr(self) -> str:
        if self.op is None:
            return f"(has-property {self.key})"
        return f"(property {self.key} {self.op} {_q(self.value)})"


@dataclass(frozen=True)
class Has:
    what: str  # one of HAS_BUILTINS
    span: Span = field(default=(0, 0), compare=False)

    def sexpr(self) -> str:
        return f"(has {self.what})"


@dataclass(frozen=True)
class LinksTo:
    ref: str
    span: Span = field(default=(0, 0), compare=False)

    def sexpr(self) -> str:
        return f"(links-to {_q(self.ref)})"


@dataclass(frozen=True)
class LinkedFrom:
    ref: str
    span: Span = field(default=(0, 0), compare=False)

    def sexpr(self) -> str:
        return f"(linked-from {_q(self.ref)})"


@dataclass(frozen=True)
class Within:
    hops: int
    ref: str
    span: Span = field(default=(0, 0), compare=False)

    def sexpr(self) -> str:
        return f"(within {self.hops} {_q(self.ref)})"


@dataclass(frozen=True)
class Modified:
    op: str  # "=", ">", ">=", "<", "<=" or "range"
    start: Period
    end: Period | None = None  # for ranges
    span: Span = field(default=(0, 0), compare=False)

    def sexpr(self) -> str:
        if self.op == "range":
            return f"(modified {self.start.text}..{self.end.text})"
        return f"(modified {self.op} {self.start.text})"


Node = Union[Word, Phrase, Near, And, Or, Not, Tag, Path, Prop, Has, LinksTo, LinkedFrom, Within, Modified]


def text_clauses(node: Node | None, positive: bool = True) -> list[Word | Phrase | Near]:
    """Words, phrases and NEAR pairs that count for ranking: not under a negation, no duplicates."""
    found: dict[str, Word | Phrase | Near] = {}

    def walk(n: Node, positive: bool) -> None:
        if isinstance(n, (Word, Phrase, Near)):
            if positive:
                found.setdefault(n.label, n)
        elif isinstance(n, And):
            for child in (*n.required, *n.any):
                walk(child, positive)
        elif isinstance(n, Or):
            for child in n.items:
                walk(child, positive)
        elif isinstance(n, Not):
            walk(n.item, not positive)

    if node is not None:
        walk(node, positive)
    return sorted(found.values(), key=lambda clause: clause.span[0])  # in the order they were written


# ----- lexer -------------------------------------------------------------------


@dataclass
class Token:
    kind: str  # ( ) OR AND NOT NEG NEAR PHRASE WORDS FIELD PROP
    start: int
    end: int
    text: str = ""
    value: object = None
    value_span: Span = (0, 0)


_KEYWORD_NEAR = re.compile(r"NEAR/(\d+)")
_DATE_SHAPED = re.compile(r"\d{4}(?:-\d{1,2}){0,2}(?:\.\.\d{4}(?:-\d{1,2}){0,2})?")
_FIELD_START = re.compile(r"([A-Za-z][A-Za-z0-9_-]*):")
_CLOSE_QUOTE = {'"': '"', "\u201c": "\u201d"}


def _suggest(name: str, options) -> str | None:
    best = min(options, key=lambda o: osa_distance(name, o))
    return best if osa_distance(name, best) <= 2 else None


def _read_quoted(query: str, start: int) -> tuple[str, int]:
    """(text, end) for a quote opening at ``start``."""
    close = query.find(_CLOSE_QUOTE[query[start]], start + 1)
    if query[start] == "\u201c" and close == -1:
        close = query.find('"', start + 1)
    if close == -1:
        raise QueryError("this quote is never closed", start, start + 1, 'add the closing " at the end of the phrase')
    return query[start + 1 : close], close + 1


def _read_link(query: str, start: int) -> tuple[str, int]:
    close = query.find("]]", start + 2)
    if close == -1:
        raise QueryError("this [[ is never closed", start, start + 2, "add ]] after the note name")
    return query[start + 2 : close], close + 2


def lex(query: str) -> list[Token]:
    tokens: list[Token] = []
    i, n = 0, len(query)
    while i < n:
        char = query[i]
        if char.isspace():
            i += 1
        elif char in "()":
            tokens.append(Token(char, i, i + 1, char))
            i += 1
        elif char in _CLOSE_QUOTE:
            text, end = _read_quoted(query, i)
            tokens.append(Token("PHRASE", i, end, query[i:end], text, (i + 1, end - 1)))
            i = end
        elif char == "-" and i + 1 < n and not query[i + 1].isspace() and query[i + 1] not in ")-":
            tokens.append(Token("NEG", i, i + 1, "-"))
            i += 1
        elif char == "[":
            if query.startswith("[[", i):
                _, end = _read_link(query, i)
                raise QueryError(
                    "a [[link]] needs a field in front of it", i, end,
                    "write links-to:[[…]], linked-from:[[…]] or within:N of:[[…]]",
                )
            tokens.append(_lex_property(query, i))
            i = tokens[-1].end
        else:
            field_match = _FIELD_START.match(query, i)
            if field_match:
                tokens.append(_lex_field(query, i, field_match))
                i = tokens[-1].end
                continue
            end = i
            while end < n and not query[end].isspace() and query[end] not in '()"\u201c':
                end += 1
            chunk = query[i:end]
            near = _KEYWORD_NEAR.fullmatch(chunk)
            if chunk in ("OR", "AND", "NOT"):
                tokens.append(Token(chunk, i, end, chunk))
            elif near:
                tokens.append(Token("NEAR", i, end, chunk, int(near.group(1))))
            else:
                words = _words_at(chunk, i)
                if words:
                    tokens.append(Token("WORDS", i, end, chunk, words))
            i = end
    return tokens


def _words_at(chunk: str, offset: int) -> list[tuple[str, int, int]]:
    """(term, start, end) for each word of ``chunk``, with offsets in the query."""
    if unicodedata.is_normalized("NFC", chunk):
        return [(normalize_term(chunk[a:b]), offset + a, offset + b) for a, b in word_spans(chunk)]
    return [(term, offset, offset + len(chunk)) for term in tokenize(chunk)]  # offsets approximate


def _lex_field(query: str, start: int, match: re.Match) -> Token:
    name = match.group(1).lower()
    if name not in FIELDS:
        suggestion = _suggest(name, FIELDS)
        hint = f"did you mean {suggestion}:? " if suggestion else ""
        hint += f'fields are {", ".join(f + ":" for f in FIELDS)}; to search for the text, put it in quotes'
        raise QueryError(f'unknown field "{match.group(1)}:"', start, match.end(), hint)
    i = match.end()
    n = len(query)
    if i < n and query[i] in _CLOSE_QUOTE:
        value, end = _read_quoted(query, i)
        value_span = (i + 1, end - 1)
    elif query.startswith("[[", i):
        value, end = _read_link(query, i)
        value_span = (i + 2, end - 2)
    else:
        end = i
        while end < n and not query[end].isspace() and query[end] != ")":
            end += 1
        value, value_span = query[i:end], (i, end)
    if not value.strip():
        raise QueryError(f"{name}: needs a value", start, end, _FIELD_EXAMPLES[name])
    return Token("FIELD", start, end, query[start:end], (name, value.strip()), value_span)


_FIELD_EXAMPLES = {
    "tag": "e.g. tag:ai-safety",
    "path": "e.g. path:20_ZETTLEKASTEN or path:\"50_MOC/discourse\"",
    "links-to": "e.g. links-to:[[Note name]]",
    "linked-from": "e.g. linked-from:[[Note name]]",
    "within": "e.g. within:2 of:[[Note name]]",
    "of": "e.g. within:2 of:[[Note name]]",
    "has": "e.g. has:tags, has:backlinks or has:status (a property)",
    "missing": "e.g. missing:tags or missing:status (a property)",
    "modified": "e.g. modified:>=2026-09-01, modified:2026-09 or modified:2026-01..2026-03",
}


def _lex_property(query: str, start: int) -> Token:
    """``[key]`` or ``[key op value]``; the value may be quoted or a [[link]]."""
    i, n = start + 1, len(query)
    while i < n and query[i] not in ":=<>]":
        i += 1
    key = query[start + 1 : i].strip()
    if i >= n:
        raise QueryError("this [ is never closed", start, start + 1, "add ] after the property, e.g. [status:done]")
    if not key:
        raise QueryError("a property needs a name", start, i + 1, "e.g. [status], [status:done] or [created>=2026-09-01]")
    if query[i] == "]":
        return Token("PROP", start, i + 1, query[start : i + 1], (key, None, None), (i, i))
    op = next(o for o in (">=", "<=", ">", "<", "=", ":") if query.startswith(o, i))
    j = i + len(op)
    while j < n and query[j] == " ":
        j += 1
    if j < n and query[j] in _CLOSE_QUOTE:
        value, value_end = _read_quoted(query, j)
        value_span = (j + 1, value_end - 1)
    elif query.startswith("[[", j):
        _, value_end = _read_link(query, j)
        value, value_span = query[j:value_end], (j, value_end)
    else:
        value_end = j
        while value_end < n and query[value_end] != "]":
            value_end += 1
        value, value_span = query[j:value_end].strip(), (j, value_end)
    close = value_end
    while close < n and query[close] == " ":
        close += 1
    if close >= n or query[close] != "]":
        raise QueryError("this [ is never closed", start, start + 1, "add ] after the value, e.g. [status:done]")
    if not value:
        raise QueryError(f"[{key}{op}] needs a value after {op}", start, close + 1, f"e.g. [{key}{op}done], or [{key}] for any value")
    return Token("PROP", start, close + 1, query[start : close + 1], (key, op, value), value_span)


# ----- parser ------------------------------------------------------------------


def _period(text: str, span: Span) -> Period:
    period = parse_period(text)
    if period is None:
        raise QueryError(f'"{text}" is not a date', *span, "write a year, month or day: 2026, 2026-09 or 2026-09-24")
    return period


def _date_target(text: str, span: Span) -> Period | tuple[Period, Period]:
    if ".." in text:
        left, right = text.split("..", 1)
        cut = span[0] + len(left)
        return _period(left, (span[0], cut)), _period(right, (cut + 2, span[1]))
    return _period(text, span)


def _reference(value: str) -> str:
    """A note reference as written: [[ ]] already removed; drop |alias and #heading."""
    return value.split("|", 1)[0].split("#", 1)[0].strip()


class _Parser:
    def __init__(self, query: str) -> None:
        self.query = query
        self.tokens = lex(query)
        self.i = 0

    def peek(self, offset: int = 0) -> Token | None:
        j = self.i + offset
        return self.tokens[j] if j < len(self.tokens) else None

    def next(self) -> Token:
        token = self.tokens[self.i]
        self.i += 1
        return token

    def at_stop(self) -> bool:
        token = self.peek()
        return token is None or token.kind in (")", "OR", "AND")

    def parse(self) -> Node | None:
        if not self.tokens:
            return None
        node = self.parse_or()
        token = self.peek()
        if token is not None:  # only ')' can stop parse_or early
            raise QueryError("this ) has no matching (", token.start, token.end, "remove it or add ( before it")
        return node

    def parse_or(self) -> Node:
        token = self.peek()
        if token is not None and token.kind in ("OR", "AND"):
            raise QueryError(f"{token.kind} needs something before it", token.start, token.end)
        items = [self.parse_and()]
        while (token := self.peek()) is not None and token.kind == "OR":
            self.next()
            if self.at_stop():
                raise QueryError("OR needs something after it", token.start, token.end)
            items.append(self.parse_and())
        if len(items) == 1:
            return items[0]
        return Or(tuple(items), (_start(items[0]), _end(items[-1])))

    def parse_and(self) -> Node:
        units: list[list] = []  # [node, required]; bare words are required only when joined by AND
        joined = False
        while not ((token := self.peek()) is None or token.kind in (")", "OR")):
            if token.kind == "AND":
                self.next()
                if not units:
                    raise QueryError("AND needs something before it", token.start, token.end)
                if self.at_stop():
                    raise QueryError("AND needs something after it", token.start, token.end)
                units[-1][1] = True
                joined = True
                continue
            if token.kind == "NEAR":
                self.parse_near(units)
                continue
            nodes = self.parse_unary()
            units.append([nodes[0], joined])
            units.extend([node, False] for node in nodes[1:])
            joined = False
        nodes = [_finish(node) for node, _ in units]
        if len(nodes) == 1:
            return nodes[0]
        required = tuple(node for node, (_, flag) in zip(nodes, units) if flag or not _bare(node))
        any_words = tuple(node for node, (_, flag) in zip(nodes, units) if not flag and _bare(node))
        return And(required, any_words, (_start(nodes[0]), _end(nodes[-1])))

    def parse_near(self, units: list[list]) -> None:
        token = self.next()
        distance = token.value
        example = "e.g. reward NEAR/3 model"
        if not 1 <= distance <= MAX_NEAR:
            raise QueryError(f"NEAR/n needs a distance from 1 to {MAX_NEAR}", token.start, token.end)
        left = units[-1][0] if units else None
        if isinstance(left, _NearOperand):
            raise QueryError("a word can only be in one NEAR/n: write `a NEAR/2 b b NEAR/2 c`", token.start, token.end)
        right = self.peek()
        right_terms: list[tuple[str, int, int]] = []
        if right is not None and right.kind == "WORDS":
            right_terms = right.value
        elif right is not None and right.kind == "PHRASE" and len(tokenize(right.value)) == 1:
            right_terms = [(tokenize(right.value)[0], right.start, right.end)]
        if not isinstance(left, Word) or not right_terms:
            raise QueryError(f"NEAR/{distance} needs a single word on each side", token.start, token.end, example)
        self.next()
        first = right_terms[0]
        units[-1][0] = _NearOperand(Near(left.term, first[0], distance, (left.span[0], first[2])))
        units.extend([Word(term, False, (a, b)), False] for term, a, b in right_terms[1:])
        following = self.peek()
        if following is not None and following.kind == "NEAR" and len(right_terms) == 1:
            raise QueryError("a word can only be in one NEAR/n: write `a NEAR/2 b b NEAR/2 c`", following.start, following.end)

    def parse_unary(self) -> list[Node]:
        token = self.peek()
        if token.kind in ("NEG", "NOT"):
            self.next()
            if self.at_stop() or self.peek().kind == "NEAR":
                raise QueryError(f"{token.text} needs something after it", token.start, token.end)
            nodes = self.parse_unary()
            if len(nodes) > 1:  # -gpt-4o: exclude the words as written, next to each other
                inner: Node = Phrase(tuple(n.term for n in nodes), (nodes[0].span[0], nodes[-1].span[1]))
            else:
                inner = nodes[0]
                if isinstance(inner, Word):
                    inner = Word(inner.term, True, inner.span)
            return [Not(inner, (token.start, _end(inner)))]
        return self.parse_primary()

    def parse_primary(self) -> list[Node]:
        token = self.next()
        kind = token.kind
        if kind == "(":
            closing = self.peek()
            if closing is not None and closing.kind == ")":
                raise QueryError("empty parentheses", token.start, closing.end)
            node = self.parse_or()
            closing = self.peek()
            if closing is None or closing.kind != ")":
                raise QueryError("this ( is never closed", token.start, token.end, "add ) after the group")
            self.next()
            return [node]
        if kind == ")":
            raise QueryError("this ) has no matching (", token.start, token.end, "remove it or add ( before it")
        if kind == "NEAR":
            raise QueryError(f"{token.text} needs a single word on each side", token.start, token.end, "e.g. reward NEAR/3 model")
        if kind == "PHRASE":
            terms = tokenize(token.value)
            if not terms:
                raise QueryError("nothing to search for inside these quotes", token.start, token.end)
            if len(terms) == 1:
                return [Word(terms[0], True, (token.start, token.end))]
            return [Phrase(tuple(terms), (token.start, token.end))]
        if kind == "WORDS":
            return [Word(term, False, (a, b)) for term, a, b in token.value]
        if kind == "PROP":
            return [self.property(token)]
        return [self.field(token)]

    def property(self, token: Token) -> Prop:
        key, op, value = token.value
        span = (token.start, token.end)
        if op is None:
            return Prop(key, span=span)
        # "2026" is both a number and a year: keep both readings, the note's value decides.
        period = None
        if _DATE_SHAPED.fullmatch(value):
            period = _date_target(value, token.value_span)  # raises "not a date" for 2026-13
            if isinstance(period, tuple) and op in COMPARISONS:
                raise QueryError(f"a range needs : or =, not {op}", *token.value_span, f"e.g. [{key}:{value}]")
        value_number = number(value)
        if op in COMPARISONS and value_number is None and period is None:
            raise QueryError(
                f'{op} compares numbers or dates, and "{value}" is neither', *token.value_span,
                "write a number, or a date as 2026, 2026-09 or 2026-09-24",
            )
        return Prop(key, op, value, value_number, period, span)

    def field(self, token: Token) -> Node:
        name, value = token.value
        span = (token.start, token.end)
        if name == "tag":
            return Tag(value.lstrip("#"), span)
        if name == "path":
            return Path(value, span)
        if name in ("links-to", "linked-from"):
            ref = _reference(value)
            if not ref:
                raise QueryError(f"{name}: needs a note", *token.value_span, _FIELD_EXAMPLES[name])
            return (LinksTo if name == "links-to" else LinkedFrom)(ref, span)
        if name == "has" or name == "missing":
            what = value.lower()
            node: Node = Has(what, span) if what in HAS_BUILTINS else Prop(value, span=span)
            return Not(node, span) if name == "missing" else node
        if name == "modified":
            match = re.fullmatch(r"(>=|<=|>|<|=)?(.*)", value)
            op, date_text = match.group(1) or "=", match.group(2)
            offset = token.value_span[0] + len(match.group(1) or "")
            target = _date_target(date_text, (offset, token.value_span[1]))
            if isinstance(target, tuple):
                if op != "=":
                    raise QueryError(f"a range needs no {op}", *token.value_span, f"e.g. modified:{date_text}")
                return Modified("range", target[0], target[1], span)
            return Modified(op, target, None, span)
        if name == "of":
            raise QueryError("of: goes with within:N", token.start, token.end, "e.g. within:2 of:[[Note name]]")
        # within:N of:R
        if not value.isdigit() or not 1 <= int(value) <= MAX_HOPS:
            raise QueryError(f"within: needs a number of links from 1 to {MAX_HOPS}", *token.value_span, "e.g. within:2 of:[[Note name]]")
        following = self.peek()
        if following is None or following.kind != "FIELD" or following.value[0] != "of":
            raise QueryError(f"within:{value} needs of:[[note]] right after it", token.start, token.end, f"e.g. within:{value} of:[[Note name]]")
        self.next()
        ref = _reference(following.value[1])
        return Within(int(value), ref, (token.start, following.end))


class _NearOperand:
    """Placeholder while parsing: a NEAR pair sitting where its left word was."""

    def __init__(self, near: Near) -> None:
        self.near = near
        self.span = near.span


def _bare(node) -> bool:
    return isinstance(node, Word) and not node.exact


def _start(node) -> int:
    return node.span[0]


def _end(node) -> int:
    return node.span[1]


def parse(query: str) -> Node | None:
    tree = _Parser(query).parse()
    return _finish(tree)


def _finish(node):
    """Replace parse-time placeholders (NEAR operands) in a finished tree."""
    if isinstance(node, _NearOperand):
        return node.near
    if isinstance(node, And):
        return And(tuple(_finish(n) for n in node.required), node.any, node.span)
    if isinstance(node, Or):
        return Or(tuple(_finish(n) for n in node.items), node.span)
    if isinstance(node, Not):
        return Not(_finish(node.item), node.span)
    return node
