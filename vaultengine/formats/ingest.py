"""`vault ingest`: turn a markdown, plain-text or saved-HTML file into a note of a type.

1. Convert. Markdown keeps its text (YAML frontmatter is dropped; its ``title`` and
   ``url``/``source``/``link`` are used). Plain text keeps its text; lines that would
   read as headings are escaped. HTML is reduced to its ``<article>``/``<main>``/
   ``<body>`` without scripts, styles, navigation, forms and embeds, links and images
   are made absolute against the page's canonical URL (or reduced to their text when
   there is none), and the rest goes through markdownify. The title comes from
   ``--title``, the document (frontmatter ``title``, ``og:title``, ``<title>`` minus a
   ``| Site`` suffix, the first H1, the first line of a text file) or the file name.
2. Fit it into the note. Its headings are shifted so that the highest is one level
   below the type's body heading; a first heading that repeats the title is dropped;
   ``key:: value`` lines are escaped (``key\\:: value``) so an imported document cannot
   add relations to the vault.
3. Duplicates. Against every note of the type: title similarity and content overlap
   (see ``similarity.py``), plus the content hash the type records. A score at or above
   its threshold, or the same hash, stops the ingest; the existing notes and their
   scores are reported and nothing is written.
4. Write through the safe write path (``write.py``): the new note must pass the check.
   Provenance goes into the type's provenance keys: the original file name, the
   SHA-256 of the file's bytes, the time of ingestion.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from vaultengine.formats import similarity
from vaultengine.formats.spec import NoteType
from vaultengine.formats.typed import TypedVault
from vaultengine.formats.write import WriteError, WritePlan, plan_new
from vaultengine.frontmatter import split_frontmatter
from vaultengine.parser import _INLINE_FIELD, normalize_text, parse_note
from vaultengine.vault import link_name

MARKDOWN_SUFFIXES = (".md", ".markdown", ".mdown", ".mkd")
TEXT_SUFFIXES = (".txt", ".text")
HTML_SUFFIXES = (".html", ".htm", ".xhtml")
_DROP_TAGS = ("script", "style", "noscript", "template", "svg", "iframe", "form", "button", "nav", "aside",
              "canvas", "video", "audio", "object", "embed", "select", "input")
_HEADING_LINE = re.compile(r"^( {0,3})(#{1,6})(?=[ \t]|$)")
_IDENTIFIER_PREFIX = re.compile(r"^[\w.:/-]*\d[\w.:/-]* - ")


@dataclass
class Document:
    kind: str  # markdown | text | html
    file_name: str
    sha256: str
    title: str
    body: str  # markdown
    url: str | None = None
    warnings: list[str] = field(default_factory=list)


# ----- conversion --------------------------------------------------------------------------------


def _decode(data: bytes, warnings: list[str]) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        warnings.append("not UTF-8; read as Windows-1252")
        return data.decode("cp1252", errors="replace")


def detect_kind(path: Path, data: bytes) -> str:
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in HTML_SUFFIXES:
        return "html"
    if suffix in TEXT_SUFFIXES:
        return "text"
    head = data[:2048].lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")) or b"<body" in head:
        return "html"
    raise WriteError(f"{path.name}: cannot tell the format from the name; use .md, .txt or .html")


def read_document(path: str | Path, title: str | None = None) -> Document:
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise WriteError(f"cannot read {path}: {exc.strerror or exc}") from None
    return document_from_bytes(path.name, data, title)


def document_from_bytes(file_name: str, data: bytes, title: str | None = None) -> Document:
    """A document from its bytes; ``file_name`` decides the kind (.md, .txt, .html)."""
    path = Path(file_name)
    kind = detect_kind(path, data)
    warnings: list[str] = []
    if kind == "html":
        found_title, body, url = _from_html(data, warnings)
    elif kind == "markdown":
        found_title, body, url = _from_markdown(_decode(data, warnings))
    else:
        found_title, body, url = _from_text(_decode(data, warnings))
    final_title = " ".join((title or found_title or path.stem).split())
    if not body.strip():
        raise WriteError(f"{path.name}: no text found to ingest")
    return Document(kind, path.name, hashlib.sha256(data).hexdigest(), final_title, body, url, warnings)


def document_to_dict(document: Document) -> dict[str, Any]:
    return {"file": document.file_name, "kind": document.kind, "title": document.title, "sha256": document.sha256,
            "url": document.url, "warnings": list(document.warnings)}


def _from_markdown(text: str) -> tuple[str | None, str, str | None]:
    text = normalize_text(text)
    block = split_frontmatter(text)
    body = text[block.end_offset:]
    meta = {str(k).lower(): v for k, v in block.data.items()}
    title = meta.get("title") if isinstance(meta.get("title"), str) else None
    url = next((meta[k] for k in ("url", "source", "link", "canonical_url")
                if isinstance(meta.get(k), str) and meta[k].startswith(("http://", "https://"))), None)
    if title is None:
        # The first heading is the title only if it is an H1 (in a vault note, "# Notes"
        # after "## References" is a section, not a title).
        headings = parse_note(body).headings
        title = headings[0].text if headings and headings[0].level == 1 else None
    return title, body, url


def _from_text(text: str) -> tuple[str | None, str, str | None]:
    text = normalize_text(text)
    first = next((line.strip() for line in text.split("\n") if line.strip()), "")
    title = first if 0 < len(first) <= 120 else None
    body = "\n".join("\\" + line.lstrip() if _HEADING_LINE.match(line) else line for line in text.split("\n"))
    return title, body, None


def _from_html(data: bytes, warnings: list[str]) -> tuple[str | None, str, str | None]:
    from bs4 import BeautifulSoup
    from markdownify import MarkdownConverter

    soup = BeautifulSoup(data, "html.parser")

    def meta(**attrs: str) -> str | None:
        tag = soup.find("meta", attrs=attrs)
        value = tag.get("content") if tag else None
        return " ".join(value.split()) if isinstance(value, str) and value.strip() else None

    canonical = soup.find("link", rel="canonical")
    url = (canonical.get("href") if canonical and canonical.get("href") else None) or meta(property="og:url")
    if url and urlparse(url).scheme not in ("http", "https"):
        url = None
    site = meta(property="og:site_name")
    title = meta(property="og:title")
    if title is None and soup.title and soup.title.get_text(strip=True):
        title = " ".join(soup.title.get_text().split())
        if " | " in title and len(title.rsplit(" | ", 1)[0].split()) >= 2:
            title = title.rsplit(" | ", 1)[0]
        if site:
            for separator in (" - ", " – ", " — "):
                if title.endswith(separator + site):
                    title = title[: -len(separator + site)]
    root = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    if title is None:
        heading = root.find("h1")
        title = " ".join(heading.get_text().split()) if heading else None
    for tag in root.find_all(_DROP_TAGS):
        tag.decompose()
    if root is soup.body:
        for tag in root.find_all(["header", "footer"], recursive=False):
            tag.decompose()
    unlinked = 0
    for anchor in root.find_all("a"):
        href = anchor.get("href")
        target = urljoin(url, href) if url and href else href
        if href and href.startswith("#"):
            anchor.unwrap()  # a jump inside the original page means nothing in a note
        elif target and urlparse(target).scheme in ("http", "https", "mailto"):
            anchor["href"] = target
        else:
            anchor.unwrap()
            unlinked += 1
    for image in root.find_all("img"):
        src = image.get("src")
        target = urljoin(url, src) if url and src else src
        if target and urlparse(target).scheme in ("http", "https"):
            image["src"] = target
        else:
            image.replace_with(f" {image.get('alt')} " if image.get("alt") else "")
    if unlinked and not url:
        warnings.append(f"{unlinked} relative links kept as text (the page has no canonical URL)")
    markdown = MarkdownConverter(heading_style="ATX", bullets="-").convert_soup(root)
    lines = [line.rstrip() for line in markdown.split("\n")]
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"
    return title, body, url


# ----- fitting the body into the note ---------------------------------------------------------------


def fit_body(body: str, title: str, top_level: int, warnings: list[str]) -> str:
    """Shift headings below ``top_level``, drop a first heading repeating the title, escape fields."""
    body = normalize_text(body).strip("\n")
    lines = body.split("\n")
    note = parse_note(body)
    headings = note.headings
    drop = None
    if headings and similarity.words(headings[0].text) == similarity.words(title):
        before = "\n".join(lines[: headings[0].line - 1])
        if not before.strip():
            drop = headings[0].line
    kept = [h for h in headings if h.line != drop]
    if kept:
        shift = max(0, top_level + 1 - min(h.level for h in kept))
        if shift:
            warnings.append(f"headings moved down {shift} level{'s' if shift > 1 else ''} to sit under the body heading")
        for heading in kept:
            index = heading.line - 1
            match = _HEADING_LINE.match(lines[index])
            if match:
                level = min(6, len(match[2]) + shift)
                lines[index] = match[1] + "#" * level + lines[index][match.end():]
    escaped = 0
    for field in note.inline_fields:
        index = field.line - 1
        match = _INLINE_FIELD.match(lines[index])
        if match:
            at = lines[index].index("::", match.start("key"))
            lines[index] = lines[index][:at] + "\\" + lines[index][at:]
            escaped += 1
    if escaped:
        warnings.append(f"{escaped} `key::` line{'s' if escaped > 1 else ''} escaped so they add no relations")
    if drop is not None:
        del lines[drop - 1]
    return "\n".join(lines).strip("\n")


# ----- duplicates -------------------------------------------------------------------------------


@dataclass
class Candidate:
    path: str
    title: float
    title_pair: tuple[str, str]
    overlap: float | None
    shared: int
    compared: int
    same_hash: bool

    @property
    def duplicate(self) -> bool:
        return self.same_hash or similarity.is_title_duplicate(self.title) or similarity.is_content_duplicate(self.overlap)

    @property
    def reasons(self) -> list[str]:
        found = []
        if self.same_hash:
            found.append("same file (content hash)")
        if similarity.is_title_duplicate(self.title):
            found.append(f"title similarity {self.title:.3f} >= {similarity.TITLE_THRESHOLD}")
        if similarity.is_content_duplicate(self.overlap):
            found.append(f"content overlap {self.overlap:.3f} >= {similarity.OVERLAP_THRESHOLD}")
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "duplicate": self.duplicate, "reasons": self.reasons,
            "title_similarity": round(self.title, 4), "titles_compared": list(self.title_pair),
            "content_overlap": None if self.overlap is None else round(self.overlap, 4),
            "shared_shingles": self.shared, "shingles_compared": self.compared, "same_hash": self.same_hash,
        }


class DuplicateFound(Exception):
    def __init__(self, document: Document, duplicates: list[Candidate], candidates: list[Candidate]) -> None:
        super().__init__(f"{document.file_name} duplicates {duplicates[0].path}")
        self.document = document
        self.duplicates = duplicates
        self.candidates = candidates


def note_titles(typed: TypedVault, path: str, note_type: NoteType) -> list[str]:
    """Titles a note is known by: its name (also without a leading identifier like
    "2411.00640 - "), aliases, H1s that are not section headings, and the first link text
    under the type's ``title_section``."""
    note = typed.vault.notes[path]
    name = link_name(path)
    titles = [name, _IDENTIFIER_PREFIX.sub("", name), *note.aliases]
    titles += [h.text for h in note.headings if h.level == 1 and not any(p.matches(h.text) for p in note_type.headings)]
    if note_type.title_section is not None:
        section = next((h for h in note.headings if note_type.title_section.matches(h.text)), None)
        if section is not None:
            end = next((h.line for h in note.headings if h.line > section.line and h.level <= section.level), None)
            inside = lambda line: line > section.line and (end is None or line < end)  # noqa: E731
            labelled = [(l.line, l.display) for l in note.links if l.display and inside(l.line)]
            labelled += [(e.line, e.display) for e in note.external_links if e.display and inside(e.line)]
            if labelled:
                titles.append(min(labelled)[1])
    return list(dict.fromkeys(t for t in titles if t and t.strip()))


def note_body(text: str) -> str:
    """A note's text without frontmatter and without its header block."""
    note = parse_note(text)
    lines = normalize_text(text).split("\n")
    start = note.body_start_line - 1
    if note.header_fields:
        start = max(start, note.header_fields[-1].line)
    return "\n".join(lines[start:])


def find_candidates(typed: TypedVault, note_type: NoteType, document: Document, body: str) -> list[Candidate]:
    root = typed.vault.root
    new_shingles = similarity.shingles(body)
    hash_key = note_type.ingest.hash_key if note_type.ingest else None
    candidates = []
    for path in typed.notes_of_type(note_type.name):
        titles = note_titles(typed, path, note_type)
        best, pair = -1.0, (document.title, titles[0] if titles else "")
        for title in titles:
            score = similarity.title_similarity(document.title, title)
            if score > best:
                best, pair = score, (document.title, title)
        text = (root / path).read_text(encoding="utf-8", errors="replace")
        overlap = similarity.content_overlap(new_shingles, similarity.shingles(note_body(text)))
        recorded = typed.metadata(path).get(hash_key) if hash_key else None
        same_hash = bool(recorded and isinstance(recorded.raw, str) and recorded.raw.strip().lower().endswith(document.sha256))
        # A byte-for-byte copy of the note file itself is the same file too.
        same_hash = same_hash or hashlib.sha256((root / path).read_bytes()).hexdigest() == document.sha256
        candidates.append(Candidate(path, max(best, 0.0), pair, overlap.score, overlap.shared, overlap.smaller, same_hash))
    candidates.sort(key=lambda c: (not c.duplicate, -max(c.title, c.overlap or 0.0), c.path))
    return candidates


# ----- plan -----------------------------------------------------------------------------------


def plan_ingest(typed: TypedVault, document: Document, type_name: str, *, folder: str | None = None,
                values: dict[str, str] | None = None, now: dt.datetime | None = None,
                found: dict[str, Any] | None = None) -> WritePlan:
    note_type = typed.spec.type_named(type_name)
    if note_type is None:
        raise WriteError(f"format {typed.spec.name!r} has no type {type_name!r}")
    spec = note_type.ingest
    if spec is None:
        raise WriteError(f"type {type_name!r} has no `ingest:` block in the format, so vault ingest cannot fill it")
    now = now or dt.datetime.now()
    body_index = next(i for i, p in enumerate(note_type.headings) if p.pattern == spec.body_heading)
    body = fit_body(document.body, document.title, typed.usual_heading_level(note_type.name, body_index), document.warnings)

    candidates = find_candidates(typed, note_type, document, body)
    if found is not None:
        found["candidates"] = candidates
    duplicates = [c for c in candidates if c.duplicate]
    if duplicates:
        raise DuplicateFound(document, duplicates, candidates)

    given = dict(values or {})
    if spec.file_key:
        given.setdefault(spec.file_key, document.file_name)
    if spec.hash_key:
        given.setdefault(spec.hash_key, f"sha256:{document.sha256}")
    if spec.date_key:
        given.setdefault(spec.date_key, "{now}")
    sections = {spec.body_heading: "\n" + body}  # a blank line under the heading, as in the vault's notes
    if spec.references_heading:
        label = document.title.replace("[", "\\[").replace("]", "\\]")
        sections[spec.references_heading] = "\n" + (f"1. [{label}]({document.url})" if document.url else f"1. {label}")
    plan = plan_new(typed, type_name, document.title, folder=folder, values=given, sections=sections, now=now)
    plan.operation = "ingest"
    return plan
