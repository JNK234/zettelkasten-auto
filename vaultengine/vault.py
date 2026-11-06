"""Load a vault from disk (or from memory) and index its files for link lookup.

Every file is indexed, not only markdown, so embeds of images, PDFs, canvases,
Excalidraw drawings and other attachments can resolve. Markdown files are also
parsed.

All lookups are case-insensitive and Unicode-NFC-normalised (macOS stores
file names in NFD, while typed link text is usually NFC).
"""

from __future__ import annotations

import os
import posixpath
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from vaultengine.model import ParsedNote
from vaultengine.parser import parse_note


def lookup_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip().lower()


def is_markdown(path: str) -> bool:
    return path.lower().endswith(".md")


def link_name(path: str) -> str:
    """Name a link uses for a file: basename without '.md' for notes, full name otherwise."""
    name = posixpath.basename(path)
    return name[:-3] if is_markdown(name) else name


def iter_vault_files(root: Path) -> Iterator[tuple[str, Path]]:
    """Yield (vault-relative NFC posix path, absolute path) for every vault file.

    Files and folders whose name starts with '.' are skipped (.obsidian, .trash,
    .git, and the index's own .vaultengine folder), as Obsidian does not show them.
    """
    for directory, subdirs, filenames in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if not d.startswith("."))
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            full = Path(directory) / filename
            yield unicodedata.normalize("NFC", full.relative_to(root).as_posix()), full


class Vault:
    def __init__(self, files: Iterable[str], notes: Mapping[str, ParsedNote], root: Path | None = None) -> None:
        self.root = root
        self.files: list[str] = sorted(set(files) | set(notes))
        self.notes: dict[str, ParsedNote] = dict(notes)
        self._by_path: dict[str, list[str]] = defaultdict(list)
        self._by_name: dict[str, list[str]] = defaultdict(list)
        self._by_alias: dict[str, list[str]] = defaultdict(list)
        for path in self.files:
            keys = {lookup_key(path)}
            if is_markdown(path):
                keys.add(lookup_key(path[:-3]))
            for key in keys:
                self._by_path[key].append(path)
            names = {lookup_key(posixpath.basename(path)), lookup_key(link_name(path))}
            for key in names:
                self._by_name[key].append(path)
        for path, note in self.notes.items():
            for alias in {lookup_key(alias) for alias in note.aliases}:
                if alias:
                    self._by_alias[alias].append(path)

    # ----- construction -------------------------------------------------

    @classmethod
    def load(cls, root: str | Path) -> "Vault":
        """Scan and parse ``root`` (see ``iter_vault_files`` for what is included)."""
        root = Path(root)
        files: list[str] = []
        notes: dict[str, ParsedNote] = {}
        for rel, full in iter_vault_files(root):
            files.append(rel)
            if is_markdown(rel):
                text = full.read_text(encoding="utf-8", errors="replace")
                notes[rel] = parse_note(text, path=rel)
        return cls(files, notes, root)

    @classmethod
    def from_files(cls, contents: Mapping[str, str | None]) -> "Vault":
        """Build a vault in memory: {"folder/Note.md": "text", "img.png": None}."""
        notes = {
            path: parse_note(text or "", path=path) for path, text in contents.items() if is_markdown(path)
        }
        return cls(contents.keys(), notes)

    # ----- lookups (each returns every matching file path) --------------

    def files_at_path(self, path: str) -> list[str]:
        """Files whose vault path equals ``path`` (the '.md' extension is optional)."""
        return list(self._by_path.get(lookup_key(path), []))

    def files_named(self, name: str) -> list[str]:
        """Files whose basename equals ``name`` (the '.md' extension is optional)."""
        return list(self._by_name.get(lookup_key(name), []))

    def notes_with_alias(self, alias: str) -> list[str]:
        return list(self._by_alias.get(lookup_key(alias), []))
