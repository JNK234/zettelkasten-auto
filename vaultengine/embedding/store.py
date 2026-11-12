"""Note vectors cached in the index, by the hash of the embedded text and the model id.

``notes.embed_hash`` (written by ``vault index``) says which text each note is embedded as;
``embeddings`` holds one vector per (hash, model id). A note needs embedding when its hash
has no vector for the current model: it was added or edited, or the model id changed. So:

* editing one note re-embeds that note only (its hash changed; every other hash is cached);
* changing the model id re-embeds every note (no hash has a vector for the new id); the
  old model's vectors stay, so switching back costs nothing, until ``--prune`` drops them;
* a vector whose text no note has any more is dropped by ``vault index``.

The note's text is read from the vault file when it is embedded. If the file changed since
the last index, its hash is not the indexed one: the vector is stored under the hash of
what was read (it is content-addressed, so it is right for that text) and that note is
reported as changed since the last index.

``note_vectors`` is for readers (``related``, ``suggest``, ``explain``) and never writes: it
follows the index (a note uses the cached vector of its indexed text), and a note without
one is embedded in memory for that run only and reported, so the command can say to run
``vault embed``. Only ``embed_notes`` (``vault embed``) stores vectors.
"""

from __future__ import annotations

import sqlite3
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from vaultengine.embedding.provider import EmbeddingProvider
from vaultengine.embedding.text import embedding_text, text_hash
from vaultengine.parser import normalize_text, parse_note_with_layout

if TYPE_CHECKING:
    import numpy as np

BATCH = 64


@dataclass
class EmbedReport:
    model_id: str
    provider: str
    dim: int
    model_source: str
    notes: int = 0
    embedded: list[str] = field(default_factory=list)  # notes whose vector was computed this run
    cached: int = 0  # notes whose vector was already there
    changed_since_index: list[str] = field(default_factory=list)  # embedded from a file newer than the index
    unreadable: list[str] = field(default_factory=list)  # in the index but not readable on disk
    empty: list[str] = field(default_factory=list)  # no token the model knows: a zero vector
    pruned: int = 0  # vectors of other model ids dropped (--prune)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _vector_bytes(vector: "np.ndarray") -> bytes:
    import numpy as np

    return np.asarray(vector, dtype="<f4").tobytes()


def note_text(root: str | Path, path: str) -> str | None:
    """The text ``path`` is embedded as, read from disk now (None when unreadable)."""
    try:
        raw = (Path(root) / path).read_bytes()
    except OSError:
        return None
    text = normalize_text(raw.decode("utf-8", errors="replace"))
    note, layout = parse_note_with_layout(text, path=path)
    return embedding_text(path, note, layout)


def status(conn: sqlite3.Connection, model_id: str | None) -> dict:
    """Vector coverage of the indexed notes for ``model_id``, without loading any model."""
    notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    have = missing = 0
    missing_paths: list[str] = []
    if model_id is not None:
        rows = conn.execute(
            """SELECT f.path, e.content_hash IS NOT NULL FROM notes n JOIN files f ON f.id = n.file_id
               LEFT JOIN embeddings e ON e.content_hash = n.embed_hash AND e.model_id = ?
               ORDER BY f.path""",
            (model_id,),
        )
        for path, found in rows:
            if found:
                have += 1
            else:
                missing += 1
                missing_paths.append(path)
    dims = [d for (d,) in conn.execute("SELECT DISTINCT dim FROM embeddings WHERE model_id = ?", (model_id,))]
    others = [
        {"model_id": m, "vectors": n, "dim": d}
        for m, n, d in conn.execute(
            "SELECT model_id, COUNT(*), MAX(dim) FROM embeddings WHERE model_id IS NOT ? GROUP BY model_id ORDER BY model_id",
            (model_id,),
        )
    ]
    total, size = conn.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(vector)), 0) FROM embeddings").fetchone()
    return {
        "model_id": model_id,
        "notes": notes,
        "with_vector": have,
        "missing": missing,
        "missing_paths": missing_paths,
        "dim": dims[0] if len(dims) == 1 else None,
        "other_models": others,
        "cache_vectors": total,
        "cache_bytes": size,
    }


def embed_notes(
    conn: sqlite3.Connection,
    root: str | Path,
    provider: EmbeddingProvider,
    *,
    full: bool = False,
    prune: bool = False,
    file_ids: set[int] | None = None,
) -> EmbedReport:
    """Embed every indexed note (or ``file_ids``) that has no vector for the provider's model; store them.

    ``full``: embed again even when a vector is cached. ``prune``: drop vectors of other model ids.
    """
    started = time.perf_counter()
    root = Path(root)
    model_id = provider.model_id
    report = EmbedReport(model_id=model_id, provider=provider.name, dim=provider.dim, model_source=provider.source)
    rows = conn.execute(
        """SELECT n.file_id, f.path, n.embed_hash, e.dim FROM notes n JOIN files f ON f.id = n.file_id
           LEFT JOIN embeddings e ON e.content_hash = n.embed_hash AND e.model_id = ?
           ORDER BY f.path""",
        (model_id,),
    ).fetchall()
    todo = []
    for file_id, path, embed_hash, dim in rows:
        if file_ids is not None and file_id not in file_ids:
            continue
        report.notes += 1
        if dim == provider.dim and not full:
            report.cached += 1
        else:
            todo.append((path, embed_hash))
    pending: dict[str, tuple[str, list[str]]] = {}  # hash -> (text, paths)
    for path, embed_hash in todo:
        text = note_text(root, path)
        if text is None:
            report.unreadable.append(path)
            continue
        digest = text_hash(text)
        if digest != embed_hash:
            report.changed_since_index.append(path)
        pending.setdefault(digest, (text, []))[1].append(path)
    if not pending and not prune:  # nothing to write: no write lock for readers
        report.seconds = round(time.perf_counter() - started, 3)
        return report
    conn.execute("BEGIN IMMEDIATE")
    try:
        items = list(pending.items())
        for start in range(0, len(items), BATCH):
            batch = items[start : start + BATCH]
            vectors = provider.embed([text for _, (text, _) in batch])
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (content_hash, model_id, dim, vector) VALUES (?, ?, ?, ?)",
                [(digest, model_id, provider.dim, _vector_bytes(vector)) for (digest, _), vector in zip(batch, vectors)],
            )
            for (_, (_, paths)), vector in zip(batch, vectors):
                report.embedded.extend(paths)
                if not vector.any():
                    report.empty.extend(paths)
        if prune:
            report.pruned = conn.execute("DELETE FROM embeddings WHERE model_id != ?", (model_id,)).rowcount
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    report.embedded.sort()
    report.seconds = round(time.perf_counter() - started, 3)
    return report


@dataclass
class NoteVectors:
    """Unit vectors of the indexed notes for one model: ``matrix[row_of[file_id]]``."""

    model_id: str
    file_ids: list[int]
    matrix: "np.ndarray"  # (notes, dim) float32
    row_of: dict[int, int]
    computed: list[str] = field(default_factory=list)  # notes with no cached vector: embedded in memory for this run
    unreadable: list[str] = field(default_factory=list)  # no cached vector and no file to embed: no vector

    def get(self, file_id: int) -> "np.ndarray | None":
        row = self.row_of.get(file_id)
        return None if row is None else self.matrix[row]


def note_vectors(conn: sqlite3.Connection, root: str | Path, provider: EmbeddingProvider,
                 texts: dict[str, str] | None = None) -> NoteVectors:
    """Every indexed note's vector for the provider's model. Reads the index, never writes it.

    A note uses the cached vector of its indexed text. A note without one is embedded in memory
    for this run only (listed in ``computed``; ``vault embed`` caches them): from the cached
    vector of the text on disk now if there is one, else by the provider. ``texts``: filled with
    path -> text for the notes read, so a caller need not read and parse them again.
    """
    import numpy as np

    rows = conn.execute(
        """SELECT n.file_id, f.path, e.vector FROM notes n JOIN files f ON f.id = n.file_id
           LEFT JOIN embeddings e ON e.content_hash = n.embed_hash AND e.model_id = ? AND e.dim = ?
           ORDER BY n.file_id""",
        (provider.model_id, provider.dim),
    ).fetchall()
    vectors: dict[int, np.ndarray] = {}
    computed: list[str] = []
    unreadable: list[str] = []
    pending: dict[str, tuple[str, list[int]]] = {}  # text hash -> (text, notes)
    for file_id, path, blob in rows:
        if blob is not None:
            vectors[file_id] = np.frombuffer(blob, dtype="<f4")
            continue
        text = note_text(root, unicodedata.normalize("NFC", path))
        if text is None:
            unreadable.append(path)
            continue
        if texts is not None:
            texts[path] = text
        computed.append(path)
        digest = text_hash(text)
        cached = conn.execute(
            "SELECT vector FROM embeddings WHERE content_hash = ? AND model_id = ? AND dim = ?",
            (digest, provider.model_id, provider.dim),
        ).fetchone()
        if cached is not None:  # edited since the last index, and that text was embedded before
            vectors[file_id] = np.frombuffer(cached[0], dtype="<f4")
        else:
            pending.setdefault(digest, (text, []))[1].append(file_id)
    items = list(pending.values())
    for start in range(0, len(items), BATCH):
        batch = items[start : start + BATCH]
        for (_, ids), vector in zip(batch, provider.embed([text for text, _ in batch])):
            for file_id in ids:
                vectors[file_id] = vector
    file_ids = sorted(vectors)
    matrix = np.stack([vectors[f] for f in file_ids]) if file_ids else np.zeros((0, provider.dim), np.float32)
    return NoteVectors(provider.model_id, file_ids, np.ascontiguousarray(matrix, dtype=np.float32),
                       {f: i for i, f in enumerate(file_ids)}, sorted(computed), sorted(unreadable))
