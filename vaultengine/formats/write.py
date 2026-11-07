"""The safe write path: `vault new`, `vault link`, `vault set` (and `vault fix`).

Every write goes through one transaction:

1. take the vault's write lock (a file in the system temp folder named after the
   vault's path, so a refused write leaves nothing behind in the vault) and roll
   back any write a crash interrupted (see *Atomicity*);
2. read the vault and plan the new file contents, editing only the lines needed;
3. the gate: check the whole vault as it would be after the write. The write is
   refused, and nothing is written, if it would add any violation anywhere. A
   violation that was already there does not block it (so a note can be repaired
   one problem at a time); a new note has no "before", so it must be clean.
   Violations are compared by what they are about, not by line number;
4. commit all files atomically, as one group;
5. update the index for the written notes only, so other pending edits stay
   "changed since the last index" for ``vault check --changed``. A written note is
   left pending too if it had unindexed changes before and still has violations.

Atomicity: each new content is written to a temporary file next to its target and
fsynced; each existing target is hard-linked to a backup; a journal listing them
is written to ``.vaultengine/journal/``; then the temporaries are renamed over the
targets (atomic per file) and read back. Removing the journal is the commit point.
If anything fails before that, every file is restored from its backup and new
files are removed. If the process dies in between, the next write command (or
``recover``) finds the journal and rolls the group back. Leftover temporaries
and backups are dot-files, which the vault scan skips.
"""

from __future__ import annotations

import datetime as dt
import errno
import json
import os
import re
import shutil
import time
import unicodedata
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from vaultengine.formats.edit import EditRefused, Lines, add_relation, set_frontmatter_value, set_header_value
from vaultengine.formats.spec import FormatSpec, MetadataKey, NoteType
from vaultengine.formats.typed import Issue, TypedVault
from vaultengine.index import IndexNeedsRebuild, VaultIndex
from vaultengine.parser import parse_note
from vaultengine.vault import Vault, link_name, lookup_key

ENGINE_DIR = ".vaultengine"
TEMP_PREFIX = ".vault-write-"
_NAME_FORBIDDEN = re.compile(r'[\\/:*?"<>|#^\[\]]')
_SLUG = re.compile(r"[^0-9a-z]+")
LOCK_TIMEOUT_SECONDS = 30.0
# Fault injection for tests only: exit abruptly after this many renames of a commit.
_CRASH_ENV = "VAULTENGINE_TEST_CRASH_AFTER_RENAMES"


class WriteError(ValueError):
    """The write cannot be planned (note not found, name taken, bad arguments). Nothing was written."""


class NoteMissing(WriteError):
    """A note named in the write does not exist (or the name matches several)."""


class WriteFailed(RuntimeError):
    """The commit failed; every file was restored. Nothing was written."""


@dataclass
class FileChange:
    path: str  # vault-relative
    old: str | None  # current content, byte-exact; None when the file is created
    new: str
    changes: list[str] = field(default_factory=list)

    @property
    def created(self) -> bool:
        return self.old is None


@dataclass
class WritePlan:
    operation: str
    files: list[FileChange] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # things that were already so


@dataclass
class Verdict:
    new_violations: list[Issue]
    tolerated: list[Issue]  # violations already there, in the notes the write touches
    after: TypedVault

    @property
    def ok(self) -> bool:
        return not self.new_violations


@dataclass
class Outcome:
    status: Literal["written", "refused", "dry-run", "unchanged"]
    plan: WritePlan
    verdict: Verdict | None = None
    index: dict[str, Any] | None = None
    recovered: list[str] = field(default_factory=list)


# ----- reading --------------------------------------------------------------------


def read_text(root: Path, path: str) -> str:
    try:
        return (root / path).read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        raise WriteError(f"{path} is not valid UTF-8; it is not touched") from None


def find_note(typed: TypedVault, ref: str) -> str:
    """A note by vault path (the .md is optional) or by name. Must be exactly one."""
    vault = typed.vault
    ref = unicodedata.normalize("NFC", ref.strip())
    matches = [p for p in vault.files_at_path(ref) if p in vault.notes]
    if not matches and "/" not in ref:
        matches = [p for p in vault.files_named(ref) if p in vault.notes]
    if not matches:
        raise NoteMissing(f"no note {ref!r}")
    if len(matches) > 1:
        raise NoteMissing(f"{ref!r} matches {len(matches)} notes: {', '.join(sorted(matches))}; give the path")
    return matches[0]


# ----- the gate ---------------------------------------------------------------------


def violation_keys(issue: Issue) -> list[tuple[str, ...]]:
    """What a violation is about, without line numbers (edits move lines)."""
    detail = issue.detail or {}
    if issue.code == "missing-heading":
        return [(issue.path, issue.code, name) for name in detail.get("headings", [])]
    if issue.code == "missing-mirror":
        return [(issue.path, issue.code, detail["relation"], detail["source"])]
    if issue.code == "heading-order":
        return [(issue.path, issue.code)]
    return [(issue.path, issue.code, issue.message)]


def after_vault(typed: TypedVault, files: list[FileChange]) -> TypedVault:
    notes = dict(typed.vault.notes)
    for change in files:
        notes[change.path] = parse_note(change.new, path=change.path)
    paths = set(typed.vault.files) | {change.path for change in files}
    return TypedVault(typed.spec, Vault(paths, notes, typed.vault.root))


def judge(typed: TypedVault, files: list[FileChange]) -> Verdict:
    after = after_vault(typed, files)
    before = Counter(key for issue in typed.issues() for key in violation_keys(issue))
    seen: Counter[tuple[str, ...]] = Counter()
    new, tolerated = [], []
    touched = {change.path for change in files}
    for issue in after.issues():
        keys = violation_keys(issue)
        fresh = False
        for key in keys:
            seen[key] += 1
            if seen[key] > before.get(key, 0):
                fresh = True
        if fresh:
            new.append(issue)
        elif issue.path in touched:
            tolerated.append(issue)
    return Verdict(new, tolerated, after)


# ----- commit ------------------------------------------------------------------------


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_file(path: Path, data: bytes, mode: int | None) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666 if mode is None else mode & 0o7777)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if path.exists():
            path.unlink()
        raise
    if mode is not None:
        os.chmod(path, mode & 0o7777)


def _journal_dir(root: Path) -> Path:
    return root / ENGINE_DIR / "journal"


def _rollback(root: Path, entries: list[dict[str, Any]]) -> None:
    for entry in reversed(entries):
        target = root / entry["target"]
        temp = root / entry["temp"]
        backup = root / entry["backup"] if entry["backup"] else None
        if temp.exists():  # never renamed: the target is as it was
            temp.unlink()
            if backup is not None and backup.exists():
                backup.unlink()
        elif backup is not None and backup.exists():
            os.replace(backup, target)
        elif entry["created"] and target.exists():
            target.unlink()
        for directory in reversed(entry.get("made_dirs", [])):
            try:
                (root / directory).rmdir()
            except OSError:
                pass


def commit(root: Path, files: list[FileChange]) -> None:
    """Write all files or none (see the module docstring). Raises WriteFailed after restoring."""
    root = Path(root)
    txid = uuid.uuid4().hex[:12]
    journal_dir = _journal_dir(root)
    journal_dir.mkdir(parents=True, exist_ok=True)
    journal = journal_dir / f"{txid}.json"
    entries: list[dict[str, Any]] = []
    crash_after = os.environ.get(_CRASH_ENV)
    try:
        for change in files:  # nothing changed since the plan was made
            target = root / change.path
            current = target.read_bytes().decode("utf-8", errors="surrogateescape") if target.exists() else None
            if current != change.old:
                what = "was created" if change.old is None else "changed" if current is not None else "was deleted"
                raise WriteFailed(f"{change.path} {what} while the write was being prepared; nothing was written")
        for number, change in enumerate(files):
            target = root / change.path
            made_dirs = []
            parent = target.parent
            missing = []
            while not parent.exists():
                missing.append(parent)
                parent = parent.parent
            for directory in reversed(missing):
                directory.mkdir()
                made_dirs.append(directory.relative_to(root).as_posix())
            temp = target.parent / f"{TEMP_PREFIX}{txid}-{number}.tmp"
            backup = target.parent / f"{TEMP_PREFIX}{txid}-{number}.bak" if change.old is not None else None
            entry = {
                "target": change.path,
                "temp": temp.relative_to(root).as_posix(),
                "backup": backup.relative_to(root).as_posix() if backup else None,
                "created": change.old is None,
                "made_dirs": made_dirs,
            }
            entries.append(entry)
            mode = target.stat().st_mode if backup else None
            _write_file(temp, change.new.encode("utf-8"), mode)
            if backup is not None:
                try:
                    os.link(target, backup)
                except OSError:
                    shutil.copy2(target, backup)
        journal.write_text(json.dumps({"txid": txid, "entries": entries}), encoding="utf-8")
        with open(journal, "rb+") as stream:
            os.fsync(stream.fileno())
        _fsync_dir(journal_dir)

        for number, entry in enumerate(entries):
            os.replace(root / entry["temp"], root / entry["target"])
            if crash_after is not None and number + 1 >= int(crash_after):
                os._exit(70)  # test-only: die between renames, leaving the journal
        for directory in {(root / entry["target"]).parent for entry in entries}:
            _fsync_dir(directory)
        for change in files:
            if (root / change.path).read_bytes() != change.new.encode("utf-8"):
                raise WriteFailed(f"{change.path} does not hold what was written; every file was restored")
    except BaseException as exc:
        _rollback(root, entries)
        if journal.exists():
            journal.unlink()
        if isinstance(exc, WriteFailed):
            raise
        if isinstance(exc, OSError):
            raise WriteFailed(f"write failed ({exc}); every file was restored") from exc
        raise
    journal.unlink()  # the commit point
    _fsync_dir(journal_dir)
    for entry in entries:
        if entry["backup"]:
            try:
                (root / entry["backup"]).unlink()
            except FileNotFoundError:
                pass


def recover(root: Path) -> list[str]:
    """Roll back writes a crash interrupted, and remove leftovers of finished ones.

    Returns the vault paths restored. Call with the write lock held.
    """
    root = Path(root)
    restored: list[str] = []
    journal_dir = _journal_dir(root)
    if journal_dir.exists():
        for journal in sorted(journal_dir.glob("*.json")):
            try:
                entries = json.loads(journal.read_text(encoding="utf-8"))["entries"]
            except (OSError, ValueError, KeyError):
                continue  # not ours, or unreadable: leave it for a person
            _rollback(root, entries)
            restored.extend(entry["target"] for entry in entries)
            journal.unlink()
    # Without a journal, temporaries and backups belong to finished or never-started writes.
    for directory, subdirs, filenames in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in (ENGINE_DIR, ".git")]
        for name in filenames:
            if name.startswith(TEMP_PREFIX) and name.endswith((".tmp", ".bak")):
                os.unlink(Path(directory) / name)
    return sorted(set(restored))


def lock_path(root: Path) -> Path:
    """The vault's write lock: outside the vault, the same for every process writing to it."""
    import hashlib
    import tempfile

    key = hashlib.sha256(str(Path(root).resolve()).encode("utf-8")).hexdigest()[:20]
    return Path(tempfile.gettempdir()) / f"vaultengine-write-{key}.lock"


@contextmanager
def write_lock(root: Path, timeout: float = LOCK_TIMEOUT_SECONDS) -> Iterator[None]:
    """One writer at a time per vault (POSIX advisory lock; a no-op where fcntl is missing)."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        yield
        return
    path = lock_path(root)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    except OSError as exc:
        raise WriteError(f"cannot open the write lock {path}: {exc.strerror or exc}") from None
    with os.fdopen(fd, "r+") as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES) or time.monotonic() > deadline:
                    raise WriteError("another vault write is in progress (could not take the write lock)") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# ----- index --------------------------------------------------------------------------


def _stale_notes(db_path: Path, root: Path) -> tuple[set[str] | None, str | None]:
    """Notes with changes the index has not seen (None if there is no index), and a rebuild reason."""
    if not db_path.exists():
        return None, None
    with VaultIndex.open_existing(db_path) as index:
        stale = index.staleness(root)
    return set(stale.added) | set(stale.changed), stale.reason


def sync_index(db_path: Path, root: Path, written: list[str], stale_before: set[str] | None,
               reason: str | None, after: TypedVault) -> dict[str, Any]:
    if stale_before is None:
        return {"status": "no-index", "synced": [], "pending": written}
    if reason:
        return {"status": "needs-rebuild", "reason": reason, "synced": [], "pending": written}
    pending = sorted(p for p in written if p in stale_before and after.issues(p))
    synced = sorted(set(written) - set(pending))
    if synced:
        try:
            with VaultIndex(db_path) as index:
                index.update(root, only=synced)
        except IndexNeedsRebuild as exc:
            return {"status": "needs-rebuild", "reason": str(exc), "synced": [], "pending": written}
    return {"status": "updated", "synced": synced, "pending": pending}


# ----- the transaction ---------------------------------------------------------------------


def run_write(spec: FormatSpec, root: str | Path, db_path: str | Path, build: Callable[[TypedVault], WritePlan],
              dry_run: bool = False, parse_cache: dict | None = None) -> Outcome:
    """``parse_cache``: see ``Vault.load``; a server passes one to avoid re-parsing unchanged notes."""
    root = Path(root)
    db_path = Path(db_path)
    with write_lock(root):
        recovered = recover(root)
        stale_before, reason = _stale_notes(db_path, root)
        typed = TypedVault(spec, Vault.load(root, cache=parse_cache))
        plan = build(typed)
        if not plan.files:
            return Outcome("unchanged", plan, recovered=recovered)
        verdict = judge(typed, plan.files)
        if not verdict.ok:
            return Outcome("refused", plan, verdict, recovered=recovered)
        if dry_run:
            return Outcome("dry-run", plan, verdict, recovered=recovered)
        commit(root, plan.files)
        index = sync_index(db_path, root.resolve(), [f.path for f in plan.files], stale_before, reason, verdict.after)
        return Outcome("written", plan, verdict, index, recovered)


def diff_text(files: list[FileChange]) -> str:
    """Unified diff of planned changes (new files diff against nothing)."""
    import difflib

    chunks = []
    for change in files:
        old = (change.old or "").splitlines(keepends=True)
        new = change.new.splitlines(keepends=True)
        fromfile = "/dev/null" if change.created else f"a/{change.path}"
        for line in difflib.unified_diff(old, new, fromfile=fromfile, tofile=f"b/{change.path}"):
            if line.endswith(("\r\n", "\n", "\r")):
                chunks.append(line.rstrip("\r\n") + "\n")
            else:
                chunks.append(line + "\n\\ No newline at end of file\n")
    return "".join(chunks)


def file_diff(change: FileChange) -> str:
    return diff_text([change])


def outcome_to_dict(outcome: Outcome, dry_run: bool) -> dict[str, Any]:
    """The shape `--json` prints and the JSON-RPC write methods return."""
    plan, verdict = outcome.plan, outcome.verdict
    return {
        "operation": plan.operation,
        "status": outcome.status,
        "dry_run": dry_run,
        "written": outcome.status == "written",
        "files": [
            {"path": f.path, "action": "create" if f.created else "modify", "changes": list(f.changes),
             "diff": file_diff(f), "content": f.new if f.created else None}
            for f in plan.files
        ],
        "already": list(plan.notes),
        "new_violations": [i.to_dict() for i in verdict.new_violations] if verdict else [],
        "tolerated": [i.to_dict() for i in verdict.tolerated] if verdict else [],
        "index": outcome.index,
        "recovered": list(outcome.recovered),
    }


# ----- operations -----------------------------------------------------------------------------


def _expand(value: str, key: MetadataKey, now: dt.datetime) -> str:
    if "{now}" in value and key.kinds[0] in ("datetime", "date") and key.date_format:
        return value.replace("{now}", now.strftime(key.date_format))
    return value


def _key(note_type: NoteType, name: str) -> MetadataKey:
    for key in note_type.metadata:
        if key.name.lower() == name.lower():
            return key
    declared = ", ".join(k.name for k in note_type.metadata) or "none"
    raise WriteError(f"{name!r} is not a metadata key of {note_type.name} (keys: {declared})")


def _one_line(value: str, what: str) -> str:
    if "\n" in value or "\r" in value:
        raise WriteError(f"{what} must be one line")
    return value.strip()


def note_name(note_type: NoteType, title: str) -> str:
    cleaned = re.sub(r"\s+", " ", _NAME_FORBIDDEN.sub(" ", title)).strip()
    cleaned = re.sub(r" +([,.;!?])", r"\1", cleaned)
    slug = _SLUG.sub("-", unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()).strip("-")
    name = note_type.new.name.replace("{title}", cleaned).replace("{slug}", slug)
    if note_type.new.max_name_length and len(name) > note_type.new.max_name_length:
        name = name[: note_type.new.max_name_length].rstrip(" -,.;")
    if not name.strip() or name.startswith("."):
        raise WriteError(f"the title {title!r} gives no usable file name")
    return name


def plan_new(
    typed: TypedVault,
    type_name: str,
    title: str,
    *,
    folder: str | None = None,
    name: str | None = None,
    values: dict[str, str] | None = None,
    relations: list[tuple[str, str]] | None = None,
    sections: dict[str, str] | None = None,
    now: dt.datetime | None = None,
) -> WritePlan:
    spec = typed.spec
    note_type = spec.type_named(type_name)
    if note_type is None:
        raise WriteError(f"format {spec.name!r} has no type {type_name!r} (types: {', '.join(t.name for t in spec.types)})")
    title = _one_line(title, "the title")
    if not title:
        raise WriteError("the title is empty")
    now = now or dt.datetime.now()
    if folder is None:
        folder = next((m.folder for m in note_type.match if m.folder), "")
    folder = folder.strip("/")
    base = _one_line(name, "the name") if name else note_name(note_type, title)
    if _NAME_FORBIDDEN.search(base) or base.startswith("."):
        raise WriteError(f"{base!r} cannot be a note name (no / \\ : * ? \" < > | # ^ [ ], no leading '.')")
    path = unicodedata.normalize("NFC", f"{folder}/{base}.md" if folder else f"{base}.md")
    if typed.spec.type_for(path) is not note_type:
        other = typed.spec.type_for(path)
        raise WriteError(f"{path} would be a {other.name if other else 'note of no type'}, not a {type_name}; "
                         f"check --folder and the name")
    taken = [p for p in typed.vault.files_named(base)] + typed.vault.files_at_path(path)
    if taken:
        raise WriteError(f"a note named {base!r} already exists: {', '.join(sorted(set(taken)))}")

    given = {k.lower(): (k, v) for k, v in (values or {}).items()}
    for key_name in given.values():
        _key(note_type, key_name[0])
    defaults = {k.lower(): v for k, v in note_type.new.defaults}
    meta_lines = []
    for key in note_type.metadata:
        if key.name.lower() in given:
            value = _one_line(given[key.name.lower()][1], key.name)
        elif key.name.lower() in defaults:
            value = defaults[key.name.lower()]
        elif key.required:
            value = ""  # the gate reports it as empty
        else:
            continue
        meta_lines.append((key, _expand(value, key, now)))

    grouped: dict[str, list[str]] = {}
    targets: dict[str, list[str]] = {}
    for relation_name, target_ref in relations or []:
        relation = spec.relation(relation_name)
        rel_name = relation.name if relation else relation_name
        target = find_note(typed, target_ref)
        grouped.setdefault(rel_name, []).append(typed.link_text(target))
        targets.setdefault(rel_name, []).append(target)
    relation_lines = [f"{rel}:: " + ", ".join(f"[[{t}]]" for t in links) for rel, links in grouped.items()]

    section_text = {}
    for heading, text in (sections or {}).items():
        pattern = next((p for p in note_type.headings if p.matches(heading)), None)
        if pattern is None:
            names = ", ".join(p.pattern for p in note_type.headings) or "none"
            raise WriteError(f"{heading!r} is not a required heading of {type_name} (headings: {names})")
        section_text[pattern.pattern] = text.rstrip("\n")

    if note_type.metadata_source == "frontmatter":
        import yaml
        body = ["---"] + [
            yaml.safe_dump({k.name: v}, allow_unicode=True, width=10_000).rstrip("\n") for k, v in meta_lines
        ] + ["---"]
    else:
        body = []
        for index, (key, value) in enumerate(meta_lines):
            if index and note_type.new.spaced_metadata:
                body.append("")
            body.append(f"{key.name}: {value}".rstrip())
    placed_relations = False
    if relation_lines and note_type.relations_heading is None:
        body.extend([""] + relation_lines)
        placed_relations = True
    if note_type.new.title_heading:
        body.extend(["", f"# {title}"])
    for index, pattern in enumerate(note_type.headings):
        if pattern.is_regex:
            raise WriteError(f"{type_name} requires a heading matching {pattern.pattern}; vault new cannot write it")
        level = typed.usual_heading_level(note_type.name, index)
        body.extend(["", "#" * level + " " + pattern.pattern])
        if relation_lines and not placed_relations and note_type.relations_heading.matches(pattern.pattern):
            body.extend(relation_lines)
            placed_relations = True
        if pattern.pattern in section_text:
            body.append(section_text[pattern.pattern])
    if relation_lines and not placed_relations:
        body.extend(["", f"## {note_type.relations_heading.pattern}"] + relation_lines)
    while body and body[0] == "":
        body.pop(0)
    content = "\n".join(body) + "\n"

    files = [FileChange(path, None, content, [f"new {type_name}: {title}"])]
    link = link_name(path)  # unique: the name was checked to be free
    names = {lookup_key(link), lookup_key(path[:-3])}
    edits = _Edits(typed)
    already = []
    for rel_name, paths in targets.items():
        relation = spec.relation(rel_name)
        if relation is None or relation.inverse is None:
            continue
        for target in paths:
            # A link to the new note may already be there, dangling until now.
            if any(e.relation == relation.inverse and lookup_key(e.target) in names
                   for e in typed.relations(target, mirrors=False)):
                already.append(f"{target} already has `{relation.inverse}:: [[{link}]]`")
                continue
            edits.add_relation(target, relation.inverse, [link])
    return WritePlan("new", files + edits.changes(), already)


class _Edits:
    """Edits to existing notes, one read and one FileChange per file."""

    def __init__(self, typed: TypedVault) -> None:
        self.typed = typed
        self.files: dict[str, tuple[str, Lines, list[str]]] = {}

    def _lines(self, path: str) -> tuple[str, Lines, list[str]]:
        if path not in self.files:
            raw = read_text(self.typed.vault.root, path)
            self.files[path] = (raw, Lines(raw), [])
        return self.files[path]

    def add_relation(self, path: str, relation: str, links: list[str]) -> None:
        _, lines, described = self._lines(path)
        described.append(add_relation(self.typed, lines, path, relation, links))

    def changes(self) -> list[FileChange]:
        return [FileChange(path, raw, lines.render(), described) for path, (raw, lines, described) in self.files.items()]


def plan_link(typed: TypedVault, source_ref: str, target_ref: str, relation_name: str) -> WritePlan:
    source = find_note(typed, source_ref)
    target = find_note(typed, target_ref)
    if source == target:
        raise WriteError("a note cannot be related to itself")
    relation = typed.spec.relation(relation_name)
    rel = relation.name if relation else relation_name.strip()
    if not re.fullmatch(r"[^\W_][\w \-]{0,63}", rel):
        raise WriteError(f"{relation_name!r} cannot be a field name")
    plan = WritePlan("link")
    wanted = [(source, rel, target)]
    if relation is not None and relation.inverse is not None:
        wanted.append((target, relation.inverse, source))
    edits = _Edits(typed)
    for path, name, other in wanted:
        # Undeclared relations are written as asked; the gate refuses them (unknown-field).
        if any(e.relation == name and e.target_path == other for e in typed.relations(path, mirrors=False)):
            plan.notes.append(f"{path} already has `{name}:: [[{typed.link_text(other)}]]`")
            continue
        edits.add_relation(path, name, [typed.link_text(other)])
    plan.files = edits.changes()
    return plan


def plan_set(typed: TypedVault, note_ref: str, key_name: str, value: str, now: dt.datetime | None = None) -> WritePlan:
    path = find_note(typed, note_ref)
    note_type = typed.spec.type_for(path)
    if note_type is None:
        raise WriteError(f"{path} has no type in format {typed.spec.name!r}; vault set only writes declared keys")
    key = _key(note_type, key_name)
    value = _expand(_one_line(value, "the value"), key, now or dt.datetime.now())
    note = typed.vault.notes[path]
    current = typed.metadata(path).get(key.name)
    if current is not None and (current.raw if isinstance(current.raw, str) else None) == value:
        return WritePlan("set", notes=[f"{path} already has {key.name}: {value}"])
    raw = read_text(typed.vault.root, path)
    lines = Lines(raw)
    source = note_type.metadata_source
    in_frontmatter = any(k.lower() == key.name.lower() for k in note.frontmatter)
    in_header = any(f.key.lower() == key.name.lower() for f in note.header_fields)
    use_frontmatter = source == "frontmatter" or (
        source == "either" and (in_frontmatter or (note.has_frontmatter and not in_header))
    )
    try:
        if use_frontmatter:
            described = set_frontmatter_value(typed, lines, path, key, value)
        else:
            described = set_header_value(typed, lines, path, key, value)
    except EditRefused as exc:
        raise WriteError(str(exc)) from None
    return WritePlan("set", [FileChange(path, raw, lines.render(), [described])])
