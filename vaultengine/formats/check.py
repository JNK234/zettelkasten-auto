"""`vault check`: the violations of a format, for the whole vault or for what changed.

"Changed" means added or modified since the last `vault index` (the index's own
change detection: size/mtime, then content hash). The whole vault is still read,
because links, mirrors and target types depend on other notes. A change can
break a note that did not change, so ``--changed`` reports violations

* in added or modified notes;
* caused by them elsewhere: a missing mirror implied by a changed note's field
  (A now ``supports:: [[B]]``, B lacks ``supported-by:: [[A]]``);
* caused by deleted notes: relation links that pointed at a deleted note's name.

Checking never writes anything, not even the index: after a failed check,
fix the notes and check again; run `vault index` once the notes are right.
``--update-baseline`` is the one option that writes, and only its own file.

A baseline is a record of the violations a vault has at one moment, so that
``--baseline`` can fail only on violations added since. It is stored per format in
``<vault>/.vaultengine/baseline-<format>.json`` and recorded only by
``--update-baseline``, always from the whole vault. Violations are compared the way
the write gate compares them (``violation_keys``): by what they are about, not by
line, so editing a note that has baselined violations does not make them new.

Renames: the baseline also lists every note path and, for each note with violations,
a fingerprint of its text (SHA-256, and a bottom-k sketch of its 5-word shingles).
A baselined note that is gone is paired with a note that did not exist when the
baseline was recorded and has the same text, or nearly (estimated Jaccard similarity
of the shingles >= RENAME_SIMILARITY); its baselined violations then follow it to
the new path. A pairing that is not clear-cut (two equally good candidates) is not
made, and those violations count as new.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vaultengine.formats.similarity import shingles
from vaultengine.formats.typed import Issue, TypedVault, violation_keys
from vaultengine.vault import link_name, lookup_key

BASELINE_VERSION = 2
SKETCH_SIZE = 32
RENAME_SIMILARITY = 0.7
MIN_SKETCH_SHINGLES = 10  # fewer: only an identical text counts as the same note


@dataclass
class CheckScope:
    mode: Literal["all", "changed"]
    checked: list[str] = field(default_factory=list)  # notes checked
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    last_indexed_at: str | None = None
    reason: str | None = None  # why a --changed run checked everything


def scope_all(typed: TypedVault) -> CheckScope:
    return CheckScope("all", checked=sorted(typed.vault.notes))


def scope_changed(typed: TypedVault, added: list[str], changed: list[str], deleted: list[str],
                  last_indexed_at: str | None, reason: str | None = None) -> CheckScope:
    """Scope for ``--changed``. With a ``reason`` (the index must be rebuilt) every note is checked."""
    notes = typed.vault.notes
    if reason:
        return CheckScope("changed", sorted(notes), last_indexed_at=last_indexed_at, reason=reason)
    added_notes = sorted(p for p in added if p in notes)
    changed_notes = sorted(p for p in changed if p in notes)
    deleted_notes = sorted(p for p in deleted if p.lower().endswith(".md"))
    return CheckScope(
        "changed", sorted(added_notes + changed_notes), added_notes, changed_notes, deleted_notes, last_indexed_at,
    )


def select_issues(typed: TypedVault, scope: CheckScope) -> list[Issue]:
    issues = typed.issues()
    if scope.mode == "all" or scope.reason:
        return issues
    checked = set(scope.checked)
    gone = {lookup_key(link_name(path)) for path in scope.deleted} | {lookup_key(path[:-3]) for path in scope.deleted}

    def involved(issue: Issue) -> bool:
        if issue.path in checked or issue.related in checked:
            return True
        if issue.code == "unresolved-target" and issue.detail:
            target = issue.detail["target"]
            return lookup_key(target) in gone or lookup_key(posixpath.basename(target)) in gone
        return False

    return [issue for issue in issues if involved(issue)]


# ----- baseline -----------------------------------------------------------------------


class BaselineError(ValueError):
    """The baseline file cannot be read or is not a baseline of this format."""


@dataclass
class Baseline:
    path: Path
    format_name: str
    recorded_at: str
    violations: list[dict[str, Any]]  # {"path", "line", "code", "message", "keys": [[...], ...]}
    paths: list[str] = field(default_factory=list)  # every note when recorded
    fingerprints: dict[str, dict[str, Any]] = field(default_factory=dict)  # note with violations -> {"sha256", "sketch"}

    def key_counts(self, renamed: dict[str, str] | None = None) -> Counter[tuple[str, ...]]:
        return Counter(_renamed_key(tuple(key), renamed or {}) for item in self.violations for key in item["keys"])


def _renamed_key(key: tuple[str, ...], renamed: dict[str, str]) -> tuple[str, ...]:
    return tuple(renamed.get(part, part) for part in key) if renamed else key


# ----- note fingerprints (for renames) -----------------------------------------------------


def _note_text(typed: TypedVault, path: str) -> str | None:
    if typed.vault.root is None:
        return None
    try:
        return (Path(typed.vault.root) / path).read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None


def fingerprint(text: str) -> dict[str, Any]:
    hashes = sorted({
        int.from_bytes(hashlib.blake2b(" ".join(shingle).encode(), digest_size=6).digest(), "big")
        for shingle in shingles(text)
    })
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "shingles": len(hashes),
        "sketch": [f"{h:012x}" for h in hashes[:SKETCH_SIZE]],
    }


def similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """1.0 for the same text; else the Jaccard similarity of the two texts' shingles,
    estimated from their bottom-k sketches; 0.0 when either text is too short."""
    if a["sha256"] == b["sha256"]:
        return 1.0
    if min(a["shingles"], b["shingles"]) < MIN_SKETCH_SHINGLES:
        return 0.0
    first, second = set(a["sketch"]), set(b["sketch"])
    bottom = sorted(first | second)[:SKETCH_SIZE]
    return sum(1 for h in bottom if h in first and h in second) / len(bottom)


def find_renames(typed: TypedVault, baseline: Baseline) -> list[dict[str, Any]]:
    """Baselined notes that are gone, paired with the new note each became (see the module doc)."""
    notes = typed.vault.notes
    gone = {p: baseline.fingerprints[p] for p in baseline.fingerprints if p not in notes}
    if not gone:
        return []
    known = set(baseline.paths)
    fresh = {}
    for path in sorted(notes):
        if path not in known:
            text = _note_text(typed, path)
            if text is not None:
                fresh[path] = fingerprint(text)
    return pair_renames(gone, fresh)


def pair_renames(gone: dict[str, dict[str, Any]], fresh: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair notes that are gone with new notes by text fingerprint: the same text, or similar
    (>= RENAME_SIMILARITY), each the other's unique best match. Ties are not guessed.
    Shared by the baseline and the page record (`vault page`)."""
    scores = {(g, f): similarity(gone[g], fresh[f]) for g in gone for f in fresh}
    scores = {pair: score for pair, score in scores.items() if score >= RENAME_SIMILARITY}

    def unique_best(candidates: dict[str, float]) -> str | None:
        if not candidates:
            return None
        top = max(candidates.values())
        best = [name for name, score in candidates.items() if score == top]
        return best[0] if len(best) == 1 else None

    renames = []
    for g in gone:
        f = unique_best({f: score for (old, f), score in scores.items() if old == g})
        # The other way round too: the new note's best match must be this gone note.
        if f is not None and unique_best({old: score for (old, new), score in scores.items() if new == f}) == g:
            score = scores[(g, f)]
            renames.append({"from": g, "to": f, "similarity": round(score, 3),
                            "how": "same text" if score == 1.0 else "similar text"})
    return renames


@dataclass
class BaselineResult:
    baseline: Baseline
    new: list[Issue]  # in scope and not in the baseline
    known: list[Issue]  # in scope and in the baseline
    fixed: list[dict[str, Any]]  # baseline entries no longer found anywhere in the vault
    recorded: bool = False  # True when this run recorded the baseline
    previous: int | None = None  # violations in the baseline this run replaced
    renames: list[dict[str, Any]] = field(default_factory=list)  # baselined notes followed to a new path

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.baseline.path),
            "recorded_at": self.baseline.recorded_at,
            "recorded_now": self.recorded,
            "previous": self.previous,
            "violations": len(self.baseline.violations),
            "new": len(self.new),
            "known": len(self.known),
            "fixed": len(self.fixed),
            "fixed_items": self.fixed,
            "renames": self.renames,
        }


def baseline_path(root: str | Path, format_name: str) -> Path:
    safe = re.sub(r"[^\w.-]+", "-", format_name).strip("-.") or "format"
    return Path(root) / ".vaultengine" / f"baseline-{safe}.json"


def record_baseline(typed: TypedVault, path: Path, now: dt.datetime | None = None) -> Baseline:
    """Write every violation of the whole vault to ``path`` (atomically) and return it."""
    now = now or dt.datetime.now(dt.timezone.utc)
    issues = typed.issues()
    fingerprints = {}
    for note in sorted({issue.path for issue in issues}):
        text = _note_text(typed, note)
        if text is not None:
            fingerprints[note] = fingerprint(text)
    baseline = Baseline(
        path, typed.spec.name, now.isoformat(timespec="seconds"),
        [{k: v for k, v in issue.to_dict().items() if k != "related"} | {"keys": [list(key) for key in violation_keys(issue)]}
         for issue in issues],
        sorted(typed.vault.notes), fingerprints,
    )
    text = baseline_text({"name": typed.spec.name, "path": typed.spec.path}, baseline.recorded_at, baseline.violations,
                         [{"path": p, **f} for p, f in fingerprints.items()], baseline.paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)
    return baseline


def baseline_text(format_ref: dict[str, Any], recorded_at: str, violations: list[dict[str, Any]],
                  notes: list[dict[str, Any]], paths: list[str]) -> str:
    """The baseline file: one item per line, so it diffs well when a vault keeps it under version control."""
    head = json.dumps({
        "baseline": BASELINE_VERSION,
        "format": format_ref,
        "recorded_at": recorded_at,
        "summary": {"violations": len(violations), "notes": len({item["path"] for item in violations})},
    }, indent=1, ensure_ascii=False)

    def rows(items: list[Any]) -> str:
        return ",\n".join("  " + json.dumps(item, ensure_ascii=False) for item in items) + ("\n" if items else "")

    return (head[:-2] + ',\n "violations": [\n' + rows(violations) + " ],\n"
            ' "notes": [\n' + rows(notes) + " ],\n"
            ' "paths": [\n' + rows(paths) + " ]\n}\n")


def baseline_follow(text: str, path_map: dict[str, str]) -> str | None:
    """A baseline file's text with notes that moved (old path -> new path; a merged note -> its survivor)
    followed: their violations, fingerprints and places in the path list. None when nothing changes, or
    when the file is not a baseline this engine reads (then it is left alone)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or data.get("baseline") != BASELINE_VERSION:
        return None
    paths = data.get("paths", [])
    if not any(p in path_map for p in paths) and not any(v.get("path") in path_map for v in data.get("violations", [])):
        return None
    violations = []
    for item in data.get("violations", []):
        item = dict(item)
        item["path"] = path_map.get(item["path"], item["path"])
        item["keys"] = [[path_map.get(part, part) for part in key] for key in item.get("keys", [])]
        violations.append(item)
    notes, seen = [], set()
    for item in data.get("notes", []):
        new = path_map.get(item["path"], item["path"])
        if new in seen:
            continue  # a merged note's fingerprint: its survivor keeps its own
        if item["path"] in path_map and any(n["path"] == new for n in data.get("notes", [])):
            continue
        seen.add(new)
        notes.append({**item, "path": new})
    new_paths = sorted({path_map.get(p, p) for p in paths})
    return baseline_text(data.get("format", {}), data.get("recorded_at", ""), violations, notes, new_paths)


def load_baseline(path: Path, format_name: str) -> Baseline:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BaselineError(f"{path}: cannot read the baseline: {exc.strerror or exc}") from None
    except json.JSONDecodeError as exc:
        raise BaselineError(f"{path}: not a baseline file (invalid JSON: {exc})") from None
    if (not isinstance(data, dict) or data.get("baseline") != BASELINE_VERSION
            or not all(isinstance(data.get(k), list) for k in ("violations", "notes", "paths"))):
        raise BaselineError(f"{path}: not a baseline file of version {BASELINE_VERSION}; record it again with --update-baseline")
    name = (data.get("format") or {}).get("name")
    if name != format_name:
        raise BaselineError(f"{path}: recorded for format {name!r}, not {format_name!r}")
    items = data["violations"]
    for index, item in enumerate(items):
        keys = item.get("keys") if isinstance(item, dict) else None
        if not isinstance(keys, list) or not all(isinstance(k, list) and all(isinstance(p, str) for p in k) for k in keys):
            raise BaselineError(f"{path}: violations[{index}] has no valid keys; record it again with --update-baseline")
    fingerprints = {}
    for index, item in enumerate(data["notes"]):
        try:
            fingerprints[item["path"]] = {"sha256": str(item["sha256"]), "shingles": int(item["shingles"]),
                                          "sketch": [str(h) for h in item["sketch"]]}
        except (TypeError, KeyError, ValueError):
            raise BaselineError(f"{path}: notes[{index}] is not a note fingerprint; record it again with --update-baseline") from None
    return Baseline(path, name, str(data.get("recorded_at", "")), items, [str(p) for p in data["paths"]], fingerprints)


def compare_to_baseline(typed: TypedVault, issues: list[Issue], baseline: Baseline) -> BaselineResult:
    """Split the issues in scope into new and known; list the baseline entries fixed since.

    Counted like the write gate: a key the baseline has twice tolerates two violations
    about it, the third is new.
    """
    renames = find_renames(typed, baseline)
    renamed = {r["from"]: r["to"] for r in renames}
    allowed = baseline.key_counts(renamed)
    seen: Counter[tuple[str, ...]] = Counter()
    new, known = [], []
    for issue in issues:
        fresh = False
        for key in violation_keys(issue):
            seen[key] += 1
            if seen[key] > allowed.get(key, 0):
                fresh = True
        (new if fresh else known).append(issue)
    # Fixed is about the whole vault, whatever the scope.
    everywhere = Counter(key for issue in typed.issues() for key in violation_keys(issue))
    used: Counter[tuple[str, ...]] = Counter()
    fixed = []
    for item in baseline.violations:
        keys = [_renamed_key(tuple(key), renamed) for key in item["keys"]]
        for key in keys:
            used[key] += 1
        if keys and all(used[key] > everywhere.get(key, 0) for key in keys):
            fixed.append({k: item.get(k) for k in ("path", "line", "code", "message")})
    return BaselineResult(baseline, new, known, fixed, renames=renames)


def check_report(typed: TypedVault, scope: CheckScope, issues: list[Issue]) -> dict[str, Any]:
    by_code = Counter(issue.code for issue in issues)
    paths = sorted({issue.path for issue in issues})
    checked = set(scope.checked)
    return {
        "format": {"name": typed.spec.name, "path": typed.spec.path},
        "scope": {
            "mode": scope.mode,
            "checked": len(scope.checked),
            "notes": scope.checked if scope.mode == "changed" else None,
            "added": scope.added,
            "changed": scope.changed,
            "deleted": scope.deleted,
            "last_indexed_at": scope.last_indexed_at,
            "reason": scope.reason,
        },
        "ok": not issues,
        "summary": {
            "violations": len(issues),
            "notes": len(paths),
            "by_code": [{"code": code, "count": count} for code, count in sorted(by_code.items(), key=lambda kv: (-kv[1], kv[0]))],
        },
        "notes": [
            {
                "path": path,
                "type": typed.type_of(path),
                "checked": path in checked,  # False: affected by a changed or deleted note
                "violations": [issue.to_dict() for issue in issues if issue.path == path],
            }
            for path in paths
        ],
    }
