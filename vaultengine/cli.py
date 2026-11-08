"""`vault` command line: index, stats, health, search, related, suggest, unlinked, formats, check, fix,
new, link, set, ingest, coverage, serve (JSON-RPC), rpc-schema.

Human-readable output by default, machine-readable with --json.
Exit codes: 0 success, 2 usage error, missing index or invalid format file,
1 when `check` (or `formats --strict`) finds violations, a write is refused, or `ingest` finds a duplicate,
3 when a write failed and was rolled back.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from collections import Counter
from typing import Any, Sequence

from dataclasses import asdict

from vaultengine import queries
from vaultengine.vault import link_name
from vaultengine import search as search_module
from vaultengine.index import IndexReport, Staleness, VaultIndex, default_db_path


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _shown(items: Sequence[Any], limit: int) -> tuple[Sequence[Any], str]:
    if limit <= 0 or len(items) <= limit:
        return items, ""
    return items[:limit], f"  … {len(items) - limit} more (--limit 0 shows all)"


# ----- index -----------------------------------------------------------------


def _format_index(report: IndexReport, limit: int) -> str:
    parsed = len(report.parsed)
    lines = [f"Indexed {report.vault} in {report.seconds:.2f}s", f"  index      {report.db}"]
    lines.append(f"  files      {report.notes} notes, {report.attachments} attachments")
    if report.full_reason:
        lines.append(f"  full run   {report.full_reason}")
    detail = f"{len(report.added)} added, {len(report.changed)} changed, {len(report.deleted)} deleted"
    if report.forced:
        detail += f", {len(report.forced)} re-parsed unchanged"
    lines.append(f"  parsed     {parsed}  ({detail})" if parsed or report.deleted else "  parsed     0  (index already up to date)")
    skipped = report.notes - report.hashed
    lines.append(f"  read       {_plural(report.hashed, 'note')} hashed, {skipped} skipped by size/mtime")
    if report.attachments_added or report.attachments_deleted:
        lines.append(f"  attachments {len(report.attachments_added)} added, {len(report.attachments_deleted)} deleted")
    lines.append(
        f"  links      {report.links_checked} resolved this run, "
        f"{report.links_changed_elsewhere} changed status in notes that were not re-parsed"
    )
    for marker, paths in (("+", report.added), ("~", report.changed), ("-", report.deleted)):
        shown, more = _shown(sorted(paths), limit)
        lines.extend(f"    {marker} {path}" for path in shown)
        if more:
            lines.append(more)
    return "\n".join(lines)


def _cmd_index(args: argparse.Namespace) -> int:
    with VaultIndex(args.db) as index:
        report = index.update(args.vault, full=args.full)
    if args.json:
        _print_json(report.to_dict())
    else:
        print(_format_index(report, args.limit))
    return 0


# ----- shared for read commands -----------------------------------------------


def _open_for_reading(args: argparse.Namespace) -> tuple[VaultIndex, dict[str, Any]] | None:
    if not Path(args.db).exists():
        print(f"error: no index at {args.db}; run `vault index {args.vault}` first", file=sys.stderr)
        return None
    index = VaultIndex.open_existing(args.db)
    stale: Staleness = index.staleness(args.vault)
    info = {
        "db": str(args.db),
        "last_indexed_at": index.meta().get("last_indexed_at"),
        "stale": stale.is_stale,
        "pending": {"added": stale.added, "changed": stale.changed, "deleted": stale.deleted, "reason": stale.reason},
    }
    if stale.is_stale:
        print(f"warning: index is out of date ({stale.summary()}); run `vault index` to refresh", file=sys.stderr)
    return index, info


def _header(title: str, vault: str, info: dict[str, Any]) -> str:
    freshness = "OUT OF DATE" if info["stale"] else "up to date"
    return f"{title}: {Path(vault).resolve()}\n  index {info['last_indexed_at']} ({freshness})"


# ----- stats -----------------------------------------------------------------


def _format_stats(data: dict[str, Any], header: str) -> str:
    links = data["links"]
    out = [header, ""]
    out.append(f"  notes          {data['notes']:>6}   ({data['notes_with_frontmatter']} with frontmatter, {data['frontmatter_errors']} frontmatter errors)")
    out.append(f"  attachments    {data['attachments']:>6}")
    out.append(f"  links          {links['total']:>6}   wikilinks {links['wikilink']} · markdown {links['markdown']} · embeds {links['embeds']}")
    out.append(
        f"    resolved     {links['resolved']:>6}   unresolved {links['unresolved']} ({links['unresolved_targets']} distinct targets)"
        f" · ambiguous {links['ambiguous']}"
    )
    out.append(f"  note edges     {data['note_edges']:>6}   distinct note→note connections")
    out.append(f"  tags           {data['tags']['uses']:>6}   uses of {data['tags']['distinct']} distinct tags")
    out.append(f"  properties     {data['properties']['distinct_keys']:>6}   distinct keys")
    out.append(f"  aliases        {data['aliases']:>6}")
    out.append(f"  headings       {data['headings']:>6}")
    out.append(f"  blocks         {data['blocks']:>6}")

    def table(title: str, rows: list[str]) -> None:
        if rows:
            out.extend(["", title, *rows])

    table("Notes per top-level folder", [f"  {row['notes']:>5}  {row['folder']}" for row in data["folders"]])
    table("Top tags (notes)", [f"  {row['notes']:>5}  #{row['tag']}" for row in data["tags"]["top"]])
    table("Top properties (notes)", [f"  {row['notes']:>5}  {row['key']}" for row in data["properties"]["top"]])
    table(
        "Most linked (linking notes / links)",
        [f"  {row['linking_notes']:>3} / {row['backlinks']:<3} {row['path']}" for row in data["most_linked"]],
    )
    table(
        "Most wanted missing notes (linking notes / links)",
        [f"  {row['linking_notes']:>3} / {row['links']:<3} {row['target']}" for row in data["most_wanted"]],
    )
    return "\n".join(out)


def _cmd_stats(args: argparse.Namespace) -> int:
    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    with index:
        data = queries.stats(index.conn, top=args.top)
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), "index": info, **data})
    else:
        print(_format_stats(data, _header("Vault stats", args.vault, info)))
    return 0


# ----- health ----------------------------------------------------------------

_HEALTH_LABELS = (
    ("unresolved_links", "unresolved links"),
    ("ambiguous_links", "ambiguous links"),
    ("missing_subpaths", "missing heading/block targets"),
    ("orphans", "orphans (no backlinks)"),
    ("dead_ends", "dead ends (no outgoing note links)"),
    ("duplicate_basenames", "duplicate basenames"),
    ("empty_notes", "empty notes"),
)


def _format_health(data: dict[str, Any], header: str, limit: int) -> str:
    summary = data["summary"]
    out = [header, ""]
    for key, label in _HEALTH_LABELS:
        extra = f"   across {summary['unresolved_targets']} missing targets" if key == "unresolved_links" else ""
        out.append(f"  {label:<36}{summary[key]:>6}{extra}")

    def section(title: str, items: list[Any], render) -> None:
        if not items:
            return
        shown, more = _shown(items, limit)
        out.extend(["", f"{title} ({len(items)})"])
        for item in shown:
            out.extend(render(item))
        if more:
            out.append(more)

    def unresolved(group):
        sources = sorted({ref["source"] for ref in group["links"]})
        first = group["links"][0]
        also = f" (+{len(sources) - 1} more notes)" if len(sources) > 1 else ""
        counts = f"{_plural(len(group['links']), 'link')} from {_plural(len(sources), 'note')}"
        return [f"  {group['target']}  — {counts}, e.g. {first['source']}:{first['line']}{also}"]

    section("Unresolved link targets", data["unresolved_links"], unresolved)
    section(
        "Ambiguous links",
        data["ambiguous_links"],
        lambda item: [f"  {item['source']}:{item['line']}  {item['raw']}", *[f"      could be {c}" for c in item["candidates"]]],
    )
    section(
        "Missing heading/block targets",
        data["missing_subpaths"],
        lambda item: [f"  {item['source']}:{item['line']}  {item['raw']}  — {item['kind']} '{item['subpath']}' not in {item['target_path']}"],
    )
    section("Orphans", data["orphans"], lambda path: [f"  {path}"])
    section("Dead ends", data["dead_ends"], lambda path: [f"  {path}"])
    section(
        "Duplicate basenames",
        data["duplicate_basenames"],
        lambda item: [f"  {item['name']}", *[f"      {path}" for path in item["paths"]]],
    )
    section("Empty notes", data["empty_notes"], lambda path: [f"  {path}"])
    return "\n".join(out)


def _cmd_health(args: argparse.Namespace) -> int:
    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    with index:
        data = queries.health(index.conn)
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), "index": info, **data})
    else:
        print(_format_health(data, _header("Vault health", args.vault, info), args.limit))
    return 0


# ----- formats -----------------------------------------------------------------


def _format_formats(data: dict[str, Any], vault: str, limit: int) -> str:
    fmt = data["format"]
    issues = data["issues"]
    only_type = data["only_type"]
    scope = f", type {only_type}" if only_type else ""
    out = [f"Format {fmt['name']} ({fmt['path']}) on {Path(vault).resolve()}{scope}", ""]
    out.append(f"  notes   {data['notes']:>6}   {data['typed']} typed, {len(data['untyped'])} untyped")
    out.append(f"  issues  {issues['total']:>6}   in {_plural(issues['notes'], 'note')}" + (f" of type {only_type}" if only_type else ""))

    out.extend(["", f"  {'Types':<14}{'notes':>6}{'clean':>7}{'issues':>8}   match"])
    for row in data["types"]:
        out.append(f"  {row['name']:<14}{row['notes']:>6}{row['clean']:>7}{row['issues']:>8}   {row['match']}")
        out.extend(f"      e.g. {path}" for path in row["examples"])
    if data["untyped"] and not only_type:
        shown, more = _shown(data["untyped"], limit)
        out.extend(["", f"Untyped notes ({len(data['untyped'])})", *[f"  {path}" for path in shown]])
        if more:
            out.append(more)

    if issues["by_code"]:
        out.extend(["", "Issues by rule (issues / notes)"])
        out.extend(f"  {row['code']:<22}{row['count']:>5} / {row['notes']}" for row in issues["by_code"])
    if issues["items"]:
        shown, more = _shown(issues["items"], limit)
        out.extend(["", f"Issues ({len(issues['items'])})"])
        for item in shown:
            where = f"{item['path']}:{item['line']}" if item["line"] else item["path"]
            out.append(f"  {where}\n      {item['code']}: {item['message']}")
        if more:
            out.append(more)

    if not only_type:
        out.extend(["", f"  {'Relations':<20}{'declared':>9}{'resolved':>9}{'unresolved':>11}{'mirrored':>9}   inverse"])
        for row in data["relations"]:
            out.append(
                f"  {row['name']:<20}{row['declared']:>9}{row['resolved']:>9}{row['unresolved'] + row['ambiguous']:>11}"
                f"{row['implied_by_mirror']:>9}   {row['inverse'] or '(one-way)'}"
            )
        out.extend(["", "Link roles (links)"])
        out.extend(f"  {row['links']:>6}  {row['role']}" for row in data["link_roles"])
        out.append(f"  {data['links_without_role']:>6}  (no role)")
    return "\n".join(out)


def _format_note(data: dict[str, Any]) -> str:
    out = [f"{data['path']}  ({data['type'] or 'no type'})"]
    out.append("  metadata" if data["metadata"] else "  metadata  (none declared or found)")
    for item in data["metadata"]:
        where = f"line {item['line']}" if item["line"] else "frontmatter"
        out.append(f"    {item['key']:<12} {str(item['raw']):<40} {where}")
    if data["relations"]:
        out.append("  relations")
        for edge in data["relations"]:
            if edge["origin"] == "mirror":
                out.append(f"    {edge['relation']:<18} [[{edge['target']}]]  (implied: mirror of a field in {edge['mirror_of']})")
            else:
                target = edge["target_path"] or edge["status"]
                out.append(f"    {edge['relation']:<18} [[{edge['target']}]]  line {edge['line']}  -> {target}")
    out.append(f"  links ({len(data['links'])})")
    for link in data["links"]:
        target = link["target_path"] or link["status"]
        out.append(f"    line {link['line']:<4} {link['role'] or '-':<16} {link['raw']}  -> {target}")
    out.append(f"  issues ({len(data['issues'])})" if data["issues"] else "  issues    none")
    for item in data["issues"]:
        where = f"line {item['line']}: " if item["line"] else ""
        out.append(f"    {where}{item['code']}: {item['message']}")
    return "\n".join(out)


def _load_spec(args: argparse.Namespace):
    from vaultengine.formats import BUNDLED_FORMATS_DIR, FormatError, load_format, resolve_format_path

    try:
        return load_format(resolve_format_path(args.format, [Path("formats"), BUNDLED_FORMATS_DIR]))
    except FormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None


def _cmd_formats(args: argparse.Namespace) -> int:
    from vaultengine.formats import TypedVault
    from vaultengine.vault import Vault

    spec = _load_spec(args)
    if spec is None:
        return 2
    if args.type and spec.type_named(args.type) is None:
        names = ", ".join(t.name for t in spec.types)
        print(f"error: format {spec.name!r} has no type {args.type!r} (types: {names})", file=sys.stderr)
        return 2
    vault = Vault.load(args.vault)
    typed = TypedVault(spec, vault)

    if args.note:
        matches = [p for p in vault.files_at_path(args.note) or vault.files_named(args.note) if p in vault.notes]
        if len(matches) != 1:
            problem = "no note" if not matches else f"{len(matches)} notes"
            print(f"error: {problem} matching {args.note!r}", file=sys.stderr)
            return 2
        data = typed.note_report(matches[0])
        _print_json(data) if args.json else print(_format_note(data))
        return 1 if args.strict and data["issues"] else 0

    data = typed.report(only_type=args.type)
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), **data})
    else:
        print(_format_formats(data, args.vault, args.limit))
    return 1 if args.strict and data["issues"]["items"] else 0


# ----- check -------------------------------------------------------------------


def _format_check(data: dict[str, Any], vault: str) -> str:
    fmt, scope, summary = data["format"], data["scope"], data["summary"]
    out = [f"vault check {Path(vault).resolve()}  (format {fmt['name']}: {fmt['path']})"]
    if scope["mode"] == "all":
        out.append(f"  checked {_plural(scope['checked'], 'note')}")
    elif scope["reason"]:
        out.append(f"  checked all {_plural(scope['checked'], 'note')}: the index needs a rebuild ({scope['reason']})")
    else:
        since = f" (indexed {scope['last_indexed_at']})" if scope["last_indexed_at"] else ""
        out.append(f"  checked {_plural(scope['checked'], 'note')} changed since the last index{since}")
        for marker, paths in (("+", scope["added"]), ("~", scope["changed"]), ("-", scope["deleted"])):
            out.extend(f"    {marker} {path}" for path in paths)
    for note in data["notes"]:
        kind = note["type"] or "no type"
        affected = "" if note["checked"] else ", not changed: affected by a changed or deleted note"
        out.extend(["", f"{note['path']}  ({kind}{affected})"])
        for item in note["violations"]:
            out.append(f"  {item['line']:>5}  {item['code']:<20} {item['message']}")
    out.append("")
    if data["ok"]:
        out.append(f"✓ no violations in {_plural(scope['checked'], 'note')} checked")
    else:
        out.append(f"✗ {_plural(summary['violations'], 'violation')} in {_plural(summary['notes'], 'note')}")
        out.extend(f"  {row['count']:>6}  {row['code']}" for row in summary["by_code"])
    return "\n".join(out)


def _cmd_check(args: argparse.Namespace) -> int:
    from vaultengine.formats import TypedVault
    from vaultengine.formats.check import check_report, scope_all, scope_changed, select_issues
    from vaultengine.vault import Vault

    spec = _load_spec(args)
    if spec is None:
        return 2
    stale = None
    if args.changed:
        if not Path(args.db).exists():
            print(f"error: --changed needs an index, and there is none at {args.db}; run `vault index {args.vault}` first",
                  file=sys.stderr)
            return 2
        with VaultIndex.open_existing(args.db) as index:
            stale = index.staleness(args.vault)
            last = index.meta().get("last_indexed_at")
    typed = TypedVault(spec, Vault.load(args.vault))
    if stale is None:
        scope = scope_all(typed)
    else:
        scope = scope_changed(typed, stale.added, stale.changed, stale.deleted, last, stale.reason)
    data = check_report(typed, scope, select_issues(typed, scope))
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), **data})
    else:
        print(_format_check(data, args.vault))
    return 0 if data["ok"] else 1


# ----- fix -------------------------------------------------------------------------


def _cmd_fix(args: argparse.Namespace) -> int:
    from vaultengine.formats.fix import FIXABLE, plan_fixes
    from vaultengine.formats.write import FileChange, WritePlan

    found = {}

    def build(typed):
        plan = plan_fixes(typed, only=tuple(args.only) if args.only else FIXABLE)
        found["plan"] = plan
        return WritePlan("fix", [FileChange(f.path, f.original, f.new, f.changes) for f in plan.files])

    return _run_write_command(args, build, extra=lambda: _fix_extra(found.get("plan"), args))


def _fix_extra(plan, args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    if plan is None:
        return [], {}
    lines = []
    if plan.skipped:
        lines.append(f"Not fixed, needs a decision ({len(plan.skipped)}):")
        lines.extend(f"  {issue.path}:{issue.line}  {issue.code}: {reason}" for issue, reason in plan.skipped)
    if plan.remaining:
        rows = ", ".join(f"{count} {code}" for code, count in sorted(plan.remaining.items(), key=lambda kv: (-kv[1], kv[0])))
        when = "after this fix" if plan.files else "now"
        lines.append(f"Violations left {when}, for you to fix: {rows}")
    data = {
        "skipped": [{**issue.to_dict(), "reason": reason} for issue, reason in plan.skipped],
        "not_fixable": dict(plan.untouched),
        "remaining": dict(plan.remaining),
    }
    return lines, data


# ----- safe writes: new, link, set ------------------------------------------------------


def _run_write_command(args: argparse.Namespace, build, extra=None) -> int:
    from vaultengine.formats.write import WriteError, WriteFailed, diff_text, run_write

    spec = _load_spec(args)
    if spec is None:
        return 2
    try:
        outcome = run_write(spec, args.vault, args.db, build, dry_run=args.dry_run)
    except WriteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except WriteFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    extra_lines, extra_data = extra() if extra else ([], {})
    plan, verdict = outcome.plan, outcome.verdict
    diff = diff_text(plan.files) if plan.files and (args.dry_run or args.diff or outcome.status == "refused") else ""
    code = 1 if outcome.status == "refused" else 0

    if args.json:
        from vaultengine.formats.write import outcome_to_dict

        _print_json({"vault": str(Path(args.vault).resolve()), **outcome_to_dict(outcome, args.dry_run),
                     "ok": outcome.status != "refused", **extra_data})
        return code

    out = []
    if outcome.recovered:
        out.append(f"recovered an interrupted write: restored {', '.join(outcome.recovered)}")
    if outcome.status == "unchanged":
        out.extend(plan.notes or [])
        out.append("Nothing to write." if plan.operation != "fix" else "Nothing to fix automatically.")
    elif outcome.status == "refused":
        count = len(verdict.new_violations)
        what = {"new": "new note", "set": "change", "link": "link", "fix": "fix"}.get(plan.operation, plan.operation)
        out.append(f"✗ refused: the {what} would add {_plural(count, 'violation')}; nothing was written")
        for path in sorted({i.path for i in verdict.new_violations}):
            out.append(f"  {path}")
            for issue in (i for i in verdict.new_violations if i.path == path):
                out.append(f"  {issue.line:>5}  {issue.code:<20} {issue.message}")
        if diff:
            out.extend(["", "The refused change:", diff.rstrip("\n")])
    else:
        if diff:
            out.append(diff.rstrip("\n"))
            out.append("")
        if plan.operation == "fix":
            verb = "Would fix" if outcome.status == "dry-run" else "Fixed"
            out.append(f"{verb} {_plural(len(plan.files), 'note')}:")
        else:
            verb = "Would write" if outcome.status == "dry-run" else "Wrote"
            out.append(f"{verb} {_plural(len(plan.files), 'note')} ({plan.operation}):")
        for change in plan.files:
            out.append(f"  {'+' if change.created else '~'} {change.path}")
            out.extend(f"      {text}" for text in change.changes)
        out.extend(f"  = {note}" for note in plan.notes)
        out.append("  check    no new violations")
        if verdict.tolerated:
            codes = ", ".join(f"{n} {c}" for c, n in sorted(Counter(i.code for i in verdict.tolerated).items()))
            out.append(f"           already there, not blocking: {codes} (run vault check)")
        index = outcome.index
        if outcome.status == "dry-run":
            out.append("Dry run: nothing was written.")
        elif index["status"] == "updated":
            pending = f"; left pending (unindexed changes with violations): {', '.join(index['pending'])}" if index["pending"] else ""
            out.append(f"  index    updated {_plural(len(index['synced']), 'note')}{pending}")
        elif index["status"] == "no-index":
            out.append("  index    none (run `vault index` to create one)")
        else:
            out.append(f"  index    not updated: it needs a full rebuild ({index['reason']}); run `vault index`")
    if extra_lines:
        out.extend([""] + extra_lines)
    print("\n".join(out))
    return code


def _pairs(items: list[str] | None, what: str) -> dict[str, str]:
    result = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"error: {what} must look like KEY=VALUE, got {item!r}")
        result[key.strip()] = value
    return result


def _read_section_text(value: str) -> str:
    if value == "-":
        return sys.stdin.read()
    if value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8")
    return value


def _cmd_new(args: argparse.Namespace) -> int:
    from vaultengine.formats.write import plan_new

    values = _pairs(args.set, "--set")
    sections = {name: _read_section_text(text) for name, text in (args.section or [])}
    relations = [(name, target) for name, target in (args.rel or [])]
    return _run_write_command(args, lambda typed: plan_new(
        typed, args.type, args.title, folder=args.folder, name=args.name, values=values,
        relations=relations, sections=sections,
    ))


def _cmd_link(args: argparse.Namespace) -> int:
    from vaultengine.formats.write import plan_link

    return _run_write_command(args, lambda typed: plan_link(typed, args.source, args.target, args.relation))


def _cmd_set(args: argparse.Namespace) -> int:
    from vaultengine.formats.write import plan_set

    return _run_write_command(args, lambda typed: plan_set(typed, args.note, args.key, args.value))


# ----- ingest and coverage ---------------------------------------------------------------


def _candidate_lines(candidate, indent: str = "      ") -> list[str]:
    from vaultengine.formats import similarity

    title_flag = "duplicate" if similarity.is_title_duplicate(candidate.title) else "below"
    lines = [f"{indent}title similarity  {candidate.title:.3f}  {title_flag} (threshold {similarity.TITLE_THRESHOLD})"
             f"   {candidate.title_pair[0]!r} ~ {candidate.title_pair[1]!r}"]
    if candidate.overlap is None:
        lines.append(f"{indent}content overlap   -      too short to compare ({candidate.compared} five-word sequences)")
    else:
        flag = "duplicate" if similarity.is_content_duplicate(candidate.overlap) else "below"
        lines.append(f"{indent}content overlap   {candidate.overlap:.3f}  {flag} (threshold {similarity.OVERLAP_THRESHOLD})"
                     f"   {candidate.shared} of {candidate.compared} five-word sequences shared")
    if candidate.same_hash:
        lines.append(f"{indent}same file         yes    identical SHA-256 (the note file, or its recorded content hash)")
    return lines


def _cmd_ingest(args: argparse.Namespace) -> int:
    from vaultengine.formats.ingest import DuplicateFound, plan_ingest, read_document
    from vaultengine.formats.write import WriteError

    try:
        document = read_document(args.file, title=args.title)
    except WriteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    found: dict[str, Any] = {}
    values = _pairs(args.set, "--set")

    def extra():
        lines = [f"  note     {w}" for w in document.warnings]
        others = [c for c in found.get("candidates", []) if not c.duplicate][:1]
        if others:
            c = others[0]
            overlap = "-" if c.overlap is None else f"{c.overlap:.3f}"
            lines.append(f"  closest  {c.path}  (title {c.title:.3f}, content overlap {overlap})")
        data = {"document": {"file": document.file_name, "kind": document.kind, "title": document.title,
                             "sha256": document.sha256, "url": document.url, "warnings": document.warnings},
                "candidates": [c.to_dict() for c in found.get("candidates", [])[:5]]}
        return lines, data

    try:
        return _run_write_command(
            args, lambda typed: plan_ingest(typed, document, args.as_type, folder=args.folder, values=values, found=found),
            extra=extra,
        )
    except DuplicateFound as dup:
        if args.json:
            _print_json({
                "vault": str(Path(args.vault).resolve()), "operation": "ingest", "status": "duplicate", "ok": False,
                "written": False,
                "document": {"file": document.file_name, "kind": document.kind, "title": document.title, "sha256": document.sha256},
                "duplicates": [c.to_dict() for c in dup.duplicates],
                "candidates": [c.to_dict() for c in dup.candidates[:5]],
            })
            return 1
        count = len(dup.duplicates)
        out = [f"✗ not ingested: {document.file_name} duplicates {_plural(count, 'existing note')}; nothing was written",
               f"  title    {document.title!r} ({document.kind}, sha256 {document.sha256[:12]}…)"]
        for candidate in dup.duplicates:
            out.extend(["", f"  {candidate.path}", *_candidate_lines(candidate)])
        print("\n".join(out))
        return 1


def _cmd_coverage(args: argparse.Namespace) -> int:
    from vaultengine.formats import TypedVault
    from vaultengine.formats.coverage import coverage
    from vaultengine.vault import Vault

    spec = _load_spec(args)
    if spec is None:
        return 2
    if spec.coverage is None:
        print(f"error: format {spec.name!r} has no `coverage:` block", file=sys.stderr)
        return 2
    typed = TypedVault(spec, Vault.load(args.vault))
    data = coverage(typed)
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), **data})
        return 0
    sources, citing = data["sources"], data["citing"]
    reasons = citing["by_reason"]
    out = [f"vault coverage {Path(args.vault).resolve()}  (format {data['format']['name']})", ""]
    out.append(f"  sources        {sources['total']:>5}   {sources['linked']} linked from other notes, "
               f"{len(sources['unlinked'])} that nothing links to")
    out.append(f"  citing notes   {citing['total']:>5}   {citing['cite_a_source']} cite a source, {citing['without_source']} cite none "
               f"({', '.join(data['citing_types'])})")
    out.append(f"    of those     {reasons['missing']} cite only missing notes, {reasons['not_a_source']} cite only non-sources, "
               f"{reasons['none']} have no citation at all")
    if sources["unlinked"]:
        shown, more = _shown(sources["unlinked"], args.limit)
        out.extend(["", f"Sources nothing links to ({len(sources['unlinked'])})", *[f"  {p}" for p in shown]])
        if more:
            out.append(more)
    if citing["notes"]:
        shown, more = _shown(citing["notes"], args.limit)
        out.extend(["", f"Notes that cite no source ({len(citing['notes'])})"])
        labels = {"missing": "cites only missing notes", "not-a-source": "cites only non-sources", "none": "no citation"}
        for note in shown:
            detail = ""
            if note["citations"]:
                first = note["citations"][0]
                also = f" (+{len(note['citations']) - 1} more)" if len(note["citations"]) > 1 else ""
                target = f" -> {first['target_path']} ({first['target_type'] or 'no type'})" if first["target_path"] else ""
                detail = f": line {first['line']} {first['raw']}{target}{also}"
            out.append(f"  {note['path']}  ({note['type']}) {labels[note['reason']]}{detail}")
        if more:
            out.append(more)
    if data["most_cited"]:
        out.extend(["", "Most cited sources (citing notes)",
                    *[f"  {row['citing_notes']:>5}  {row['path']}" for row in data["most_cited"]]])
    print("\n".join(out))
    return 0


# ----- search ------------------------------------------------------------------

_FIELD_NAMES = {"title": "title", "aliases": "alias", "headings": "heading", "tags": "tag", "body": "body"}


class _Style:
    """ANSI styling when writing to a colour terminal; «marks» otherwise."""

    def __init__(self, color: bool) -> None:
        self.color = color

    def _wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.color and text else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def width(self, pieces) -> int:
        """Characters the pieces take on screen (the «» marks count when not in colour)."""
        return sum(len(p.text) + (2 if p.mark and not self.color else 0) for p in pieces)

    def pieces(self, pieces, base: str = "") -> str:
        out = []
        for piece in pieces:
            if not piece.mark:
                out.append(self._wrap(base, piece.text) if base else piece.text)
            elif self.color:
                out.append(self._wrap("1;33" if piece.mark == "exact" else "4;33", piece.text))
            else:
                out.append(f"«{piece.text}»")
        return "".join(out)


def _use_color(choice: str) -> bool:
    if choice == "always":
        return True
    if choice == "never":
        return False
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"


def _where(match) -> str:
    parts = []
    for name, f in match.fields.items():
        parts.append(_FIELD_NAMES[name] + (f" ×{f.tf}" if f.tf > 1 else ""))
    label = match.clause if match.exact else f"{match.clause} ≈ {match.term} ({_plural(match.edits, 'edit')})"
    return f"{label}: {', '.join(parts)}"


def _clause_line(label: str, stats) -> str:
    if stats.kind == "word":
        found = f"{_plural(stats.df, 'note')}" if stats.df else "no notes"
        if stats.required:
            found += " (exact word, required)"
        if stats.variants:
            shown = [f"{term} ({_plural(edits, 'edit')}, {_plural(stats.variant_df[term], 'note')})"
                     for term, edits in list(stats.variants.items())[:4]]
            more = f", +{len(stats.variants) - 4} more" if len(stats.variants) > 4 else ""
            found += " · close spellings: " + ", ".join(shown) + more
        elif not stats.df and not stats.required:
            found += ", no close spellings"
        return f"{label}  {found}"
    kind = "phrase" if stats.kind == "phrase" else "near"
    return f"{label}  {_plural(stats.df, 'note')} ({kind}, required)"


def _group_label(hit, clauses: int) -> str:
    if not hit.matches:
        return "matches the filters, none of the words"
    if hit.exact == 0:
        return "close spellings only"
    if clauses == 1:
        return "exact"
    return f"{hit.exact} of {clauses} exactly"


def _group(hit) -> tuple[int, bool]:
    return hit.exact, bool(hit.matches)


def _format_search(result, header: str, contexts, style: _Style, width: int, explain: bool) -> str:
    out = [header, ""]
    if not result.tree:
        out.append("  empty query")
        return "\n".join(out)
    for warning in result.warnings:
        out.append(f"  note: {warning}")
    if result.clauses:
        pad = max(len(label) for label in result.clauses)
        for label, stats in result.clauses.items():
            out.append("  " + _clause_line(label.ljust(pad), stats))
        order = "exact matches first, then score" if any(s.variants for s in result.clauses.values()) or len(result.clauses) > 1 else "by score"
    else:
        order = "by path (no words to rank by)"
    out.append(f"  {result.total} of {_plural(result.notes, 'note')} match · {order}")
    tiers = {_group(hit) for hit in result.hits} if result.clauses else set()
    previous = None
    for rank, (hit, context) in enumerate(zip(result.hits, contexts), 1):
        if len(tiers) > 1 and _group(hit) != previous:
            out.extend(["", "  " + style.dim(f"── {_group_label(hit, len(result.clauses))} ──")])
            previous = _group(hit)
        score = f"{hit.score:.2f}" if result.clauses else ""
        prefix = f"{rank:>4}. "
        room = max(10, width - len(prefix) - len(score) - 2)
        title_pieces = context.title
        if style.width(title_pieces) > room:  # long titles: cut, keeping the marks that fit
            cut, kept = room - 1, []
            for piece in title_pieces:
                if cut <= 0:
                    break
                kept.append(type(piece)(piece.text[:cut], piece.mark))
                cut -= style.width([kept[-1]])
            title_pieces = kept + [type(title_pieces[0])("…")]
        title_len = style.width(title_pieces)
        out.append("")
        padding = " " * (width - len(prefix) - title_len - len(score)) if score else ""
        out.append(prefix + style.bold(style.pieces(title_pieces)) + padding + score)
        indent = " " * len(prefix)
        out.append(indent + style.dim(hit.path))
        if hit.matches:
            out.append(indent + " · ".join(_where(m) for m in hit.matches.values()))
        if context.snippet is not None:
            snip = context.snippet
            text = ("… " if snip.truncated_start else "") + style.pieces(snip.pieces) + (" …" if snip.truncated_end else "")
            out.append(indent + text)
        if explain:
            for match in hit.matches.values():
                weight = f" × {match.weight:g} (fuzzy)" if not match.exact else ""
                out.append(indent + style.dim(f"  {match.clause}: tf~ {match.tf:.4f} -> {match.score:.4f}{weight}"))
                for name, f in match.fields.items():
                    out.append(indent + style.dim(
                        f"    {name:<8} tf {f.tf:<3} len {f.length:<5} avg {f.avg_length:8.2f}  w*tf/norm {f.weighted_tf:.4f}"))
    if result.total > len(result.hits):
        out.extend(["", f"  … {result.total - len(result.hits)} more (--limit 0 shows all)"])
    facets = result.facets
    if facets is not None and result.total:
        def block(title: str, rows, render) -> None:
            if rows:
                out.extend(["", f"  {title}"])
                out.extend(f"  {n:>5}  {render(name)}" for name, n in rows)

        block("Tags (notes)", facets.tags, lambda name: f"#{name}")
        block("Folders (notes)", facets.folders, lambda name: f"{name}/" if name else "(vault root)")
    return "\n".join(out)


def _cmd_search(args: argparse.Namespace) -> int:
    from vaultengine.snippets import hit_context

    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    with index:
        try:
            result = search_module.search(
                index.conn, args.query, limit=args.limit or None, fuzzy=not args.no_fuzzy, facets=args.facets or None
            )
        except search_module.QueryError as exc:
            if args.json:
                _print_json({"error": exc.to_dict(), "query": args.query})
            else:
                print(exc.render(args.query), file=sys.stderr)
            return 2
    if args.tree:
        print(result.tree or "(empty query)")
        return 0
    width = max(40, min(shutil.get_terminal_size((100, 24)).columns, 120))
    contexts = [hit_context(args.vault, hit, args.query, width=width - 6) for hit in result.hits]
    if args.json:
        data = result.to_dict()
        for hit, context in zip(data["hits"], contexts):
            hit["snippet"] = context.snippet.to_dict() if context.snippet else None
        _print_json({"vault": str(Path(args.vault).resolve()), "index": info, **data})
    else:
        header = _header(f"Search {args.query!r}", args.vault, info)
        print(_format_search(result, header, contexts, _Style(_use_color(args.color)), width, args.explain))
    return 0


# ----- related -----------------------------------------------------------------


def _format_related(result, header: str, style: _Style, width: int, explain: bool) -> str:
    from vaultengine.related import WEIGHTS

    seed = result.seed
    if seed.path is None:
        title = f"Related to {seed.label} (no note yet; {_plural(len(seed.linkers), 'note')} link to it)"
    else:
        title = f"Related to {seed.label}"
    out = [style.bold(title), header, ""]
    weights = " + ".join(f"{WEIGHTS[k]:g} × {k}" for k in ("graph", "text", "tags"))
    out.append(f"  score = {weights} · {_plural(result.compared, 'note')} compared")
    if not result.hits:
        out.append("  nothing related found")
    for rank, hit in enumerate(result.hits, 1):
        prefix = f"{rank:>4}. "
        status = "linked" if hit.linked else "not linked"
        right = f"{hit.score:.2f}  {status:>10}"
        room = max(10, width - len(prefix) - len(right) - 2)
        name = hit.title if len(hit.title) <= room else hit.title[: room - 1] + "…"
        padding = " " * (width - len(prefix) - len(name) - len(right))
        shown_status = style.dim(status) if hit.linked else style._wrap("1;33", status)
        out.append("")
        out.append(prefix + style.bold(name) + padding + f"{hit.score:.2f}  " + " " * (10 - len(status)) + shown_status)
        indent = " " * len(prefix)
        out.append(indent + (hit.reason or style.dim("(weak: only distant links)")))
        if explain:
            out.append(indent + style.dim(f"graph {hit.graph:.2f} · text {hit.text:.2f} · tags {hit.tags:.2f} · {hit.path}"))
    return "\n".join(out)


def _cmd_related(args: argparse.Namespace) -> int:
    from vaultengine.querylang import QueryError
    from vaultengine.related import Relatedness

    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    with index:
        engine = Relatedness(index.conn)
        try:
            seed = engine.seed(args.note)
        except QueryError as exc:
            print(f"error: {exc.message}" + (f"\n  {exc.hint}" if exc.hint else ""), file=sys.stderr)
            return 2
        result = engine.related(seed, limit=args.limit or None, unlinked_only=args.unlinked)
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), "index": info, "compared": result.compared, **result.to_dict()})
        return 0
    width = max(40, min(shutil.get_terminal_size((100, 24)).columns, 120))
    header = "  " + (seed.path or "missing note") + f" · index {info['last_indexed_at']}" + (" (OUT OF DATE)" if info["stale"] else "")
    print(_format_related(result, header, _Style(_use_color(args.color)), width, args.explain))
    return 0


# ----- suggest -----------------------------------------------------------------


def _format_links(pairs, style: _Style, width: int, explain: bool) -> list[str]:
    from vaultengine.connect import WEIGHTS

    out = [style.bold("  Links to add") + f"  (notes with no link between them; score = {WEIGHTS['shared']:g} × shared neighbours"
           f" + {WEIGHTS['cited']:g} × cited together + {WEIGHTS['text']:g} × similar text)"]
    if not pairs:
        out.extend(["", "    nothing to suggest"])
    for rank, pair in enumerate(pairs, 1):
        a, b = link_name(pair.a), link_name(pair.b)
        score = f"{pair.score:.2f}"
        prefix = f"  {rank:>3}. "
        line = f"{a}  ↔  {b}"
        room = width - len(prefix) - len(score) - 2
        if len(line) > room:
            half = (room - 5) // 2
            a = a if len(a) <= half else a[: half - 1] + "…"
            b = b if len(b) <= half else b[: half - 1] + "…"
            line = f"{a}  ↔  {b}"
        out.extend(["", prefix + style.bold(line) + " " * max(2, width - len(prefix) - len(line) - len(score)) + score])
        out.append(" " * len(prefix) + pair.reason)
        if explain:
            out.append(" " * len(prefix) + style.dim(
                f"shared {pair.shared:.2f} · cited {pair.cited:.2f} · text {pair.text:.2f} · {pair.a} ↔ {pair.b}"))
    return out


def _format_suggestions(result, header: str, style: _Style, links=None, width: int = 100, explain: bool = False) -> str:
    out = [header, ""]
    if links is not None:
        out.extend(_format_links(links, style, width, explain))
        if result is None:
            return "\n".join(out)
        out.append("")
    out.append(style.bold("  Notes to write") + f"  ({result.total_missing} link targets have no note;"
               f" {len(result.notes)} most wanted, linked from 2+ notes)")
    for rank, item in enumerate(result.notes, 1):
        mentions = f" · {item.mentions} more mention it" if item.mentions else ""
        out.extend(["", f"  {rank:>3}. {style.bold(item.name)}", f"       {_plural(item.linking_notes, 'note')} link here{mentions}"])
        where = " · ".join(f"{folder or '(vault root)'} ({n})" for folder, n in item.folders)
        out.append(f"       in: {where}")
        if item.about:
            out.append(f"       about: {', '.join(item.about)}")
    if result.same_idea:
        out.extend(["", "  Overlapping names (the same idea twice, or a topic and a narrower one):"])
        for group in result.same_idea:
            out.append("    " + " · ".join(f"{name} ({n})" for name, n in group))
    if result.existing:
        out.extend(["", "  Probably an existing note (the link's words are all in its title or alias):"])
        for name, n, path in result.existing:
            out.append(f"    [[{name}]] ({_plural(n, 'note')}) → {path}")
    out.extend(["", "  Draft one: vault suggest <vault> --draft \"<name>\"  (add --write to create it)"])
    return "\n".join(out)


def _cmd_suggest(args: argparse.Namespace) -> int:
    from vaultengine.connect import LinkSuggester
    from vaultengine.querylang import QueryError
    from vaultengine.suggest import Suggester

    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    with index:
        suggester = Suggester(index.conn)
        if args.draft is None:
            links = None
            if not args.notes:
                for_note = None
                if args.for_note:
                    try:
                        seed = suggester.engine.seed(args.for_note)
                    except QueryError as exc:
                        print(f"error: {exc.message}", file=sys.stderr)
                        return 2
                    if seed.path is None:
                        print(f'error: "{args.for_note}" is not a note yet; only notes can be linked', file=sys.stderr)
                        return 2
                    for_note = suggester.engine.graph.note_ids[seed.node]
                links = LinkSuggester(index.conn, suggester.engine).suggestions(
                    limit=args.limit or None, for_note=for_note, path=args.path
                )
            result = None if args.links or args.for_note else suggester.suggestions(limit=args.limit or None, min_links=args.min_links)
            if args.json:
                data = {"vault": str(Path(args.vault).resolve()), "index": info}
                if links is not None:
                    data["links"] = [pair.to_dict() for pair in links]
                if result is not None:
                    data.update(result.to_dict())
                _print_json(data)
            else:
                width = max(60, min(shutil.get_terminal_size((100, 24)).columns, 140))
                header = _header("Suggestions", args.vault, info)
                print(_format_suggestions(result, header, _Style(_use_color(args.color)), links, width, args.explain))
            return 0
        try:
            path, text = suggester.draft(args.draft)
        except KeyError:
            print(f'error: no link points to a missing note called "{args.draft}"', file=sys.stderr)
            print("  run `vault suggest` to list them", file=sys.stderr)
            return 2
    if args.folder is not None:
        path = f"{args.folder.strip('/')}/{path.rsplit('/', 1)[-1]}" if args.folder.strip("/") else path.rsplit("/", 1)[-1]
    if not args.write:
        print(f"# would be written to {path} (add --write to create it)\n")
        print(text, end="")
        return 0
    target = Path(args.vault) / path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError:
        print(f"error: {path} already exists; nothing written", file=sys.stderr)
        return 2
    print(f"Wrote {path}. Run `vault index {args.vault}` to link it up.")
    return 0


# ----- unlinked ----------------------------------------------------------------


def _cmd_unlinked(args: argparse.Namespace) -> int:
    from vaultengine.connect import MentionFinder
    from vaultengine.querylang import QueryError
    from vaultengine.related import Relatedness

    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    with index:
        finder = MentionFinder(index.conn, args.vault, min_words=args.min_words)
        if args.note:
            try:
                seed = Relatedness(index.conn).seed(args.note)
            except QueryError as exc:
                print(f"error: {exc.message}", file=sys.stderr)
                return 2
            if seed.path is None:
                print(f'error: "{args.note}" is not a note yet, so there is nothing to link to', file=sys.stderr)
                return 2
            target = next(f for f, p in finder.paths.items() if p == seed.path)
            mentions = finder.of(target)
        else:
            mentions = finder.all()
    by_target: dict[str, list] = {}
    for mention in mentions:
        by_target.setdefault(mention.target, []).append(mention)
    order = sorted(by_target, key=lambda t: (-len({m.source for m in by_target[t]}), t))
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), "index": info, "mentions": [asdict(m) for m in mentions]})
        return 0
    style = _Style(_use_color(args.color))
    out = [_header("Unlinked mentions", args.vault, info), ""]
    notes = len({m.source for m in mentions})
    scope = f"of {link_name(seed.path)}" if args.note else "of any note"
    verb = "does" if notes == 1 else "do"
    out.append(f"  {_plural(len(mentions), 'plain-text mention')} {scope} in {_plural(notes, 'note')} that {verb} not link to it yet")
    shown = order if args.limit == 0 or args.note else order[: args.limit]
    for target in shown:
        items = by_target[target]
        names = sorted({m.name for m in items})
        out.extend(["", "  " + style.bold(link_name(target)) + f"  ({_plural(len({m.source for m in items}), 'note')}"
                    + (f"; as {', '.join(repr(n) for n in names)}" if names != [link_name(target)] else "") + ")"])
        per = items if args.note or args.all else items[:3]
        for m in per:
            flat = lambda text: " ".join(text.split()) if text.strip() else ""
            words = flat(m.text)
            mark = style._wrap("1;33", words) if style.color else f"«{words}»"
            before = flat(m.before) + (" " if m.before[-1:].isspace() else "")
            after = (" " if m.after[:1].isspace() else "") + flat(m.after)
            out.append(f"    {m.source}:{m.line}  {before}{mark}{after}")
        if len(per) < len(items):
            out.append(f"    … {len(items) - len(per)} more (--all shows every mention)")
    if len(shown) < len(order):
        out.extend(["", f"  … {len(order) - len(shown)} more notes are mentioned (--limit 0 shows all)"])
    single = [t for t in shown if any(len(m.name.split()) == 1 and "-" not in m.name for m in by_target[t])]
    if single and not args.note and args.min_words < 2:
        out.append("  one-word titles match every use of the word; --min-words 2 leaves them out")
    print("\n".join(out))
    return 0


# ----- rpc ---------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    from vaultengine.formats import BUNDLED_FORMATS_DIR, FormatError, load_format, resolve_format_path
    from vaultengine.rpc.server import serve_stdio
    from vaultengine.service import VaultService

    format_path = None
    if args.format:
        try:
            format_path = resolve_format_path(args.format, [Path("formats"), BUNDLED_FORMATS_DIR])
            load_format(format_path)  # fail now, not on the first call
        except FormatError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    return serve_stdio(VaultService(args.vault, args.db, format_path))


def _cmd_rpc_schema(args: argparse.Namespace) -> int:
    from vaultengine.rpc.schema import schema_text

    text = schema_text()
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


# ----- entry point -----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vault", description="Index and inspect an Obsidian vault.")
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str, handler, json_flag: bool = True, vault_required: bool = False) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
        if vault_required:  # writes name their vault; also lets options sit between positionals
            sub.add_argument("vault", help="vault folder")
        else:
            sub.add_argument("vault", nargs="?", default=".", help="vault folder (default: current folder)")
        sub.add_argument("--db", help="index file (default: VAULT/.vaultengine/index.sqlite)")
        if json_flag:
            sub.add_argument("--json", action="store_true", help="print JSON instead of text")
        sub.set_defaults(handler=handler, needs_vault=True)
        return sub

    index = add("index", "Create or incrementally update the index.", _cmd_index)
    index.add_argument("--full", action="store_true", help="re-parse every note and re-resolve every link")
    index.add_argument("--limit", type=int, default=10, help="max paths listed per change type (0 = all)")
    stats = add("stats", "Summary numbers for the indexed vault.", _cmd_stats)
    stats.add_argument("--top", type=int, default=10, help="rows in each top-N table")
    health = add("health", "Link and structure problems in the indexed vault.", _cmd_health)
    health.add_argument("--limit", type=int, default=10, help="max items listed per problem type (0 = all)")
    search = add(
        "search",
        "Full-text search ranked by BM25F (title > headings/aliases/tags > body).",
        _cmd_search,
    )
    search.description = (
        "Search the vault with text and structure. Words rank (and forgive typos); "
        '"phrases", NEAR/n, tag:, path:, [property], links-to:, linked-from:, within:N of:, '
        "has:, missing:, modified:, -, NOT, OR and ( ) narrow. See README.md for the full syntax."
    )
    search.add_argument("query", help='e.g.  reward hacking links-to:[[GRPO]] -path:10_SOURCES')
    search.add_argument("--limit", type=int, default=10, help="number of results (0 = all)")
    search.add_argument("--explain", action="store_true", help="show the score breakdown per clause and field")
    search.add_argument("--no-fuzzy", action="store_true", help="match words only as typed")
    search.add_argument("--facets", type=int, default=8, help="rows of tag and folder counts (0 = all)")
    search.add_argument("--tree", action="store_true", help="print the syntax tree of the query and stop")
    search.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="colour output (default: auto)")
    related = add("related", "Notes related to a note: links, tags and text, each with a reason.", _cmd_related)
    related.description = (
        "Notes related to a note, including ones it does not link to yet: personalized PageRank over the "
        "link graph (missing notes included), blended with text similarity and shared tags. Each result says why."
    )
    related.add_argument("note", help="note name, path or alias (as in a link), or the target of a link to a missing note")
    related.add_argument("--limit", type=int, default=10, help="number of results (0 = all)")
    related.add_argument("--unlinked", action="store_true", help="only notes with no link to or from the note")
    related.add_argument("--explain", action="store_true", help="show the three signals behind each score")
    related.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="colour output (default: auto)")
    suggest = add("suggest", "Links to add between notes, notes to write, and drafts of them.", _cmd_suggest)
    suggest.add_argument("--limit", type=int, default=10, help="items per list (0 = all)")
    only = suggest.add_mutually_exclusive_group()
    only.add_argument("--links", action="store_true", help="only the links to add")
    only.add_argument("--notes", action="store_true", help="only the notes to write")
    suggest.add_argument("--for", dest="for_note", metavar="NOTE", help="links to add to or from this note")
    suggest.add_argument("--path", help="links between notes whose paths both contain this text, e.g. --path 20_ZETTLEKASTEN")
    suggest.add_argument("--explain", action="store_true", help="show the three kinds of evidence as numbers")
    suggest.add_argument("--min-links", type=int, default=2, help="only missing notes linked from at least this many notes")
    suggest.add_argument("--draft", metavar="NAME", help="print a draft for this missing note, built from the notes that link to it")
    suggest.add_argument("--write", action="store_true", help="with --draft: create the note (never overwrites)")
    suggest.add_argument("--folder", help="with --draft: folder for the note (default: where most linking notes are)")
    suggest.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="colour output (default: auto)")
    unlinked = add("unlinked", "Plain-text mentions of notes (titles and aliases) that are not links yet.", _cmd_unlinked)
    unlinked.description = (
        "Plain-text mentions of a note's title or alias that are not links yet. Titles match as words, so "
        "'reward hacking' mentions reward-hacking. Never inside code, links or URLs, never a note that already "
        "links to it, never a note mentioning itself."
    )
    unlinked.add_argument("note", nargs="?", help="only mentions of this note (name, path or alias); default: every note")
    unlinked.add_argument("--limit", type=int, default=10, help="notes listed when no note is given (0 = all)")
    unlinked.add_argument("--all", action="store_true", help="show every mention, not the first three per note")
    unlinked.add_argument("--min-words", type=int, default=1, help="ignore titles and aliases shorter than this many words")
    unlinked.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="colour output (default: auto)")
    formats = commands.add_parser(
        "formats",
        help="Check notes against a format file (note types, headings, metadata, relations, link roles).",
        description="Type every note with a format file and report where notes depart from it. "
        "Reads the vault files directly; no index is needed.",
    )
    formats.add_argument("vault", nargs="?", default=".", help="vault folder (default: current folder)")
    formats.add_argument("--format", "-f", required=True, help="format file, or a name in ./formats/ or the bundled formats/ (e.g. brain)")
    formats.add_argument("--type", help="only this note type")
    formats.add_argument("--note", help="show one note as the format sees it (vault path or note name)")
    formats.add_argument("--limit", type=int, default=10, help="max issues and untyped notes listed (0 = all)")
    formats.add_argument("--strict", action="store_true", help="exit with code 1 if any issue is found")
    formats.add_argument("--json", action="store_true", help="print JSON instead of text")
    formats.set_defaults(handler=_cmd_formats, needs_vault=True, uses_index=False)
    check = add("check", "Report every violation of a format file, per note, with line numbers (exit 1 if any).", _cmd_check)
    check.add_argument("--format", "-f", required=True, help="format file, or a name in ./formats/ or the bundled formats/")
    check.add_argument(
        "--changed", action="store_true",
        help="only notes added or changed since the last `vault index`, plus violations they cause in other notes",
    )
    def writer(name: str, help_text: str, handler) -> argparse.ArgumentParser:
        sub = add(name, help_text, handler, vault_required=True)
        sub.add_argument("--format", "-f", required=True, help="format file, or a name in ./formats/ or the bundled formats/")
        sub.add_argument("--dry-run", "-n", action="store_true", help="check and show the diff; write nothing")
        sub.add_argument("--diff", action="store_true", help="also print the diff of what is written")
        return sub

    fix = writer("fix", "Repair missing mirror relations and missing required headings; nothing else.", _cmd_fix)
    fix.add_argument("--only", action="append", choices=["missing-mirror", "missing-heading"],
                     help="repair only this kind (repeatable)")
    new = writer("new", "Create a note of a type from the format (refused if it would fail vault check).", _cmd_new)
    new.add_argument("--type", "-t", required=True, help="note type from the format")
    new.add_argument("--title", required=True, help="the note's title (H1, and the file name per the type's new.name)")
    new.add_argument("--folder", help="folder for the note (default: the type's match folder)")
    new.add_argument("--name", help="file name without .md (default: from the title)")
    new.add_argument("--set", action="append", metavar="KEY=VALUE", help="a metadata value (repeatable); {now} works for dates")
    new.add_argument("--rel", action="append", nargs=2, metavar=("RELATION", "NOTE"),
                     help="a relation to an existing note, mirror written there (repeatable)")
    new.add_argument("--section", action="append", nargs=2, metavar=("HEADING", "TEXT"),
                     help="text under a required heading; @file reads a file, - reads stdin (repeatable)")
    link = writer("link", "Relate note A to note B (A relation:: [[B]]) and write the mirror in B.", _cmd_link)
    link.add_argument("source", help="note A: vault path or note name")
    link.add_argument("target", help="note B: vault path or note name")
    link.add_argument("--as", dest="relation", required=True, help="relation name, e.g. supports")
    setter = writer("set", "Set a metadata value of a note (refused if it would fail vault check).", _cmd_set)
    setter.add_argument("note", help="vault path or note name")
    setter.add_argument("key", help="a metadata key of the note's type")
    setter.add_argument("value", help="the value as written in the note, e.g. '#supported'; {now} works for dates")
    ingest = writer("ingest", "Turn a markdown, text or HTML file into a note (refused if it duplicates one).", _cmd_ingest)
    ingest.add_argument("file", help="the file to ingest (.md, .txt, .html)")
    ingest.add_argument("--as", dest="as_type", required=True, help="note type, e.g. source")
    ingest.add_argument("--title", help="title to use instead of the document's")
    ingest.add_argument("--folder", help="folder for the note (default: the type's match folder)")
    ingest.add_argument("--set", action="append", metavar="KEY=VALUE", help="a metadata value (repeatable)")
    cov = add("coverage", "Sources nothing links to, and notes that cite no source.", _cmd_coverage)
    cov.add_argument("--format", "-f", required=True, help="format file, or a name in ./formats/ or the bundled formats/")
    cov.add_argument("--limit", type=int, default=10, help="max notes listed per list (0 = all)")
    cov.set_defaults(uses_index=False)
    serve = add("serve", "Serve the JSON-RPC 2.0 API on stdin/stdout (one JSON message per line).", _cmd_serve, json_flag=False)
    serve.add_argument("--format", "-f", help="format for vault.check/fix/coverage and the note.* writes (name or file)")
    schema = commands.add_parser("rpc-schema", help="Print the JSON Schema of the JSON-RPC API.")
    schema.add_argument("--output", help="write to this file instead of stdout")
    schema.set_defaults(handler=_cmd_rpc_schema, needs_vault=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.needs_vault:
        return args.handler(args)
    if not Path(args.vault).is_dir():
        print(f"error: vault folder not found: {args.vault}", file=sys.stderr)
        return 2
    if getattr(args, "uses_index", True):
        args.db = Path(args.db) if args.db else default_db_path(args.vault)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
