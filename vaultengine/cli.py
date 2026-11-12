"""`vault` command line: index, stats, health, search, related, suggest, unlinked, embed, config show,
formats, check, fix, new, link, set, ingest, coverage, serve (JSON-RPC), rpc-schema.

Settings (vault, index, format, embedding model) come from flags, then VAULTENGINE_* environment
variables, then the `engine:` section of config/config.yaml, then defaults (``settings.py``).

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
from vaultengine.index import IndexReport, Staleness, VaultIndex


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


def _run_write_command(args: argparse.Namespace, build, extra=None, on_outcome=None) -> int:
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
    if on_outcome is not None:
        on_outcome(outcome)
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
        what = {"new": "new note", "set": "change", "link": "link", "fix": "fix", "link-it": "link",
                "unlink": "unlink"}.get(plan.operation, plan.operation)
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


def _related_weights(args: argparse.Namespace, unlinked: bool = False) -> dict[str, float]:
    """engine.related.weights, or engine.related.unlinked_weights when what is wanted is not linked yet
    (`related --unlinked`, `discover`, `explore`: the mix `vault experiment` measured for exactly that)."""
    related = args.settings.engine.related
    return (related.unlinked_weights if unlinked else related.weights).model_dump()


def _tag_rule(args: argparse.Namespace):
    """Which links are tags (tagrule.py): from the format (--format, engine.format), else fields named tags.
    None after printing the error when the format cannot be loaded."""
    from vaultengine.formats import FormatError
    from vaultengine.tagrule import rule_for_format

    try:
        return rule_for_format(getattr(args, "format", None) or args.settings.engine.format)
    except FormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None


def _text_notes(unavailable: str | None, uncached: int, vault: str) -> list[str]:
    """What to tell about the text signal: off (and why), or computed without the cache."""
    if unavailable:
        return [f"  text signal unavailable: {unavailable}",
                "    (the other signals still rank; their weights are rescaled to sum to 1)"]
    if uncached:
        return [f"  {_plural(uncached, 'note')} had no cached vector: embedded in memory for this run only;"
                f" `vault embed {vault}` caches them"]
    return []


def _text_status(text) -> dict[str, Any]:
    """The text signal's state for --json (``text``: a related.TextSimilarity that was used)."""
    on = text.available()
    return {"available": on, "model_id": text.vectors.model_id if on else None, "unavailable": text.unavailable,
            "uncached": len(text.uncached)}


def _format_related(result, header: str, style: _Style, width: int, explain: bool, vault: str = ".") -> str:
    seed = result.seed
    if seed.path is None:
        title = f"Related to {seed.label} (no note yet; {_plural(len(seed.linkers), 'note')} link to it)"
    else:
        title = f"Related to {seed.label}"
    out = [style.bold(title), header, ""]
    weights = " + ".join(f"{result.weights[k]:.2g} × {k}" for k in ("graph", "cocitation", "coupling", "text", "tags")
                         if result.weights.get(k))
    model = f" · text: {result.model_id} embeddings" if result.model_id else ""
    out.append(f"  score = {weights} · {_plural(result.compared, 'note')} compared{model}")
    out.extend(_text_notes(result.text_unavailable, result.uncached, vault))
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
            extra = "".join(f" · {k} {getattr(hit, k):.2f}" for k in ("cocitation", "coupling") if result.weights.get(k))
            out.append(indent + style.dim(f"graph {hit.graph:.2f} · text {hit.text:.2f} (cosine {hit.cosine:.2f}) · tags {hit.tags:.2f}"
                                          f"{extra} · {hit.path}"))
    return "\n".join(out)


def _cmd_related(args: argparse.Namespace) -> int:
    from vaultengine.querylang import QueryError
    from vaultengine.related import Relatedness

    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    rule = _tag_rule(args)
    if rule is None:
        return 2
    with index:
        engine = Relatedness(index.conn, lambda: args.settings.provider(), args.vault, rule,
                             _related_weights(args, unlinked=args.unlinked))
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
    print(_format_related(result, header, _Style(_use_color(args.color)), width, args.explain, args.vault))
    return 0


# ----- explain -----------------------------------------------------------------


def _format_explain(rel, header: str, style: _Style, vault: str, limit: int, tag_rule: str = "") -> str:
    from vaultengine.related import tag_label
    from vaultengine.relations import LABELS

    a, b = link_name(rel.a), link_name(rel.b)
    out = [style.bold(f"How {a} and {b} relate"), header, ""]
    out.append(f"  score {rel.score:.2f} = the sum of weight × value  (weights: engine.explore.weights, rescaled to sum to 1)")
    text = rel.signal("text")
    out.extend(_text_notes(text.unavailable, rel.uncached, vault))
    out.extend(["", f"  {'signal':<14}{'value':>6}   {'weight':>6}   {'contribution':>12}"])
    for s in rel.signals:
        if not s.available:
            out.append(f"  {LABELS[s.name]:<14}{'—':>6}   {'—':>6}   {'unavailable':>12}")
        else:
            out.append(f"  {LABELS[s.name]:<14}{s.value:>6.2f} × {s.weight:>6.2f} = {s.contribution:>12.3f}")

    def more(items: list, shown: int) -> list[str]:
        return [f"      … {len(items) - shown} more (--limit 0 shows all)"] if len(items) > shown else []

    cut = lambda items: items if limit <= 0 else items[:limit]
    width = lambda names: max((len(n) for n in names), default=0)

    d = rel.signal("direct").evidence
    out += ["", style.bold(f"  direct links  {rel.signal('direct').value:.2f}") + "  (1 if either links to the other)"]
    for src, dst, lines, tagged in ((a, b, d["a_to_b"], d["a_to_b_tag_lines"]), (b, a, d["b_to_a"], d["b_to_a_tag_lines"])):
        marks = ", ".join(f"{n} (a tag link)" if n in tagged else str(n) for n in lines)
        where = f"{_plural(len(lines), 'link')} (line{'s' if len(lines) > 1 else ''} {marks})" if lines else "none"
        out.append(f"    {src} → {dst}: {where}")

    c = rel.signal("cocitation")
    e = c.evidence
    out += ["", style.bold(f"  co-citation  {c.value:.2f}") + "  (notes that link to both; each weighs 1 / ln(1 + the notes it links to))"]
    if e["both"]:
        rows = cut(e["both"])
        w = width(link_name(x["note"]) for x in rows)
        out += [f"    {link_name(x['note']):<{w}}  {x['weight']:.2f}  (links to {x['links_to']})" for x in rows] + more(e["both"], len(rows))
        total = sum(x["weight"] for x in e["both"])
        out.append(f"    = {total:.2f} / √(({e['total_a']:.2f} + 1) × ({e['total_b']:.2f} + 1)) = {c.value:.2f}"
                   f"   [{e['citing_a']} and {e['citing_b']} notes link to each in all, weighing {e['total_a']:.2f} and {e['total_b']:.2f}]")
    else:
        out.append(f"    no note links to both  (notes linking to {a}: {e['citing_a']}; to {b}: {e['citing_b']})")

    c = rel.signal("coupling")
    e = c.evidence
    out += ["", style.bold(f"  coupling  {c.value:.2f}") + "  (what both link to; each weighs 1 / ln(1 + the notes linking to it))"]
    if e["both"]:
        rows = cut(e["both"])
        names = [x["target"] if x["missing"] else link_name(x["target"]) for x in rows]
        names = [n + (" (no note yet)" if x["missing"] else "") for n, x in zip(names, rows)]
        w = width(names)
        out += [f"    {n:<{w}}  {x['weight']:.2f}  (linked from {x['linked_from']})" for n, x in zip(names, rows)] + more(e["both"], len(rows))
        total = sum(x["weight"] for x in e["both"])
        out.append(f"    = {total:.2f} / √(({e['total_a']:.2f} + 1) × ({e['total_b']:.2f} + 1)) = {c.value:.2f}"
                   f"   [{e['references_a']} and {e['references_b']} references in all, weighing {e['total_a']:.2f} and {e['total_b']:.2f}]")
    else:
        out.append(f"    nothing both link to  ({a} links to {_plural(e['references_a'], 'node')}, {b} to {e['references_b']})")

    t = rel.signal("tags")
    e = t.evidence
    out += ["", style.bold(f"  shared tags  {t.value:.2f}") + f"  (each tag weighs ln({e['notes']} / notes with it); weighted Jaccard;"
            f" tag links: {tag_rule})"]
    fmt = lambda tags: ", ".join(f"{tag_label(x['tag'])} {x['weight']:.2f}" for x in tags) or "none"
    out.append(f"    shared: {fmt(e['shared'])}")
    out.append(f"    only in {a}: {fmt(e['only_a'])}")
    out.append(f"    only in {b}: {fmt(e['only_b'])}")
    if e["union_weight"]:
        out.append(f"    = {e['shared_weight']:.2f} / {e['union_weight']:.2f} = {t.value:.2f}")
    if e["left_out"]:
        out.append("    left out (on more than a third of the notes): "
                   + ", ".join(f"{tag_label(x['tag'])} ({x['notes']} notes)" for x in e["left_out"]))

    out += [""]
    if not text.available:
        out += [style.bold("  text  unavailable") + f"  ({text.unavailable})"]
    else:
        e = text.evidence
        out.append(style.bold(f"  text  {text.value:.2f}") + f"  (centered cosine {e['cosine']:.3f} of the two embeddings,"
                   f" {e['model_id']}; the vault's mean vector subtracted; 0 if negative)")
        if "words" in e:
            shown = e["words"]
            out.append("    words in both notes, by their share of the cosine: "
                       + (" · ".join(f"{x['word']} {x['share']:+.3f}" for x in shown) or "none"))
            out.append(f"    = all {e['shared_words']} words in both {e['shared_total']:+.3f}, words in one note only"
                       f" {e['one_note_total']:+.3f}, the vault's average note {e['baseline']:+.3f}"
                       f"  (sum {e['shared_total'] + e['one_note_total'] + e['baseline']:.3f})")
            if not e["matches_vectors"]:
                out.append("    (a note changed since the last index: the words are read from it now, the cosine is the indexed one)")
    return "\n".join(out)


def _cmd_explain(args: argparse.Namespace) -> int:
    from vaultengine.operations import OperationError

    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    rule = _tag_rule(args)
    if rule is None:
        return 2
    with index:
        try:
            data, rel = _workbench(args, index).explain(args.a, args.b, args.words)
        except OperationError as exc:
            return _operation_error(exc)
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), "index": info, **data})
        return 0
    header = (f"  {rel.a} ↔ {rel.b}\n  index {info['last_indexed_at']}"
              + (" (OUT OF DATE)" if info["stale"] else " (up to date)"))
    print(_format_explain(rel, header, _Style(_use_color(args.color)), args.vault, args.limit, rule.description))
    return 0


# ----- map and deepen ----------------------------------------------------------


def _workbench(args: argparse.Namespace, index):
    """The explore commands' operations (operations.py) on this vault, index and settings."""
    from vaultengine.operations import Workbench

    return Workbench(index.conn, args.vault, args.settings, getattr(args, "format", None) or args.settings.engine.format)


def _operation_error(exc) -> int:
    print(f"error: {exc.message}" + (f"\n  {exc.hint}" if exc.hint else ""), file=sys.stderr)
    return 2


def _short(name: str, room: int) -> str:
    return name if len(name) <= room else name[: room - 1] + "…"


def _cmd_map(args: argparse.Namespace) -> int:
    from vaultengine.shape import percentile
    from vaultengine.vault import link_name

    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    rule = _tag_rule(args)
    if rule is None:
        return 2
    explore = args.settings.engine.explore
    if args.since:
        return _map_since(args, index, info, rule)
    with index:
        work = _workbench(args, index)
        pages_from = work.pages_from
        if args.meaning is None:
            args.meaning = explore.meaning
        data, shape, engine = work.map(args.limit, args.meaning, args.record)
        changes = data.get("meaning", {}).get("changes")
        recorded = data.get("recorded")
        uncached = len(engine.text.uncached) if engine is not None else 0
        summary = shape.summary()
        bridges = shape.bridges()
        hub, auth = shape.hits
        if args.json:
            _print_json({"vault": str(Path(args.vault).resolve()), "index": info, **data})
            return 0
    sg = shape.sg
    width = max(60, min(shutil.get_terminal_size((100, 24)).columns, 140))
    style = _Style(_use_color(args.color))
    p = explore.percentile
    out = [style.bold(f"Map of {Path(args.vault).resolve()}"), f"  index {info['last_indexed_at']}"
           + (" (OUT OF DATE)" if info["stale"] else " (up to date)"), ""]
    left = ", ".join(f"{t['tag']} on {t['notes']} of {summary['notes']} notes" for t in summary["left_out_tags"])
    out.append(f"  graph        {summary['notes']} notes, {summary['missing_notes']} missing notes, {summary['tags']} #tags"
               + (f" (left out: {left})" if left else "") + f" · {summary['edges']} edges · tag links: {summary['tag_rule']}")
    out.append(f"  communities  Louvain, seed {summary['seed']}, resolution {summary['resolution']:g} (the same every run)"
               f" · modularity {summary['modularity']:.2f} · {summary['communities']} communities hold"
               f" {summary['notes_in_communities']} notes; {_plural(summary['alone'], 'note')} in none")
    out.append(f"  cutoffs      at or above the {p:g}th percentile of the vault (engine.explore.percentile)")
    out.append(f"  topic pages  {pages_from}")
    if args.meaning:
        if shape.meaning_unavailable:
            out.append(f"  meaning      unavailable: {shape.meaning_unavailable} (the communities are from links and tags only)")
        else:
            out.append(f"  meaning      {_plural(len(shape.meaning_edges), 'edge')} added for the communities: mutual nearest"
                       f" neighbours by embedding at or above the {explore.meaning_percentile:g}th percentile of every note's"
                       f" nearest-neighbour similarity ({shape.meaning_cutoff:.3f})")
            out.extend("  " + line for line in _text_notes(None, uncached, args.vault))
    communities = shape.communities
    shown = communities if not args.limit else communities[: args.limit]
    out += ["", style.bold("  Communities") + "  (largest first; key phrases: its notes with the phrase / all notes with it)"]
    for c in shown:
        head = f"  {c.number:>3}. {c.name}"
        out += ["", style.bold(_short(head, width - 12)) + f"  {len(c.notes)} notes"]
        phrases = " · ".join(f"{ph.phrase} {ph.in_community}/{ph.in_vault}" for ph in c.phrases[:5])
        out.append(f"       key phrases  {_short(phrases, width - 20)}")
        out.append(f"       hubs         {_short(', '.join(link_name(sg.labels[v]) for v in c.hubs), width - 20)}")
        if shape.topic_page is not None:
            pages = ", ".join(f"{link_name(sg.labels[v])} (links {n})" for v, n in c.topic_pages[:2])
            more = f" +{len(c.topic_pages) - 2} more" if len(c.topic_pages) > 2 else ""
            out.append(f"       topic pages  {_short(pages + more, width - 20) if pages else 'none'}")
    if len(shown) < len(communities):
        out.append(f"  … {len(communities) - len(shown)} more communities (--limit 0 shows all)")
    if shape.topic_page is not None:
        bare = [c for c in communities if not c.topic_pages]
        out += ["", style.bold("  No topic page") + f"  ({len(bare)} of {len(communities)} communities; a topic page belongs to"
                " the community most of its direct links go to)"]
        out += [f"    {c.number:>3}. {_short(c.name, width - 24)}  {len(c.notes)} notes" for c in bare]
    if changes is not None:
        out += ["", style.bold("  What the meaning edges change") + f"  (communities {changes['communities_before']} →"
                f" {changes['communities_after']}; notes in none {len(changes['stranded_before'])} →"
                f" {len(changes['stranded_after'])})"]
        for item in changes["placed"]:
            out.append(f"    placed  {_short(link_name(item['note']), width - 40)}  → community {item['community']}")
        for number, sources in sorted(changes["joined"].items()):
            out.append(f"    joined  community {number} takes notes from communities {', '.join(map(str, sources))} of the"
                       " map without meaning edges")
        for source, numbers in sorted(changes["split"].items()):
            out.append(f"    split   community {source} of the map without them is now {', '.join(map(str, numbers))}")
        moved = changes["moved"]
        out.append(f"    moved   {_plural(len(moved), 'note')} changed community"
                   + (": " + ", ".join(f"{_short(link_name(m['note']), 40)} ({m['from']} → {m['to']})" for m in moved[:6])
                      + (f" +{len(moved) - 6}" if len(moved) > 6 else "") if moved else ""))
    b = shape.betweenness
    values = [r["betweenness"] for r in bridges]
    cut = percentile(values, p)
    kept = [r for r in bridges if r["betweenness"] >= cut and r["betweenness"] > 0]
    how = (f"exact, from all {b.sources} notes" if b.exact else
           f"estimated from a seeded sample of {b.sources} of {summary['notes']} notes (exact would cost {b.visits:,} edge"
           f" visits; engine.explore.path_budget is {b.budget:,})")
    out += ["", style.bold("  Bridge notes") + f"  (betweenness between notes, {how}; {len(kept)} at or above the"
            f" {p:g}th percentile, {cut:.3f}; constraint: low = its contacts are not tied to each other)"]
    out.append(f"         {'betweenness':<13} {'constraint':<13} note  → communities its links reach")
    for rank, r in enumerate(kept[: args.limit or None], 1):
        con = f"{r['constraint']:.3f} (p{r['constraint_percentile']:.0f})" if r["constraint"] is not None else "—"
        bet = f"{r['betweenness']:.3f} (p{r['betweenness_percentile']:.0f})"
        spans = ", ".join(map(str, r["spans"][:5]))
        out.append(f"    {rank:>3}. {bet:<13} {con:<13} {_short(link_name(r['path']), width - 50)}  → {spans}")
    for title, scores, missing in (("Hubs", hub, False), ("Authorities", auth, True)):
        ranked = sorted(scores.items(), key=lambda item: (-item[1], shape.graph.labels[item[0]]))
        hcut = percentile([v for _, v in ranked], p)
        top = [(v, x) for v, x in ranked if x >= hcut and x > 0]
        what = "notes whose body links point at the best authorities" if title == "Hubs" else "what the best hubs link to"
        out += ["", style.bold(f"  {title}") + f"  ({what}; HITS, largest 1.00; {len(top)} at or above the {p:g}th percentile)"]
        for rank, (v, x) in enumerate(top[: args.limit or None], 1):
            name = shape.graph.labels[v] if shape.graph.is_missing(v) else link_name(shape.graph.labels[v])
            out.append(f"    {rank:>3}. {x:.2f}  {_short(name, width - 20)}" + ("  (no note yet)" if shape.graph.is_missing(v) else ""))
    if recorded:
        out += ["", f"  recorded     snapshot {recorded['at']} ({recorded['bytes']:,} bytes) in {recorded['memory']}:"
                f" {recorded['communities']} communities, {recorded['gaps']} gaps ({recorded['basis']}),"
                f" {recorded['missing']} missing notes · `vault map --since {recorded['at'][:10]}` compares"]
    print("\n".join(out))
    return 0


def _map_since(args: argparse.Namespace, index, info, rule) -> int:
    from vaultengine.operations import OperationError
    from vaultengine.vault import link_name

    with index:
        try:
            diff = _workbench(args, index).map_since(args.since)
        except OperationError as exc:
            return _operation_error(exc)
    at, how = diff["snapshot"], diff["chosen"]
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), "index": info, **diff})
        return 0
    style = _Style(_use_color(args.color))
    width = max(60, min(shutil.get_terminal_size((100, 24)).columns, 140))
    n0, n1 = diff["notes"]
    l0, l1 = diff["links"]
    c0, c1 = diff["communities"]
    out = [style.bold(f"Map of {Path(args.vault).resolve()} since {at}"),
           f"  {how} · index {info['last_indexed_at']}" + (" (OUT OF DATE)" if info["stale"] else " (up to date)"),
           f"  notes {n0} → {n1} · links between notes {l0} → {l1} · communities {c0} → {c1}"]
    closed = [g for g in diff["gaps"] if g["status"] == "closed"]
    out += ["", style.bold("  Gaps") + f"  ({len(closed)} of {len(diff['gaps'])} recorded gaps closed; by {diff['gap_basis']})"]
    for g in diff["gaps"]:
        x, y = g["communities"]
        out.append(f"    {g['status']:<7} communities {x} and {y} then: {_short(g['names'][0], 40)} / {_short(g['names'][1], 40)}")
        out.append(f"            {g['why']} (then {g['links_then']:g}, where chance predicted {g['expected_then']:.1f})")
        for a, b in g["added"][:3]:
            out.append(f"            new link {_short(link_name(a), 45)} ↔ {_short(link_name(b), 45)}")
    if not diff["gaps"]:
        out.append("    none were recorded")
    out += ["", style.bold("  Communities") + f"  (formed {len(diff['formed'])}, gone {len(diff['gone'])}: matched by shared"
            " notes, Jaccard below 0.5 = no match)"]
    for c in diff["formed"]:
        out.append(f"    formed  {c['community']}. {_short(c['name'], width - 40)}  {c['notes']} notes"
                   f" (best match then: {c['best_match']}, {c['overlap']:.2f})")
    for c in diff["gone"]:
        out.append(f"    gone    {c['community']}. {_short(c['name'], width - 40)}  {c['notes']} notes then"
                   f" (best match now {c['overlap']:.2f})")
    if not diff["formed"] and not diff["gone"]:
        out.append("    every community then has one now, and every one now had one then")
    out += ["", style.bold("  Missing notes written") + f"  ({len(diff['written'])})"]
    for w in diff["written"]:
        out.append(f"    [[{w['name']}]] ({w['linked_from']} notes linked it) is now {w['note']}")
    if not diff["written"]:
        out.append("    none")
    if diff["topic_pages"]:
        n = len(diff["topic_pages"])
        out += ["", style.bold("  Topic pages") + f"  ({_plural(n, 'community')} without one then {'has' if n == 1 else 'have'}"
                " one now)".replace("communitys", "communities")]
        for t in diff["topic_pages"]:
            out.append(f"    community {t['community']} ({_short(t['name'], 40)}): {', '.join(link_name(p) for p in t['topic_pages'])}")
    print("\n".join(out))
    return 0


from vaultengine.operations import DEEPEN_LENSES as LENSES  # noqa: E402


def _cmd_deepen(args: argparse.Namespace) -> int:
    from vaultengine.operations import OperationError
    from vaultengine.vault import link_name

    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    rule = _tag_rule(args)
    if rule is None:
        return 2
    explore = args.settings.engine.explore
    with index:
        try:
            data = _workbench(args, index).deepen(args.note, args.lens, args.limit)
        except OperationError as exc:
            return _operation_error(exc)
    path, results = data["note"], data["lenses"]
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), "index": info, "note": path,
                     "percentile": explore.percentile, "lenses": results})
        return 0
    style = _Style(_use_color(args.color))
    width = max(60, min(shutil.get_terminal_size((100, 24)).columns, 140))
    p = f"{explore.percentile:g}th percentile"
    out = [style.bold(f"Deepen {link_name(path)}"), f"  {path} · index {info['last_indexed_at']}"
           + (" (OUT OF DATE)" if info["stale"] else ""),
           f"  cutoffs: at or above the {p} of the vault (engine.explore.percentile)"]
    if "neighbourhood" in results:
        r = results["neighbourhood"]
        out += ["", style.bold("  Neighbourhood") + f"  (vault related: {r['kept']} of {r['compared']} notes at or above the {p}"
                f" of this note's scores, {r['cutoff']:.2f})"]
        out += ["  " + line for line in _text_notes(r["text_unavailable"], r["uncached"], args.vault)]
        for rank, h in enumerate(r["hits"], 1):
            status = "linked" if h["linked"] else "not linked"
            out.append(f"    {rank:>3}. {_short(link_name(h['path']), width - 30)}  {h['score']:.2f}  {status}")
            out.append(f"         {h['reason'] or '(only distant links)'}")
        if r["kept"] > len(r["hits"]):
            out.append(f"         … {r['kept'] - len(r['hits'])} more (--limit 0 shows all)")
    for key, title, what in (("cited", "Cited together", "notes that link to both it and them"),
                             ("foundations", "Same foundations", "what it and they both link to")):
        if key not in results:
            continue
        r = results[key]
        scope = "every pair" if r["exact"] else "a seeded sample"
        out += ["", style.bold(f"  {title}") + f"  ({what}; {r['found']} found, {r['kept']} at or above the {p} of the vault's"
                f" {r['pairs_in_vault']} values ({scope}), {r['cutoff']:.2f})"]
        rows = r["notes"] or r["below_cutoff"]
        if not r["notes"]:
            out.append("    none stands out in this vault" + ("; the strongest:" if rows else ""))
        for rank, row in enumerate(rows, 1):
            ev = row["evidence"]
            names = ", ".join(_short(e["name"], 40) for e in ev[:3]) + (f" +{len(ev) - 3}" if len(ev) > 3 else "")
            out.append(f"    {rank:>3}. {_short(link_name(row['path']), width - 30)}  {row['value']:.2f} (p{row['percentile']:.0f})")
            out.append(f"         {'by' if key == 'cited' else 'both link'} {names}")
    if "backbone" in results:
        r = results["backbone"]
        out.append("")
        if r["community"] is None:
            out.append(style.bold("  Backbone") + "  (this note is in no community of two or more notes)")
        else:
            out.append(style.bold("  Backbone") + f"  (community {r['community']}: the longest path through the strongest ties"
                       f" of its {r['community_notes']} notes, {r['length']} notes long)")
            for i, note in enumerate(r["chain"]):
                mark = "  ◀ this note" if note == path else ""
                out.append(f"    {link_name(note)}{mark}")
                if i < len(r["steps"]):
                    out.append(f"      │ {r['steps'][i]}")
            if r["note_on_chain"] is None and r["joins_at"]:
                out.append(f"    this note is {_plural(r['steps_off'], 'step')} off it, joining at {link_name(r['joins_at'])}")
    if "cluster" in results:
        r = results["cluster"]
        out.append("")
        if r["community"] is None:
            out.append(style.bold("  Cluster") + "  (this note is in no community of two or more notes)")
        else:
            out.append(style.bold("  Cluster") + f"  (community {r['community']}: {r['name']}, {r['notes']} notes)")
            out.append("    key phrases  " + _short(" · ".join(f"{x['phrase']} {x['in_community']}/{x['in_vault']}"
                                                            for x in r["phrases"][:5]), width - 18))
            out.append("    hubs         " + ", ".join(link_name(h) for h in r["hubs"]))
            if data["topic_pages_known"]:
                pages = ", ".join(f"{link_name(t['path'])} (links {t['reaches']})" for t in r["topic_pages"][:2])
                out.append("    topic page   " + (pages or "none"))
            out.append(f"    this note    weight {r['weight_inside']:g} inside (p{r['inside_percentile']:.0f} of the members);"
                       f" {r['outside_share']:.0%} of its weight goes to other communities (participation {r['participation']:.2f})")
            for n in r["neighbours"]:
                out.append(f"    next to      {n['community']}. {_short(n['name'], 50)}  (weight {n['weight']:g}, through "
                           + ", ".join(_short(link_name(t), 40) for t in n["through"][:2]) + ")")
    print("\n".join(out))
    return 0


from vaultengine.operations import DISCOVER_LENSES as DISCOVER  # noqa: E402


def _cmd_discover(args: argparse.Namespace) -> int:
    from vaultengine.operations import OperationError
    from vaultengine.vault import link_name

    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    rule = _tag_rule(args)
    if rule is None:
        return 2
    explore = args.settings.engine.explore
    with index:
        try:
            data, engine = _workbench(args, index).discover(args.note, args.lens, args.limit, args.steps, args.seed)
        except OperationError as exc:
            return _operation_error(exc)
        uncached = len(engine.text.uncached)
    results, seed = data["lenses"], data["seed"]
    node_label = data["note"]
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), "index": info, "note": args.note and node_label,
                     "percentile": explore.percentile, "seed": seed, "lenses": results})
        return 0
    style = _Style(_use_color(args.color))
    width = max(60, min(shutil.get_terminal_size((100, 24)).columns, 140))
    p = f"{explore.percentile:g}th percentile"
    title = f"Discover around {link_name(node_label)}" if node_label is not None else f"Discover in {Path(args.vault).resolve()}"
    out = [style.bold(title), f"  index {info['last_indexed_at']}" + (" (OUT OF DATE)" if info["stale"] else " (up to date)"),
           f"  cutoffs: the {p} of the vault (engine.explore.percentile) · seed {seed} · tag links: {engine.graph.tag_rule}"]
    out.extend(_text_notes(engine.text.unavailable, uncached, args.vault))

    def items(rows, note_key=None):
        for rank, row in enumerate(rows, 1):
            head = f"    {rank:>3}. "
            if note_key:
                out.append(head + _short(link_name(row[note_key]), width - 10))
                out.append(" " * len(head) + row["reason"])
            else:
                out.append(head + row["reason"])

    if "hidden" in results:
        r = results["hidden"]
        scope = "" if r["exact"] else f" from a seeded sample of {r['sources']} notes (engine.explore.path_budget)"
        out += ["", style.bold("  Hidden link") + f"  (notes it has no link with, reached through two or more middles: notes, missing"
                f" notes, tags or key terms both have; {r['kept']} of {r['found']} at or above the {p} of {r['pairs_in_vault']}"
                f" unlinked pairs{scope}, {r['cutoff']:.2f})"]
        items(r["results"], "note") if r["results"] else out.append("    none stands out")
    if "role" in results:
        r = results["role"]
        out += ["", style.bold("  Same role elsewhere") + f"  (notes in other communities whose ten link features sit at the same"
                f" percentiles; {r['kept']} at or below the {100 - explore.percentile:g}th percentile of {r['pairs_in_vault']}"
                f" pairs' role distances, {r['cutoff']:.3f})"]
        items(r["results"], "note") if r["results"] else out.append("    none stands out")
    if "bridge" in results:
        r = results["bridge"]
        reach = ", ".join(f"{x['community']} ({x['weight']:g})" for x in r["reaches"])
        out += ["", style.bold("  Bridge") + f"  (betweenness {r['betweenness']:.3f}, p{r['betweenness_percentile']:.0f};"
                f" constraint {r['constraint']:.3f}, p{r['constraint_percentile']:.0f}; in community {r['community']}; its links and"
                f" tags reach {reach}; {'exact' if r['exact'] else f'sampled from {r['sources']} notes'})"]
        items(r["brokers"]) if r["brokers"] else out.append("    no shortest path between notes runs through it")
    for lens, title, what in (("gap", "Gap", "communities close in meaning with fewer links between them than chance predicts"),
                              ("surprise", "Surprise", "links between notes far apart in meaning, in different communities"),
                              ("unwritten", "Unwritten", "missing notes linked from the most communities; the same idea merged first")):
        if lens not in results:
            continue
        out += ["", style.bold(f"  {title}") + f"  ({what})"]
        for variant, label in (("with_embeddings", "with embeddings"), ("without_embeddings", "without them")):
            r = results[lens][variant]
            if "unavailable" in r:
                out.append(f"    {label}: unavailable ({r['unavailable']})")
                continue
            if lens == "gap":
                basis = "mean embeddings" if variant == "with_embeddings" else "key-term profiles"
                out.append(f"    {label} (closeness of {basis} at or above the {p} of {r['pairs']} community pairs,"
                           f" {r['cutoff']:.2f}; {r['found']} found)")
                for rank, row in enumerate(r["results"], 1):
                    x, y = row["communities"]
                    out.append(f"    {rank:>3}. communities {x} and {y}: {_short(row['names'][0], 40)} / {_short(row['names'][1], 40)}")
                    out.append(f"         {row['reason']}")
                    for j in row["join"]:
                        out.append(f"         join {_short(link_name(j['a']), 40)} ↔ {_short(link_name(j['b']), 40)}"
                                   f" ({j['score']:.2f}, via {', '.join(_short(m, 30) for m in j['middles'][:3])})")
            elif lens == "surprise":
                basis = "embedding" if variant == "with_embeddings" else "key-term"
                out.append(f"    {label} ({basis} similarity at or below the {100 - explore.percentile:g}th percentile of"
                           f" {r['linked_pairs']} linked pairs, {r['cutoff']:.2f}; {r['found']} found)")
                for rank, row in enumerate(r["results"], 1):
                    out.append(f"    {rank:>3}. {_short(link_name(row['a']), 45)} → {_short(link_name(row['b']), 45)}")
                    out.append(f"         {row['reason']}")
            else:
                merge = (f"merged by embedding of the names at or above the {p}, {r['merge_cutoff']:.2f}"
                         if r["by"] != "name words" else "merged by name words")
                out.append(f"    {label} ({merge}: {r['merged']} groups; linked from at least {r['cutoff']:g} communities,"
                           f" the {p} of {r['groups']}; {r['found']} found)")
                for rank, row in enumerate(r["results"], 1):
                    out.append(f"    {rank:>3}. [[{row['names'][0]}]]" + (f" +{len(row['names']) - 1}" if len(row["names"]) > 1 else ""))
                    out.append(f"         {row['reason']}")
            if not r["results"]:
                out.append("         none")
    if "wander" in results:
        r = results["wander"]
        out += ["", style.bold("  Wander") + f"  (a random walk on links, seed {r['seed']}: the same seed, the same walk;"
                f" {r['steps']} steps through {_plural(r['communities'], 'community', )})".replace("communitys", "communities")]
        out.append(f"    {link_name(r['walk'][0])}")
        for hop in r["hops"]:
            out.append(f"      → {_short(link_name(hop['to']), width - 10)}")
            out.append(f"        {hop['reason']}")
    print("\n".join(out))
    return 0


# ----- explore, marks, link it, unlink -------------------------------------------


def _memory(args: argparse.Namespace, must_exist: bool = False):
    """The explore memory (engine.memory); None after printing an error when it must exist and does not."""
    from vaultengine.memory import Memory

    path = args.settings.engine.memory_path
    if must_exist and not Path(path).is_file():
        print(f"error: nothing remembered yet ({path} does not exist): run `vault explore` first", file=sys.stderr)
        return None
    return Memory(path)


def _explorer(args: argparse.Namespace, index, rule, memory, per_lens: int | None = None, refresh: bool = False,
              keep_stats: bool = True):
    """The explorer for this vault and settings (``operations.Workbench.explorer``); ``rule`` is read from args."""
    return _workbench(args, index).explorer(memory, per_lens, refresh, keep_stats)


def _format_explore(out: dict, explorer, args: argparse.Namespace, info: dict, style: "_Style", width: int,
                    memory_path: Path, lenses: list[str]) -> str:
    from vaultengine.explore import space

    seed, sections = out["seed"], out["sections"]
    settings = explorer.settings
    if seed is None:
        title = f"Explore {Path(args.vault).resolve()}: where to start"
    elif seed.path is None:
        title = f"Explore {out['start']} (no note yet; {_plural(len(seed.linkers), 'note')} link to it)"
    else:
        title = f"Explore {_short(explorer.name(seed.path), width - 10)}"
    lines = [style.bold(title), f"  {seed.path if seed is not None and seed.path else 'the whole vault'} · index "
             f"{info['last_indexed_at']}" + (" (OUT OF DATE)" if info["stale"] else " (up to date)")]
    used = space(sections)
    if seed is None:
        lines.append(f"  proposals from {len(sections)} vault-wide lenses (engine.explore.start_lenses), up to"
                     f" {settings.per_lens} each; `vault explore <the start named>` goes there")
    else:
        budget = {s.lens.kind: s.budget for s in sections if s.applies}
        lines.append(f"  {len(sections)} lenses (engine.explore.lenses) · equal space: deepen {used['deepen']} results,"
                     f" discover {used['discover']} (up to {budget.get('deepen', settings.per_lens)} and"
                     f" {budget.get('discover', settings.per_lens)} a section) · related: engine.related.unlinked_weights")
    cut = out["duplicate_cutoff"]
    basis = "embedding cosine" if out["meaning"] else "key-term cosine"
    lines.append(f"  near-duplicates left out of a section: {basis} ≥ {cut:.2f} (the {settings.duplicate_percentile:g}th"
                 " percentile of note pairs) or the same note")
    lines.append("  sections ordered by the share of results you keep: (kept + 1) / (marked + 2), ties in config order"
                 + (" · --fresh: what you saw or dismissed before is left out" if out["fresh"] else ""))
    lines.append(f"  memory {memory_path} · `vault mark keep|dismiss 1.2` · `vault link-it 1.2` · `vault kept`")
    st = out["stats"]
    if st["mode"] == "reused":
        lines.append(f"  vault-wide numbers (distributions, key terms) from {st['computed_at']}: {st['changed_since']} notes changed"
                     f" since; recomputed when more than {st['threshold']} have, or with --refresh")
    lines.extend(_text_notes(explorer.engine.text.unavailable, len(explorer.engine.text.uncached), args.vault))
    for number, s in enumerate(sections, 1):
        marked = s.kept + s.dismissed
        rate = (f"kept {s.kept} of {marked} you marked ({s.kept / marked:.0%})" if marked else "no marks yet")
        counts = f"{len(s.results)} of {s.candidates}"
        if s.hidden:
            counts += f", {s.hidden} seen or dismissed before"
        if s.duplicates:
            counts += f", {s.duplicates} near-duplicates"
        lines += ["", style.bold(f"  {number}. {s.lens.title}") + f"  ({s.lens.kind} · {s.lens.what}; {counts} · {rate})"]
        if not s.results:
            lines.append(f"       {s.note or 'nothing found'}")
        for r in s.results:
            flag = f"  [{'kept' if r.mark == 'keep' else 'dismissed'}]" if r.mark else ""
            go = f"  → explore {r.start_at}" if seed is None and r.start_at else ""
            lines.append(f"    {r.ref:>5}  {_short(r.title, width - 12 - len(flag) - len(go))}{flag}" + style.dim(go))
            lines.append(f"           {s.lens.name} · {r.evidence}")
    return "\n".join(lines)


def _cmd_explore(args: argparse.Namespace) -> int:
    from contextlib import nullcontext

    from vaultengine.operations import OperationError, explore_json

    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    rule = _tag_rule(args)
    if rule is None:
        return 2
    memory_path = args.settings.engine.memory_path
    memory = _memory(args) if not args.no_record or Path(memory_path).is_file() else None  # --no-record creates nothing
    with index, (memory or nullcontext()):
        work = _workbench(args, index)
        try:
            out, explorer = work.explore(args.note, args.lens, memory, args.fresh, args.per_lens, args.refresh,
                                         record=not args.no_record)
        except OperationError as exc:
            return _operation_error(exc)
        if args.json:
            _print_json({"vault": str(Path(args.vault).resolve()), "index": info,
                         **explore_json(out, memory_path, args.fresh, args.timing, explorer.engine.text)})
            return 0
        width = max(60, min(shutil.get_terminal_size((100, 24)).columns, 140))
        print(_format_explore(out, explorer, args, info, _Style(_use_color(args.color)), width, memory_path, out["lenses"]))
    return 0


def _cmd_mark(args: argparse.Namespace) -> int:
    from vaultengine.operations import OperationError, mark

    verdict, refs = args.words[0], args.words[1:]
    if verdict not in ("keep", "dismiss", "clear") or not refs:
        print("error: vault mark keep|dismiss|clear REF... (REF: a result of the last `vault explore`, e.g. 2.3)",
              file=sys.stderr)
        return 2
    memory = _memory(args, must_exist=True)
    if memory is None:
        return 2
    with memory:
        try:
            data = mark(memory, verdict, refs)
        except OperationError as exc:
            return _operation_error(exc)
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), **data})
        return 0
    done = {"keep": "kept", "dismiss": "dismissed", "clear": "cleared"}[verdict]
    for r in data["marked"]:
        print(f"{r['ref']:>5}  {done if r['changed'] else 'not marked'}  {r['lens']} · {r['title']}")
    for rate in data["keep_rates"]:
        print(f"  {rate['lens']}: kept {rate['kept']} of {rate['kept'] + rate['dismissed']} marked")
    return 0


def _cmd_kept(args: argparse.Namespace) -> int:
    from vaultengine.operations import kept

    data = kept(args.settings.engine.memory_path, args.lens)
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), **data})
        return 0
    rows = data["kept"]
    width = max(60, min(shutil.get_terminal_size((100, 24)).columns, 140))
    out = [f"Kept in {Path(args.vault).resolve()}", f"  memory {data['memory']} · {_plural(len(rows), 'result')}"]
    for r in rows:
        start = r["start"] or "no starting point"
        out += ["", f"  {_short(r['title'], width - 4)}", f"    {r['lens']} · {r['evidence']}",
                f"    from {start} · kept {r['at']}"]
    if not rows:
        out.append("  nothing kept yet (`vault mark keep 2.3` after `vault explore`)")
    print("\n".join(out))
    return 0


def _cmd_link_it(args: argparse.Namespace) -> int:
    from vaultengine.operations import LinkIt, OperationError

    if len(args.words) not in (1, 2):
        print("error: vault link-it REF (a result of the last `vault explore`), or vault link-it NOTE NOTE", file=sys.stderr)
        return 2
    ref, pair = (args.words[0], (None, None)) if len(args.words) == 1 else (None, tuple(args.words))
    try:
        write = LinkIt(args.settings.engine.memory_path, args.settings.engine.explore.link_section, ref, *pair, args.reason)
    except OperationError as exc:
        return _operation_error(exc)
    args.dry_run = not args.write
    outcome_of = {}

    def after(outcome):
        outcome_of["outcome"] = outcome
        write.after(outcome)

    def extra():
        outcome = outcome_of["outcome"]
        lines = []
        if args.dry_run:
            lines.append("`--write` writes it, through the same gate; `vault unlink` undoes it.")
        elif write.result is not None and outcome.status == "written":
            lines.append(f"Kept result {write.result['ref']} ({write.result['lens']}): a link made counts as a keep.")
        return lines, write.extra(outcome)

    return _run_write_command(args, write.build, extra, after)


def _cmd_unlink(args: argparse.Namespace) -> int:
    from vaultengine.operations import Unlink

    if len(args.words) != 2:
        print("error: vault unlink NOTE NOTE", file=sys.stderr)
        return 2
    section = args.settings.engine.explore.link_section
    write = Unlink(args.settings.engine.memory_path, section, *args.words)
    args.dry_run = not args.write
    outcome_of = {}

    def after(outcome):
        outcome_of["outcome"] = outcome
        write.after(outcome)

    def extra():
        lines = []
        if args.dry_run:
            lines.append("`--write` removes it.")
        if not write.record:
            lines.append(f"(no link-it record for these two: only list lines linking the other note under {section} are"
                         " removed, and the section stays)")
        return lines, write.extra(outcome_of["outcome"])

    return _run_write_command(args, write.build, extra, after)


# ----- experiment --------------------------------------------------------------


def _cmd_experiment(args: argparse.Namespace) -> int:
    import datetime
    import shlex
    import time as clock

    from vaultengine import experiment
    from vaultengine.embedding import EmbeddingError
    from vaultengine.formats import BUNDLED_FORMATS_DIR, load_format, resolve_format_path

    if args.seeds < 2:
        print("error: --seeds must be 2 or more: with one seed there is no seed-to-seed noise to compare with", file=sys.stderr)
        return 2
    if not 0 < args.hide <= 0.5:
        print("error: --hide must be above 0 and at most 0.5 (0.1 to 0.2 is meant)", file=sys.stderr)
        return 2
    if args.queries < 1:
        print("error: --queries must be 1 or more", file=sys.stderr)
        return 2
    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    rule = _tag_rule(args)
    if rule is None:
        return 2
    name = getattr(args, "format", None) or args.settings.engine.format
    spec = load_format(resolve_format_path(name, [Path("formats"), BUNDLED_FORMATS_DIR])) if name else None
    try:
        provider = args.settings.provider()
    except EmbeddingError as exc:
        print(f"note: no embedding model ({exc}): the options that need text are left out", file=sys.stderr)
        provider = None
    engine = args.settings.engine
    seeds = [engine.explore.seed + i for i in range(args.seeds)]
    started = clock.perf_counter()
    with index:
        out = experiment.run(
            index.conn, args.vault, tag_rule=rule, spec=spec, provider=provider,
            related_weights=engine.related.unlinked_weights.model_dump(), text_variant="whole",
            resolution=engine.explore.resolution, without=args.without or [], seeds=seeds, hide=args.hide,
            meaning=(engine.explore.meaning_neighbours, engine.explore.meaning_percentile) if engine.explore.meaning else None,
            queries=args.queries, questions=tuple(args.question or ("relatedness", "text", "communities")),
            progress=lambda line: print(f"  {line}", file=sys.stderr) if not args.quiet else None,
        )
    seconds = clock.perf_counter() - started
    date = datetime.date.today().isoformat()
    command = "vault " + " ".join(shlex.quote(a) for a in (args.argv or ["experiment"]))
    report = experiment.markdown(out, str(Path(args.vault).resolve()), date, seconds, command)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), "seconds": seconds, "decisions": out["decisions"]})
    else:
        print(report, end="")
    return 0


# ----- suggest -----------------------------------------------------------------


def _format_links(pairs, style: _Style, width: int, explain: bool, weights: dict | None = None,
                  text_notes: list[str] | None = None) -> list[str]:
    from vaultengine.connect import WEIGHTS

    weights = weights or WEIGHTS
    names = {"shared": "shared neighbours", "cited": "cited together", "tags": "shared tags", "text": "similar text"}
    formula = " + ".join(f"{weights[k]:.2g} × {names[k]}" for k in ("shared", "cited", "tags", "text") if weights[k])
    out = [style.bold("  Links to add") + f"  (notes with no link between them; score = {formula})"]
    out.extend(text_notes or [])
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
                f"shared {pair.shared:.2f} · cited {pair.cited:.2f} · tags {pair.tags:.2f} · text {pair.text:.2f} · {pair.a} ↔ {pair.b}"))
    return out


def _format_suggestions(result, header: str, style: _Style, links=None, width: int = 100, explain: bool = False,
                        weights: dict | None = None, text_notes: list[str] | None = None) -> str:
    out = [header, ""]
    if links is not None:
        out.extend(_format_links(links, style, width, explain, weights, text_notes))
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
    from vaultengine.related import Relatedness
    from vaultengine.suggest import Suggester

    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    rule = _tag_rule(args)
    if rule is None:
        return 2
    with index:
        suggester = Suggester(index.conn, Relatedness(index.conn, lambda: args.settings.provider(), args.vault, rule, _related_weights(args)))
        text = suggester.engine.text
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
                link_suggester = LinkSuggester(index.conn, suggester.engine)
                links = link_suggester.suggestions(limit=args.limit or None, for_note=for_note, path=args.path)
            result = None if args.links or args.for_note else suggester.suggestions(limit=args.limit or None, min_links=args.min_links)
            if args.json:
                data = {"vault": str(Path(args.vault).resolve()), "index": info}
                if links is not None:
                    data["links"] = [pair.to_dict() for pair in links]
                    data["text"] = _text_status(text)
                    data["weights"] = link_suggester.weights
                if result is not None:
                    data.update(result.to_dict())
                _print_json(data)
            else:
                width = max(60, min(shutil.get_terminal_size((100, 24)).columns, 140))
                header = _header("Suggestions", args.vault, info)
                extra = {}
                if links is not None:
                    extra = {"weights": link_suggester.weights,
                             "text_notes": _text_notes(text.unavailable, len(text.uncached), args.vault)}
                print(_format_suggestions(result, header, _Style(_use_color(args.color)), links, width, args.explain, **extra))
            return 0
        try:
            path, draft = suggester.draft(args.draft)
        except KeyError:
            print(f'error: no link points to a missing note called "{args.draft}"', file=sys.stderr)
            print("  run `vault suggest` to list them", file=sys.stderr)
            return 2
        for line in _text_notes(text.unavailable, len(text.uncached), args.vault):  # the draft's "See Also" used it
            print("note:" + line[1:], file=sys.stderr)
    if args.folder is not None:
        path = f"{args.folder.strip('/')}/{path.rsplit('/', 1)[-1]}" if args.folder.strip("/") else path.rsplit("/", 1)[-1]
    if not args.write:
        print(f"# would be written to {path} (add --write to create it)\n")
        print(draft, end="")
        return 0
    target = Path(args.vault) / path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "x", encoding="utf-8") as handle:
            handle.write(draft)
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


# ----- embed -------------------------------------------------------------------


def _provider(args: argparse.Namespace):
    """The configured embedding provider, or None after printing what to fix."""
    from vaultengine.embedding import EmbeddingError

    try:
        return args.settings.provider()
    except EmbeddingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("  the embedding model is set in config/config.yaml (engine.embedding); see `vault config show`", file=sys.stderr)
        return None


def _size(n: int) -> str:
    return f"{n} B" if n < 1024 else f"{n / 1024:.1f} KiB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MiB"


def _cmd_embed_status(args: argparse.Namespace) -> int:
    from vaultengine.embedding.store import status

    opened = _open_for_reading(args)
    if opened is None:
        return 2
    index, info = opened
    engine = args.settings.engine
    model_path = engine.embedding.model_path
    with index:
        data = status(index.conn, engine.model_id)
    data["provider"] = engine.embedding.provider
    data["model_path"] = str(model_path) if model_path else None
    data["model_found"] = bool(model_path and Path(model_path).is_dir())
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), "index": info, **data})
        return 0
    found = "found" if data["model_found"] else "NOT FOUND"
    out = [_header("Embeddings", args.vault, info), ""]
    out.append(f"  model      {data['model_id']} · {data['provider']} · {data['model_path']} ({found})")
    dims = f" ({data['dim']} dims)" if data["dim"] else ""
    pending = info["pending"]
    waiting = len(pending["added"]) + len(pending["changed"])
    if data["missing"]:
        state = f"{data['missing']} missing: run `vault embed {args.vault}`"
    elif info["stale"]:
        state = "every indexed note has one, but the index is out of date"
    else:
        state = "all up to date"
    out.append(f"  vectors    {data['with_vector']} of {data['notes']} indexed notes{dims} · {state}")
    if info["stale"]:
        detail = pending["reason"] or f"{len(pending['added'])} added, {len(pending['changed'])} changed, {len(pending['deleted'])} deleted"
        out.append(f"  pending    {detail} since the last index: `vault embed {args.vault}` indexes"
                   + (f" and embeds {'it' if waiting == 1 else 'them'}" if waiting else " first"))
    shown, more = _shown(data["missing_paths"] if data["with_vector"] else [], args.limit)  # all missing: no list
    out.extend(f"    - {path}" for path in shown)
    if more:
        out.append(more)
    if data["other_models"]:
        others = " · ".join(f"{m['model_id']}: {m['vectors']} vectors ({m['dim']} dims)" for m in data["other_models"])
        out.append(f"  other ids  {others}  (kept for switching back; `vault embed --prune` drops them)")
    else:
        out.append("  other ids  none")
    out.append(f"  cache      {data['cache_vectors']} vectors, {_size(data['cache_bytes'])} in {args.db}")
    print("\n".join(out))
    return 0


def _cmd_embed(args: argparse.Namespace) -> int:
    from vaultengine.embedding.store import embed_notes

    if args.status:
        return _cmd_embed_status(args)
    provider = _provider(args)  # before touching the index: a missing model changes nothing
    if provider is None:
        return 2
    with VaultIndex(args.db) as index:
        indexed = index.update(args.vault)
        report = embed_notes(index.conn, args.vault, provider, full=args.full, prune=args.prune)
    if args.json:
        _print_json({"vault": str(Path(args.vault).resolve()), "index": indexed.to_dict(), **report.to_dict()})
        return 0
    out = [f"Embedded {Path(args.vault).resolve()} in {indexed.seconds + report.seconds:.2f}s",
           f"  model      {provider.describe()} · {provider.source}"]
    parsed = len(indexed.parsed)
    if parsed or indexed.deleted:
        out.append(f"  index      {parsed} parsed ({len(indexed.added)} added, {len(indexed.changed)} changed, "
                   f"{len(indexed.deleted)} deleted)" + (f" · all re-parsed: {indexed.full_reason}" if indexed.full_reason else ""))
    else:
        out.append("  index      already up to date")
    out.append(f"  vectors    {len(report.embedded)} embedded, {report.cached} cached ({report.notes} notes)"
               + (" · --full: every note embedded again" if args.full else ""))
    shown, more = _shown(report.embedded, args.limit)
    out.extend(f"    + {path}" for path in shown)
    if more:
        out.append(more)
    if report.empty:
        out.append(f"  empty      {_plural(len(report.empty), 'note')} with no word the model knows (a zero vector:"
                   " never similar to anything)")
    if report.changed_since_index:
        out.append(f"  changed    {_plural(len(report.changed_since_index), 'note')} changed while embedding; "
                   "run `vault embed` again")
    if report.unreadable:
        out.append(f"  unreadable {', '.join(report.unreadable)}")
    if args.prune:
        out.append(f"  pruned     {_plural(report.pruned, 'vector')} of other model ids")
    print("\n".join(out))
    return 0


# ----- config ------------------------------------------------------------------


def _cmd_config_show(args: argparse.Namespace) -> int:
    settings = args.settings
    data = settings.to_dict()
    if args.json:
        _print_json(data)
        return 0
    found = {"--config": "from --config", "$VAULTENGINE_CONFIG": "from $VAULTENGINE_CONFIG",
             "repository default": "the repository's; --config FILE or $VAULTENGINE_CONFIG for another"}
    config = f"{settings.config_file}  ({found[settings.config_found_by]})" if settings.config_file else "none found"
    out = ["Engine settings · precedence: flag > env > file > default", f"  config file  {config}", ""]
    rows = data["settings"]
    key_width = max(len(r["key"]) for r in rows)
    value_width = max(len(r["value"] or "(not set)") for r in rows)
    for row in rows:
        if row["key"] == "explore":
            out.append(f"  {'explore':<{key_width}}  (no settings yet: the explore engine's go here)")
            continue
        value = row["value"] or "(not set)"
        source = row["source"]
        if source.startswith("file "):
            source = "file"
        note = ""
        if row["key"] in ("vault", "embedding.model_path") and row["value"] and not Path(row["value"]).is_dir():
            note = "  ✗ not found"
        out.append(f"  {row['key']:<{key_width}}  {value:<{value_width}}  {source}{note}")
    out += ["", "  env: VAULTENGINE_<KEY>, \"__\" between levels (VAULTENGINE_EMBEDDING__MODEL_PATH)",
            "  flags: the vault argument, --db, --format, and --setting KEY=VALUE for any key"]
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
    return serve_stdio(VaultService(args.vault, args.db, format_path, settings=args.settings))


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

    def settings_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", metavar="FILE", help="config file (default: $VAULTENGINE_CONFIG, else config/config.yaml)")
        sub.add_argument("--setting", action="append", metavar="KEY=VALUE", default=[],
                         help="any engine setting, e.g. embedding.model_id=x (repeatable; see `vault config show`)")

    def add(name: str, help_text: str, handler, json_flag: bool = True, vault_required: bool = False,
            parent=commands, words: str | None = None) -> argparse.ArgumentParser:
        sub = parent.add_parser(name, help=help_text, description=help_text)
        if words:  # `vault mark [VAULT] keep 2.3`: the vault, when given, is the first word (a folder)
            sub.add_argument("words", nargs="+", metavar="ARG", help=words)
            sub.set_defaults(vault=None, vault_in_words=True)
        elif vault_required:  # writes name their vault; also lets options sit between positionals
            sub.add_argument("vault", help="vault folder")
        else:
            sub.add_argument("vault", nargs="?", default=None,
                             help="vault folder (default: engine.vault in the config, else the current folder)")
        sub.add_argument("--db", help="index file (default: engine.index in the config, else VAULT/.vaultengine/index.sqlite)")
        settings_args(sub)
        if json_flag:
            sub.add_argument("--json", action="store_true", help="print JSON instead of text")
        sub.set_defaults(handler=handler, needs_vault=True)
        return sub

    def format_arg(sub: argparse.ArgumentParser, required: bool = True, help_text: str | None = None) -> None:
        sub.add_argument("--format", "-f", help=help_text or "format file, or a name in ./formats/ or the bundled formats/"
                         " (default: engine.format in the config)")
        sub.set_defaults(needs_format=required)

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
        "link graph (missing notes included), blended with text similarity (embeddings from the model in "
        "engine.embedding, see `vault embed`) and shared tags. Each result says why."
    )
    related.add_argument("note", help="note name, path or alias (as in a link), or the target of a link to a missing note")
    related.add_argument("--limit", type=int, default=10, help="number of results (0 = all)")
    related.add_argument("--unlinked", action="store_true", help="only notes with no link to or from the note")
    related.add_argument("--explain", action="store_true", help="show the three signals behind each score")
    related.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="colour output (default: auto)")
    format_arg(related, required=False, help_text="format whose tag fields say which links are tags (default: engine.format;"
               " without one, fields named tags)")
    explain = add("explain", "How two notes relate, signal by signal, with the evidence for each.", _cmd_explain)
    explain.description = (
        "How two notes relate: direct links, co-citation (notes that link to both), bibliographic coupling (what both "
        "link to), shared tags and embedding similarity, each with its weight (engine.explore.weights), its contribution "
        "to the score, and what it rests on. Without the embedding model the text signal is marked unavailable."
    )
    explain.add_argument("a", help="a note: name, path or alias")
    explain.add_argument("b", help="another note")
    explain.add_argument("--words", type=int, default=8, help="words listed behind the text score")
    explain.add_argument("--limit", type=int, default=8, help="notes listed per signal (0 = all)")
    explain.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="colour output (default: auto)")
    format_arg(explain, required=False, help_text="format whose tag fields say which links are tags (default: engine.format;"
               " without one, fields named tags)")
    vmap = add("map", "The shape of the vault from links and tags: communities, bridge notes, hubs and authorities.", _cmd_map)
    vmap.description = (
        "Communities (Louvain, seeded: the same every run) named by their key phrases and hub notes, the ones with no "
        "topic page (notes the format types as topic or page), bridge notes (betweenness and Burt's constraint) and hubs vs "
        "authorities (HITS). From links and tags only; cutoffs are percentiles of the vault (engine.explore.percentile)."
    )
    vmap.add_argument("--limit", type=int, default=10, help="items per list (0 = all)")
    vmap.add_argument("--meaning", action=argparse.BooleanOptionalAction, default=None,
                      help="also join notes that are each other's nearest neighbours by embedding, for the communities,"
                           " and show what that changes (default: engine.explore.meaning)")
    vmap.add_argument("--record", action="store_true", help="also save a small snapshot of the map in engine.memory"
                      " (for --since later); without it the map writes nothing")
    vmap.add_argument("--since", metavar="DATE", help="instead of the map: what changed since the snapshot of DATE"
                      " (2026-11-10 or 2026-11-10T14:30): gaps closed, communities formed, missing notes written")
    vmap.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="colour output (default: auto)")
    format_arg(vmap, required=False, help_text="format whose tag fields say which links are tags (default: engine.format)")
    deepen = add("deepen", "Five lenses on one note: neighbourhood, cited together, same foundations, backbone, cluster.",
                 _cmd_deepen)
    deepen.add_argument("note", help="note name, path or alias")
    deepen.add_argument("--lens", action="append", choices=LENSES, help="only this lens (repeatable; default: all five)")
    deepen.add_argument("--limit", type=int, default=8, help="notes per lens (0 = all that stand out)")
    deepen.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="colour output (default: auto)")
    format_arg(deepen, required=False, help_text="format whose tag fields say which links are tags (default: engine.format)")
    disc = add("discover", "What the vault does not say yet: hidden links, same roles, bridges, gaps, surprises,"
               " unwritten notes, a seeded wander.", _cmd_discover)
    disc.description = (
        "Discover lenses. With a note: hidden link (Swanson's A-B-C), same role elsewhere, bridge, wander, and the gaps, "
        "surprises and unwritten notes around it; without one: gaps, surprises and unwritten notes in the whole vault. "
        "Every result says why; every cutoff is a percentile of the vault (engine.explore.percentile); embeddings only "
        "measure distance (gap, surprise) and merge names (unwritten), and those lenses also show what they find without "
        "them. The same index, settings and seed give the same output."
    )
    disc.add_argument("note", nargs="?", help="note name, path or alias (default: lenses over the whole vault)")
    disc.add_argument("--lens", action="append", choices=DISCOVER, help="only this lens (repeatable)")
    disc.add_argument("--limit", type=int, default=3, help="results per lens (0 = all that stand out)")
    disc.add_argument("--steps", type=int, default=8, help="wander: steps to take")
    disc.add_argument("--seed", type=int, help="seed for wander and sampling (default: engine.explore.seed)")
    disc.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="colour output (default: auto)")
    format_arg(disc, required=False, help_text="format: which links are tags, which notes are topic pages (default: engine.format)")
    from vaultengine.explore import REGISTRY as EXPLORE_LENSES

    expl = add("explore", "Explore from a note (or a missing note): the deepen and discover lenses, a section each;"
               " without one, where to start.", _cmd_explore)
    expl.description = (
        "The deepen and discover lenses through one registry: a section per lens (engine.explore.lenses), 3 to 5 results "
        "each (engine.explore.per_lens), every result with its lens and one line of evidence, near-duplicates left out, "
        "deepen and discover given equal space. Without a note it proposes where to start from what the vault-wide lenses "
        "find (engine.explore.start_lenses). Remembers what it showed (engine.memory): --fresh hides what you saw or "
        "dismissed; your keep rate per lens orders the sections. Never writes a note."
    )
    expl.add_argument("note", nargs="?", help="note name, path or alias, or a missing note (default: where to start)")
    expl.add_argument("--lens", action="append", choices=list(EXPLORE_LENSES), help="only this lens (repeatable)")
    expl.add_argument("--per-lens", type=int, choices=(3, 4, 5), help="results per section (default: engine.explore.per_lens)")
    expl.add_argument("--fresh", action="store_true", help="leave out what you saw or dismissed before, from this start")
    expl.add_argument("--no-record", action="store_true", help="do not remember this run (not seen, no refs for mark/link-it)")
    expl.add_argument("--timing", action="store_true", help="with --json: each lens's seconds (the only part that varies)")
    expl.add_argument("--refresh", action="store_true", help="recompute the stored vault-wide numbers now (see stats.py)")
    expl.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="colour output (default: auto)")
    format_arg(expl, required=False, help_text="format: which links are tags, which notes are topic pages (default: engine.format)")
    mark = add("mark", "Keep or dismiss results of the last `vault explore` (remembered in engine.memory, never in notes).",
               _cmd_mark, words="[VAULT] keep|dismiss|clear REF... (REF: 2.3 = section 2, result 3 of the last explore)")
    kept = add("kept", "What you kept from `vault explore`, with its lens and evidence.", _cmd_kept)
    kept.add_argument("--lens", choices=list(EXPLORE_LENSES), help="only this lens")
    linkit = add("link-it", "Make an explore result a real link: [[other note]] and a reason under See Also, in both notes.",
                 _cmd_link_it, words="[VAULT] REF (a result of the last explore), or [VAULT] NOTE NOTE")
    linkit.description = (
        "Write `- [[other]] — reason` under the section engine.explore.link_section (See Also) in both notes, through "
        "the safe write path (refused if vault check would get worse). A dry run unless --write; `vault unlink` undoes it."
    )
    format_arg(linkit)
    linkit.add_argument("--write", action="store_true", help="write it (default: a dry run that shows the diff)")
    linkit.add_argument("--diff", action="store_true", help="with --write: also print the diff")
    linkit.add_argument("--reason", help="the one-line reason (default: the result's lens and evidence)")
    unlink = add("unlink", "Undo `vault link-it`: remove the See Also lines linking the two notes, in both.", _cmd_unlink,
                 words="[VAULT] NOTE NOTE")
    format_arg(unlink)
    unlink.add_argument("--write", action="store_true", help="write it (default: a dry run that shows the diff)")
    unlink.add_argument("--diff", action="store_true", help="with --write: also print the diff")
    exp = add("experiment", "Compare relatedness, community and text options on the vault's own notes, seeded.",
              _cmd_experiment)
    exp.description = (
        "Hide a seeded share of the links (and everything derived from them) and see which relatedness signals and "
        "which embedded text recover them (recall at 5 and 10, MRR); find communities with and without meaning edges "
        "and compare them with the Tags: topics and topic pages (NMI, ARI), stranded notes and stability; time each. "
        "Writes a table per question with the winner, the margin, and whether it beats the seed-to-seed noise."
    )
    exp.add_argument("--seeds", type=int, default=5, help="seeds (hidden links, Louvain); from engine.explore.seed on")
    exp.add_argument("--hide", type=float, default=0.15, help="share of the linked note pairs hidden per seed (0.1-0.2)")
    exp.add_argument("--queries", type=int, default=200, help="hidden links scored per seed at most")
    exp.add_argument("--without", action="append", metavar="PATH", help="also run without the notes under PATH (repeatable)")
    exp.add_argument("--question", action="append", choices=("relatedness", "text", "communities"), help="only this question")
    exp.add_argument("--output", metavar="FILE", help="also write the report here (e.g. docs/experiments/<date>.md)")
    exp.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    format_arg(exp, required=False, help_text="format: tag links, template headings, topic pages (default: engine.format)")
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
    format_arg(suggest, required=False, help_text="format whose tag fields say which links are tags (default: engine.format;"
               " without one, fields named tags)")
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
    formats.add_argument("vault", nargs="?", default=None,
                         help="vault folder (default: engine.vault in the config, else the current folder)")
    format_arg(formats)
    settings_args(formats)
    formats.add_argument("--type", help="only this note type")
    formats.add_argument("--note", help="show one note as the format sees it (vault path or note name)")
    formats.add_argument("--limit", type=int, default=10, help="max issues and untyped notes listed (0 = all)")
    formats.add_argument("--strict", action="store_true", help="exit with code 1 if any issue is found")
    formats.add_argument("--json", action="store_true", help="print JSON instead of text")
    formats.set_defaults(handler=_cmd_formats, needs_vault=True, uses_index=False)
    check = add("check", "Report every violation of a format file, per note, with line numbers (exit 1 if any).", _cmd_check)
    format_arg(check)
    check.add_argument(
        "--changed", action="store_true",
        help="only notes added or changed since the last `vault index`, plus violations they cause in other notes",
    )
    def writer(name: str, help_text: str, handler) -> argparse.ArgumentParser:
        sub = add(name, help_text, handler, vault_required=True)
        format_arg(sub)
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
    format_arg(cov)
    cov.add_argument("--limit", type=int, default=10, help="max notes listed per list (0 = all)")
    cov.set_defaults(uses_index=False)
    serve = add("serve", "Serve the JSON-RPC 2.0 API on stdin/stdout (one JSON message per line).", _cmd_serve, json_flag=False)
    format_arg(serve, required=False, help_text="format for vault.check/fix/coverage and the note.* writes (name or file;"
               " default: engine.format in the config)")
    embed = add("embed", "Embed the notes that have no vector for the configured model, and cache the vectors in the index.",
                _cmd_embed)
    embed.description = (
        "Bring the index up to date, then embed every note whose text has no cached vector for the configured "
        "model (engine.embedding in config/config.yaml): new and edited notes, or all of them when the model id "
        "changed. Vectors are cached in the index by the hash of the note's text and the model id."
    )
    embed.add_argument("--status", action="store_true", help="only report how many notes have a vector; change nothing")
    embed.add_argument("--full", action="store_true", help="embed every note again, cached or not")
    embed.add_argument("--prune", action="store_true", help="drop the cached vectors of other model ids")
    embed.add_argument("--limit", type=int, default=10, help="max paths listed (0 = all)")
    config = commands.add_parser("config", help="Show the engine settings and where each comes from.",
                                 description="Engine settings: flag > env (VAULTENGINE_*) > config file (engine:) > default.")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    show = add("show", "Print every engine setting with its value and where it came from (flag, env, file or default).",
               _cmd_config_show, parent=config_commands)
    format_arg(show, required=False)
    show.set_defaults(uses_index=False, vault_may_be_missing=True)
    schema = commands.add_parser("rpc-schema", help="Print the JSON Schema of the JSON-RPC API.")
    schema.add_argument("--output", help="write to this file instead of stdout")
    schema.set_defaults(handler=_cmd_rpc_schema, needs_vault=False)
    return parser


def _settings(args: argparse.Namespace):
    """The engine settings for this command: its flags over env over the config file over defaults."""
    from vaultengine.settings import SettingsError, load_settings

    flags = [("vault", args.vault, "<vault>"), ("index", getattr(args, "db", None), "--db"),
             ("format", getattr(args, "format", None), "--format")]
    for item in getattr(args, "setting", None) or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SettingsError(f"--setting {item!r}: expected KEY=VALUE, e.g. embedding.model_id=potion-base-8M")
        flags.insert(0, (key.strip(), value, f"--setting {key.strip()}"))  # the dedicated flags win over --setting
    return load_settings(flags, config_file=getattr(args, "config", None))


def main(argv: Sequence[str] | None = None) -> int:
    from vaultengine.settings import SettingsError

    args = build_parser().parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    if not args.needs_vault:
        return args.handler(args)
    if getattr(args, "command", None) in ("discover", "explore") and args.note is None and args.vault and not Path(args.vault).is_dir():
        args.note, args.vault = args.vault, None  # `vault discover NOTE` with the vault from the settings
    if getattr(args, "vault_in_words", False) and len(args.words) > 1 and Path(args.words[0]).is_dir():
        args.vault, args.words = args.words[0], args.words[1:]
    try:
        args.settings = _settings(args)
    except SettingsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    engine = args.settings.engine
    args.vault = str(engine.vault)
    if hasattr(args, "needs_format"):
        args.format = engine.format
        if args.needs_format and not args.format:
            print("error: no format: give --format NAME (e.g. --format brain), or set engine.format in "
                  "config/config.yaml or VAULTENGINE_FORMAT", file=sys.stderr)
            return 2
    if not Path(args.vault).is_dir() and not getattr(args, "vault_may_be_missing", False):
        print(f"error: vault folder not found: {args.vault}", file=sys.stderr)
        return 2
    if getattr(args, "uses_index", True):
        args.db = engine.index_path
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
