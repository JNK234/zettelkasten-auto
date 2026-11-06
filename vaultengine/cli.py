"""`vault` command line: index, stats, health, serve (JSON-RPC), rpc-schema.

Human-readable output by default, machine-readable with --json.
Exit codes: 0 success, 2 usage error or missing index.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from vaultengine import queries
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


# ----- rpc ---------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    from vaultengine.rpc.server import serve_stdio
    from vaultengine.service import VaultService

    return serve_stdio(VaultService(args.vault, args.db))


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

    def add(name: str, help_text: str, handler, json_flag: bool = True) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
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
    add("serve", "Serve the JSON-RPC 2.0 API on stdin/stdout (one JSON message per line).", _cmd_serve, json_flag=False)
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
    args.db = Path(args.db) if args.db else default_db_path(args.vault)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
