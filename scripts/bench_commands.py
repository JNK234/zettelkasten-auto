"""Time every `vault` command on a generated vault (wall clock, one process per command, as a person runs them).

    python scripts/bench_commands.py DIR [--label 5000] [--skip-experiment]

DIR: a generated vault (`scripts/bench_shape.py discover N DIR`); it is changed (links written, a note created, a
source ingested), so give it a copy. Its index and the engine's files (.vaultengine) are removed first, so the first
`vault index` is a full one and explore starts with no stored numbers. The format commands use a small format
written for these vaults (every note a `note` with a `Tags:` line of tag links, `related::` relations, a `sources/`
folder for ingest). Prints one line per command: the command, the seconds, the exit code.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAULT = str(ROOT / ".venv" / "bin" / "vault")

FORMAT = """format: 1
name: bench
description: the vaults scripts/bench_shape.py generates
metadata_source: inline
relations:
  related:
    inverse: related
link_roles:
  see-also: [See Also]
coverage:
  sources: [source]
  citing: [note]
  citation_roles: [see-also]
types:
  source:
    match: {folder: sources}
    headings: [References, Notes]
    metadata:
      Tags: {kind: links, link_role: tag}
      Original: {kind: text}
      Content hash: {kind: text}
      Ingested: {kind: datetime, format: "%d-%m-%Y %H:%M"}
    ingest:
      body_heading: Notes
      references_heading: References
      provenance: {file: Original, hash: Content hash, date: Ingested}
  note:
    match: {name: "n*"}
    metadata:
      Tags: {required: true, kind: links, link_role: tag}
    new:
      name: "n-{slug}"
      defaults: {Tags: "[[topic-1]]"}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vault")
    parser.add_argument("--label", default="")
    parser.add_argument("--skip-experiment", action="store_true")
    parser.add_argument("--json", help="also write the timings here")
    args = parser.parse_args()
    vault = Path(args.vault).resolve()
    shutil.rmtree(vault / ".vaultengine", ignore_errors=True)
    work = Path(tempfile.mkdtemp(prefix="bench-commands-"))
    fmt = work / "bench.yaml"
    fmt.write_text(FORMAT, encoding="utf-8")
    doc = work / "an-article.md"
    doc.write_text("# An article about n00042\n\nA few words on retrieval, evaluation and reward.\n", encoding="utf-8")
    f = ["-f", str(fmt)]
    v = str(vault)
    rows: list[dict] = []

    def run(label: str, *command: str, expect: int | None = 0, stdin: str | None = None) -> str:
        started = time.perf_counter()
        proc = subprocess.run([VAULT, *command], capture_output=True, text=True, input=stdin, cwd=ROOT)
        seconds = time.perf_counter() - started
        rows.append({"command": label, "seconds": round(seconds, 1), "exit": proc.returncode})
        print(f"{label:<58} {seconds:8.1f} s   exit {proc.returncode}", flush=True)
        if expect is not None and proc.returncode != expect:
            print(proc.stderr[-2000:], file=sys.stderr)
        return proc.stdout

    run("vault index (first, full)", "index", v)
    run("vault index (nothing changed)", "index", v)
    run("vault embed --full (every note)", "embed", v, "--full")
    run("vault embed (all cached)", "embed", v)
    run("vault embed --status", "embed", v, "--status")
    run("vault config show", "config", "show", v)
    run("vault stats", "stats", v)
    run("vault health", "health", v)
    run("vault search 'retrieval reward'", "search", v, "retrieval reward")
    run("vault related n00042", "related", v, "n00042")
    run("vault related n00042 --unlinked", "related", v, "n00042", "--unlinked")
    run("vault explain n00042 n00043", "explain", v, "n00042", "n00043")
    run("vault suggest", "suggest", v)
    run("vault unlinked n00042", "unlinked", v, "n00042")
    run("vault map", "map", v)
    run("vault map --record", "map", v, "--record")
    run("vault deepen n00042", "deepen", v, "n00042")
    run("vault discover n00042", "discover", v, "n00042")
    run("vault discover (whole vault)", "discover", v)
    run("vault explore n00042 (first: nothing stored)", "explore", v, "n00042")
    run("vault explore n00042 (numbers stored)", "explore", v, "n00042")
    out = run("vault explore n00042 --json (again, for the refs)", "explore", v, "n00042", "--json")
    results = [r for s in json.loads(out)["sections"] for r in s["results"]]
    linkable = next(r["ref"] for r in results if r["link"])
    run("vault explore (no start)", "explore", v)
    run("vault explore n00042 (refs again)", "explore", v, "n00042")
    run(f"vault mark keep {results[0]['ref']}", "mark", v, "keep", results[0]["ref"])
    run("vault kept", "kept", v)
    run(f"vault link-it {linkable} (dry run)", "link-it", v, linkable, *f)
    run(f"vault link-it {linkable} --write", "link-it", v, linkable, *f, "--write")
    run("vault explore n00042 (2 notes changed: numbers reused)", "explore", v, "n00042")
    run("vault explore n00042 --fresh (same state again)", "explore", v, "n00042", "--fresh")
    run("vault map --since 2000-01-01", "map", v, "--since", "2000-01-01")
    a, b = next(r["link"] for r in results if r["ref"] == linkable)
    run("vault unlink (dry run)", "unlink", v, Path(a).stem, Path(b).stem, *f)
    run("vault unlink --write", "unlink", v, Path(a).stem, Path(b).stem, *f, "--write")
    run("vault formats", "formats", v, *f, expect=None)
    run("vault check", "check", v, *f, expect=None)
    run("vault check --changed", "check", v, *f, "--changed", expect=None)
    run("vault coverage", "coverage", v, *f, expect=None)
    run("vault fix --dry-run", "fix", v, *f, "--dry-run", expect=None)
    run("vault new (a note)", "new", v, *f, "--type", "note", "--title", "Bench note")
    run("vault link n00042 n00043 --as related", "link", v, *f, "n00042", "n00043", "--as", "related")
    run("vault set n00042 Tags", "set", v, *f, "n00042", "Tags", "[[topic-1]] [[topic-2]]")
    run("vault ingest (a markdown file)", "ingest", v, *f, str(doc), "--as", "source", "--folder", "sources")
    rpc = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "explore.run", "params": {"note": "n00042", "record": False}})
    run("vault serve: one explore.run call (start to answer)", "serve", v, *f, stdin=rpc + "\n")
    run("vault rpc-schema", "rpc-schema")
    if not args.skip_experiment:
        run("vault experiment --seeds 2", "experiment", v, "--seeds", "2", "--quiet")
    shutil.rmtree(work, ignore_errors=True)
    if args.json:
        Path(args.json).write_text(json.dumps({"label": args.label, "rows": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
