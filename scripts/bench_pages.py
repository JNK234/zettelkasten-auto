#!/usr/bin/env python3
"""Generate a vault that follows formats/core.yaml, and time `vault pipeline` and `vault page check` on it.

    .venv/bin/python scripts/bench_pages.py generate 5000 /tmp/bench-5k     # 5,000 notes
    .venv/bin/python scripts/bench_pages.py time /tmp/bench-5k 3 --build     # build the index, then time each command
    .venv/bin/python scripts/bench_pages.py time-moves /tmp/bench-5k         # vault next, a move and a merge (they write)
    .venv/bin/python scripts/bench_pages.py time-every /tmp/bench-5k --build # every command of the reading pipeline (writes)

Shaped like the sample vault: 20% sources (~26 KB each), 75% zettels (~3 KB, 2-4 tags from a
vocabulary of 150, 1-2 source links, 3-6 zettel links), 5% pages made from 8-25 related zettels by the
same code `vault page new` uses (blocks, index line, engine state). Deterministic for a given size.
Each command runs as a fresh process, as from the shell: nothing is cached between runs but the index.
The timing edits 20 zettels in the vault (appends a line) to time reads with a stale index.
"""

from __future__ import annotations

import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

WORDS = ("reward model policy gradient grader verifier environment agent trajectory rollout sample batch "
         "benchmark evaluation judge rubric signal noise bias variance estimator baseline advantage critic value "
         "token context window memory retrieval search plan tool call state action observation feedback loss").split()


def sentence(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def paragraph(rng: random.Random, size: int) -> str:
    out = []
    while sum(len(s) + 1 for s in out) < size:
        out.append(sentence(rng, rng.randint(8, 20)))
    return " ".join(out)


def generate(total: int, root: Path) -> None:
    from vaultengine import Vault
    from vaultengine.formats import TypedVault, load_format
    from vaultengine.formats.pages import STATE_PATH, Pages, sha256_text

    rng = random.Random(total)
    n_sources, n_pages = total // 5, total // 20
    n_zettels = total - n_sources - n_pages - 1  # and the index
    tags = [f"topic-{i:03d}" for i in range(150)]
    sources = [f"{2400 + i // 100}.{i % 100:05d} - Paper {i}" for i in range(n_sources)]
    zettels = [f"zettel-{i:05d}-{rng.choice(WORDS)}-{rng.choice(WORDS)}" for i in range(n_zettels)]
    zettel_tags = {}
    for folder in ("10_SOURCES", "20_ZETTLEKASTEN", "50_MOC"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(sources):
        body = "\n\n".join(paragraph(rng, 2500) for _ in range(10))
        others = " ".join(f"[[{rng.choice(sources)}]]" for _ in range(3))
        (root / "10_SOURCES" / f"{name}.md").write_text(
            f"Created: 01-01-2026 10:00\nTags: [[{rng.choice(tags)}]]\nSource: https://arxiv.org/abs/{name[:10]}\n\n"
            f"## References\n\n1. [{name}](https://arxiv.org/abs/{name[:10]})\n\n# Notes\n\n{body}\n\n## Connections\n\n{others}\n")
    for i, name in enumerate(zettels):
        # Zettels cluster: each draws its tags and neighbours from a topic neighbourhood.
        topic = i % len(tags)
        mine = sorted({tags[(topic + rng.randint(0, 3)) % len(tags)] for _ in range(rng.randint(2, 4))})
        zettel_tags[name] = mine
        cited = rng.sample(sources, rng.randint(1, 2))
        neighbours = [zettels[(i + rng.randint(-300, 300)) % n_zettels] for _ in range(rng.randint(3, 6))]
        body = "\n\n".join(paragraph(rng, 500) for _ in range(5))
        (root / "20_ZETTLEKASTEN" / f"{name}.md").write_text(
            f"Created: 01-01-2026 10:00\nTags: {' '.join(f'[[{t}]]' for t in mine)}\n\n# {name}\n\n## What This Is\n\n"
            f"{body}\n\n## Tensions\n\n" + "\n".join(f"- [[{n}]]" for n in neighbours)
            + "\n\n# References\n\n" + "\n".join(f"1. [[{c}]]" for c in cited) + "\n")
    (root / "50_MOC" / "index.md").write_text("# Index\n\n<!-- vault:pages -->\n<!-- /vault:pages -->\n")

    typed = TypedVault(load_format(REPO / "formats" / "core.yaml"), Vault.load(root))
    pages = Pages(typed)
    state = {"version": 1, "pages": {}}
    index_lines = []
    for p in range(n_pages):
        start = rng.randrange(n_zettels)
        chosen = [f"20_ZETTLEKASTEN/{zettels[(start + k * rng.randint(1, 5)) % n_zettels]}.md" for k in range(rng.randint(8, 25))]
        chosen = list(dict.fromkeys(chosen))
        title = f"Page {p:04d} on {rng.choice(WORDS)} and {rng.choice(WORDS)}"
        slug = title.lower().replace(" ", "-")
        path = f"50_MOC/{slug}.md"
        text = (f"Created: 01-01-2026 10:00\nTags: {pages.default_tags(chosen)}\n\n# {title}\n\n{pages.blocks_text(chosen)}\n\n"
                f"## Notes\n\n{paragraph(rng, 1500)}\n")
        (root / path).write_text(text)
        state["pages"][path] = {"refreshed_at": "2026-01-01T10:00:00",
                                "zettels": {z: sha256_text(pages.text(z) or "") for z in sorted(chosen)}}
        cited, _ = pages.sources_of(chosen)
        index_lines.append(f"- [[{slug}]]: {title} ({len(chosen)} zettels, {len(cited)} sources)")
    (root / "50_MOC" / "index.md").write_text("# Index\n\n<!-- vault:pages -->\n" + "\n".join(sorted(index_lines))
                                              + "\n<!-- /vault:pages -->\n")
    (root / STATE_PATH).parent.mkdir(exist_ok=True)
    (root / STATE_PATH).write_text(json.dumps(state))
    size = sum(f.stat().st_size for f in root.rglob("*.md"))
    print(f"{root}: {n_sources} sources, {n_zettels} zettels, {n_pages} pages, index; {size / 1e6:.0f} MB of notes")


def run(args: list[str]) -> float:
    start = time.perf_counter()
    result = subprocess.run([str(REPO / ".venv" / "bin" / "vault"), *args], capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if result.returncode not in (0, 1):
        raise SystemExit(f"vault {' '.join(args)} failed ({result.returncode}): {result.stderr}")
    if "no usable index" in result.stderr:
        raise SystemExit(f"vault {' '.join(args)} did not use the index: {result.stderr}")
    return elapsed


def time_all(root: Path, repeat: int = 3, build: bool = False) -> None:
    """Wall time of each command as a fresh process: the index, then reads through it, then the same
    reads after 20 zettels changed since the index (parsed from disk), then the incremental index."""
    page = sorted((root / "50_MOC").glob("page-*.md"))[0].relative_to(root).as_posix()
    zettel = sorted((root / "20_ZETTLEKASTEN").glob("*.md"))[0].stem
    notes = sum(1 for _ in root.rglob("*.md"))
    print(f"{root}: {notes} notes; best and median of {repeat} runs (seconds)", flush=True)

    def show(label: str, args: list[str], times: int = repeat) -> None:
        runs = [run(args) for _ in range(times)]
        print(f"  {label:<44} best {min(runs):7.2f}   median {statistics.median(runs):7.2f}", flush=True)

    if build:
        db = root / ".vaultengine" / "index.sqlite"
        for extra in (db, db.with_name("index.sqlite-wal"), db.with_name("index.sqlite-shm")):
            extra.unlink(missing_ok=True)
        show("vault index (full build, once)", ["index", str(root)], 1)
    show("vault index (nothing changed)", ["index", str(root)])
    reads = {
        "vault page check PAGE": ["page", "check", str(root), page, "-f", "core"],
        "vault pipeline": ["pipeline", str(root), "-f", "core"],
        "vault note ZETTEL": ["note", str(root), zettel, "-f", "core"],
        "vault page check (every page)": ["page", "check", str(root), "-f", "core"],
    }
    for label, args in reads.items():
        show(label, args, 1 if "every page" in label else repeat)
    # 20 zettels edited since the index: read through the index, those 20 parsed from disk.
    changed = sorted((root / "20_ZETTLEKASTEN").glob("*.md"))[100:120]
    for path in changed:
        path.write_text(path.read_text() + "\nEdited after the index.\n")
    for label in ("vault page check PAGE", "vault pipeline"):
        show(f"{label} (20 notes changed since index)", reads[label])
    show("vault index (20 notes changed)", ["index", str(root)], 1)


def time_moves(root: Path) -> None:
    """vault next, then a move and a merge of the zettel most notes link to (each a dry run, then written),
    then that every link still resolves (`vault health`)."""
    import sqlite3

    conn = sqlite3.connect(root / ".vaultengine" / "index.sqlite")
    rows = conn.execute(
        """SELECT t.path, COUNT(DISTINCT l.source_id) AS n FROM links l JOIN files t ON t.id = l.target_id
           WHERE t.path LIKE '20_ZETTLEKASTEN/%' AND l.status = 'resolved' GROUP BY t.path ORDER BY n DESC, t.path LIMIT 2"""
    ).fetchall()
    conn.close()
    (hub, linkers), (other, _) = rows
    print(f"{root}: the zettel most linked to: {hub} ({linkers} notes link to it)", flush=True)

    def show(label: str, args: list[str]) -> None:
        print(f"  {label:<44} {run(args):7.2f} s", flush=True)

    def health() -> str:
        text = subprocess.run([str(REPO / ".venv" / "bin" / "vault"), "health", str(root)], capture_output=True, text=True).stdout
        return "; ".join(" ".join(line.split()) for line in text.splitlines()
                         if line.strip().startswith(("unresolved links", "ambiguous links", "missing heading")))

    print(f"  before: {health()}", flush=True)
    show("vault next", ["next", str(root), "-f", "core"])
    new_name = Path(hub).stem + "-moved"
    show("vault move (dry run)", ["move", str(root), hub, new_name, "-f", "core"])
    show("vault move --write", ["move", str(root), hub, new_name, "-f", "core", "--write"])
    moved = f"20_ZETTLEKASTEN/{new_name}.md"
    show("vault merge (dry run)", ["merge", str(root), other, moved, "-f", "core"])
    show("vault merge --write", ["merge", str(root), other, moved, "-f", "core", "--write"])
    print(f"  after:  {health()}", flush=True)


def time_every(root: Path, build: bool = False) -> None:
    """Every command of the reading pipeline, in the order of docs/workflow.md, each a fresh process: the index,
    the check and its baseline, pipeline, note, pages, findings, ideas, move, merge, next, then the index after 20
    notes changed. Reads and dry runs are run 3 times (best and median shown: a shared machine is noisy); the
    writes run once, after their dry run. The vault is changed (a page, an idea, a move, a merge): time a fresh copy."""
    import sqlite3

    db = root / ".vaultengine" / "index.sqlite"
    rows: list[tuple[str, float, float]] = []

    def show(label: str, args: list[str], repeat: bool = True) -> str:
        start = time.perf_counter()
        result = subprocess.run([str(REPO / ".venv" / "bin" / "vault"), *args], capture_output=True, text=True)
        runs = [time.perf_counter() - start]
        if result.returncode not in (0, 1):
            raise SystemExit(f"vault {' '.join(args)} failed ({result.returncode}): {result.stderr}")
        if "no usable index" in result.stderr:
            raise SystemExit(f"vault {' '.join(args)} did not use the index: {result.stderr}")
        if repeat:
            runs += [run(args) for _ in range(2)]
        rows.append((label, min(runs), statistics.median(runs)))
        print(f"  {label:<48} best {min(runs):7.2f} s   median {statistics.median(runs):7.2f} s", flush=True)
        return result.stdout

    core = ["-f", "core"]
    if build:
        for extra in (db, db.with_name("index.sqlite-wal"), db.with_name("index.sqlite-shm")):
            extra.unlink(missing_ok=True)
        show("vault index (full build)", ["index", str(root)], repeat=False)
    notes = sum(1 for _ in root.rglob("*.md"))
    print(f"{root}: {notes} notes", flush=True)
    show("vault index (nothing changed)", ["index", str(root)])
    page = sorted((root / "50_MOC").glob("page-*.md"))[0].relative_to(root).as_posix()
    zettels = sorted(p.stem for p in (root / "20_ZETTLEKASTEN").glob("*.md"))
    conn = sqlite3.connect(db)
    (hub, _), (other, _) = conn.execute(
        """SELECT t.path, COUNT(DISTINCT l.source_id) AS n FROM links l JOIN files t ON t.id = l.target_id
           WHERE t.path LIKE '20_ZETTLEKASTEN/%' AND l.status = 'resolved' GROUP BY t.path ORDER BY n DESC, t.path LIMIT 2"""
    ).fetchall()
    conn.close()

    show("vault check (whole vault)", ["check", str(root), *core])
    show("vault check --update-baseline", ["check", str(root), *core, "--update-baseline"], repeat=False)
    show("vault check --baseline", ["check", str(root), *core, "--baseline"])
    show("vault pipeline", ["pipeline", str(root), *core])
    show("vault note ZETTEL", ["note", str(root), zettels[0], *core])
    show("vault page check PAGE", ["page", "check", str(root), page, *core])
    show("vault page check (every page)", ["page", "check", str(root), *core])
    source = " ".join(f"[[{z}]]" for z in zettels[:3])
    new_page = ["page", "new", str(root), "Bench page", "--from", source, "--tags", "[[topic-000]]", *core]
    show("vault page new (dry run)", new_page)
    show("vault page new --write", new_page + ["--write"], repeat=False)
    show("vault page refresh (dry run)", ["page", "refresh", str(root), "bench-page", *core])
    exclude = ["page", "exclude", str(root), "bench-page", zettels[3], *core]
    show("vault page exclude (dry run)", exclude)
    show("vault page exclude --write", exclude + ["--write"], repeat=False)
    found = json.loads(show("vault findings", ["findings", str(root), *core, "--json", "--limit", "0"]))
    finding = found["findings"][0]["id"]
    idea = ["idea", "new", str(root), "--from", finding, *core]
    show("vault idea new (dry run)", idea)
    made = json.loads(show("vault idea new --write", idea + ["--write", "--json"], repeat=False))
    idea_path = made["idea"]["path"]
    show("vault idea set (dry run)", ["idea", "set", str(root), idea_path, "done", *core])
    show("vault idea set --write", ["idea", "set", str(root), idea_path, "done", *core, "--write"], repeat=False)
    new_name = Path(hub).stem + "-moved"
    show("vault move (dry run)", ["move", str(root), hub, new_name, *core])
    show("vault move --write", ["move", str(root), hub, new_name, *core, "--write"], repeat=False)
    moved = f"20_ZETTLEKASTEN/{new_name}.md"
    show("vault merge (dry run)", ["merge", str(root), other, moved, *core])
    show("vault merge --write", ["merge", str(root), other, moved, *core, "--write"], repeat=False)
    show("vault next", ["next", str(root), *core])
    show("vault check --baseline (after the writes)", ["check", str(root), *core, "--baseline"])
    for path in sorted((root / "20_ZETTLEKASTEN").glob("*.md"))[100:120]:
        path.write_text(path.read_text() + "\nEdited after the index.\n")
    show("vault check --baseline (20 notes changed)", ["check", str(root), *core, "--baseline"])
    show("vault index (20 notes changed)", ["index", str(root)], repeat=False)
    print("\n| command | best (s) | median (s) |\n|---|---|---|")
    for label, best, median in rows:
        print(f"| `{label}` | {best:.1f} | {median:.1f} |")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "generate":
        target = Path(sys.argv[3])
        if target.exists() and any(target.iterdir()):
            raise SystemExit(f"{target} is not empty")
        generate(int(sys.argv[2]), target)
    elif len(sys.argv) >= 3 and sys.argv[1] == "time-moves":
        time_moves(Path(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "time-every":
        time_every(Path(sys.argv[2]), build="--build" in sys.argv)
    elif len(sys.argv) >= 3 and sys.argv[1] == "time":
        time_all(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3, build="--build" in sys.argv)
    else:
        raise SystemExit(__doc__)
