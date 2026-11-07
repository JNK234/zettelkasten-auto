#!/usr/bin/env python3
"""Check the documentation's links: README.md, docs/*.md and tui/README.md.

* every relative link points at a file or folder that exists;
* every #anchor exists in its target (GitHub's heading slugs: lower-cased, punctuation
  removed, each space a '-', '-1', '-2'... for repeated headings);
* every docs/*.md is linked from README.md, and links back to it.

Links inside code (fenced blocks, `inline code`) are not links and are skipped.
Exit code 0 when everything resolves, 1 otherwise (each problem is printed).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"!?\[(?:[^\[\]]|\[[^\]]*\])*\]\(([^()\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^\s*(```|~~~)")


def prose(text: str) -> str:
    """The text with fenced code blocks and inline code spans blanked (line numbers kept)."""
    out, fence = [], False
    for line in text.split("\n"):
        if FENCE.match(line):
            fence = not fence
            out.append("")
            continue
        out.append("" if fence else re.sub(r"`[^`]*`", "``", line))
    return "\n".join(out)


def slug(heading: str) -> str:
    text = re.sub(r"`", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)  # github-slugger keeps letters, digits, '_', '-' and spaces
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    found: dict[str, int] = {}
    result = set()
    fence = False
    for line in path.read_text(encoding="utf-8").split("\n"):
        if FENCE.match(line):
            fence = not fence
            continue
        match = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if fence or not match:
            continue
        base = slug(match[1])
        count = found.get(base, 0)
        found[base] = count + 1
        result.add(base if count == 0 else f"{base}-{count}")
    return result


def main() -> int:
    files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "tui" / "README.md"]
    problems: list[str] = []
    linked_from_readme: set[Path] = set()
    checked = 0
    for source in files:
        text = prose(source.read_text(encoding="utf-8"))
        for number, line in enumerate(text.split("\n"), start=1):
            for match in LINK.finditer(line):
                target = match[1]
                if re.match(r"^[a-z][a-z0-9+.-]*:", target):  # https:, mailto:
                    continue
                checked += 1
                where = f"{source.relative_to(ROOT)}:{number}"
                file_part, _, anchor = target.partition("#")
                resolved = (source.parent / unquote(file_part)).resolve() if file_part else source
                if not resolved.exists():
                    problems.append(f"{where}: {target} -> no such file {resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved}")
                    continue
                if source.name == "README.md" and source.parent == ROOT:
                    linked_from_readme.add(resolved)
                if anchor:
                    if resolved.is_dir() or resolved.suffix != ".md":
                        problems.append(f"{where}: {target} -> an anchor into a file that is not markdown")
                    elif unquote(anchor) not in anchors(resolved):
                        problems.append(f"{where}: {target} -> no heading with the anchor #{anchor} in {resolved.relative_to(ROOT)}")
    for doc in sorted((ROOT / "docs").glob("*.md")):
        if doc.resolve() not in linked_from_readme:
            problems.append(f"docs/{doc.name} is not linked from README.md")
        if "../README.md" not in doc.read_text(encoding="utf-8"):
            problems.append(f"docs/{doc.name} does not link back to README.md")
    for problem in problems:
        print(problem)
    docs = len(list((ROOT / "docs").glob("*.md")))
    print(f"{checked} links in {len(files)} files checked, {docs} docs linked from README.md: "
          + ("all resolve" if not problems else f"{len(problems)} problems"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
