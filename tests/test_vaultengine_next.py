"""`vault next` on a vault with one planted item of each kind, and nothing else that needs doing.

Proved: the queue is exactly the planted items, in a stable order (the same across runs and hash
seeds); each item's command resolves it; relative cutoffs list the upper half of a kind; the trust
signal is shown by `vault note --json` and is never written to a note.
"""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS = Path(__file__).parent
REPO_ROOT = TESTS.parent


def run_vault(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "vaultengine", *args], capture_output=True, text=True, cwd=REPO_ROOT,
                          env={**os.environ, **(env or {})})


def source(body: str = "") -> str:
    return f"Created: 01-01-2026 10:00\nTags: [[ml]]\nSource: https://example.com\n\n# S\n\n{body}\n"


def zettel(tags: str, body: str) -> str:
    return f"Created: 01-01-2026 10:00\nTags: {tags}\n\n# Z\n\n{body}\n"


def idea(status: str) -> str:
    return (f"Created: 01-01-2026 10:00\nTags: [[ml]]\nStatus: {status}\n\n# I\n\n<!-- vault:finding -->\n"
            "- id: link-0123456789\n<!-- /vault:finding -->\n")


FILES = {
    "10_SOURCES/S1.md": source(),
    "10_SOURCES/S2.md": source(),
    "10_SOURCES/S3.md": source(),
    # extract: no zettel links S4. Its dangling links only spread the counts the findings' percentiles come from.
    "10_SOURCES/S4.md": source("Loose ends: [[alpha]] [[beta]] [[gamma]] [[delta]] [[epsilon]] [[zeta]]."),
    # on the page (made below); p2 is edited after it, so the page needs a refresh
    "20_ZETTLEKASTEN/p1.md": zettel("[[grading]]", "Graders check end states. From [[S1]]."),
    "20_ZETTLEKASTEN/p2.md": zettel("[[grading]]", "Partial credit needs a gate. From [[S1]]."),
    "20_ZETTLEKASTEN/p3.md": zettel("[[grading]]", "Rubrics drift. From [[S1]]."),
    # behind: the tag grading and a link to p1 in common with the page
    "20_ZETTLEKASTEN/z-behind.md": zettel("[[grading]]", "Builds on [[p1]]. From [[S3]]."),
    # page: three zettels in no page share the tag memory; write: all three link [[reward-hacking]];
    # merge: two link [[end-state-grading]], which is probably grading-the-end-state
    "20_ZETTLEKASTEN/c1.md": zettel("[[memory]]", "From [[S2]]. See [[reward-hacking]], [[end-state-grading]]."),
    "20_ZETTLEKASTEN/c2.md": zettel("[[memory]]", "From [[S2]]. See [[reward-hacking]], [[end-state-grading]]."),
    "20_ZETTLEKASTEN/c3.md": zettel("[[memory]]", "From [[S2]]. See [[reward-hacking]]."),
    "20_ZETTLEKASTEN/grading-the-end-state.md": zettel("[[other]]", "From [[S3]]."),
    # idea: open and nothing links to it; a done one, and an open one a zettel builds on, are not listed
    "50_MOC/ideas/Lonely.md": idea("#open"),
    "50_MOC/ideas/Done.md": idea("#done"),
    "50_MOC/ideas/Used.md": idea("#open"),
    "20_ZETTLEKASTEN/builder.md": zettel("[[other]]", "Builds on the idea [[Used]]. From [[S3]]."),
}
PLANTED = [
    ("write", "reward-hacking"),
    ("refresh", "50_MOC/grading.md"),
    ("behind", "20_ZETTLEKASTEN/z-behind.md"),
    ("merge", "end-state-grading"),
    ("extract", "10_SOURCES/S4.md"),
    ("page", "memory"),
    ("idea", "50_MOC/ideas/Lonely.md"),
]


class Planted:
    def __init__(self, files=FILES):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name) / "vault"
        for path, text in files.items():
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
            (self.root / path).write_text(text)
        made = run_vault("page", "new", str(self.root), "Grading", "--from", "[[p1]] [[p2]] [[p3]]", "-f", "core", "--write")
        assert made.returncode == 0, made.stdout + made.stderr
        (self.root / "20_ZETTLEKASTEN/p2.md").write_text(zettel("[[grading]]", "Partial credit needs a correctness gate. From [[S1]]."))
        assert run_vault("index", str(self.root)).returncode == 0

    def queue(self, env=None) -> dict:
        result = run_vault("next", str(self.root), "-f", "core", "--limit", "0", "--json", env=env)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    def close(self):
        self.dir.cleanup()


class NextTests(unittest.TestCase):
    def setUp(self):
        self.vault = Planted()

    def tearDown(self):
        self.vault.close()

    def test_exactly_the_planted_items_in_order(self):
        self.assertEqual(run_vault("check", str(self.vault.root), "-f", "core").returncode, 0)  # a clean vault otherwise
        items = self.vault.queue()["items"]
        self.assertEqual([(i["kind"], i["subject"]) for i in items], PLANTED)
        # The order: trust first (the missing note three notes rely on), then kind, as documented.
        self.assertGreater(items[0]["trust"], 0)
        self.assertEqual({i["trust"] for i in items[1:]}, {0})
        text = run_vault("next", str(self.vault.root), "-f", "core").stdout
        self.assertIn("1. write    write [[reward-hacking]]", text)
        self.assertIn("run  vault page refresh", text)

    def test_stable_across_runs_and_hash_seeds(self):
        outputs = {json.dumps(self.vault.queue(env={"PYTHONHASHSEED": str(seed)}), sort_keys=True) for seed in (0, 1, 2, 3)}
        self.assertEqual(len(outputs), 1)

    def test_each_command_resolves_its_item(self):
        items = self.vault.queue()["items"]
        for item in items:
            with self.subTest(item=item["kind"]):
                args = shlex.split(item["command"].replace("<what the zettel says>", "What S4 says"))
                self.assertEqual(args[0], "vault")
                result = run_vault(*args[1:])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        left = {i["id"] for i in self.vault.queue()["items"]}
        self.assertFalse(left & {i["id"] for i in items}, "an item is still there after its command ran")
        self.assertEqual(run_vault("check", str(self.vault.root), "-f", "core").returncode, 0)

    def test_trust_is_shown_and_never_written(self):
        before = {p: (self.vault.root / p).read_bytes() for p in FILES if (self.vault.root / p).exists()}
        data = json.loads(run_vault("note", str(self.vault.root), "S1", "-f", "core", "--json").stdout)
        trust = data["trust"]
        # S1: linked from p1, p2, p3 and the page's sources block; cited by the page through its zettels.
        self.assertEqual((trust["links_in"], trust["pages"], trust["ideas"]), (4, 1, 0))
        self.assertGreater(trust["percentile"], 50)
        self.assertIn("trust      ", run_vault("note", str(self.vault.root), "S1", "-f", "core").stdout)
        run_vault("next", str(self.vault.root), "-f", "core")
        self.assertEqual({p: (self.vault.root / p).read_bytes() for p in before}, before)


class CutoffTests(unittest.TestCase):
    def test_upper_half_of_unextracted_sources(self):
        # Four sources no zettel links to; two are linked from other notes (more trusted): only they are listed.
        files = dict(FILES)
        files["10_SOURCES/S5.md"] = source()
        files["10_SOURCES/S6.md"] = source()
        files["10_SOURCES/S3.md"] = source("Compare [[S5]] and [[S6]].")
        files["10_SOURCES/S2.md"] = source("Compare [[S5]].")
        vault = Planted(files)
        self.addCleanup(vault.close)
        data = vault.queue()
        extract = [i["subject"] for i in data["items"] if i["kind"] == "extract"]
        self.assertEqual(extract, ["10_SOURCES/S5.md", "10_SOURCES/S6.md"])
        self.assertEqual(data["counts"]["extract"], {"listed": 2, "all": 3})  # S4 is the third, below the median


if __name__ == "__main__":
    unittest.main()
