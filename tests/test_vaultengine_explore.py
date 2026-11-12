"""`vault explore`, its memory (mark, kept, --fresh, keep rates), link-it / unlink and map history (--record, --since).

Most tests run the real commands on a copy of the sample vault (tests/fixtures/vault), each class with its own copy
and memory file, so what one remembers cannot leak into another.
"""

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from vaultengine.cli import main as cli_main
from vaultengine.explore import REGISTRY, Explorer, Result, Section
from vaultengine.linkit import clean_reason, note_reason
from vaultengine.memory import Memory, MemoryStoreError

from embedding_model import requires_model
from test_vaultengine_embedding import NO_ENGINE_ENV

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "vault"
NOTE = "kl-penalty-in-rlhf"


def run(*args, columns="130"):
    out, err = io.StringIO(), io.StringIO()
    with patch.dict(os.environ, {**NO_ENGINE_ENV, "COLUMNS": columns}, clear=True), redirect_stdout(out), redirect_stderr(err):
        code = cli_main([str(a) for a in args])
    return code, out.getvalue(), err.getvalue()


def notes_hash(root: Path) -> str:
    """Every file of the vault except the engine's own folder: names and bytes."""
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".vaultengine" not in p.parts):
        h.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


class SampleCopy(unittest.TestCase):
    """A fresh copy of the sample vault, indexed and embedded, per class."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "vault"
        shutil.copytree(SAMPLE, cls.root)
        code, _, err = run("embed", cls.root)
        assert code == 0, err
        cls.memory = cls.root / ".vaultengine" / "memory.sqlite"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def explore(self, *args, json_out=True):
        code, out, err = run("explore", self.root, *args, "-f", "brain", *(["--json"] if json_out else ["--color", "never"]))
        self.assertEqual(code, 0, err)
        return json.loads(out) if json_out else out


@requires_model
class ExploreOutputTests(SampleCopy):
    def test_explore_twice_gives_identical_output(self):
        first = self.explore(NOTE, json_out=False)
        second = self.explore(NOTE, json_out=False)
        self.assertEqual(first, second)
        self.assertEqual(self.explore(NOTE), self.explore(NOTE))
        self.assertEqual(self.explore(json_out=False), self.explore(json_out=False))  # the no-start proposals too

    def test_every_result_carries_its_lens_and_one_line_of_evidence(self):
        for start in (NOTE, None, "llm-as-a-judge-framework"):  # a note, no start, a missing note
            data = self.explore(*([start] if start else []))
            results = [r for s in data["sections"] for r in s["results"]]
            self.assertGreater(len(results), 5, start)
            for s in data["sections"]:
                for r in s["results"]:
                    self.assertEqual(r["lens"], s["lens"])
                    self.assertIn(r["lens"], REGISTRY)
                    self.assertTrue(r["evidence"].strip(), r)
                    self.assertNotIn("\n", r["evidence"])
                    self.assertRegex(r["ref"], r"^\d+\.\d+$")
        text = self.explore(NOTE, json_out=False).splitlines()
        refs = [i for i, line in enumerate(text) if re.match(r"^\s+\d+\.\d+  ", line)]
        self.assertTrue(refs)
        for i in refs:  # the line under each result: "<lens> · <evidence>"
            lens, _, evidence = text[i + 1].strip().partition(" · ")
            self.assertIn(lens, REGISTRY, text[i + 1])
            self.assertTrue(evidence)

    def test_no_near_duplicates_within_a_section(self):
        from vaultengine.cli import _explorer, _tag_rule
        from vaultengine.index import VaultIndex, default_db_path
        from vaultengine.settings import load_settings

        data = self.explore(NOTE)
        cut = data["duplicate_cutoff"]
        settings = load_settings([("vault", str(self.root), "test"), ("format", "brain", "test")], env={})
        args = type("Args", (), {"settings": settings, "vault": str(self.root), "format": "brain"})()
        with VaultIndex(default_db_path(self.root)) as index:
            explorer = _explorer(args, index, _tag_rule(args), None)
            self.assertAlmostEqual(explorer.duplicate_cutoff, cut)
            dropped = 0
            for start in (NOTE, None, "reward-hacking-as-specification-gaming"):
                for s in self.explore(*([start] if start else []))["sections"]:
                    dropped += s["duplicates"]
                    for i, a in enumerate(s["results"]):
                        for b in s["results"][i + 1:]:
                            self.assertNotEqual(a["target"], b["target"])
                            self.assertFalse(set(a["notes"]) & set(b["notes"]), (a["title"], b["title"]))
                            for x in a["notes"]:
                                for y in b["notes"]:
                                    sim = explorer.similarity(x, y)
                                    self.assertTrue(sim is None or sim < cut, (s["lens"], x, y, sim))
        self.assertGreater(dropped, 0)  # the rule did leave something out: not a vacuous pass

    def test_deepen_and_discover_get_equal_space(self):
        data = self.explore(NOTE)
        kinds = {REGISTRY[s["lens"]].kind for s in data["sections"]}
        self.assertEqual(kinds, {"deepen", "discover"})
        budget = {k: sum(s["budget"] for s in data["sections"] if REGISTRY[s["lens"]].kind == k) for k in kinds}
        self.assertEqual(budget["deepen"], budget["discover"])
        self.assertTrue(all(3 <= s["budget"] <= 5 and len(s["results"]) <= s["budget"] for s in data["sections"]))

    def test_no_start_proposals_come_from_what_the_lenses_find(self):
        from vaultengine.cli import _explorer, _tag_rule
        from vaultengine.index import VaultIndex, default_db_path
        from vaultengine.settings import load_settings

        data = self.explore()
        self.assertEqual([s["lens"] for s in data["sections"]], ["gap", "surprise", "unwritten", "topicless"])
        settings = load_settings([("vault", str(self.root), "test"), ("format", "brain", "test")], env={})
        args = type("Args", (), {"settings": settings, "vault": str(self.root), "format": "brain"})()
        with VaultIndex(default_db_path(self.root)) as index:
            x = _explorer(args, index, _tag_rule(args), None)
            d = x.discover
            gaps = d.gaps(True, None, 0)["results"]
            surprises = {(r["a"], r["b"]) for r in d.surprises(True, None, 0)["results"]}
            unwritten = {f"[[{r['names'][0]}]]" for r in d.unwritten(True, None, 0)["results"]}
            bare = {c.name for c in x.shape.communities if not c.topic_pages}
        joins = {(g["join"][0]["a"], g["join"][0]["b"]) for g in gaps if g["join"]}
        by_lens = {s["lens"]: s["results"] for s in data["sections"]}
        self.assertTrue(all(r["kind"] == "pair" and tuple(r["notes"]) in joins for r in by_lens["gap"] if r["notes"]))
        self.assertTrue(all(tuple(r["notes"]) in surprises for r in by_lens["surprise"]))
        self.assertTrue(all(r["target"] in unwritten for r in by_lens["unwritten"]))
        self.assertTrue(all(r["target"].removeprefix("community: ") in bare for r in by_lens["topicless"]))
        for s in data["sections"]:
            self.assertTrue(s["results"], s["lens"])
            self.assertTrue(all(r["start_at"] for r in s["results"]), s["lens"])  # each says where it would start


@requires_model
class MemoryTests(SampleCopy):
    def refs(self, data, lens):
        return [r for s in data["sections"] if s["lens"] == lens for r in s["results"]]

    def test_a_dismissed_note_stays_gone_across_sessions_with_fresh(self):
        data = self.explore("reward-hacking-as-specification-gaming")
        first = self.refs(data, "hidden")[0]
        code, out, err = run("mark", self.root, "dismiss", first["ref"])
        self.assertEqual(code, 0, err)
        self.assertIn("dismissed", out)
        # Later sessions: separate processes' worth of opening the memory file, --fresh leaves it out everywhere.
        for _ in range(2):
            fresh = self.explore("reward-hacking-as-specification-gaming", "--fresh")
            targets = {r["target"] for s in fresh["sections"] for r in s["results"]}
            self.assertNotIn(first["target"], targets)
        # Without --fresh it is there, marked.
        plain = self.explore("reward-hacking-as-specification-gaming")
        again = [r for r in self.refs(plain, "hidden") if r["target"] == first["target"]]
        self.assertEqual([r["mark"] for r in again], ["dismiss"])
        text = self.explore("reward-hacking-as-specification-gaming", json_out=False)
        self.assertIn("[dismissed]", text)
        with Memory(self.memory) as memory:  # remembered in the engine's store
            self.assertIn(first["target"], memory.dismissed_targets("20_ZETTLEKASTEN/reward-hacking-as-specification-gaming.md"))

    def test_fresh_hides_what_was_seen_and_fills_from_further_down(self):
        start = "obfuscated-reward-hacking"
        seen = self.explore(start)
        fresh = self.explore(start, "--fresh")
        for s_seen, s_fresh in zip(sorted(seen["sections"], key=lambda s: s["lens"]), sorted(fresh["sections"], key=lambda s: s["lens"])):
            before = {r["target"] for r in s_seen["results"]}
            self.assertFalse(before & {r["target"] for r in s_fresh["results"]}, s_seen["lens"])
            if s_seen["found"] > len(before) + s_seen["duplicates"]:  # there was more to show
                self.assertTrue(s_fresh["results"] or s_fresh["duplicates"], s_seen["lens"])

    def test_the_keep_rate_orders_the_sections_and_is_shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_file = Path(tmp) / "memory.sqlite"
            setting = ("--setting", f"memory={memory_file}")
            code, out, err = run("explore", self.root, NOTE, "-f", "brain", "--json", *setting)
            before = [s["lens"] for s in json.loads(out)["sections"]]
            self.assertEqual(before, list(REGISTRY_DEFAULT))  # no marks: the configured order
            # keep twice from role (section 7), dismiss twice from cluster (section 5)
            run("mark", self.root, "keep", "7.1", "7.2", *setting)
            run("mark", self.root, "dismiss", "5.1", "5.2", *setting)
            code, out, err = run("explore", self.root, NOTE, "-f", "brain", "--json", *setting)
            after = [s["lens"] for s in json.loads(out)["sections"]]
            self.assertEqual(after[0], "role")
            self.assertEqual(after[-1], "cluster")
            self.assertEqual(after[1:-1], [n for n in before if n not in ("role", "cluster")])  # ties keep the config order
            code, text, _ = run("explore", self.root, NOTE, "-f", "brain", "--color", "never", *setting)
            self.assertIn("Same role elsewhere  (discover · ", text)
            self.assertIn("kept 2 of 2 you marked (100%)", text.split("\n  1. ")[1].split("\n")[0])
            self.assertIn("kept 0 of 2 you marked (0%)", [l for l in text.splitlines() if "Cluster  (" in l][0])

    def test_kept_lists_lens_and_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            setting = ("--setting", f"memory={Path(tmp) / 'm.sqlite'}")
            code, out, _ = run("explore", self.root, NOTE, "-f", "brain", "--json", *setting)
            first = json.loads(out)["sections"][0]["results"][0]
            run("mark", self.root, "keep", first["ref"], *setting)
            code, out, _ = run("kept", self.root, "--json", *setting)
            kept = json.loads(out)["kept"]
            self.assertEqual([(k["lens"], k["target"], k["evidence"]) for k in kept],
                             [(first["lens"], first["target"], first["evidence"])])
            code, text, _ = run("kept", self.root, *setting)
            self.assertIn(f"    {first['lens']} · {first['evidence']}", text)
            code, _, err = run("mark", self.root, "keep", "99.9", *setting)
            self.assertEqual(code, 2)
            self.assertIn("no result 99.9 in the last explore run", err)


REGISTRY_DEFAULT = ("neighbourhood", "cited", "foundations", "backbone", "cluster", "hidden", "role", "gap", "surprise", "unwritten")


@requires_model
class LinkItTests(SampleCopy):
    def check_count(self):
        code, out, _ = run("check", self.root, "-f", "brain", "--json")
        return json.loads(out)["summary"]["violations"]

    def test_link_it_writes_both_sides_passes_check_and_unlink_removes_both(self):
        original = {p: (self.root / p).read_bytes() for p in ("20_ZETTLEKASTEN/kl-penalty-in-rlhf.md",)}
        violations = self.check_count()
        data = self.explore(NOTE)
        hidden = next(r for s in data["sections"] if s["lens"] == "hidden" for r in s["results"])
        a, b = hidden["link"]
        before = {p: (self.root / p).read_bytes() for p in (a, b)}
        untouched = notes_hash(self.root)
        code, out, err = run("link-it", self.root, hidden["ref"], "-f", "brain")
        self.assertEqual(code, 0, err)
        self.assertIn("Dry run: nothing was written.", out)
        self.assertEqual(notes_hash(self.root), untouched)  # a dry run writes nothing
        code, out, err = run("link-it", self.root, hidden["ref"], "-f", "brain", "--write")
        self.assertEqual(code, 0, err)
        self.assertIn("check    no new violations", out)
        name = lambda p: Path(p).stem
        for path, other in ((a, b), (b, a)):
            text = (self.root / path).read_text()
            line = next(l for l in text.splitlines() if l.startswith(f"- [[{name(other)}]] — "))
            self.assertIn("hidden link: ", line)
            self.assertNotIn("#", line.partition(" — ")[2])
            self.assertIn("See Also", text)
        self.assertEqual(self.check_count(), violations)  # vault check: nothing new
        code, out, _ = run("related", self.root, a, "--json")
        self.assertTrue(next(h for h in json.loads(out)["hits"] if h["path"] == b)["linked"])
        with Memory(self.memory) as memory:
            self.assertEqual(memory.kept()[-1]["target"], hidden["target"])  # a link made counts as a keep
        code, out, err = run("unlink", self.root, name(a), name(b), "-f", "brain")
        self.assertIn("Dry run: nothing was written.", out)
        self.assertEqual({p: (self.root / p).read_bytes() != before[p] for p in (a, b)}, {a: True, b: True})
        code, out, err = run("unlink", self.root, name(a), name(b), "-f", "brain", "--write")
        self.assertEqual(code, 0, err)
        self.assertEqual({p: (self.root / p).read_bytes() for p in (a, b)}, before)  # byte for byte
        self.assertEqual(self.check_count(), violations)
        self.assertEqual(original["20_ZETTLEKASTEN/kl-penalty-in-rlhf.md"], (self.root / "20_ZETTLEKASTEN/kl-penalty-in-rlhf.md").read_bytes())

    def test_link_it_under_an_existing_section_and_no_double_link(self):
        # ppo-as-trpo-simplification has a "# See Also" with five links already: the new line goes after them.
        path = self.root / "20_ZETTLEKASTEN/ppo-as-trpo-simplification.md"
        other = self.root / f"20_ZETTLEKASTEN/{NOTE}.md"
        originals = {path: path.read_bytes(), other: other.read_bytes()}
        try:
            code, out, err = run("link-it", self.root, "ppo-as-trpo-simplification", NOTE, "-f", "brain",
                                 "--reason", "per-token #kl and [[x]] $a$", "--write")
            self.assertEqual(code, 0, err)
            lines = path.read_text().splitlines()
            at = lines.index("- [[natural-gradient-step-size-adaptation]]")
            self.assertEqual(lines[at + 1], f"- [[{NOTE}]] — per-token kl and x a")  # under it, cleaned
            self.assertEqual(lines.count("# See Also"), 1)
            self.assertIn("# See Also\n- [[ppo-as-trpo-simplification]] — per-token kl and x a", other.read_text())
            code, out, _ = run("link-it", self.root, "ppo-as-trpo-simplification", NOTE, "-f", "brain", "--write")
            self.assertIn("Nothing to write.", out)  # both already link each other there
            code, out, err = run("unlink", self.root, "ppo-as-trpo-simplification", NOTE, "-f", "brain", "--write")
            self.assertEqual(code, 0, err)
            self.assertEqual({p: p.read_bytes() for p in originals}, originals)  # its section and lines stay, the rest goes
        finally:
            for p, data in originals.items():
                p.write_bytes(data)
            run("index", self.root)

    def test_unlink_gives_back_a_note_without_a_final_newline(self):
        # The neighbourhood result kl-penalty-in-rlhf ↔ RLHF: 10_SOURCES/RLHF.md has no final newline. Link it, write,
        # unlink: both notes byte for byte as before (RLHF.md used to come back one newline longer).
        a, b = self.root / f"20_ZETTLEKASTEN/{NOTE}.md", self.root / "10_SOURCES/RLHF.md"
        originals = {a: a.read_bytes(), b: b.read_bytes()}
        self.assertFalse(originals[b].endswith(b"\n"))
        with tempfile.TemporaryDirectory() as tmp:
            setting = ("--setting", f"memory={Path(tmp) / 'm.sqlite'}")
            try:
                code, out, err = run("explore", self.root, NOTE, "-f", "brain", "--json", *setting)
                self.assertEqual(code, 0, err)
                result = next(r for s in json.loads(out)["sections"] if s["lens"] == "neighbourhood"
                              for r in s["results"] if r["link"] and "10_SOURCES/RLHF.md" in r["link"])
                code, out, err = run("link-it", self.root, result["ref"], "-f", "brain", "--write", *setting)
                self.assertEqual(code, 0, err)
                self.assertIn("\n\n# See Also\n- [[kl-penalty-in-rlhf]] — neighbourhood: ", b.read_text(encoding="utf-8"))
                self.assertFalse(b.read_bytes().endswith(b"\n"))  # still no final newline
                code, out, err = run("unlink", self.root, NOTE, "RLHF", "-f", "brain", "--write", *setting)
                self.assertEqual(code, 0, err)
                self.assertEqual({p: p.read_bytes() for p in originals}, originals)
            finally:
                for p, data in originals.items():
                    p.write_bytes(data)
                run("index", self.root)

    def test_a_result_that_is_not_two_notes_cannot_be_linked(self):
        data = self.explore()
        missing = next(r for s in data["sections"] if s["lens"] == "unwritten" for r in s["results"])
        code, _, err = run("link-it", self.root, missing["ref"], "-f", "brain")
        self.assertEqual(code, 2)
        self.assertIn("is not two notes to link", err)


@requires_model
class NothingWrittenTests(SampleCopy):
    def test_reads_write_no_note_and_map_writes_no_memory_without_record(self):
        untouched = notes_hash(self.root)
        self.assertFalse(self.memory.exists())
        run("map", self.root, "-f", "brain", "--json")
        self.assertFalse(self.memory.exists())  # a read does not write, not even the engine's store
        self.explore(NOTE, "--no-record")
        self.assertFalse(self.memory.exists())
        self.explore(NOTE)
        self.explore()
        data = self.explore(NOTE)
        run("mark", self.root, "keep", data["sections"][0]["results"][0]["ref"])
        run("kept", self.root)
        run("link-it", self.root, data["sections"][0]["results"][0]["ref"], "-f", "brain")  # dry run
        code, _, _ = run("map", self.root, "-f", "brain", "--record", "--json")
        self.assertEqual(code, 0)
        stamp = self.memory.read_bytes()
        code, out, err = run("map", self.root, "-f", "brain", "--since", "2020-01-01", "--json")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.memory.read_bytes(), stamp)  # --since only reads
        self.assertEqual(notes_hash(self.root), untouched)


@requires_model
class MapHistoryTests(SampleCopy):
    def test_since_reports_a_gap_closed_after_a_link(self):
        code, out, err = run("map", self.root, "-f", "brain", "--record", "--json")
        self.assertEqual(code, 0, err)
        recorded = json.loads(out)["recorded"]
        data = self.explore()
        gap = next(r for s in data["sections"] if s["lens"] == "gap" for r in s["results"] if r["link"])
        code, out, err = run("map", self.root, "-f", "brain", "--since", recorded["at"][:10], "--json")
        before = json.loads(out)
        self.assertTrue(all(g["status"] == "open" for g in before["gaps"]))  # nothing changed yet
        code, out, err = run("link-it", self.root, gap["ref"], "-f", "brain", "--write")
        self.assertEqual(code, 0, err)
        code, out, err = run("map", self.root, "-f", "brain", "--since", recorded["at"][:10], "--json")
        after = json.loads(out)
        closed = [g for g in after["gaps"] if g["status"] == "closed"]
        self.assertEqual(len(closed), 1, after["gaps"])
        self.assertEqual(closed[0]["added"], [sorted(gap["link"])])
        self.assertGreaterEqual(closed[0]["links_now"], closed[0]["expected_now"])
        code, text, _ = run("map", self.root, "-f", "brain", "--since", recorded["at"][:10], "--color", "never")
        self.assertIn("(1 of", text)
        self.assertIn("    closed  communities", text)

    def test_since_reports_a_missing_note_now_written(self):
        run("map", self.root, "-f", "brain", "--record")
        target = self.root / "20_ZETTLEKASTEN" / "reward-model-overoptimization-hump.md"
        target.write_text("# reward-model-overoptimization-hump\n\nWritten now.\n")
        try:
            run("index", self.root)
            code, out, err = run("map", self.root, "-f", "brain", "--since", "2000-01-01", "--json")
            self.assertEqual(code, 0, err)
            written = json.loads(out)["written"]
            self.assertIn({"name": "reward-model-overoptimization-hump", "linked_from": 12,
                           "note": "20_ZETTLEKASTEN/reward-model-overoptimization-hump.md"}, written)
        finally:
            target.unlink()
            run("index", self.root)


class UnitTests(unittest.TestCase):
    def test_reason_is_one_clean_line(self):
        self.assertEqual(clean_reason("a\n b  #tag [[x]] $y$ %%c%%"), "a b tag x y c")
        long = clean_reason("word " * 100)
        self.assertLessEqual(len(long), 160)
        self.assertTrue(long.endswith("…"))
        self.assertEqual(note_reason("Hidden link", "no link either way; 3 middles in common: 'kl' (4 notes); score 2.10"
                                                    " (p95 of 10 unlinked pairs)"), "hidden link: 3 middles in common: 'kl' (4 notes)")

    def test_budget_is_equal_per_kind_and_between_three_and_five(self):
        settings = type("S", (), {"per_lens": 4})()
        x = Explorer.__new__(Explorer)
        x.settings = settings
        make = lambda names: [Section(REGISTRY[n]) for n in names]
        five_five = make(REGISTRY_DEFAULT)
        x.budget(five_five, five_five)
        self.assertEqual({s.budget for s in five_five}, {4})
        two_six = make(["neighbourhood", "cluster", "hidden", "role", "gap", "surprise", "unwritten", "bridge"])
        x.budget(two_six, two_six)  # 32 results, 16 each: deepen 8 a section capped at 5, discover 16 / 6 -> 3 (floor 3)
        self.assertEqual([s.budget for s in two_six], [5, 5, 3, 3, 3, 3, 3, 3])

    def test_memory_refs_and_snapshot_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Memory(Path(tmp) / "m.sqlite") as memory:
                with self.assertRaisesRegex(MemoryStoreError, "no explore run yet"):
                    memory.result("1.1")
                memory.record_session("a.md", False, ["hidden"], [Result("hidden", "note", "b.md", "b", "why", ("b.md",), ("a.md", "b.md"), ref="1.1").to_dict()])
                r = memory.result("1.1")
                self.assertEqual((r["start"], r["link"], r["notes"]), ("a.md", ("a.md", "b.md"), ["b.md"]))
                self.assertEqual(memory.seen_targets("a.md"), {"b.md"})
                with self.assertRaisesRegex(MemoryStoreError, "no map snapshot yet"):
                    memory.snapshot_since("2026-01-01")
                memory.conn.execute("INSERT INTO snapshots (at, data) VALUES ('2026-01-05T10:00:00', x'00'), ('2026-01-07T10:00:00', x'00')")
                self.assertEqual(memory.snapshot_since("2026-01-06")[1], "2026-01-07T10:00:00")  # the first at or after
                self.assertEqual(memory.snapshot_since("2026-01-01")[1], "2026-01-05T10:00:00")
                self.assertEqual(memory.snapshot_since("2026-02-01")[1], "2026-01-07T10:00:00")  # else the last before
                with self.assertRaisesRegex(MemoryStoreError, "give a date"):
                    memory.snapshot_since("last week")

    def test_lens_settings_are_validated(self):
        from vaultengine.settings import SettingsError, load_settings

        with self.assertRaisesRegex(SettingsError, "unknown lens 'nope'"):
            load_settings([("explore.lenses", "neighbourhood,nope", "test")], env={})
        with self.assertRaisesRegex(SettingsError, "need a starting point"):
            load_settings([("explore.start_lenses", "gap,hidden", "test")], env={})
        with self.assertRaisesRegex(SettingsError, "only run without a starting point"):
            load_settings([("explore.lenses", "gap,topicless", "test")], env={})
        with self.assertRaisesRegex(SettingsError, "per_lens"):
            load_settings([("explore.per_lens", "6", "test")], env={})


PARTNER = "# Partner\n\nA note to link with.\n"

# Each note, by what its ending or See Also section is like: link-it then unlink must give back these bytes.
ROUND_TRIP = {
    "no final newline": "# Title\n\nSome text, and no line break after it.",
    "CRLF line endings": "# Title\r\n\r\nSome text.\r\nMore text.\r\n",
    "a See Also section already": "# Title\n\nSome text.\n\n## See Also\n- [[elsewhere]]\n",
    "CRLF and no final newline": "# Title\r\n\r\nSome text.",
    "a See Also section, last, no final newline": "# Title\n\nText.\n\n## See Also\n- [[elsewhere]]",
    "a See Also section, CRLF, then another section": "# Title\r\n\r\n## See Also\r\n- [[elsewhere]]\r\n\r\n## Notes\r\nmore\r\n",
    "an empty See Also heading, no final newline": "# Title\n\nText.\n\n## See Also",
    "a blank line at the end": "# Title\n\nSome text.\n\n",
    "a blank line at the end, CRLF": "# Title\r\n\r\nSome text.\r\n\r\n",
    "spaces on its last line, no final newline": "# Title\n\nSome text.\n   ",
    "a byte order mark and CRLF": "\ufeff# Title\r\n\r\nSome text.\r\n",
    "no heading, one line, no final newline": "Just one line.",
}


class UnlinkRoundTripTests(unittest.TestCase):
    """link-it --write then unlink --write gives back both notes byte for byte, whatever their line endings, their
    ending, and whether they had a See Also section. Each case links a note of that kind with a plain partner, and
    with a second note of the same kind (both sides of the write)."""

    def round_trip(self, files: dict[str, str], a: str, b: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "vault"
            root.mkdir()
            for name, text in files.items():
                (root / name).write_bytes(text.encode("utf-8"))
            original = {name: (root / name).read_bytes() for name in files}
            setting = ("--setting", f"memory={Path(tmp) / 'memory.sqlite'}")
            self.assertEqual(run("index", root)[0], 0)
            code, out, err = run("link-it", root, a, b, "-f", "brain", "--write", *setting)
            self.assertEqual(code, 0, err)
            for name, other in ((a, b), (b, a)):
                linked = (root / f"{name}.md").read_bytes()
                self.assertIn(f"- [[{other}]] — linked by hand (vault link-it)".encode(), linked)
                if b"\r\n" in original[f"{name}.md"]:
                    self.assertNotRegex(linked, rb"[^\r]\n", "a CRLF note got a bare LF line")
                self.assertEqual(linked.endswith((b"\n", b"\r")), original[f"{name}.md"].endswith((b"\n", b"\r")),
                                 "link-it changed whether the note ends with a line break")
            code, out, err = run("unlink", root, a, b, "-f", "brain", "--write", *setting)
            self.assertEqual(code, 0, err)
            self.assertEqual({name: (root / name).read_bytes() for name in files}, original)

    def test_each_kind_of_note_comes_back_byte_for_byte(self):
        for kind, text in ROUND_TRIP.items():
            with self.subTest(kind=kind, partner="plain"):
                self.round_trip({"note.md": text, "partner.md": PARTNER}, "note", "partner")
            with self.subTest(kind=kind, partner="the same kind"):
                self.round_trip({"one.md": text, "two.md": text}, "one", "two")

    def test_a_memory_file_of_schema_1_gains_the_blank_line_columns(self):
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite"
            conn = sqlite3.connect(path)
            conn.executescript(
                "CREATE TABLE links (id INTEGER PRIMARY KEY, at TEXT NOT NULL, a TEXT NOT NULL, b TEXT NOT NULL, lens TEXT,"
                " section TEXT NOT NULL, line_a TEXT NOT NULL, line_b TEXT NOT NULL, created_a INTEGER NOT NULL,"
                " created_b INTEGER NOT NULL, undone TEXT);"
                "INSERT INTO links VALUES (1, '2026-01-01T00:00:00', 'a.md', 'b.md', NULL, 'See Also', '- [[b]]', '- [[a]]', 1, 0, NULL);"
                "PRAGMA user_version = 1;")
            conn.close()
            with Memory(path, readonly=True) as memory:  # read as it is, not changed
                self.assertEqual(memory.link_between("b.md", "a.md")["created_b"], 1)
            with Memory(path) as memory:
                self.assertEqual(memory.conn.execute("PRAGMA user_version").fetchone()[0], 2)
                old = memory.link_between("b.md", "a.md")
                self.assertEqual((old["blank_a"], old["blank_b"]), (None, None))  # not recorded then
                memory.record_link("c.md", "d.md", None, "See Also", "- [[d]]", "- [[c]]", True, False, True, False)
                new = memory.link_between("d.md", "c.md")
                self.assertEqual((new["created_a"], new["created_b"], new["blank_a"], new["blank_b"]), (0, 1, 0, 1))


@requires_model
class RelatedWeightsTests(SampleCopy):
    def test_plain_related_keeps_linked_notes_where_they_were_and_unlinked_uses_the_measured_mix(self):
        code, out, _ = run("related", self.root, NOTE, "--limit", "3", "--color", "never")
        self.assertIn("score = 0.5 × graph + 0.3 × text + 0.2 × tags", out)
        self.assertRegex(out, r"1\. direct-preference-optimization-dpo +0\.87 +linked")
        code, out, _ = run("related", self.root, NOTE, "--limit", "3", "--unlinked", "--color", "never")
        self.assertIn("score = 0.1 × graph + 0.05 × cocitation + 0.05 × coupling + 0.7 × text + 0.1 × tags", out)
        self.assertRegex(out, r"1\. RLHF +0\.74 +not linked")


if __name__ == "__main__":
    unittest.main()
