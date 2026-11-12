"""The vault-wide numbers `vault explore` keeps between runs (stats.py): the same output with and without them,
when they are reused, and that a reused run recomputes what is about particular notes."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from vaultengine.index import VaultIndex, default_db_path
from vaultengine.operations import Workbench
from vaultengine.settings import load_settings
from vaultengine.stats import Stats, fingerprint

from approx import assert_same
from embedding_model import requires_model

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "vault"


def state(**notes):
    return {f"{name}.md": digest for name, digest in notes.items()}


class StatsUnitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "stats.sqlite"
        self.calls = []

    def compute(self, name, value):
        def run():
            self.calls.append(name)
            return value
        return run

    def stats(self, current, share=0.02, refresh=False, key="k"):
        return Stats(self.path, key, current, share, refresh)

    def test_exact_reuses_everything_reused_only_the_tolerant_and_fresh_nothing(self):
        base = state(**{f"n{i}": f"h{i}" for i in range(100)})
        s = self.stats(base)
        self.assertEqual(s.mode, "fresh")
        s.get("dist", self.compute("dist", [1, 2, 3]), tolerant=True)
        s.get("betweenness", self.compute("betweenness", {1: 0.5}))
        s.save()
        s = self.stats(base)
        self.assertEqual(s.mode, "exact")
        self.assertEqual((s.get("dist", self.compute("dist", None), tolerant=True), s.get("betweenness", self.compute("b", None))),
                         ([1, 2, 3], {1: 0.5}))
        self.assertEqual(self.calls, ["dist", "betweenness"])  # nothing computed again
        edited = dict(base, **{"n5.md": "changed"})  # 1 of 100 notes: at most 2 (2%) reuse
        s = self.stats(edited)
        self.assertEqual((s.mode, s.changed), ("reused", {"n5.md"}))
        self.assertEqual(s.get("dist", self.compute("dist", None), tolerant=True), [1, 2, 3])  # a distribution: reused
        self.assertEqual(s.get("betweenness", self.compute("betweenness", {1: 0.7})), {1: 0.7})  # about notes: now
        s.save()
        s = self.stats(edited)  # the same state again: the overlay has what was computed for it
        self.assertEqual(s.get("betweenness", self.compute("betweenness", None)), {1: 0.7})
        self.assertEqual(self.calls, ["dist", "betweenness", "betweenness"])
        many = dict(base, **{f"n{i}.md": "changed" for i in range(10)})
        self.assertEqual(self.stats(many).mode, "fresh")  # 10 changed: more than 2
        self.assertEqual(self.stats(base, refresh=True).mode, "fresh")
        self.assertEqual(self.stats(base, key="other settings").mode, "fresh")

    def test_a_run_saving_twice_keeps_both(self):
        base = state(a="1")
        s = self.stats(base)
        s.get("one", self.compute("one", 1))
        s.save()
        s.get("two", self.compute("two", 2))
        s.save()  # a second run in the same process: adds to the set, does not replace it
        s = self.stats(base)
        self.assertEqual((s.mode, s.get("one", self.compute("x", None)), s.get("two", self.compute("y", None))), ("exact", 1, 2))

    def test_key_terms_reused_for_unchanged_notes_only(self):
        base = state(a="1", b="2", c="3")
        s = self.stats(base, share=0.5)
        s.key_terms(lambda: {"a.md": {"x": 1.0}, "b.md": {"y": 1.0}, "c.md": {"z": 1.0}}, None)
        s.save()
        s = self.stats(dict(state(a="1", b="changed"), **{"d.md": "4"}), share=1.0)  # b edited, c deleted, d new
        self.assertEqual(s.mode, "reused")
        terms = s.key_terms(lambda: self.fail("not everything"), lambda path: {"new": 1.0})
        self.assertEqual(terms, {"a.md": {"x": 1.0}, "b.md": {"new": 1.0}, "d.md": {"new": 1.0}})

    def test_the_fingerprint_names_the_settings(self):
        self.assertEqual(fingerprint(seed=42, model="m"), fingerprint(model="m", seed=42))
        self.assertNotEqual(fingerprint(seed=42), fingerprint(seed=43))


@requires_model
class StatsOnTheSampleTests(unittest.TestCase):
    """Explore with the kept numbers gives what explore without them gives, for every kind of start."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "vault"
        shutil.copytree(SAMPLE, cls.root)
        with VaultIndex(default_db_path(cls.root)) as ix:
            ix.update(cls.root)
        cls.settings = load_settings([("vault", str(cls.root), "test"), ("format", "brain", "test")], env={})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        (self.root / ".vaultengine" / "stats.sqlite").unlink(missing_ok=True)  # each test starts with none kept

    def runs(self, keep_stats, starts):
        from vaultengine.explore import REGISTRY

        out = {}
        lenses = [name for name, lens in REGISTRY.items() if lens.from_start]
        with VaultIndex(default_db_path(self.root)) as ix:
            work = Workbench(ix.conn, self.root, self.settings, "brain")
            explorer = work.explorer(None, keep_stats=keep_stats)
            for start in starts:
                run = explorer.run(start, lenses if start else self.settings.engine.explore.start_lenses)
                out[start] = [(s.lens.name, s.budget, s.duplicates, [r.to_dict() for r in s.results]) for s in run["sections"]]
                mode = run["stats"]["mode"]
        return out, mode

    def test_same_output_without_kept_numbers_with_fresh_ones_and_with_stored_ones(self):
        paths = sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*.md"))
        starts = [None, "rl-environment-design", "llm-as-a-judge-framework"] + paths[::6]  # 30 notes, 2 missing, none
        plain, mode = self.runs(False, starts)
        self.assertEqual(mode, "off")
        first, mode = self.runs(True, starts)
        self.assertEqual(mode, "fresh")
        stored, mode = self.runs(True, starts)
        self.assertEqual(mode, "exact")
        self.assertEqual(first, plain)
        self.assertEqual(stored, plain)

    def test_a_small_edit_reuses_the_numbers_and_says_so(self):
        from vaultengine.cli import main as cli_main
        import io
        from contextlib import redirect_stdout

        def explore(*extra):
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli_main(["explore", str(self.root), "kl-penalty-in-rlhf", "-f", "brain", "--json", "--no-record",
                                 "--timing", *extra])
            self.assertEqual(code, 0)
            return json.loads(out.getvalue())

        explore()
        note = self.root / "20_ZETTLEKASTEN" / "ppo-as-trpo-simplification.md"
        original = note.read_bytes()
        try:
            note.write_bytes(original + b"\nOne more line.\n")
            with VaultIndex(default_db_path(self.root)) as ix:
                ix.update(self.root)
            data = explore()
            self.assertEqual((data["numbers"]["source"], data["numbers"]["changed_since"], data["numbers"]["stored"]),
                             ("reused", 1, "reused"))
            self.assertEqual(explore("--refresh")["numbers"]["source"], "current")
        finally:
            note.write_bytes(original)
            with VaultIndex(default_db_path(self.root)) as ix:
                ix.update(self.root)


if __name__ == "__main__":
    unittest.main()


@requires_model
class GoldenTests(unittest.TestCase):
    """Every explore answer on the sample vault - from all 180 notes, the 100 missing notes linked three or more
    times, and without a start, with all twelve lenses - equals what explore gave before the vault-wide numbers were
    kept and before betweenness, constraint and HITS were rewritten (tests/fixtures/explore_golden.json.gz, made by
    that code). Once computing everything, then from the numbers stored on disk."""

    LENSES = ["neighbourhood", "cited", "foundations", "backbone", "cluster", "hidden", "role", "bridge", "gap",
              "surprise", "unwritten", "wander"]

    @staticmethod
    def serialise(out):
        return {"start": out["start"], "cut": out["duplicate_cutoff"], "sections": [
            {"lens": s.lens.name, "budget": s.budget, "found": s.candidates, "dup": s.duplicates, "note": s.note,
             "results": [r.to_dict() for r in s.results]} for s in out["sections"]]}

    def test_every_start_on_the_sample_is_unchanged(self):
        import gzip

        golden = json.loads(gzip.decompress((SAMPLE.parent / "explore_golden.json.gz").read_bytes()))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "vault"
            shutil.copytree(SAMPLE, root)
            with VaultIndex(default_db_path(root)) as ix:
                ix.update(root)
                settings = load_settings([("vault", str(root), "test"), ("format", "brain", "test")], env={})
                for pass_, starts in (("computed", sorted(golden)), ("stored", sorted(golden)[::7])):
                    explorer = Workbench(ix.conn, root, settings, "brain").explorer(None)
                    for start in starts:  # keys: a path, "[[missing note]]", or "" (no start)
                        ref = start[2:-2] if start.startswith("[[") else start
                        out = explorer.run(ref or None, self.LENSES if start else settings.engine.explore.start_lenses)
                        with self.subTest(numbers=pass_, start=start):
                            assert_same(self, json.loads(json.dumps(self.serialise(out), sort_keys=True)), golden[start])
                    self.assertEqual(out["stats"]["mode"], "fresh" if pass_ == "computed" else "exact")
