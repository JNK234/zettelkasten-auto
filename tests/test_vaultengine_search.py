"""Full-text search: tokenizer, field extraction, postings storage, incremental upkeep, BM25F."""

import io
import json
import math
import os
import random
import tempfile
import unicodedata
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from vaultengine import index as index_module
from vaultengine import search as search_module
from vaultengine.cli import main as cli_main
from vaultengine.index import VaultIndex, default_db_path
from vaultengine.parser import parse_note_with_layout
from vaultengine.search import FIELD_WEIGHTS, VALUE_GAP, Near, Phrase, QueryError, Word, near_pairs, note_fields, parse_query, postings, search
from vaultengine.snippets import hit_context
from vaultengine.spelling import close_terms, edit_budget, osa_distance
from vaultengine.tokenize import tokenize, word_segments, word_spans

try:
    from uniseg.wordbreak import words as uniseg_words
except ImportError:  # dev dependency; `uv sync` installs it
    uniseg_words = None

FIXTURES = Path(__file__).parent / "fixtures"
OLD_TIME = 1_600_000_000


# ----- tokenizer ---------------------------------------------------------------


class TokenizerTests(unittest.TestCase):
    def test_lowercases_and_splits_on_word_boundaries(self):
        self.assertEqual(tokenize("Reward Hacking, as Specification-Gaming!"), ["reward", "hacking", "as", "specification", "gaming"])

    def test_keeps_numbers_and_uax29_joins(self):
        self.assertEqual(tokenize("2503.11926 - Monitoring (CoT)"), ["2503.11926", "monitoring", "cot"])
        self.assertEqual(tokenize("3,000 steps at lr 1e-6, v1.2.3"), ["3,000", "steps", "at", "lr", "1e", "6", "v1.2.3"])
        self.assertEqual(tokenize("don't, e.g. it’s"), ["don't", "e.g", "it’s"])
        self.assertEqual(tokenize("snake_case __init__ GPT-4o"), ["snake_case", "__init__", "gpt", "4o"])

    def test_drops_punctuation_symbols_and_emoji(self):
        self.assertEqual(tokenize("a — b → c · 👍 🇩🇪 ___ #tag/nested"), ["a", "b", "c", "tag", "nested"])

    def test_quote_does_not_stick_to_the_next_word(self):
        # regex's \b gives "'active" here; UAX #29 does not join a quote after '='.
        self.assertEqual(tokenize("status='active' on 'evals'."), ["status", "active", "on", "evals"])

    def test_unicode_normalisation(self):
        self.assertEqual(tokenize("Cafe\u0301 CAFÉ café"), ["café", "café", "café"])  # NFD and NFC agree
        self.assertEqual(tokenize("co\u00adoperate"), ["cooperate"])  # soft hyphen removed
        self.assertEqual(tokenize("Straße ΣΊΣΥΦΟΣ"), ["straße", "σίσυφος"])  # lowercase (final sigma), not casefold

    def test_scripts_without_spaces_are_one_character_per_term(self):
        self.assertEqual(tokenize("東京に住む カタカナ"), ["東", "京", "に", "住", "む", "カタカナ"])

    def test_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(" -- !! \n"), [])

    ALPHABET = "aZéאב0٣カ_:·,;.'\"\u0301\u00ad\u200d =-#東に🇩👍\n\r—→–§×’"

    @unittest.skipUnless(uniseg_words, "uniseg not installed (dev dependency)")
    def test_matches_an_independent_uax29_implementation(self):
        """Random strings over one character of every Word_Break class, against uniseg."""

        def words_with_letters(segments):
            fmt = {"\u00ad"}  # the Format character in the alphabet; tokenize removes it
            kept = ["".join(c for c in s if c not in fmt) for s in segments]
            return [s for s in kept if any(unicodedata.category(c)[0] in "LN" for c in s)]

        rng = random.Random(7)
        for _ in range(30_000):
            text = "".join(rng.choice(self.ALPHABET) for _ in range(rng.randint(1, 10)))
            self.assertEqual(words_with_letters(word_segments(text)), words_with_letters(uniseg_words(text)), repr(text))

    @unittest.skipUnless(uniseg_words, "uniseg not installed (dev dependency)")
    def test_matches_independent_implementation_on_sample_vault(self):
        # ASCII text is covered by the fast-path test below; uniseg is slow, so check a
        # fixed sample of the lines that take the full rules (every line passed when run in full).
        lines = sorted({
            line
            for path in (FIXTURES / "vault").rglob("*.md")
            for line in unicodedata.normalize("NFC", path.read_text(encoding="utf-8")).split("\n")
            if not line.isascii()
        })
        for line in random.Random(3).sample(lines, 400):
            expected = [w for w in uniseg_words(line) if any(unicodedata.category(c)[0] in "LN" for c in w)]
            self.assertEqual(word_segments(line), expected, line)

    def test_ascii_fast_path_equals_full_rules(self):
        from vaultengine import tokenize as module

        rng = random.Random(11)
        alphabet = "aZq09_:.',;\"- #=\t!?()[]`"
        for _ in range(30_000):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
            fast = [w for w in module._ASCII_WORD.findall(text) if w.strip("_")]
            full = [w for w in module._WORD.findall(text) if w[0].isalnum() or not module._X_ONLY.fullmatch(w)]
            self.assertEqual(fast, full, repr(text))


# ----- fields ------------------------------------------------------------------


def fields_of(text, path="Note.md"):
    note, layout = parse_note_with_layout(text, path)
    return {name: [term for value in values for term in value] for name, values in note_fields(path, note, layout).items()}


class FieldTests(unittest.TestCase):
    def test_five_fields(self):
        text = (
            "---\naliases: [PPO, Clipped objective]\ntags: [rl/policy-gradient]\nauthor: Schulman\n---\n"
            "# Proximal Policy Optimization\n"
            "Clipped surrogate objective. #optimisation\n"
            "## Why it works ##\n"
            "See [[TRPO|trust region]] ^block-1\n"
        )
        self.assertEqual(
            fields_of(text, "Concepts/PPO paper.md"),
            {
                "title": ["ppo", "paper"],
                "aliases": ["ppo", "clipped", "objective"],
                "headings": ["proximal", "policy", "optimization", "why", "it", "works"],
                "tags": ["rl", "policy", "gradient", "optimisation"],
                "body": ["clipped", "surrogate", "objective", "see", "trpo", "trust", "region"],
            },
        )  # 'author' is not indexed; heading lines, #tags and ^block ids are not body text

    def test_code_and_comments_are_not_indexed(self):
        text = (
            "Intro `inline code` text\n"
            "```python\nsecret_fenced = 1\n```\n"
            "~~~\ntilde fence\n~~~\n"
            "## Heading with `code span`\n"
            "%% hidden comment %%\n"
            "> ```\n> quoted fence\n> ```\n"
            "Outro\n"
        )
        fields = fields_of(text)
        self.assertEqual(fields["body"], ["intro", "text", "outro"])
        self.assertEqual(fields["headings"], ["heading", "with"])

    def test_tags_inside_code_or_links_are_neither_tags_nor_removed(self):
        fields = fields_of("[[Note#Section]] and `#notatag` #real")
        self.assertEqual(fields["tags"], ["real"])
        self.assertEqual(fields["body"], ["note", "section", "and"])

    def test_each_distinct_tag_once(self):
        self.assertEqual(fields_of("#RL #rl #rl\n#other")["tags"], ["rl", "other"])

    def test_positions_and_lengths(self):
        note, layout = parse_note_with_layout("---\naliases: [a b, b]\n---\n# b\nb c b\n", "b.md")
        positions, lengths = postings(note_fields("b.md", note, layout))
        self.assertEqual(lengths, {"title": 1, "aliases": 3, "headings": 1, "body": 3})
        self.assertEqual(positions[("b", "title")], [0])
        self.assertEqual(positions[("b", "aliases")], [1, 2 + VALUE_GAP])  # second alias starts after the gap
        self.assertEqual(positions[("b", "body")], [0, 2])
        self.assertEqual(positions[("c", "body")], [1])
        self.assertEqual(positions[("a", "aliases")], [0])


# ----- index storage and incremental upkeep ------------------------------------


class TempVault:
    def __init__(self, files):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "vault"
        self.root.mkdir()
        self.clock = OLD_TIME
        for path, text in files.items():
            self.write(path, text)

    def write(self, path, text):
        full = self.root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text, encoding="utf-8")
        self.clock += 10
        os.utime(full, (self.clock, self.clock))

    def cleanup(self):
        self._tmp.cleanup()


class IndexTestCase(unittest.TestCase):
    FILES: dict = {}

    def setUp(self):
        self.vault = TempVault(self.FILES)
        self.ix = VaultIndex(default_db_path(self.vault.root))
        self.ix.update(self.vault.root)

    def tearDown(self):
        self.ix.close()
        self.vault.cleanup()

    def file_id(self, path):
        return self.ix.conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()[0]

    def rows(self, path):
        return self.ix.conn.execute(
            "SELECT term, field, tf, positions FROM postings WHERE file_id = ? ORDER BY term, field", (self.file_id(path),)
        ).fetchall()

    def search(self, query, limit=None):
        return search(self.ix.conn, query, limit=limit)


class PostingsStorageTests(IndexTestCase):
    FILES = {
        "A.md": "---\naliases: [Alpha note]\n---\n# Alpha heading\nalpha beta alpha #gamma\n",
        "B.md": "beta only `alpha in code`",
        "C.md": "gamma delta",
    }

    def test_one_row_per_term_note_field_with_positions(self):
        self.assertEqual(
            [tuple(row) for row in self.rows("A.md")],
            [
                ("a", "title", 1, "[0]"),
                ("alpha", "aliases", 1, "[0]"),
                ("alpha", "body", 2, "[0,2]"),
                ("alpha", "headings", 1, "[0]"),
                ("beta", "body", 1, "[1]"),
                ("gamma", "tags", 1, "[0]"),
                ("heading", "headings", 1, "[1]"),
                ("note", "aliases", 1, "[1]"),
            ],
        )
        lengths = self.ix.conn.execute(
            "SELECT field, length FROM field_lengths WHERE file_id = ? ORDER BY field", (self.file_id("A.md"),)
        ).fetchall()
        self.assertEqual([tuple(r) for r in lengths], [("aliases", 2), ("body", 3), ("headings", 2), ("tags", 1), ("title", 1)])

    def log_writes(self):
        """Record which notes' search rows every later statement inserts, updates or deletes."""
        conn = self.ix.conn
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS touched (tbl TEXT, op TEXT, file_id INTEGER)")
        for table in ("postings", "field_lengths"):
            for op, row in (("INSERT", "NEW"), ("DELETE", "OLD"), ("UPDATE", "NEW")):
                conn.execute(
                    f"CREATE TEMP TRIGGER IF NOT EXISTS log_{table}_{op} AFTER {op} ON main.{table} "
                    f"BEGIN INSERT INTO touched VALUES ('{table}', '{op}', {row}.file_id); END"
                )

    def touched(self):
        path_of = dict(self.ix.conn.execute("SELECT id, path FROM files"))
        return {(tbl, op, path_of.get(file_id, file_id)) for tbl, op, file_id in self.ix.conn.execute("SELECT * FROM touched")}

    def test_editing_one_note_rewrites_only_its_rows(self):
        others = {path: self.rows(path) for path in ("A.md", "C.md")}
        self.log_writes()
        self.vault.write("B.md", "beta epsilon")
        report = self.ix.update(self.vault.root)
        self.assertEqual(report.changed, ["B.md"])
        self.assertEqual({path for _, _, path in self.touched()}, {"B.md"})
        self.assertEqual(
            self.touched(),
            {(t, op, "B.md") for t in ("postings", "field_lengths") for op in ("INSERT", "DELETE")},
        )
        self.assertEqual({path: self.rows(path) for path in others}, others)
        self.assertEqual([tuple(r)[:3] for r in self.rows("B.md")], [("b", "title", 1), ("beta", "body", 1), ("epsilon", "body", 1)])
        self.assertEqual([h.path for h in self.search("epsilon").hits], ["B.md"])

    def test_unchanged_run_writes_nothing(self):
        self.log_writes()
        self.ix.update(self.vault.root)
        self.assertEqual(self.touched(), set())

    def test_deleting_a_note_removes_its_rows(self):
        gone = self.file_id("C.md")
        self.log_writes()
        (self.vault.root / "C.md").unlink()
        self.ix.update(self.vault.root)
        self.assertEqual({path for _, _, path in self.touched()}, {gone})  # the files row is gone too
        self.assertEqual({op for _, op, _ in self.touched()}, {"DELETE"})
        for table in ("postings", "field_lengths"):
            self.assertEqual(self.ix.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE file_id = ?", (gone,)).fetchone()[0], 0)
        self.assertEqual(self.search("delta").total, 0)
        self.assertEqual(self.search("gamma").clauses["gamma"].df, 1)  # only A.md now
        self.assertEqual(self.search("gamma").notes, 2)

    def test_rename_moves_rows_to_the_new_title(self):
        (self.vault.root / "C.md").rename(self.vault.root / "Delta notes.md")
        self.ix.update(self.vault.root)
        self.assertEqual([(h.path, list(h.matches["delta"].fields)) for h in self.search("delta").hits], [("Delta notes.md", ["title", "body"])])
        self.assertEqual(self.search("c").total, 0)

    def test_search_version_change_rereads_every_note(self):
        with patch.object(index_module, "SEARCH_VERSION", "999"):
            report = self.ix.update(self.vault.root)
        self.assertEqual(report.forced, ["A.md", "B.md", "C.md"])
        self.assertIn("search index version", report.full_reason)


# ----- BM25F -------------------------------------------------------------------

# Five tiny notes. Every term frequency and field length used below is counted by
# hand from these texts and written out as a literal.
BM25F_NOTES = {
    # title    reward hacking                      (2)
    # aliases  specification gaming                (2)
    # body     a policy exploits the reward        (5)
    "Reward hacking.md": "---\naliases: [Specification gaming]\n---\nA policy exploits the reward.\n",
    # title    policy notes                        (2)
    # tags     rl                                  (1)   '#rl' is a tag, not a heading (no space)
    # headings reward shaping                      (2)
    # body     shaping the reward changes the policy reward hacking follows   (9)
    "Policy notes.md": "#rl\n## Reward shaping\nShaping the reward changes the policy. Reward hacking follows.\n",
    # title    hacking log                         (2)
    # body     hacking again and hacking more inline too   (7)   'reward' only appears in code
    "Hacking log.md": "Hacking again and hacking more.\n\n```python\nreward = 1\n```\n\nInline `reward` too.\n",
    # title    gardening                           (1)
    # body     tomatoes need sun keep a log        (6)
    "Gardening.md": "Tomatoes need sun. Keep a log.\n",
    # title    misc                                (1)
    # tags     reward hacking                      (2)   from the tag 'reward-hacking'
    # body     see the reward model                (4)
    "Misc.md": "---\ntags: [reward-hacking]\n---\nSee the reward model.\n",
}

N = 5
AVG = {
    "title": (2 + 2 + 2 + 1 + 1) / 5,  # 1.6
    "aliases": 2 / 1,  # only "Reward hacking" has aliases
    "headings": 2 / 1,  # only "Policy notes" has headings
    "tags": (1 + 2) / 2,  # "Policy notes" and "Misc"
    "body": (5 + 9 + 7 + 6 + 4) / 5,  # 6.2
}
W = {"title": 3, "aliases": 2, "headings": 2, "tags": 2, "body": 1}
K1, B = 1.2, 0.75


def hand_idf(df):
    return math.log(1 + (N - df + 0.5) / (df + 0.5))


def hand_term_score(df, occurrences):
    """occurrences: [(field, tf, field length)] of one term in one note."""
    tf = sum(W[f] * tf / ((1 - B) + B * length / AVG[f]) for f, tf, length in occurrences)
    return hand_idf(df) * tf * (K1 + 1) / (K1 + tf)


class BM25FTests(IndexTestCase):
    FILES = BM25F_NOTES

    def scores(self, query):
        return {hit.path: hit.score for hit in self.search(query).hits}

    def test_fixture_matches_hand_counts(self):
        # Guard: if tokenising or field extraction changed, the hand counts below are wrong, not the scorer.
        lengths = {
            (path, field): length
            for path, field, length in self.ix.conn.execute(
                "SELECT f.path, l.field, l.length FROM field_lengths l JOIN files f ON f.id = l.file_id"
            )
        }
        self.assertEqual(
            lengths,
            {
                ("Reward hacking.md", "title"): 2, ("Reward hacking.md", "aliases"): 2, ("Reward hacking.md", "body"): 5,
                ("Policy notes.md", "title"): 2, ("Policy notes.md", "tags"): 1, ("Policy notes.md", "headings"): 2,
                ("Policy notes.md", "body"): 9,
                ("Hacking log.md", "title"): 2, ("Hacking log.md", "body"): 7,
                ("Gardening.md", "title"): 1, ("Gardening.md", "body"): 6,
                ("Misc.md", "title"): 1, ("Misc.md", "tags"): 2, ("Misc.md", "body"): 4,
            },
        )

    def test_reward_hacking_scores_match_hand_computation(self):
        df_reward = 3  # Reward hacking (title, body), Policy notes (headings, body), Misc (tags, body); not Hacking log (code only)
        df_hacking = 4  # Reward hacking (title), Policy notes (body), Hacking log (title, body), Misc (tags)
        expected = {
            "Reward hacking.md": hand_term_score(df_reward, [("title", 1, 2), ("body", 1, 5)])
            + hand_term_score(df_hacking, [("title", 1, 2)]),
            "Policy notes.md": hand_term_score(df_reward, [("headings", 1, 2), ("body", 2, 9)])
            + hand_term_score(df_hacking, [("body", 1, 9)]),
            "Hacking log.md": hand_term_score(df_hacking, [("title", 1, 2), ("body", 2, 7)]),
            "Misc.md": hand_term_score(df_reward, [("tags", 1, 2), ("body", 1, 4)])
            + hand_term_score(df_hacking, [("tags", 1, 2)]),
        }
        result = self.search("reward hacking")
        self.assertEqual((result.notes, result.total), (N, 4))  # Gardening does not match
        self.assertEqual({t: s.df for t, s in result.clauses.items()}, {"reward": df_reward, "hacking": df_hacking})
        self.assertAlmostEqual(result.clauses["reward"].idf, math.log(1 + 2.5 / 3.5), delta=1e-12)
        self.assertEqual([s.variants for s in result.clauses.values()], [{}, {}])  # no spelling variants in this fixture
        self.assertEqual(set(self.scores("reward hacking")), set(expected))
        for path, score in self.scores("reward hacking").items():
            self.assertAlmostEqual(score, expected[path], delta=1e-6, msg=path)
        self.assertEqual([h.path for h in result.hits], sorted(expected, key=lambda p: -expected[p]))
        # The same numbers as literals, for reading (worked out separately with a calculator script).
        literal = [("Reward hacking.md", 1.324250), ("Misc.md", 1.205611), ("Policy notes.md", 1.125469), ("Hacking log.md", 0.496053)]
        self.assertEqual([h.path for h in result.hits], [path for path, _ in literal])
        for hit, (_, score) in zip(result.hits, literal):
            self.assertAlmostEqual(hit.score, score, delta=1e-6, msg=hit.path)

    def test_single_field_cases_match_hand_computation(self):
        cases = {
            "gaming": {"Reward hacking.md": hand_term_score(1, [("aliases", 1, 2)])},
            "shaping": {"Policy notes.md": hand_term_score(1, [("headings", 1, 2), ("body", 1, 9)])},
            "rl": {"Policy notes.md": hand_term_score(1, [("tags", 1, 1)])},
            "log": {"Hacking log.md": hand_term_score(2, [("title", 1, 2)]), "Gardening.md": hand_term_score(2, [("body", 1, 6)])},
            "the": {
                "Reward hacking.md": hand_term_score(3, [("body", 1, 5)]),
                "Policy notes.md": hand_term_score(3, [("body", 2, 9)]),
                "Misc.md": hand_term_score(3, [("body", 1, 4)]),
            },
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                got = self.scores(query)
                self.assertEqual(set(got), set(expected))
                for path in expected:
                    self.assertAlmostEqual(got[path], expected[path], delta=1e-6, msg=path)

    def test_title_match_outranks_body_only_match(self):
        # 'log': title of "Hacking log", only in the body of "Gardening".
        self.assertEqual([h.path for h in self.search("log").hits], ["Hacking log.md", "Gardening.md"])
        # 'hacking': only in the title of "Reward hacking", only in the body of "Policy notes" (a longer body too,
        # but the title still wins with everything else equal: same df, same tf).
        hits = {hit.path: hit for hit in self.search("hacking").hits}
        self.assertEqual(list(hits["Reward hacking.md"].matches["hacking"].fields), ["title"])
        self.assertEqual(list(hits["Policy notes.md"].matches["hacking"].fields), ["body"])
        self.assertGreater(hits["Reward hacking.md"].score, hits["Policy notes.md"].score)

    def test_title_beats_body_even_against_repetition(self):
        # Saturation: three body mentions of a term do not beat one title mention.
        self.vault.write("Verifier.md", "A short note.")
        self.vault.write("Other.md", "verifier verifier verifier in a note")
        self.ix.update(self.vault.root)
        self.assertEqual([h.path for h in self.search("verifier").hits], ["Verifier.md", "Other.md"])

    def test_query_handling(self):
        base = self.scores("reward hacking")
        self.assertEqual(self.scores("REWARD, hacking!"), base)  # same tokenizer as the index
        self.assertEqual(self.scores("reward reward hacking"), base)  # repeated terms count once
        self.assertEqual(self.scores("hacking reward"), base)
        unknown = self.search("reward zzzunknown")
        self.assertEqual(unknown.clauses["zzzunknown"].df, 0)
        self.assertEqual({h.path: h.score for h in unknown.hits}, self.scores("reward"))
        empty = self.search(" -- ")
        self.assertEqual((empty.clauses, empty.total, empty.hits), ({}, 0, []))
        self.assertEqual(len(self.search("reward hacking", limit=2).hits), 2)
        self.assertEqual(self.search("reward hacking", limit=2).total, 4)

    def test_explanation_adds_up(self):
        for hit in self.search("reward hacking").hits:
            self.assertAlmostEqual(hit.score, sum(m.score for m in hit.matches.values()), delta=1e-12)
            for match in hit.matches.values():
                self.assertAlmostEqual(match.tf, sum(f.weighted_tf for f in match.fields.values()), delta=1e-12)


class SampleVaultReferenceTests(unittest.TestCase):
    """SQL-side search vs a direct in-memory BM25F over the same notes (checks the SQL aggregation at scale).

    Spelling variants are off here: this checks plain BM25F. Fuzzy matching has its own tests below.
    """

    _shared = None

    @classmethod
    def shared_index(cls):
        """One index of the sample vault for every test class that only reads it."""
        if cls._shared is None:
            cls._tmp = tempfile.TemporaryDirectory()
            cls._shared = VaultIndex(Path(cls._tmp.name) / "index.sqlite")
            cls._shared.update(FIXTURES / "vault")
        return cls._shared

    @classmethod
    def setUpClass(cls):
        cls.ix = cls.shared_index()
        cls.docs = {}
        for path in sorted((FIXTURES / "vault").rglob("*.md")):
            rel = path.relative_to(FIXTURES / "vault").as_posix()
            note, layout = parse_note_with_layout(path.read_text(encoding="utf-8"), rel)
            cls.docs[rel] = {f: [t for v in values for t in v] for f, values in note_fields(rel, note, layout).items()}

    def reference(self, query):
        terms = set(tokenize(query))
        n = len(self.docs)
        nonempty = {f: [len(d[f]) for d in self.docs.values() if d[f]] for f in FIELD_WEIGHTS}
        avg = {f: sum(v) / len(v) for f, v in nonempty.items() if v}
        scores = {}
        for term in terms:
            df = sum(any(term in d[f] for f in d) for d in self.docs.values())
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for path, d in self.docs.items():
                tf = sum(
                    FIELD_WEIGHTS[f] * d[f].count(term) / (0.25 + 0.75 * len(d[f]) / avg[f]) for f in d if term in d[f]
                )
                if tf:
                    scores[path] = scores.get(path, 0) + idf * tf * 2.2 / (1.2 + tf)
        return scores

    def test_same_scores_as_reference(self):
        for query in ("reward hacking", "trajectory synthesis", "GRPO", "the environment", "2503.11926", "kl penalty rlhf"):
            with self.subTest(query=query):
                expected = self.reference(query)
                got = {h.path: h.score for h in search(self.ix.conn, query, limit=None, fuzzy=False).hits}
                self.assertEqual(set(got), set(expected))
                for path, score in expected.items():
                    self.assertAlmostEqual(got[path], score, delta=1e-9, msg=path)


# ----- spelling ----------------------------------------------------------------


class SpellingTests(unittest.TestCase):
    def test_edit_budget_by_length(self):
        self.assertEqual([edit_budget(w) for w in ("the", "tho", "gain", "policy", "hacking", "gradient", "zettelkasten")], [0, 0, 1, 1, 1, 2, 2])

    def test_words_with_digits_are_never_fuzzy(self):
        self.assertEqual([edit_budget(w) for w in ("2503.11926", "gpt4o", "v1.2", "h100s")], [0, 0, 0, 0])

    def test_distance_counts_a_swap_as_one_edit(self):
        self.assertEqual(osa_distance("zettelkatsen", "zettelkasten"), 1)
        self.assertEqual(osa_distance("policy", "polcy"), 1)  # deletion
        self.assertEqual(osa_distance("policy", "police"), 1)  # substitution
        self.assertEqual(osa_distance("hackng", "hacking"), 1)  # insertion
        self.assertEqual(osa_distance("zetelkasten", "zettlekasten"), 2)

    def test_vocabulary_walk_equals_checking_every_term(self):
        rng = random.Random(0)
        alphabet = "abcaébb"
        for _ in range(2000):
            vocabulary = sorted({"".join(rng.choice(alphabet) for _ in range(rng.randint(1, 7))) for _ in range(rng.randint(1, 40))})
            word = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 7)))
            budget = rng.randint(0, 2)
            expected = {t: osa_distance(word, t) for t in vocabulary if osa_distance(word, t) <= budget}
            self.assertEqual(close_terms(word, budget, vocabulary), expected, (word, budget, vocabulary))


# ----- query language ------------------------------------------------------------


class QueryParserTests(unittest.TestCase):
    def test_words_phrases_and_near(self):
        self.assertEqual(
            parse_query('Reward hacking "exact phrase here" "single" policy NEAR/3 model'),
            [Word("reward"), Word("hacking"), Phrase(("exact", "phrase", "here")), Word("single", exact=True), Near("policy", "model", 3)],
        )

    def test_quotes(self):
        self.assertEqual(parse_query("\u201ccurly quotes\u201d"), [Phrase(("curly", "quotes"))])
        with self.assertRaisesRegex(QueryError, "never closed"):
            parse_query('"unclosed phrase')
        with self.assertRaisesRegex(QueryError, "nothing to search for inside these quotes"):
            parse_query('"!!" word')
        self.assertEqual(parse_query('"Reward-Hacking"'), [Phrase(("reward", "hacking"))])

    def test_near_operands_are_the_adjacent_words(self):
        self.assertEqual(parse_query("gpt-4o NEAR/2 model card"), [Word("gpt"), Near("4o", "model", 2), Word("card")])
        self.assertEqual(parse_query('"exact" NEAR/1 b'), [Near("exact", "b", 1)])

    def test_near_must_be_uppercase_with_a_distance(self):
        self.assertEqual(parse_query("near miss NEAR"), [Word("near"), Word("miss")])
        self.assertEqual(parse_query("a near/2 b"), [Word("a"), Word("near"), Word("2"), Word("b")])

    def test_duplicates_removed(self):
        self.assertEqual(parse_query("policy Policy POLICY"), [Word("policy")])

    def test_errors(self):
        for query, message in (
            ("NEAR/3 model", "single word on each side"),
            ("model NEAR/3", "single word on each side"),
            ('"a phrase" NEAR/2 b', "single word on each side"),
            ("a NEAR/2 b NEAR/2 c", "only be in one NEAR"),
            ("a NEAR/0 b", "from 1 to 99"),
            ("a NEAR/100 b", "from 1 to 99"),
        ):
            with self.subTest(query=query), self.assertRaisesRegex(QueryError, message):
                parse_query(query)

    def test_near_pairs(self):
        self.assertEqual(near_pairs([0], [3], 3), 1)
        self.assertEqual(near_pairs([0], [4], 3), 0)
        self.assertEqual(near_pairs([5], [2, 8], 3), 2)  # either order
        self.assertEqual(near_pairs([0, 2], [0, 2], 2, same=True), 1)  # one pair of distinct occurrences


# ----- fuzzy words -------------------------------------------------------------

# title    policy                  (1)   body  a policy maps states to actions          (6)
# title    polcy notes             (2)   body  notes about the polcy gradient           (5)
# title    police                  (1)   body  the police arrived                       (3)
# title    long                    (1)   body  words x11, policy                        (12)
# title    numbers                 (1)   body  paper 2503.11926 and 2503.11927 and v1.2 (6)
FUZZY_NOTES = {
    "Policy.md": "A policy maps states to actions.\n",
    "Polcy notes.md": "Notes about the polcy gradient.\n",
    "Police.md": "The police arrived.\n",
    "Long.md": "words " * 11 + "policy.\n",
    "Numbers.md": "Paper 2503.11926 and 2503.11927 and v1.2.\n",
}
FUZZY_AVG = {"title": (1 + 2 + 1 + 1 + 1) / 5, "body": (6 + 5 + 3 + 12 + 6) / 5}


def fuzzy_hand_score(weight, df, occurrences):
    tf = sum(W[f] * tf / ((1 - B) + B * length / FUZZY_AVG[f]) for f, tf, length in occurrences)
    return weight * math.log(1 + (5 - df + 0.5) / (df + 0.5)) * tf * (K1 + 1) / (K1 + tf)


class FuzzyTests(IndexTestCase):
    FILES = FUZZY_NOTES

    def test_misspelled_word_finds_the_note(self):
        result = self.search("plicy")  # one edit (5 letters)
        self.assertEqual(result.clauses["plicy"].df, 0)
        self.assertEqual(result.clauses["plicy"].variants, {"policy": 1})
        self.assertEqual({h.path for h in result.hits}, {"Policy.md", "Long.md"})
        match = result.hits[0].matches["plicy"]
        self.assertEqual((match.exact, match.term, match.edits), (False, "policy", 1))

    def test_a_swap_is_one_edit(self):
        self.assertEqual(self.search("ploicy").clauses["ploicy"].variants, {"policy": 1})
        self.assertEqual({h.path for h in self.search("ploicy").hits}, {"Policy.md", "Long.md"})

    def test_two_edits_for_long_words_only(self):
        self.assertEqual(self.search("gradyents").clauses["gradyents"].variants, {"gradient": 2})  # 9 letters
        self.assertEqual([h.path for h in self.search("gradyents").hits], ["Polcy notes.md"])
        self.assertEqual(self.search("gradxnt").total, 0)  # 7 letters, 2 edits away: too far

    def test_short_words_and_numbers_are_exact_only(self):
        self.assertEqual(self.search("tho").total, 0)  # 'the' is one edit away, but 'tho' has 3 letters
        result = self.search("2503.11926")
        self.assertEqual((result.clauses["2503.11926"].variants, [h.path for h in result.hits]), ({}, ["Numbers.md"]))

    def test_exact_matches_always_outrank_spelling_variants(self):
        hits = self.search("policy").hits
        self.assertEqual([(h.path, h.exact) for h in hits], [("Policy.md", 1), ("Long.md", 1), ("Police.md", 0), ("Polcy notes.md", 0)])
        # The rule is doing work here: by score alone both variant notes would beat the exact match in Long.md.
        scores = {h.path: h.score for h in hits}
        self.assertGreater(scores["Police.md"], scores["Long.md"])
        self.assertGreater(scores["Polcy notes.md"], scores["Long.md"])

    def test_fuzzy_scores_match_hand_computation(self):
        df = 2  # 'policy' is in 2 notes; its variants 'polcy' and 'police' in 1 each: the variants use the larger df
        expected = {
            "Policy.md": fuzzy_hand_score(1, 2, [("title", 1, 1), ("body", 1, 6)]),
            "Long.md": fuzzy_hand_score(1, 2, [("body", 1, 12)]),
            "Police.md": fuzzy_hand_score(0.5, df, [("title", 1, 1), ("body", 1, 3)]),  # one edit: half weight
            "Polcy notes.md": fuzzy_hand_score(0.5, df, [("title", 1, 2), ("body", 1, 5)]),
        }
        result = self.search("policy")
        self.assertAlmostEqual(result.clauses["policy"].fuzzy_idf, math.log(1 + 3.5 / 2.5), delta=1e-12)
        for hit in result.hits:
            self.assertAlmostEqual(hit.score, expected[hit.path], delta=1e-6, msg=hit.path)

    def test_exact_notes_score_as_without_fuzzy(self):
        with_fuzzy = {h.path: h.score for h in self.search("policy").hits if h.exact}
        self.assertEqual(with_fuzzy, {h.path: h.score for h in self.search("policy", limit=None).hits if h.exact})
        self.assertEqual(with_fuzzy, {h.path: h.score for h in search(self.ix.conn, "policy", fuzzy=False).hits})

    def test_multi_word_queries_rank_by_exact_clauses_then_score(self):
        hits = self.search("policy gradient").hits
        tiers = [h.exact for h in hits]
        self.assertEqual(tiers, sorted(tiers, reverse=True))
        self.assertEqual(hits[-1].path, "Police.md")  # only a spelling variant of one word
        polcy = next(h for h in hits if h.path == "Polcy notes.md")
        self.assertEqual((polcy.exact, polcy.matches["policy"].term, polcy.matches["gradient"].exact), (1, "polcy", True))

    def test_quoted_word_is_exact_and_required(self):
        self.assertEqual(self.search('"plicy"').total, 0)
        self.assertEqual([h.path for h in self.search('"policy"').hits], ["Policy.md", "Long.md"])
        self.assertEqual([h.path for h in self.search('"police" policy').hits], ["Police.md"])

    def test_variants_follow_index_updates(self):
        self.vault.write("New.md", "The polisy draft.")
        self.ix.update(self.vault.root)
        self.assertEqual(self.search("policy").clauses["policy"].variants, {"polcy": 1, "police": 1, "polisy": 1})
        (self.vault.root / "Police.md").unlink()
        self.ix.update(self.vault.root)
        self.assertEqual(self.search("policy").clauses["policy"].variants, {"polcy": 1, "polisy": 1})


# ----- phrases and NEAR -----------------------------------------------------------

# body values (split at headings) and lengths:
#   P1  [reward hacking is common] [more reward] [hacking later]       8   headings: next, then
#   P2  [hacking reward reversed]                                      3
#   P3  [reward model hacking]                                         3
#   P4  [reward hacking again and reward hacking]                      6   aliases: reward | hacking
#   P5  [plain text]                                                   2   (the phrase is in a code block)
PHRASE_NOTES = {
    "P1.md": "reward hacking is common.\n## Next\nmore reward\n## Then\nhacking later.\n",
    "P2.md": "hacking reward, reversed.\n",
    "P3.md": "reward model hacking.\n",
    "P4.md": "---\naliases: [Reward, Hacking]\n---\nReward-Hacking again, and reward hacking.\n",
    "P5.md": "```\nreward hacking\n```\nplain text\n",
}


class PhraseTests(IndexTestCase):
    FILES = PHRASE_NOTES

    def test_only_adjacent_words_in_order(self):
        result = self.search('"reward hacking"')
        self.assertEqual({h.path for h in result.hits}, {"P1.md", "P4.md"})
        tf = {h.path: h.matches['"reward hacking"'].fields["body"].tf for h in result.hits}
        self.assertEqual(tf, {"P1.md": 1, "P4.md": 2})  # not across P1's headings; P4's hyphenated form counts
        self.assertNotIn("aliases", next(h for h in result.hits if h.path == "P4.md").matches['"reward hacking"'].fields)

    def test_phrase_scores_match_hand_computation(self):
        avg_body = (8 + 3 + 3 + 6 + 2) / 5
        phrase_idf = math.log(1 + (5 - 2 + 0.5) / (2 + 0.5))

        def score(tf, length):
            x = tf / ((1 - B) + B * length / avg_body)
            return phrase_idf * x * (K1 + 1) / (K1 + x)

        got = {h.path: h.score for h in self.search('"reward hacking"').hits}
        self.assertAlmostEqual(got["P1.md"], score(1, 8), delta=1e-6)
        self.assertAlmostEqual(got["P4.md"], score(2, 6), delta=1e-6)
        self.assertEqual(list(got), ["P4.md", "P1.md"])

    def test_phrase_is_required(self):
        self.assertEqual({h.path for h in self.search('"reward hacking" common OR "reward hacking" again').hits}, {"P1.md", "P4.md"})
        self.assertEqual([h.path for h in self.search('"reward hacking" again').hits], ["P4.md"])  # a word must match too
        self.assertEqual(self.search('"hacking reward" model').total, 0)  # P2 has the phrase, not 'model'
        self.assertEqual([h.path for h in self.search('"hacking reward"').hits], ["P2.md"])

    def test_longer_phrases_and_misses(self):
        self.assertEqual([h.path for h in self.search('"reward hacking is common"').hits], ["P1.md"])
        self.assertEqual(self.search('"reward hacking later"').total, 0)
        self.assertEqual(self.search('"plain text reward"').total, 0)
        self.assertEqual(self.search('"reward hackng"').total, 0)  # no spelling help inside quotes


# body positions:  N1 alpha 0, beta 3 | N2 beta 0, alpha 1 | N3 alpha 0, (heading) beta 101
#                  N5 alpha 0 and 2, beta 1 | Alpha.md: alpha only in the title, beta in the body
NEAR_NOTES = {
    "N1.md": "alpha one two beta\n",
    "N2.md": "beta alpha\n",
    "N3.md": "alpha\n## Heading\nbeta\n",
    "Alpha.md": "beta words\n",
    "N5.md": "alpha beta alpha\n",
}


class NearTests(IndexTestCase):
    FILES = NEAR_NOTES

    def paths(self, query):
        return {h.path for h in self.search(query).hits}

    def test_distance_boundary_and_order(self):
        self.assertEqual(self.paths("alpha NEAR/3 beta"), {"N1.md", "N2.md", "N5.md"})
        self.assertEqual(self.paths("alpha NEAR/2 beta"), {"N2.md", "N5.md"})  # N1 is 3 apart
        self.assertEqual(self.paths("beta NEAR/1 alpha"), {"N2.md", "N5.md"})

    def test_same_field_and_section_only(self):
        self.assertNotIn("N3.md", self.paths("alpha NEAR/99 beta"))  # a heading between them
        self.assertNotIn("Alpha.md", self.paths("alpha NEAR/99 beta"))  # title and body

    def test_pairs_and_hand_score(self):
        hits = {h.path: h for h in self.search("alpha NEAR/3 beta").hits}
        self.assertEqual({p: h.matches["alpha NEAR/3 beta"].fields["body"].tf for p, h in hits.items()}, {"N1.md": 1, "N2.md": 1, "N5.md": 2})
        avg_body = (4 + 2 + 1 + 2 + 3) / 5  # N3 has 2 body words but in two values; Alpha.md 2
        avg_body = (4 + 2 + 2 + 2 + 3) / 5
        near_idf = math.log(1 + (5 - 3 + 0.5) / (3 + 0.5))
        for path, tf, length in (("N1.md", 1, 4), ("N2.md", 1, 2), ("N5.md", 2, 3)):
            x = tf / ((1 - B) + B * length / avg_body)
            self.assertAlmostEqual(hits[path].score, near_idf * x * (K1 + 1) / (K1 + x), delta=1e-6, msg=path)

    def test_same_word_twice(self):
        self.assertEqual(self.paths("alpha NEAR/2 alpha"), {"N5.md"})
        self.assertEqual(self.paths("alpha NEAR/1 alpha"), set())

    def test_near_is_exact_and_required(self):
        self.assertEqual(self.paths("alpa NEAR/3 beta"), set())
        self.assertEqual(self.paths("alpha NEAR/2 beta"), {"N2.md", "N5.md"})
        self.assertEqual(self.paths("alpha NEAR/2 beta words"), set())  # and at least one bare word must match


# ----- snippets ----------------------------------------------------------------


def marked(context):
    return [(p.text, p.mark) for p in context.snippet.pieces if p.mark]


class SnippetTests(IndexTestCase):
    FILES = {
        **FUZZY_NOTES,
        "Heading only.md": "# Big topical word\nNothing else here.\n",
        "Alias only.md": "---\naliases: [Nickname here]\n---\nPlain.\n",
        "Tag only.md": "Text #topic/sub-area\n",
        "Far apart.md": "alpha " + "filler " * 40 + "then beta alpha at the end.\n",
        "Phrase.md": "---\naliases: [x]\n---\nSee Reward-Hacking and reward   hacking\nhere.\n",
    }

    def context(self, query, path, width=100):
        hit = next(h for h in self.search(query).hits if h.path == path)
        return hit_context(self.vault.root, hit, query, width)

    def test_body_snippet_marks_the_match(self):
        context = self.context("policy", "Policy.md")
        self.assertEqual(context.snippet.field, "body")
        self.assertEqual(context.snippet.text, "A policy maps states to actions")
        self.assertEqual(context.snippet.to_dict()["marks"], [[2, 8, "exact"]])
        self.assertEqual([(p.text, p.mark) for p in context.title], [("Policy", "exact")])

    def test_fuzzy_match_is_marked_as_fuzzy(self):
        context = self.context("policy", "Polcy notes.md")
        self.assertEqual(marked(context), [("polcy", "fuzzy")])
        self.assertEqual([(p.text, p.mark) for p in context.title], [("Polcy", "fuzzy"), (" notes", None)])

    def test_phrase_is_one_mark_and_whitespace_collapses(self):
        context = self.context('"reward hacking"', "Phrase.md")
        self.assertEqual(marked(context), [("Reward-Hacking", "exact"), ("reward hacking", "exact")])

    def test_heading_alias_and_tag_snippets(self):
        self.assertEqual((self.context("topical", "Heading only.md").snippet.field, self.context("topical", "Heading only.md").snippet.text),
                         ("headings", "# Big topical word"))
        self.assertEqual(self.context("nickname", "Alias only.md").snippet.text, "alias: Nickname here")
        tag = self.context("area", "Tag only.md").snippet
        self.assertEqual((tag.field, tag.text, marked(self.context("area", "Tag only.md"))), ("tags", "#topic/sub-area", [("area", "exact")]))

    def test_window_with_most_clauses_wins(self):
        snippet = self.context("alpha beta", "Far apart.md").snippet
        self.assertEqual([(t, m) for t, m in marked(self.context("alpha beta", "Far apart.md"))], [("beta", "exact"), ("alpha", "exact")])
        self.assertTrue(snippet.truncated_start)

    def test_width_is_respected(self):
        snippet = self.context("alpha beta", "Far apart.md", width=30).snippet
        self.assertLessEqual(len(snippet.text), 30)
        self.assertIn("beta alpha", snippet.text)

    def test_snippet_reads_the_current_file(self):
        hit = next(h for h in self.search("policy").hits if h.path == "Policy.md")
        self.vault.write("Policy.md", "Rewritten, the policy changed.\n")
        self.assertEqual(hit_context(self.vault.root, hit, "policy").snippet.text, "Rewritten, the policy changed")
        (self.vault.root / "Policy.md").unlink()
        self.assertIsNone(hit_context(self.vault.root, hit, "policy").snippet)


# ----- properties on the sample vault ---------------------------------------------


class SampleVaultQueryProperties(unittest.TestCase):
    """Phrase and NEAR against brute force over the notes' words, and the ordering rule, on the sample vault."""

    @classmethod
    def setUpClass(cls):
        cls.ix = SampleVaultReferenceTests.shared_index()

    def values(self):
        for path in sorted((FIXTURES / "vault").rglob("*.md")):
            rel = path.relative_to(FIXTURES / "vault").as_posix()
            note, layout = parse_note_with_layout(path.read_text(encoding="utf-8"), rel)
            yield rel, note_fields(rel, note, layout)

    def test_phrases_and_near_match_brute_force(self):
        notes = dict(self.values())
        rng = random.Random(5)
        bodies = [(p, v) for p, f in notes.items() for v in f["body"] if len(v) > 6]
        queries = []
        for _ in range(25):
            path, value = rng.choice(bodies)
            i = rng.randrange(len(value) - 3)
            queries.append(("phrase", tuple(value[i : i + rng.randint(2, 3)])))
            queries.append(("near", (value[i], value[i + rng.randint(1, 3)], rng.randint(1, 4))))
        for kind, q in queries:
            expected = set()
            for path, fields in notes.items():
                for values in fields.values():
                    for v in values:
                        if kind == "phrase" and any(tuple(v[j : j + len(q)]) == q for j in range(len(v))):
                            expected.add(path)
                        if kind == "near":
                            left = [i for i, t in enumerate(v) if t == q[0]]
                            right = [j for j, t in enumerate(v) if t == q[1]]
                            if any(i != j and abs(i - j) <= q[2] for i in left for j in right):
                                expected.add(path)
            query = '"' + " ".join(q) + '"' if kind == "phrase" else f"{q[0]} NEAR/{q[2]} {q[1]}"
            with self.subTest(query=query):
                self.assertEqual({h.path for h in search(self.ix.conn, query, limit=None).hits}, expected)

    def test_exact_before_fuzzy_and_tiers_ordered(self):
        words = [row[0] for row in self.ix.conn.execute("SELECT term FROM terms WHERE df >= 2 AND length(term) >= 4 ORDER BY term")]
        rng = random.Random(9)
        for word in rng.sample(words, 60) + ["root", "kills", "zettel"]:
            with self.subTest(word=word):
                hits = search(self.ix.conn, word, limit=None).hits
                flags = [h.exact for h in hits]
                self.assertEqual(flags, sorted(flags, reverse=True))  # every exact match before every variant-only one
                for tier in set(flags):
                    scores = [h.score for h in hits if h.exact == tier]
                    self.assertEqual(scores, sorted(scores, reverse=True))
        for query in ("reward hackng", "trajectroy synthesis", "polcy gradient estimate"):
            hits = search(self.ix.conn, query, limit=None).hits
            self.assertEqual([h.exact for h in hits], sorted((h.exact for h in hits), reverse=True), query)

    def test_word_spans_match_indexed_words(self):
        for path in sorted((FIXTURES / "vault").rglob("*.md")):
            text = unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))
            self.assertEqual([text[a:b] for a, b in word_spans(text)], word_segments(text), path.name)


# ----- CLI ---------------------------------------------------------------------


class SearchCliTests(unittest.TestCase):
    def setUp(self):
        self.vault = TempVault(FUZZY_NOTES)
        self.run_cli("index", str(self.vault.root))

    def tearDown(self):
        self.vault.cleanup()

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {"COLUMNS": "80"}), redirect_stdout(out), redirect_stderr(err):
            code = cli_main(list(args))
        return code, out.getvalue(), err.getvalue()

    def search(self, *args):
        return self.run_cli("search", str(self.vault.root), *args)

    def test_needs_an_index(self):
        other = TempVault({"A.md": "a"})
        try:
            code, _, err = self.run_cli("search", str(other.root), "reward")
        finally:
            other.cleanup()
        self.assertEqual(code, 2)
        self.assertIn("no index", err)

    def test_readable_output(self):
        code, out, err = self.search("policy", "--color", "never")
        self.assertEqual((code, err), (0, ""))
        lines = out.splitlines()
        for expected in (
            "  policy  2 notes · close spellings: polcy (1 edit, 1 note), police (1 edit, 1 note)",
            "  4 of 5 notes match · exact matches first, then score",
            "  ── exact ──",
            "      Policy.md",
            "      policy: title, body",
            "      A «policy» maps states to actions",
            "  ── close spellings only ──",
            "      policy ≈ police (1 edit): title, body",
            "      The «police» arrived",
        ):
            self.assertIn(expected, lines)
        titles = [line for line in lines if line[:6].strip().rstrip(".").isdigit()]
        self.assertEqual([t.split()[1] for t in titles], ["«Policy»", "Long", "«Police»", "«Polcy»"])
        self.assertEqual({len(t) for t in titles}, {80})  # scores right-aligned to the terminal width
        self.assertTrue(titles[0].endswith(" 1.52"))  # fuzzy_hand_score(1, 2, [("title", 1, 1), ("body", 1, 6)])
        self.assertNotIn("\x1b[", out)

    def test_colour(self):
        _, out, _ = self.search("policy", "--color", "always")
        self.assertIn("\x1b[1;33mpolicy\x1b[0m", out)  # exact match
        self.assertIn("\x1b[4;33mpolice\x1b[0m", out)  # spelling variant
        self.assertNotIn("«", out)
        _, auto, _ = self.search("policy")  # not a terminal
        self.assertNotIn("\x1b[", auto)

    def test_phrase_near_and_explain(self):
        _, out, _ = self.search('"about the polcy" notes', "--explain", "--color", "never")
        self.assertIn('  "about the polcy"  1 note (phrase, required)', out)
        self.assertIn('      "about the polcy": body · notes: title, body', out)
        self.assertIn("      «Notes» «about the polcy» gradient", out)  # 'notes' is a query word too
        self.assertIn("        notes: tf~ ", out)
        _, out, _ = self.search("states NEAR/3 policy", "--color", "never")
        self.assertIn("      states NEAR/3 policy: body", out)
        self.assertIn("      A «policy» maps «states» to actions", out)

    def test_errors_and_options(self):
        code, out, err = self.search("NEAR/3 policy")
        self.assertEqual((code, out), (2, ""))
        self.assertEqual(err, "error: NEAR/3 needs a single word on each side\n  NEAR/3 policy\n  ^^^^^^\n  e.g. reward NEAR/3 model\n")
        _, out, _ = self.search("policy", "--no-fuzzy", "--color", "never")
        self.assertIn("  2 of 5 notes match · by score", out)
        _, out, _ = self.search("!!")
        self.assertIn("empty query", out)
        self.vault.write("New.md", "policy")
        _, _, err = self.search("policy")
        self.assertIn("index is out of date (1 added", err)

    def test_json(self):
        _, out, _ = self.search("policy", "--json", "--limit", "0")
        data = json.loads(out)
        self.assertEqual((data["total"], len(data["hits"])), (4, 4))
        self.assertEqual(data["clauses"]["policy"]["variants"], {"polcy": 1, "police": 1})
        first, last = data["hits"][0], data["hits"][-1]
        self.assertEqual((first["exact"], first["snippet"]["marks"]), (1, [[2, 8, "exact"]]))
        self.assertEqual((last["exact"], last["matches"]["policy"]["term"], last["matches"]["policy"]["edits"]), (0, "polcy", 1))
        self.assertFalse(data["index"]["stale"])


def tearDownModule():
    if SampleVaultReferenceTests._shared is not None:
        SampleVaultReferenceTests._shared.close()
        SampleVaultReferenceTests._tmp.cleanup()
        SampleVaultReferenceTests._shared = None  # other modules that share it build it again


if __name__ == "__main__":
    unittest.main()
