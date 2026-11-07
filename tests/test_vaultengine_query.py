"""The vault query language: syntax trees, errors, every filter on a hand-built vault, and the sample vault."""

import io
import json
import os
import re
import unittest
from collections import defaultdict, deque
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from vaultengine import LinkGraph, Vault
from vaultengine.cli import main as cli_main
from vaultengine.index import VaultIndex, default_db_path
from vaultengine.parser import parse_note_with_layout
from vaultengine.querylang import QueryError, parse
from vaultengine.search import note_fields, search
from vaultengine.vault import lookup_key

from test_vaultengine_search import FIXTURES, TempVault

SAMPLE = FIXTURES / "vault"


def tree(query):
    return parse(query).sexpr()


# ----- syntax trees ----------------------------------------------------------------


class SyntaxTreeTests(unittest.TestCase):
    def check(self, cases):
        for query, expected in cases:
            with self.subTest(query=query):
                self.assertEqual(tree(query), expected)

    def test_the_three_questions(self):
        self.check([
            ("reward hacking links-to:[[grpo-group-relative-policy-optimization]] -path:10_SOURCES",
             '(and (links-to "grpo-group-relative-policy-optimization") (not (path "10_SOURCES")) (any reward hacking))'),
            ("within:2 of:[[obfuscated-reward-hacking]] tag:ai-safety", '(and (within 2 "obfuscated-reward-hacking") (tag ai-safety))'),
            ("missing:status", "(not (has-property status))"),
        ])

    def test_precedence_not_then_and_then_or(self):
        self.check([
            ("a OR b c", "(or a (and (any b c)))"),
            ("a b OR c", "(or (and (any a b)) c)"),
            ("(a OR b) c", "(and (or a b) (any c))"),
            ("NOT a OR b", "(or (not (exact a)) b)"),
            ("-a b", "(and (not (exact a)) (any b))"),
            ("a AND b OR c AND d", "(or (and a b) (and c d))"),
            ("a OR b OR c", "(or a b c)"),
            ("((a))", "a"),
            ("x -(a OR b)", "(and (not (or a b)) (any x))"),
            ("NOT -a", "(not (not (exact a)))"),
            ("-(tag:x OR path:y) z", '(and (not (or (tag x) (path "y"))) (any z))'),
        ])

    def test_explicit_and_makes_words_required(self):
        self.check([
            ("a AND b c", "(and a b (any c))"),
            ("a b AND c", "(and b c (any a))"),
            ("tag:x AND a", "(and (tag x) a)"),
        ])

    def test_text_forms(self):
        self.check([
            ('"reward hacking" model', '(and (phrase "reward hacking") (any model))'),
            ('"Reward"', "(exact reward)"),
            ("\u201ccurly quotes\u201d", '(phrase "curly quotes")'),
            ("reward NEAR/3 model gpt-4o", "(and (near reward model 3) (any gpt 4o))"),
            ("a NEAR/2 gpt-4o", "(and (near a gpt 2) (any 4o))"),
            ("(a NEAR/2 b) OR c", "(or (near a b 2) c)"),
            ("-gpt-4o", '(not (phrase "gpt 4o"))'),
            ('-"reward hacking"', '(not (phrase "reward hacking"))'),
            ("a or b and not c", "(and (any a or b and not c))"),  # lowercase keywords are words
            ("near miss", "(and (any near miss))"),
            ('"tag:ai"', "(exact tag:ai)"),  # quoted: text, not a field
            ("- -- !! reward", "reward"),  # stray punctuation is ignored
        ])

    def test_fields(self):
        self.check([
            ("tag:#AI/Safety", "(tag AI/Safety)"),
            ('path:"50_MOC/discourse" path:Notes/', '(and (path "50_MOC/discourse") (path "Notes/"))'),
            ("links-to:[[CLM - A fixed typed contract]]", '(links-to "CLM - A fixed typed contract")'),
            ("links-to:[[Note#Heading|shown text]]", '(links-to "Note")'),
            ('linked-from:"Some note" links-to:Plain', '(and (linked-from "Some note") (links-to "Plain"))'),
            ("within:3 of:[[Hub]]", '(within 3 "Hub")'),
            ("-within:1 of:Hub", '(not (within 1 "Hub"))'),
            ("has:tags has:Backlinks has:status", "(and (has tags) (has backlinks) (has-property status))"),
            ("missing:tags missing:Created", "(and (not (has tags)) (not (has-property Created)))"),
            ("TAG:x Path:y", '(and (tag x) (path "y"))'),  # field names are case-insensitive
        ])

    def test_properties_and_dates(self):
        self.check([
            ("[type]", "(has-property type)"),
            ("[status:baby] [status=#baby]", '(and (property status : "baby") (property status = "#baby"))'),
            ('[type:"key insight"]', '(property type : "key insight")'),
            ("[created>=2026-09] [rating<3]", '(and (property created >= "2026-09") (property rating < "3"))'),
            ("[created:2026-01..2026-03]", '(property created : "2026-01..2026-03")'),
            ("[supports=[[Hub]]]", '(property supports = "[[Hub]]")'),
            ("[my key:value]", '(property my key : "value")'),
            ("modified:2026-09", "(modified = 2026-09)"),
            ("modified:>=2026-09-01 modified:<2027", "(and (modified >= 2026-09-01) (modified < 2027))"),
            ("modified:2026-01..2026-03", "(modified 2026-01..2026-03)"),
        ])

    def test_spans_point_into_the_query(self):
        node = parse("reward -path:10_SOURCES")
        self.assertEqual(node.any[0].span, (0, 6))
        self.assertEqual(node.required[0].span, (7, 23))
        near = parse("gpt-4o NEAR/2 model").required[0]
        self.assertEqual(near.span, (4, 19))  # from "4o" to "model"

    def test_empty(self):
        self.assertIsNone(parse(""))
        self.assertIsNone(parse("  !! "))


# ----- errors ------------------------------------------------------------------


class ErrorTests(unittest.TestCase):
    CASES = [
        # query, message (regex), (start, end)
        ("tga:ai-safety reward", r'unknown field "tga:"', (0, 4)),
        ("reward (tag:x OR b", r"this \( is never closed", (7, 8)),
        ("a OR", r"OR needs something after it", (2, 4)),
        ("OR a", r"OR needs something before it", (0, 2)),
        ("a OR OR b", r"OR needs something after it", (2, 4)),
        ("(a OR) b", r"OR needs something after it", (3, 5)),
        ("a AND", r"AND needs something after it", (2, 5)),
        ("AND a", r"AND needs something before it", (0, 3)),
        ("a NOT", r"NOT needs something after it", (2, 5)),
        ("a -", None, None),  # a lone '-' is punctuation, not an error
        ("a)", r"this \) has no matching \(", (1, 2)),
        ("() a", r"empty parentheses", (0, 2)),
        ('a "unclosed', r"this quote is never closed", (2, 3)),
        ('""', r"nothing to search for inside these quotes", (0, 2)),
        ("[status", r"this \[ is never closed", (0, 1)),
        ("[]", r"a property needs a name", (0, 2)),
        ("[status:]", r"needs a value after :", (0, 9)),
        ("[status>baby]", r'> compares numbers or dates, and "baby" is neither', (8, 12)),
        ("[created>2026-13]", r'"2026-13" is not a date', (9, 16)),
        ("[created>2026..2027]", r"a range needs : or =", (9, 19)),
        ("within:2", r"within:2 needs of:\[\[note\]\] right after it", (0, 8)),
        ("within:2 tag:x", r"within:2 needs of:", (0, 8)),
        ("of:[[Hub]]", r"of: goes with within:N", (0, 10)),
        ("within:0 of:[[Hub]]", r"within: needs a number of links from 1 to 10", (7, 8)),
        ("within:two of:[[Hub]]", r"within: needs a number", (7, 10)),
        ("modified:2026-02-30", r'"2026-02-30" is not a date', (9, 19)),
        ("modified:>2026..2027", r"a range needs no >", (9, 20)),
        ("links-to:", r"links-to: needs a value", (0, 9)),
        ("links-to:[[Hub", r"this \[\[ is never closed", (9, 11)),
        ("links-to:[[|x]]", r"links-to: needs a note", (11, 13)),
        ("[[Hub]]", r"a \[\[link\]\] needs a field in front of it", (0, 7)),
        ("tag:", r"tag: needs a value", (0, 4)),
        ("reward NEAR/3", r"NEAR/3 needs a single word on each side", (7, 13)),
        ("NEAR/3 model", r"NEAR/3 needs a single word on each side", (0, 6)),
        ('"a b" NEAR/2 c', r"NEAR/2 needs a single word on each side", (6, 12)),
        ("a NEAR/2 tag:x", r"NEAR/2 needs a single word on each side", (2, 8)),
        ("a NEAR/2 b NEAR/2 c", r"a word can only be in one NEAR/n", (11, 17)),
        ("a NEAR/100 b", r"NEAR/n needs a distance from 1 to 99", (2, 10)),
        ("https://example.com", r'unknown field "https:"', (0, 6)),
    ]

    def test_messages_and_spans(self):
        for query, message, span in self.CASES:
            with self.subTest(query=query):
                if message is None:
                    parse(query)
                    continue
                with self.assertRaisesRegex(QueryError, message) as caught:
                    parse(query)
                self.assertEqual((caught.exception.start, caught.exception.end), span)

    def test_render_points_at_the_problem(self):
        with self.assertRaises(QueryError) as caught:
            parse("tga:ai-safety reward")
        self.assertEqual(
            caught.exception.render("tga:ai-safety reward").splitlines()[:3],
            ['error: unknown field "tga:"', "  tga:ai-safety reward", "  ^^^^"],
        )
        self.assertIn("did you mean tag:?", caught.exception.render("tga:ai-safety reward"))
        with self.assertRaises(QueryError) as caught:
            parse("reward (tag:x OR b")
        self.assertEqual(caught.exception.render("reward (tag:x OR b").splitlines()[1:3], ["  reward (tag:x OR b", "         ^"])


# ----- properties: where they come from, how values compare -----------------------------


class PropertyExtractionTests(unittest.TestCase):
    def fields(self, text):
        from vaultengine.properties import body_fields

        note, layout = parse_note_with_layout(text)
        return [(f.source, f.key, f.value) for f in body_fields(layout.text, layout.masked)]

    def test_header_fields_are_only_the_leading_run(self):
        text = "Created: 01-02-2026 10:00\n\nTags:\n\nStatus: #baby\n\n# Title\nNote: not a field\nStatus: also not\n"
        self.assertEqual(self.fields(text), [("header", "Created", "01-02-2026 10:00"), ("header", "Tags", ""), ("header", "Status", "#baby")])
        self.assertEqual(self.fields("Intro sentence.\nStatus: late\n"), [])
        self.assertEqual(self.fields("Two words: no\nStatus: x\n"), [])  # a key is one word

    def test_header_after_frontmatter_and_values_keep_inline_code(self):
        text = "---\ntitle: x\n---\nCommand: `git status: ok`\n"
        self.assertEqual(self.fields(text), [("header", "Command", "`git status: ok`")])

    def test_inline_fields(self):
        text = "# T\nplain:: 1\n- listed:: [[A]], [[B]]\n> quoted:: yes\nEmpty::\n```\ncode:: no\n```\n`inline:: no`\nurl https://x.y\n"
        self.assertEqual(
            self.fields(text),
            [("inline", "plain", "1"), ("inline", "listed", "[[A]], [[B]]"), ("inline", "quoted", "yes"), ("inline", "Empty", "")],
        )

    def test_value_items_and_dates(self):
        from vaultengine.properties import items, parse_date, parse_period

        self.assertEqual(items(json.dumps("#baby  #processed")), {"baby  #processed", "baby", "processed"})
        self.assertEqual(items(json.dumps(["A", "[[Note|shown]]"])), {"a", "note"})
        self.assertEqual(parse_date("24-09-2026 19:20"), datetime(2026, 9, 24, 19, 20))  # day first, this project's format
        self.assertEqual(parse_date("2026-09-24T19:20:05Z"), datetime(2026, 9, 24, 19, 20, 5))
        self.assertIsNone(parse_date("31-02-2026"))
        self.assertIsNone(parse_date("09/24/2026"))  # ambiguous formats are not guessed
        december = parse_period("2026-12")
        self.assertEqual((december.start, december.end), (datetime(2026, 12, 1), datetime(2027, 1, 1)))


# ----- every filter on a hand-built vault ------------------------------------------
#
# Resolved links (note -> note):  Paper A -> Hub,  Hub -> Leaf one,  Leaf one -> Leaf two,
# Leaf two -> Far,  Far -> Hub (through the inline field supports:: [[Hub]]).
# Unresolved: Paper A and Old -> [[GRPO]];  Hub -> [[Missing note]].
# Two notes are called Index (an ambiguous reference). Lonely links nowhere and nothing links to it.
# Tags: Paper A rl/policy-gradient, ai-safety (frontmatter); Hub ai-safety and draft (the #draft in
# "Status: #draft" is an inline tag, as in Obsidian); Leaf one done; Leaf two rl; Old rl/value-based.

QUERY_NOTES = {
    "Sources/Paper A.md": (
        "---\ntags: [rl/policy-gradient, ai-safety]\naliases: [PA]\ncreated: 2026-03-05\nrating: 4\n---\n"
        "# Paper A\nReward hacking in policy gradient. Links [[GRPO]] and [[Hub]].\n"
    ),
    "Notes/Hub.md": "Created: 12-06-2026 10:00\nStatus: #draft\n\n# Hub\nHub note linking [[Leaf one]] and [[Missing note]]. #ai-safety\n",
    "Notes/Leaf one.md": "Status: #done\n\nLeaf talks about reward. Links [[Leaf two]].\n",
    "Notes/Leaf two.md": "Robot learning notes. Links [[Far]]. #rl\n",
    "Notes/Far.md": "Far away note about hacking.\nsupports:: [[Hub]]\n",
    "Archive/Old.md": "---\nrating: 2\ncreated: 2025-01-10\n---\nOld GRPO notes link [[GRPO]]. #rl/value-based\n",
    "Notes/Index.md": "index\n",
    "Archive/Index.md": "index\n",
    "Lonely.md": "Nothing links here, and I link nowhere.\n",
}
ALL = set(QUERY_NOTES)
PAPER, HUB, LEAF1, LEAF2, FAR, OLD = (
    "Sources/Paper A.md", "Notes/Hub.md", "Notes/Leaf one.md", "Notes/Leaf two.md", "Notes/Far.md", "Archive/Old.md"
)
INDEXES, LONELY = {"Notes/Index.md", "Archive/Index.md"}, "Lonely.md"
MODIFIED = {PAPER: datetime(2026, 1, 15, 12), HUB: datetime(2026, 2, 20, 9)}  # everything else: 2020


class QueryVaultTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vault = TempVault(QUERY_NOTES)
        for path, when in MODIFIED.items():
            os.utime(cls.vault.root / path, (when.timestamp(), when.timestamp()))
        cls.ix = VaultIndex(default_db_path(cls.vault.root))
        cls.ix.update(cls.vault.root)

    @classmethod
    def tearDownClass(cls):
        cls.ix.close()
        cls.vault.cleanup()

    def run_query(self, query, **kwargs):
        return search(self.ix.conn, query, limit=None, **kwargs)

    def paths(self, query):
        return {hit.path for hit in self.run_query(query).hits}

    def check(self, cases):
        for query, expected in cases:
            with self.subTest(query=query):
                self.assertEqual(self.paths(query), set(expected))


class FilterTests(QueryVaultTestCase):
    def test_tags_include_nested_tags(self):
        self.check([
            ("tag:rl", {PAPER, LEAF2, OLD}),
            ("tag:rl/policy-gradient", {PAPER}),
            ("tag:rl/policy", set()),  # a nested tag matches by whole segments
            ("tag:#AI-Safety", {PAPER, HUB}),  # '#' optional, case-insensitive; frontmatter and inline tags
        ])

    def test_path(self):
        self.check([("path:Notes", {HUB, LEAF1, LEAF2, FAR, "Notes/Index.md"}), ("path:notes/leaf", {LEAF1, LEAF2}),
                    ('path:"Paper A"', {PAPER}), ("-path:Notes -path:Archive", {PAPER, LONELY})])

    def test_links_to(self):
        self.check([
            ("links-to:[[Hub]]", {PAPER, FAR}),  # Far's link is in an inline field
            ("links-to:Notes/Hub", {PAPER, FAR}),
            ("links-to:[[hub.md]]", {PAPER, FAR}),
            ("links-to:[[PA]]", set()),  # an alias of Paper A: resolves, nobody links to it
            ("links-to:[[Notes/Index]]", set()),
            ("links-to:[[GRPO]]", {PAPER, OLD}),  # no such note: the links written to it
        ])
        self.assertEqual(self.run_query("links-to:[[GRPO]]").warnings,
                         ["[[GRPO]] is not a file in the vault; using the links written to it (2 notes)"])
        self.assertEqual(self.run_query("links-to:[[Nothing]]").warnings,
                         ["[[Nothing]] is not a file in the vault, and no note links to it"])

    def test_ambiguous_reference_is_an_error_at_its_span(self):
        with self.assertRaisesRegex(QueryError, r"\[\[Index\]\] could be 2 notes: Archive/Index.md, Notes/Index.md") as caught:
            self.run_query("reward links-to:[[Index]]")
        self.assertEqual((caught.exception.start, caught.exception.end), (7, 25))
        self.assertEqual(caught.exception.hint, "write the path, e.g. [[Archive/Index]]")

    def test_linked_from(self):
        self.check([("linked-from:[[Hub]]", {LEAF1}), ("linked-from:[[Paper A]]", {HUB}), ("linked-from:[[Lonely]]", set())])
        with self.assertRaisesRegex(QueryError, r"\[\[Missing note\]\] is not a note in the vault"):
            self.run_query("linked-from:[[Missing note]]")

    def test_within_hops_either_direction(self):
        self.check([
            ("within:1 of:[[Hub]]", {PAPER, LEAF1, FAR}),
            ("within:2 of:[[Hub]]", {PAPER, LEAF1, FAR, LEAF2}),
            ("within:1 of:[[Paper A]]", {HUB}),
            ("within:2 of:[[Paper A]]", {HUB, LEAF1, FAR}),
            ("within:3 of:[[Paper A]]", {HUB, LEAF1, FAR, LEAF2}),
            ("within:10 of:[[Lonely]]", set()),
            ("within:1 of:[[GRPO]]", {PAPER, OLD}),  # a missing note: its linkers are one hop away
            ("within:2 of:[[GRPO]]", {PAPER, OLD, HUB}),
        ])

    def test_has_and_missing(self):
        self.check([
            ("has:tags", {PAPER, HUB, LEAF1, LEAF2, OLD}),
            ("has:aliases", {PAPER}),
            ("has:headings", {PAPER, HUB}),
            ("has:frontmatter", {PAPER, OLD}),
            ("has:links", {PAPER, HUB, LEAF1, LEAF2, FAR}),  # a resolved link to another note
            ("has:backlinks", {HUB, LEAF1, LEAF2, FAR}),
            ("has:unresolved", {PAPER, HUB, OLD}),
            ("has:properties", {PAPER, HUB, LEAF1, FAR, OLD}),
            ("missing:backlinks", {PAPER, OLD, LONELY} | INDEXES),
            ("missing:status", ALL - {HUB, LEAF1}),
            ("has:status", {HUB, LEAF1}),  # a header field
            ("has:supports", {FAR}),  # an inline field
            ("has:rating", {PAPER, OLD}),  # frontmatter
        ])

    def test_property_values(self):
        self.check([
            ("[status=done]", {LEAF1}),
            ("[status=#draft]", {HUB}),
            ("[status:dra]", {HUB}),  # ':' is a substring test
            ("[Status=DRAFT]", {HUB}),  # keys and values are case-insensitive
            ("[supports=hub] [supports:Hub]", {FAR}),  # a [[link]] value matches its target
            ("[rating>=3]", {PAPER}),
            ("[rating<3]", {OLD}),
            ("[rating>4]", set()),
            ("[rating=4]", {PAPER}),
            ("[rating:4]", {PAPER}),
            ("[rating>2026]", set()),  # a number, not a year: ratings are numbers
        ])

    def test_property_dates(self):
        self.check([
            ("[created>=2026]", {PAPER, HUB}),
            ("[created<2026-03-05]", {OLD}),
            ("[created>2026-03-05]", {HUB}),  # after that day: Paper A (on it) is out
            ("[created>=2026-03-05]", {PAPER, HUB}),
            ("[created:2026-06]", {HUB}),  # Hub's header date is day-first: 12-06-2026
            ("[created=2026-06-12]", {HUB}),
            ("[created:2025..2026-03]", {OLD, PAPER}),
        ])

    def test_modified_dates(self):
        self.check([
            ("modified:2026", {PAPER, HUB}),
            ("modified:2026-02", {HUB}),
            ("modified:<2026", ALL - {PAPER, HUB}),
            ("modified:2026-01..2026-01", {PAPER}),
            ("modified:>=2026-02-20", {HUB}),
            ("modified:>2026-02-20", set()),
        ])

    def test_text_with_filters(self):
        self.check([
            ("reward tag:ai-safety", {PAPER}),  # Leaf one mentions reward but has no tag
            ("reward hacking", {PAPER, LEAF1, FAR}),  # any of the words
            ("reward AND hacking", {PAPER}),
            ("hacking -path:Sources", {FAR}),
            ("reward OR tag:rl", {PAPER, LEAF1, LEAF2, OLD}),
            ('"policy gradient" OR far', {PAPER, FAR, LEAF2}),  # Leaf two's [[Far]] link is text too
            ("(tag:ai-safety OR tag:rl) -has:frontmatter", {HUB, LEAF2}),
            ("NOT (tag:rl OR path:Notes)", {LONELY, "Archive/Index.md"}),
            ("-tag:rl -path:Notes -path:Archive", {LONELY}),
        ])

    def test_negated_words_are_exact(self):
        self.assertEqual(self.paths("root"), {LEAF2})  # 'robot' is one edit from 'root'
        self.assertEqual(self.paths("path:Notes -root"), {HUB, LEAF1, LEAF2, FAR, "Notes/Index.md"})  # not excluded
        self.assertEqual(self.paths("path:Notes -robot"), {HUB, LEAF1, FAR, "Notes/Index.md"})
        self.assertEqual(self.paths("path:Notes -(root OR rewards)"), {HUB, LEAF1, LEAF2, FAR, "Notes/Index.md"})

    def test_ranking_text_scores_filters_do_not(self):
        result = self.run_query("reward hacking path:Notes OR reward hacking path:Sources")
        self.assertEqual([(h.path, h.exact) for h in result.hits][0], (PAPER, 2))  # both words exactly first
        self.assertEqual({h.path for h in result.hits[1:]}, {LEAF1, FAR})
        self.assertGreater(result.hits[1].score, result.hits[2].score)
        no_text = self.run_query("tag:rl")
        self.assertEqual([h.path for h in no_text.hits], sorted([PAPER, LEAF2, OLD]))  # no words: by path
        self.assertEqual({h.score for h in no_text.hits}, {0.0})
        plain = {h.path: h.score for h in search(self.ix.conn, "reward hacking", limit=None).hits}
        filtered = {h.path: h.score for h in self.run_query("reward hacking path:Notes").hits}
        self.assertEqual(filtered, {p: s for p, s in plain.items() if p.startswith("Notes/")})  # filters don't change scores

    def test_facets_count_the_whole_result(self):
        result = search(self.ix.conn, "tag:rl OR tag:ai-safety", limit=1)
        self.assertEqual(result.total, 4)
        self.assertEqual(result.facets.tags, [("rl", 3), ("ai-safety", 2), ("draft", 1), ("rl/policy-gradient", 1), ("rl/value-based", 1)])
        self.assertEqual(result.facets.folders, [("Notes", 2), ("Archive", 1), ("Sources", 1)])
        root = search(self.ix.conn, "nowhere", limit=None)
        self.assertEqual(root.facets.folders, [("", 1)])  # Lonely.md is in the vault root

    def test_incremental_updates_reach_filters(self):
        vault = TempVault(QUERY_NOTES)
        try:
            with VaultIndex(default_db_path(vault.root)) as ix:
                ix.update(vault.root)
                vault.write("Lonely.md", "Status: #done\n\nNow I link [[Hub]]. #rl\n")
                ix.update(vault.root)
                self.assertEqual({h.path for h in search(ix.conn, "links-to:[[Hub]]", limit=None).hits}, {PAPER, FAR, LONELY})
                self.assertEqual({h.path for h in search(ix.conn, "[status=done]", limit=None).hits}, {LEAF1, LONELY})
                self.assertIn(LONELY, {h.path for h in search(ix.conn, "within:1 of:[[Hub]] tag:rl", limit=None).hits})
                (vault.root / "Notes/Leaf one.md").unlink()
                ix.update(vault.root)
                self.assertEqual({h.path for h in search(ix.conn, "within:2 of:[[Hub]]", limit=None).hits}, {PAPER, FAR, LEAF2, LONELY})
        finally:
            vault.cleanup()


# ----- the sample vault, against independent computations ----------------------------


class SampleVaultQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_vaultengine_search import SampleVaultReferenceTests

        cls.ix = SampleVaultReferenceTests.shared_index()
        cls.vault = Vault.load(SAMPLE)  # parsed and resolved in memory, without the SQLite index
        cls.graph = LinkGraph(cls.vault)
        cls.words = {}
        for path, text in ((p.relative_to(SAMPLE).as_posix(), p.read_text(encoding="utf-8")) for p in SAMPLE.rglob("*.md")):
            note, layout = parse_note_with_layout(text, path)
            cls.words[path] = {t for values in note_fields(path, note, layout).values() for v in values for t in v}

    def paths(self, query, **kwargs):
        return {hit.path for hit in search(self.ix.conn, query, limit=None, **kwargs).hits}

    def test_question_1_reward_hacking_linking_to_grpo_outside_sources(self):
        query = "reward hacking links-to:[[grpo-group-relative-policy-optimization]] -path:10_SOURCES"
        key = "grpo-group-relative-policy-optimization"
        linkers = {src for src, targets in self.graph.unresolved_links.items() if any(lookup_key(t) == key for t in targets)}
        self.assertEqual(len(linkers), 10)
        expected = {p for p in linkers if not p.startswith("10_SOURCES/") and self.words[p] & {"reward", "hacking"}}
        self.assertEqual(self.paths(query, fuzzy=False), expected)
        self.assertEqual(
            [h.path for h in search(self.ix.conn, query, limit=None).hits],
            [
                "20_ZETTLEKASTEN/instance-specific-rubric-as-reward-function.md",
                "20_ZETTLEKASTEN/the-critic-deletion-tension-in-agentic-rl.md",
                "20_ZETTLEKASTEN/trajectory-as-the-atomic-unit-of-policy-optimization.md",
                "20_ZETTLEKASTEN/zero-gradient-prompt-filtering.md",
            ],
        )

    def test_question_2_two_hops_tagged_ai_safety(self):
        origin = "20_ZETTLEKASTEN/obfuscated-reward-hacking.md"
        neighbours = defaultdict(set)
        for source, targets in self.graph.resolved_links.items():
            for target in targets:
                if target != source and target.endswith(".md"):
                    neighbours[source].add(target)
                    neighbours[target].add(source)
        distance, queue = {origin: 0}, deque([origin])
        while queue:
            current = queue.popleft()
            for other in neighbours[current]:
                if other not in distance and distance[current] < 2:
                    distance[other] = distance[current] + 1
                    queue.append(other)
        tagged = {p for p, note in self.vault.notes.items() if "ai-safety" in {t.lower() for t in note.tag_names}}
        expected = {p for p, d in distance.items() if 1 <= d <= 2} & tagged
        self.assertEqual(len(expected), 12)
        self.assertEqual(self.paths("within:2 of:[[obfuscated-reward-hacking]] tag:ai-safety"), expected)

    def test_question_3_missing_a_property(self):
        def header_keys(text):
            keys = set()
            for line in text.split("\n"):
                if line.strip():
                    match = re.fullmatch(r"([A-Za-z][\w-]*):(\s.*)?", line)
                    if not match:
                        break
                    keys.add(match.group(1).lower())
            return keys

        texts = {p.relative_to(SAMPLE).as_posix(): p.read_text(encoding="utf-8") for p in SAMPLE.rglob("*.md")}
        expected = {p for p, t in texts.items() if "status" not in header_keys(t)}  # the sample has no frontmatter
        self.assertEqual(
            expected,
            {
                "50_MOC/comparisons/proactive-vs-reactive-memory-management.md",
                "50_MOC/topics/emergent-misalignment-from-reward-hacking.md",
                "50_MOC/topics/llm-judge-score-bias-and-contrastive-decoding.md",
            },
        )
        self.assertEqual(self.paths("missing:status"), expected)
        self.assertEqual(self.paths("path:20_ZETTLEKASTEN missing:created"),
                         {p for p, t in texts.items() if p.startswith("20_ZETTLEKASTEN/") and "created" not in header_keys(t)})

    def test_more_filters_against_independent_counts(self):
        notes = self.vault.notes
        self.assertEqual(self.paths("tag:ai-safety"), {p for p, n in notes.items() if "ai-safety" in n.tag_names})
        self.assertEqual(self.paths("-path:10_SOURCES -path:50_MOC"), {p for p in notes if p.startswith("20_ZETTLEKASTEN/")})
        texts = {p.relative_to(SAMPLE).as_posix(): p.read_text(encoding="utf-8") for p in SAMPLE.rglob("*.md")}
        self.assertEqual(self.paths("[strength=B]"), {p for p, t in texts.items() if re.search(r"^Strength: B\s*$", t, re.M)})
        backlinked = {t for src, targets in self.graph.resolved_links.items() for t in targets if t != src}
        self.assertEqual(self.paths("missing:backlinks"), set(notes) - backlinked)
        self.assertEqual(len(self.paths("missing:backlinks")), 41)  # = the orphans in `vault health`
        target = "20_ZETTLEKASTEN/reward-hacking-as-specification-gaming.md"
        self.assertEqual(self.paths("links-to:[[reward-hacking-as-specification-gaming]]"),
                         {src for src, targets in self.graph.resolved_links.items() if target in targets and src != target})

    def test_facets_against_independent_counts(self):
        result = search(self.ix.conn, "links-to:[[reward-hacking-as-specification-gaming]]", limit=None, facets=None)
        ids = {h.path for h in result.hits}
        folders = defaultdict(int)
        for path in ids:
            parts = path.split("/")[:-1]
            for depth in range(1, len(parts) + 1):
                folders["/".join(parts[:depth])] += 1
        self.assertEqual(dict(result.facets.folders), dict(folders))
        tags = defaultdict(int)
        for path in ids:
            for tag in {t.lower() for t in self.vault.notes[path].tag_names}:
                tags[tag] += 1
        self.assertEqual(dict(result.facets.tags), dict(tags))


# ----- CLI ---------------------------------------------------------------------


class QueryCliTests(QueryVaultTestCase):
    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {"COLUMNS": "80"}), redirect_stdout(out), redirect_stderr(err):
            code = cli_main(["search", str(self.vault.root), *args])
        return code, out.getvalue(), err.getvalue()

    def test_results_with_facets_and_warnings(self):
        code, out, err = self.run_cli("tag:rl OR links-to:[[GRPO]]", "--color", "never")
        self.assertEqual((code, err), (0, ""))
        for line in (
            "  note: [[GRPO]] is not a file in the vault; using the links written to it (2 notes)",
            "  3 of 9 notes match · by path (no words to rank by)",
            "  Tags (notes)",
            "      3  #rl",
            "      1  #rl/policy-gradient",
            "  Folders (notes)",
            "      1  Archive/",
        ):
            self.assertIn(line, out.splitlines())

    def test_filter_only_group_is_labelled(self):
        _, out, _ = self.run_cli("reward OR tag:rl", "--color", "never")
        self.assertIn("  ── matches the filters, none of the words ──", out.splitlines())

    def test_tree(self):
        code, out, _ = self.run_cli("reward -path:Sources OR [rating>3]", "--tree")
        self.assertEqual((code, out), (0, '(or (and (not (path "Sources")) (any reward)) (property rating > "3"))\n'))

    def test_errors(self):
        code, out, err = self.run_cli("reward (tag:x OR b")
        self.assertEqual((code, out), (2, ""))
        self.assertEqual(err, "error: this ( is never closed\n  reward (tag:x OR b\n         ^\n  add ) after the group\n")
        code, out, _ = self.run_cli("links-to:[[Index]]", "--json")
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out)["error"]["start"], 0)
        self.assertIn("could be 2 notes", json.loads(out)["error"]["message"])


def tearDownModule():
    from test_vaultengine_search import tearDownModule as close_shared_index

    close_shared_index()


if __name__ == "__main__":
    unittest.main()
