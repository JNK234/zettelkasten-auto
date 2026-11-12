"""JSON-RPC 2.0 server tests.

Recorded sessions live in tests/fixtures/rpc/<session>.jsonl, one exchange per
line: {"request": <message>, "response": <message or null>} or
{"raw": "<unparseable line>", "response": ...}. They are replayed through a
real `python -m vaultengine serve` subprocess over stdio.

To re-record after an intended behaviour change (then review the diff):
    VAULTENGINE_RECORD=1 python -m pytest tests/test_vaultengine_rpc.py -k replay
"""

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jsonschema

from vaultengine.fuzzy import fuzzy_match
from vaultengine.rpc.models import METHODS
from vaultengine.rpc.schema import SCHEMA_PATH, schema_text
from vaultengine.rpc.server import RpcServer, serve_stdio
from vaultengine.service import VaultService

FIXTURES = Path(__file__).parent / "fixtures"
SESSIONS = {
    "sample_vault": FIXTURES / "vault",
    "resolver_vault": FIXTURES / "resolver_vault",
    "sample_vault_format": FIXTURES / "vault",  # served with --format brain: check, fix, coverage, writes, ingest
    "sample_vault_explore": FIXTURES / "vault",  # served with --format brain: explore, marks, link it, map history
    # served with --format core: the baseline, pipeline, note facts, pages, findings, ideas, move, merge, next
    "sample_vault_core": FIXTURES / "vault",
}
SESSION_FORMATS = {"sample_vault_format": "brain", "sample_vault_explore": "brain", "sample_vault_core": "core"}
REPO_ROOT = Path(__file__).parent.parent
BRAIN_PATH = REPO_ROOT / "formats" / "brain.yaml"
RECORD = os.environ.get("VAULTENGINE_RECORD") == "1"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def schema_validator(definition):
    return jsonschema.Draft202012Validator({"$defs": SCHEMA["$defs"], "$ref": f"#/$defs/{definition}"})


_DATES = re.compile(r"\b(\d{2})-(\d{2})-(\d{4}) \d{2}:\d{2}\b"
                    r"|\b(\d{4})-(\d{2})-(\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2}|Z)?)?\b")


def _today() -> set[tuple[str, str, str]]:
    import datetime as dt

    return {(d.strftime("%Y"), d.strftime("%m"), d.strftime("%d"))
            for d in (dt.datetime.now(), dt.datetime.now(dt.timezone.utc))}


def _now_to_placeholder(text: str) -> str:
    """Dates of today (a new note's Created line, a log line, a refresh time) become <NOW>; others stay."""
    today = _today()

    def swap(match: re.Match) -> str:
        day, month, year = (match[1], match[2], match[3]) if match[1] else (match[6], match[5], match[4])
        return "<NOW>" if (year, month, day) in today else match[0]

    return _DATES.sub(swap, text)


TIMESTAMP_KEYS = {"last_indexed_at", "recorded_at", "refreshed_at", "modified", "at", "snapshot", "computed_at"}


def normalize(value, vault_root):
    """Replace run-specific values (temp paths, timings, timestamps) with placeholders of the same type.
    Today's dates and times inside text (a new note's Created line, a log line, a diff) become <NOW>."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "seconds" and isinstance(item, (int, float)):
                out[key] = 0.0  # keeps the schema type (number)
            elif key in TIMESTAMP_KEYS and isinstance(item, str):
                out[key] = "<TIMESTAMP>"
            else:
                out[key] = normalize(item, vault_root)
        return out
    if isinstance(value, list):
        return [normalize(item, vault_root) for item in value]
    if isinstance(value, str):
        return _now_to_placeholder(value.replace(vault_root, "<VAULT>").replace(str(REPO_ROOT), "<REPO>"))
    return value


def load_session(name):
    path = FIXTURES / "rpc" / f"{name}.jsonl"
    return path, [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wire_line(exchange):
    return exchange["raw"] if "raw" in exchange else json.dumps(exchange["request"], ensure_ascii=False)


class TempVaultCopy:
    def __init__(self, source):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve() / "vault"
        # copy2 keeps mtimes, so runs are deterministic. An index someone built in the
        # fixture (e.g. by trying the CLI on it) is never copied: sessions start unindexed.
        shutil.copytree(source, self.root, ignore=shutil.ignore_patterns(".vaultengine"))
        # Files fresh from a checkout are "racily clean" (modified within 2 s of an
        # index run) and get re-hashed on the next run, which would change recorded
        # counts. Pin every mtime well in the past so sessions don't depend on timing.
        for path in self.root.rglob("*"):
            if path.is_file():
                os.utime(path, (1_600_000_000, 1_600_000_000))

    def cleanup(self):
        self._tmp.cleanup()


class RecordedSessionReplayTests(unittest.TestCase):
    def replay(self, name):
        path, exchanges = load_session(name)
        vault = TempVaultCopy(SESSIONS[name])
        try:
            if RECORD:
                self.record(path, exchanges, vault)
                return
            stdin = "".join(wire_line(e) + "\n" for e in exchanges)
            serve_args = ["--format", SESSION_FORMATS[name]] if name in SESSION_FORMATS else []
            proc = subprocess.run(
                [sys.executable, "-m", "vaultengine", "serve", str(vault.root), *serve_args],
                input=stdin.encode("utf-8"), capture_output=True, cwd=REPO_ROOT, timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
            outputs = [json.loads(line) for line in proc.stdout.decode("utf-8").splitlines()]
            expected = [e for e in exchanges if e["response"] is not None]
            self.assertEqual(len(outputs), len(expected), "number of response lines")
            for number, (exchange, actual) in enumerate(zip(expected, outputs), start=1):
                with self.subTest(exchange=number, sent=wire_line(exchange)[:80]):
                    self.assertEqual(normalize(actual, str(vault.root)), exchange["response"])
        finally:
            vault.cleanup()

    def record(self, path, exchanges, vault):
        from vaultengine.settings import load_settings

        name = path.stem
        # The settings `vault serve <vault> [--format NAME]` has: the vault (and format) from its flags.
        fmt = SESSION_FORMATS.get(name)
        flags = [("vault", str(vault.root), "<vault>")] + ([("format", fmt, "--format")] if fmt else [])
        settings = load_settings(flags, env={k: v for k, v in os.environ.items() if not k.startswith("VAULTENGINE_")})
        server = RpcServer(VaultService(vault.root, format_path=REPO_ROOT / "formats" / f"{fmt}.yaml" if fmt else None,
                                        settings=settings))
        try:
            for exchange in exchanges:
                reply = server.handle_text(wire_line(exchange))
                exchange["response"] = None if reply is None else normalize(json.loads(reply), str(vault.root))
        finally:
            server.service.close()
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in exchanges), encoding="utf-8")

    def test_replay_sample_vault(self):
        self.replay("sample_vault")

    def test_replay_resolver_vault(self):
        self.replay("resolver_vault")

    def test_replay_sample_vault_format(self):
        self.replay("sample_vault_format")

    def test_replay_sample_vault_explore(self):
        self.replay("sample_vault_explore")

    def test_replay_sample_vault_core(self):
        self.replay("sample_vault_core")


class RecordedSessionContentTests(unittest.TestCase):
    """Checks on the recordings themselves, so they can't silently encode wrong behaviour."""

    def exchanges(self):
        for name in SESSIONS:
            _, exchanges = load_session(name)
            yield from ((name, e) for e in exchanges)

    def test_every_recorded_response_matches_the_json_schema(self):
        response_schema = schema_validator("Response")
        for name, exchange in self.exchanges():
            responses = exchange["response"] if isinstance(exchange["response"], list) else [exchange["response"]]
            for response in filter(None, responses):
                with self.subTest(session=name, id=response.get("id")):
                    response_schema.validate(response)

    def test_successful_requests_match_the_json_schema(self):
        request_schema = schema_validator("Request")
        for name, exchange in self.exchanges():
            requests = exchange.get("request")
            responses = exchange["response"]
            pairs = zip(requests, responses) if isinstance(requests, list) and isinstance(responses, list) else [(requests, responses)]
            for request, response in pairs:
                if isinstance(response, dict) and "result" in response:
                    request_schema.validate(request)

    def test_each_method_is_covered_by_a_recording(self):
        recorded = {e["request"].get("method") for _, e in self.exchanges() if isinstance(e.get("request"), dict)}
        self.assertTrue(set(METHODS) <= recorded, set(METHODS) - recorded)

    def test_each_error_code_is_covered_by_a_recording(self):
        codes = set()
        for _, exchange in self.exchanges():
            responses = exchange["response"] if isinstance(exchange["response"], list) else [exchange["response"]]
            codes.update(r["error"]["code"] for r in filter(None, responses) if "error" in r)
        # -32005 (write failed, rolled back) needs a crash or a concurrent edit; see WriteFailedErrorTests.
        self.assertEqual(codes, {-32700, -32600, -32601, -32602, -32001, -32002, -32003, -32004, -32006, -32007, -32008})


class CoreSessionContentTests(unittest.TestCase):
    """What the sample_vault_core recording must show (served with --format core)."""

    @classmethod
    def setUpClass(cls):
        _, exchanges = load_session("sample_vault_core")
        cls.by_id = {e["request"]["id"]: (e["request"], e["response"]) for e in exchanges}

    def result(self, number):
        request, response = self.by_id[number]
        self.assertIn("result", response, (request, response))
        return response["result"]

    def error(self, number):
        return self.by_id[number][1]["error"]["code"]

    def test_baseline(self):
        self.assertEqual(self.error(2), -32008)  # none recorded yet
        everything = self.result(3)
        self.assertFalse(everything["ok"])
        recorded = self.result(4)["baseline"]
        self.assertTrue(recorded["recorded_now"])
        self.assertEqual(recorded["violations"], everything["summary"]["violations"])
        compared = self.result(5)
        self.assertTrue(compared["ok"])
        self.assertEqual((compared["baseline"]["new"], compared["baseline"]["known"]), (0, recorded["violations"]))
        self.assertEqual(self.error(6), -32602)  # baseline and update_baseline together
        after_move = self.result(28)  # the moved note keeps its baselined violations: still nothing new
        self.assertTrue(after_move["ok"])

    def test_pages_ideas_findings(self):
        self.assertEqual(self.result(10)["status"], "dry-run")
        self.assertEqual(self.result(11)["from"][0]["query"], "verification")
        self.assertEqual(self.error(12), -32602)  # no zettels, no queries
        written = self.result(13)
        self.assertEqual((written["status"], written["page"]["path"]), ("written", "50_MOC/verification-gaps.md"))
        self.assertEqual(self.result(14)["pages"][0]["path"], "50_MOC/verification-gaps.md")
        self.assertEqual(self.result(16)["index"]["status"], "unchanged")  # only the engine state changes
        self.assertEqual(self.result(18)["total"], 94)
        self.assertEqual({f["method"] for f in self.result(19)["findings"]}, {"link"})
        self.assertEqual(self.error(20), -32002)
        idea = self.result(22)
        self.assertEqual(idea["finding"]["id"], "missing-32eacad6a1")
        self.assertIn("<!-- vault:finding -->", idea["idea"]["content"])
        self.assertEqual(self.result(23)["status"], "refused")  # "maybe" is not a status of the format
        self.assertEqual(self.result(24)["status"], "written")

    def test_merge_move_next(self):
        merge = self.result(25)
        self.assertEqual(merge["status"], "dry-run")
        self.assertEqual([(s["link"], s["action"]) for s in merge["self_links"]],
                         [("[[pass-hat-k-reliability-metric]]", "line removed")])
        self.assertEqual([f["action"] for f in merge["files"]].count("delete"), 1)
        move = self.result(27)
        self.assertEqual(move["moved"], {"20_ZETTLEKASTEN/agent-test-time-scaling-limited-by-verification-gap.md":
                                         "20_ZETTLEKASTEN/verification-gap-limits-test-time-scaling.md"})
        upcoming = self.result(29)
        self.assertEqual(len(upcoming["items"]), 3)
        self.assertTrue(all(item["command"].startswith("vault ") for item in upcoming["items"]))


class SameAsCliTests(unittest.TestCase):
    """The JSON-RPC methods return what the CLI prints with --json (less its "vault" and "ok")."""

    @classmethod
    def setUpClass(cls):
        cls.vault = TempVaultCopy(SESSIONS["sample_vault_core"])
        cls.core = REPO_ROOT / "formats" / "core.yaml"
        cls.service = VaultService(cls.vault.root, format_path=cls.core)
        cls.service.build()

    @classmethod
    def tearDownClass(cls):
        cls.service.close()
        cls.vault.cleanup()

    def cli(self, *args):
        result = subprocess.run([sys.executable, "-m", "vaultengine", args[0], str(self.vault.root), *args[1:], "-f",
                                 str(self.core), "--json"], capture_output=True, text=True, cwd=REPO_ROOT)
        data = json.loads(result.stdout)
        data.pop("vault", None)
        data.pop("ok", None)
        return data

    def test_read_methods(self):
        pipeline = self.service.pipeline()
        pipeline.pop("limit")
        self.assertEqual(pipeline, self.cli("pipeline"))
        facts = self.service.note_facts("a-bounty-list-is-a-map-of-what-cannot-be-verified")
        self.assertEqual(facts, self.cli("note", "a-bounty-list-is-a-map-of-what-cannot-be-verified"))
        found = self.service.findings()
        cli = self.cli("findings", "--limit", "0")
        self.assertEqual((found["total"], found["thresholds"], found["findings"]), (cli["total"], cli["thresholds"], cli["findings"]))
        upcoming = self.service.next(limit=None)
        cli = self.cli("next", "--limit", "0")
        self.assertEqual((upcoming["total"], upcoming["counts"]), (cli["total"], cli["counts"]))
        self.assertEqual([(i["id"], i["trust"]) for i in upcoming["items"]], [(i["id"], i["trust"]) for i in cli["items"]])


class ProtocolErrorTests(unittest.TestCase):
    """The error contract, asserted explicitly (also covered by the recordings)."""

    @classmethod
    def setUpClass(cls):
        cls.vault = TempVaultCopy(SESSIONS["resolver_vault"])
        cls.server = RpcServer(VaultService(cls.vault.root))
        cls.server.handle_text('{"jsonrpc":"2.0","id":0,"method":"index.build"}')

    @classmethod
    def tearDownClass(cls):
        cls.server.service.close()
        cls.vault.cleanup()

    def call(self, message):
        text = message if isinstance(message, str) else json.dumps(message)
        reply = self.server.handle_text(text)
        return None if reply is None else json.loads(reply)

    def assertError(self, message, code, expected_id="missing"):
        reply = self.call(message)
        self.assertEqual(reply["error"]["code"], code, reply)
        self.assertEqual(reply["jsonrpc"], "2.0")
        if expected_id != "missing":
            self.assertEqual(reply["id"], expected_id)
        return reply

    def test_malformed_requests_return_invalid_request(self):
        cases = [
            ({"jsonrpc": "1.0", "id": 1, "method": "index.status"}, 1),
            ({"id": 2, "method": "index.status"}, 2),
            ({"jsonrpc": "2.0", "id": 3, "method": 42}, 3),
            ({"jsonrpc": "2.0", "id": 4, "method": "index.status", "params": "x"}, 4),
            ({"jsonrpc": "2.0", "id": 5, "method": "index.status", "extra": True}, 5),
            ({"jsonrpc": "2.0", "id": {"not": "valid"}, "method": "index.status"}, None),
            ({"jsonrpc": "2.0", "id": 1.5, "method": "index.status"}, None),
            ({"jsonrpc": "2.0", "id": False, "method": "index.status"}, None),
            ({"jsonrpc": "2.0", "method": 7}, None),  # invalid even without id: still answered
            ("42", None),
            ('"index.status"', None),
            ("[]", None),
        ]
        for message, expected_id in cases:
            with self.subTest(message=message):
                self.assertError(message, -32600, expected_id)

    def test_batch_of_invalid_items_returns_one_error_each(self):
        replies = self.call("[1, 2, 3]")
        self.assertEqual([r["error"]["code"] for r in replies], [-32600] * 3)

    def test_unknown_method_returns_method_not_found(self):
        reply = self.assertError({"jsonrpc": "2.0", "id": 7, "method": "notes.delete"}, -32601, 7)
        self.assertEqual(reply["error"]["data"]["method"], "notes.delete")
        self.assertError({"jsonrpc": "2.0", "id": 8, "method": "rpc.discover"}, -32601, 8)
        self.assertError({"jsonrpc": "2.0", "id": 9, "method": "INDEX.BUILD"}, -32601, 9)

    def test_notifications_never_get_a_response(self):
        self.assertIsNone(self.call({"jsonrpc": "2.0", "method": "index.status"}))
        self.assertIsNone(self.call({"jsonrpc": "2.0", "method": "notes.delete"}))
        self.assertIsNone(self.call({"jsonrpc": "2.0", "method": "notes.pick", "params": {}}))
        self.assertIsNone(self.call('[{"jsonrpc": "2.0", "method": "index.status"}]'))

    def test_null_id_is_a_request_not_a_notification(self):
        reply = self.call({"jsonrpc": "2.0", "id": None, "method": "index.status"})
        self.assertIsNone(reply["id"])
        self.assertTrue(reply["result"]["built"])

    def test_parse_errors(self):
        for text in ["{", "not json", '{"jsonrpc": "2.0", "id": 1, "method": "x", "params": {"a": NaN}}']:
            with self.subTest(text=text):
                self.assertError(text, -32700, None)

    def test_invalid_params(self):
        for params in [{}, {"query": ""}, {"query": "x", "limit": 0}, {"query": "x", "limit": 201}, {"query": 5}, {"query": "x", "nope": 1}, ["x"]]:
            with self.subTest(params=params):
                reply = self.assertError({"jsonrpc": "2.0", "id": 1, "method": "notes.pick", "params": params}, -32602, 1)
                self.assertIn("data", reply["error"])

    def test_server_errors(self):
        self.assertError({"jsonrpc": "2.0", "id": 1, "method": "note.get", "params": {"path": "Nope.md"}}, -32002, 1)
        empty = TempVaultCopy(SESSIONS["resolver_vault"])
        try:
            server = RpcServer(VaultService(empty.root))
            reply = json.loads(server.handle_text('{"jsonrpc":"2.0","id":1,"method":"health.report"}'))
            self.assertEqual(reply["error"]["code"], -32001)
            self.assertFalse((empty.root / ".vaultengine").exists(), "read methods must not create the index")
        finally:
            empty.cleanup()

    def test_result_that_violates_its_model_is_an_internal_error(self):
        with patch("vaultengine.queries.stats", return_value={"unexpected": 1}), patch("sys.stderr", io.StringIO()):
            self.assertError({"jsonrpc": "2.0", "id": 1, "method": "vault.stats"}, -32603, 1)


class FormatMethodTests(unittest.TestCase):
    """Format methods: what the recordings cannot show (a failed commit, a changed format file, no format)."""

    def setUp(self):
        self.vault = TempVaultCopy(SESSIONS["sample_vault"])
        self.service = VaultService(self.vault.root, format_path=BRAIN_PATH)
        self.server = RpcServer(self.service)

    def tearDown(self):
        self.service.close()
        self.vault.cleanup()

    def call(self, method, params=None):
        message = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        return json.loads(self.server.handle_text(json.dumps(message)))

    def test_write_failed_is_error_32005_and_nothing_changes(self):
        from vaultengine.formats.write import WriteFailed

        before = {p: p.read_bytes() for p in self.vault.root.rglob("*.md")}
        with patch("vaultengine.formats.write.commit", side_effect=WriteFailed("disk full; every file was restored")):
            reply = self.call("note.set", {"note": "CLM - Grade the observed end state rather than the trajectory",
                                           "key": "Status", "value": "#contested"})
        self.assertEqual(reply["error"]["code"], -32005)
        self.assertEqual(reply["error"]["message"], "disk full; every file was restored")
        self.assertEqual({p: p.read_bytes() for p in self.vault.root.rglob("*.md")}, before)

    def test_the_format_file_is_reloaded_when_it_changes(self):
        with tempfile.TemporaryDirectory() as scratch:
            custom = Path(scratch) / "custom.yaml"
            custom.write_text(BRAIN_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            service = VaultService(self.vault.root, format_path=custom)
            server = RpcServer(service)
            first = json.loads(server.handle_text('{"jsonrpc":"2.0","id":1,"method":"vault.check","params":{"limit":1}}'))
            self.assertEqual(first["result"]["summary"]["violations"], 102)
            marker = "Tags as #hashtags.\n      Created: {required: true"
            text = custom.read_text(encoding="utf-8")
            self.assertIn(marker, text)
            custom.write_text(text.replace(marker, "Tags as #hashtags.\n      Created: {required: false"), encoding="utf-8")
            os.utime(custom, ns=(custom.stat().st_mtime_ns + 10**9,) * 2)
            second = json.loads(server.handle_text('{"jsonrpc":"2.0","id":2,"method":"vault.check","params":{"limit":1}}'))
            self.assertEqual(second["result"]["summary"]["violations"], 87)  # the 15 zettels without Created are fine now
            custom.write_text("format: 1\ntypes: {}\n", encoding="utf-8")
            os.utime(custom, ns=(custom.stat().st_mtime_ns + 2 * 10**9,) * 2)
            broken = json.loads(server.handle_text('{"jsonrpc":"2.0","id":3,"method":"vault.check"}'))
            self.assertEqual(broken["error"]["code"], -32003)
            self.assertIn("Invalid format file", broken["error"]["message"])
            service.close()

    def test_the_parse_cache_never_serves_a_stale_note(self):
        first = self.call("vault.check", {"limit": 1})["result"]["summary"]["violations"]
        claim = self.vault.root / "50_MOC/discourse/rl-environment-design/CLM - Grade the observed end state rather than the trajectory.md"
        mtime = claim.stat().st_mtime_ns
        claim.write_text(claim.read_text(encoding="utf-8").replace("Status: #supported", "Status: #settled"), encoding="utf-8")
        os.utime(claim, ns=(mtime, mtime))  # same mtime: the cache must go by content
        self.assertEqual(self.call("vault.check", {"limit": 1})["result"]["summary"]["violations"], first + 1)
        (self.vault.root / "10_SOURCES/RLHF.md").unlink()  # a deleted note leaves the cache
        after_delete = self.call("vault.check", {})["result"]
        self.assertNotIn("10_SOURCES/RLHF.md", [n["path"] for n in after_delete["notes"]])
        (self.vault.root / "20_ZETTLEKASTEN/brand-new.md").write_text("no header\n", encoding="utf-8")
        after_add = self.call("vault.check", {})["result"]
        self.assertIn("20_ZETTLEKASTEN/brand-new.md", [n["path"] for n in after_add["notes"]])

    def test_reads_do_not_create_an_index(self):
        self.assertIn("result", self.call("vault.check", {"limit": 1}))
        self.assertIn("result", self.call("vault.coverage", {"limit": 1}))
        self.assertFalse((self.vault.root / ".vaultengine").exists())

    def test_invalid_params(self):
        for method, params in [
            ("vault.check", {"changed": "yes"}), ("vault.check", {"limit": 0}), ("vault.fix", {"only": []}),
            ("vault.fix", {"only": ["everything"]}), ("note.new", {"type": "zettel"}), ("note.link", {"source": "a", "target": "b"}),
            ("note.ingest", {}), ("note.ingest", {"path": "a.md", "content": "x", "filename": "a.md"}),
            ("note.new", {"type": "zettel", "title": "t", "relations": [{"relation": "supports"}]}),
        ]:
            with self.subTest(method=method, params=params):
                self.assertEqual(self.call(method, params)["error"]["code"], -32602)

    def test_serve_refuses_a_bad_format_at_start(self):
        proc = subprocess.run([sys.executable, "-m", "vaultengine", "serve", str(self.vault.root), "--format", "nope"],
                              input=b"", capture_output=True, cwd=REPO_ROOT, timeout=60)
        self.assertEqual(proc.returncode, 2)
        self.assertIn(b"no format named 'nope'", proc.stderr)


class SearchAndConnectApiTests(unittest.TestCase):
    """search.query, note.preview, note.related, links.suggest, notes.suggest, mentions.unlinked on the sample vault."""

    @classmethod
    def setUpClass(cls):
        cls.vault = TempVaultCopy(SESSIONS["sample_vault"])
        cls.server = RpcServer(VaultService(cls.vault.root))
        cls.server.handle_text('{"jsonrpc":"2.0","id":0,"method":"index.build"}')

    @classmethod
    def tearDownClass(cls):
        cls.server.service.close()
        cls.vault.cleanup()

    def call(self, method, **params):
        reply = json.loads(self.server.handle_text(json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})))
        schema_validator("Response").validate(reply)
        return reply

    def test_query_syntax_error_points_at_the_problem(self):
        query = "reward (tag:ai-safety OR path:50_MOC"
        error = self.call("search.query", query=query)["error"]
        self.assertEqual((error["code"], error["message"]), (-32006, "this ( is never closed"))
        self.assertEqual(error["data"], {"query": query, "start": 7, "end": 8, "hint": "add ) after the group"})
        self.assertEqual(query[error["data"]["start"] : error["data"]["end"]], "(")
        unknown = self.call("search.query", query="tga:x")["error"]
        self.assertEqual((unknown["code"], unknown["data"]["start"], unknown["data"]["end"]), (-32006, 0, 4))
        self.assertIn("did you mean tag:?", unknown["data"]["hint"])

    def test_search_hits_snippets_and_facets(self):
        result = self.call("search.query", query="reward hacking", limit=3, facets=2)["result"]
        self.assertEqual((result["total"], result["order"], len(result["hits"])), (106, "exact_then_score", 3))
        top = result["hits"][0]
        self.assertEqual(top["path"], "50_MOC/topics/emergent-misalignment-from-reward-hacking.md")
        title = top["title"]
        self.assertEqual([title[m["start"] : m["end"]] for m in top["title_marks"]], ["reward", "hacking"])
        text = top["snippet"]["text"]
        self.assertTrue(all(text[m["start"] : m["end"]].lower() in ("reward", "hacking") for m in top["snippet"]["marks"]))
        self.assertEqual(len(result["facets"]["tags"]), 2)

    def test_preview_marks_every_match_as_search_counts_it(self):
        result = self.call("note.preview", path="20_ZETTLEKASTEN/monitorability-tax.md", query='"monitorability tax"')["result"]
        marked = [line["text"][m["start"] : m["end"]] for line in result["lines"] for m in line["marks"]]
        self.assertTrue(marked and all(word.lower() == "monitorability tax" for word in marked), marked)
        self.assertEqual(result["marks"], len(marked))
        self.assertEqual(result["first_match_line"], next(l["number"] for l in result["lines"] if l["marks"]))
        plain = self.call("note.preview", path="20_ZETTLEKASTEN/monitorability-tax.md")["result"]
        self.assertEqual((plain["marks"], plain["matched"]), (0, False))
        self.assertEqual(len(plain["lines"]), len(result["lines"]))
        self.assertEqual(self.call("note.preview", path="nope.md")["error"]["code"], -32002)

    def test_related_suggest_and_unlinked(self):
        related = self.call("note.related", note="kl-penalty-in-rlhf", limit=3)["result"]
        self.assertEqual(related["seed"]["path"], "20_ZETTLEKASTEN/kl-penalty-in-rlhf.md")
        # Tag links are tags: trust-region shares the tag [[policy-gradient-methods]] and comes before RLHF.
        self.assertIn("shares [[policy-gradient-methods]]", related["hits"][1]["reason"])
        self.assertEqual(related["hits"][2]["title"], "RLHF")
        self.assertFalse(related["hits"][2]["linked"])
        missing = self.call("note.related", note="rl-environment-design", limit=1)["result"]
        self.assertEqual((missing["seed"]["path"], len(missing["seed"]["linkers"])), (None, 52))
        self.assertEqual(self.call("note.related", note="nope")["error"]["code"], -32002)
        links = self.call("links.suggest", limit=2, path="20_ZETTLEKASTEN")["result"]["links"]
        self.assertTrue(all(p["a"].startswith("20_ZETTLEKASTEN/") and p["b"].startswith("20_ZETTLEKASTEN/") for p in links))
        self.assertEqual(self.call("links.suggest", note="nope")["error"]["code"], -32002)
        notes = self.call("notes.suggest", limit=1)["result"]
        self.assertEqual((notes["total_missing"], notes["notes"][0]["name"]), (612, "rl-environment-design"))
        unlinked = self.call("mentions.unlinked", limit=2, per_note=1)["result"]
        self.assertEqual((unlinked["total"], unlinked["notes"], len(unlinked["targets"])), (595, 79, 2))
        self.assertEqual([len(t["mentions"]) for t in unlinked["targets"]], [1, 1])
        one = self.call("mentions.unlinked", note="monitorability-tax")["result"]
        self.assertEqual((one["note"], one["total"]), ("20_ZETTLEKASTEN/monitorability-tax.md", 4))


class StdioTransportTests(unittest.TestCase):
    def run_stdio(self, text_lines, service):
        out = io.BytesIO()
        with patch("sys.stderr", io.StringIO()):
            serve_stdio(service, stdin=io.BytesIO(b"".join(text_lines)), stdout=out)
        return [json.loads(line) for line in out.getvalue().decode("utf-8").splitlines()]

    def test_blank_lines_ignored_and_bad_utf8_is_a_parse_error(self):
        vault = TempVaultCopy(SESSIONS["resolver_vault"])
        try:
            replies = self.run_stdio(
                [b"\n", b"   \n", b"\xff\xfe\n", b'{"jsonrpc":"2.0","id":1,"method":"index.status"}\n'], VaultService(vault.root)
            )
            self.assertEqual([r.get("error", {}).get("code") for r in replies], [-32700, None])
        finally:
            vault.cleanup()

    def test_prints_inside_handlers_cannot_corrupt_stdout(self):
        vault = TempVaultCopy(SESSIONS["resolver_vault"])
        try:
            service = VaultService(vault.root)
            original = service.status

            def noisy_status():
                print("debug noise from somewhere deep in the engine")
                return original()

            service.status = noisy_status
            # As in production, protocol bytes go to sys.stdout.buffer, the same
            # stream a stray print() would write to.
            shared = io.BytesIO()
            fake_stdout = io.TextIOWrapper(shared, encoding="utf-8", write_through=True)
            stdin = io.BytesIO(b'{"jsonrpc":"2.0","id":1,"method":"index.status"}\n')
            with patch("sys.stdout", fake_stdout), patch("sys.stderr", io.StringIO()) as stderr:
                serve_stdio(service, stdin=stdin)
            lines = shared.getvalue().decode("utf-8").splitlines()
            self.assertEqual(len(lines), 1, lines)
            self.assertIn("result", json.loads(lines[0]))
            self.assertIn("debug noise", stderr.getvalue())
        finally:
            vault.cleanup()

    def test_line_separator_characters_are_escaped(self):
        from vaultengine.rpc.server import encode

        line = encode({"text": "a\u2028b\u2029c\nd"})
        self.assertNotIn("\u2028", line)
        self.assertEqual(len(line.splitlines()), 1)
        self.assertEqual(json.loads(line)["text"], "a\u2028b\u2029c\nd")


class SchemaTests(unittest.TestCase):
    def test_committed_schema_is_up_to_date(self):
        self.assertEqual(
            SCHEMA_PATH.read_text(encoding="utf-8"), schema_text(),
            "run `vault rpc-schema --output vaultengine/rpc/schema.json`",
        )

    def test_schema_is_valid_draft_2020_12(self):
        jsonschema.Draft202012Validator.check_schema(SCHEMA)

    def test_schema_describes_every_method(self):
        self.assertEqual(list(SCHEMA["x-methods"]), list(METHODS))
        for name, method in SCHEMA["x-methods"].items():
            for part in ("params", "result", "request", "response"):
                self.assertIn(method[part]["$ref"].split("/")[-1], SCHEMA["$defs"], f"{name}.{part}")

    def test_schema_rejects_what_the_server_rejects(self):
        request = schema_validator("Request")
        request.validate({"jsonrpc": "2.0", "id": 1, "method": "notes.pick", "params": {"query": "x"}})
        for bad in [
            {"jsonrpc": "2.0", "id": 1, "method": "notes.pick", "params": {}},
            {"jsonrpc": "2.0", "id": 1, "method": "notes.pick", "params": {"query": "x", "limit": 0}},
            {"jsonrpc": "1.0", "id": 1, "method": "index.status"},
            {"jsonrpc": "2.0", "id": 1, "method": "notes.delete"},
        ]:
            with self.subTest(bad=bad):
                self.assertFalse(request.is_valid(bad))


class FuzzyTests(unittest.TestCase):
    def test_subsequence_matching(self):
        self.assertIsNotNone(fuzzy_match("alignmnt fakng", "Alignment Faking in Large Language Models"))
        self.assertIsNotNone(fuzzy_match("rlhf ppo", "PPO and RLHF - Deep Dive"))  # tokens in any order
        self.assertIsNone(fuzzy_match("ppox", "PPO and RLHF"))
        self.assertIsNone(fuzzy_match("   ", "anything"))

    def test_positions_point_at_matched_characters(self):
        match = fuzzy_match("grpo", "PPO vs GRPO")
        self.assertEqual("".join("PPO vs GRPO"[i] for i in match.positions).lower(), "grpo")

    def test_ranking(self):
        def score(query, text):
            return fuzzy_match(query, text).score

        self.assertGreater(score("ppo", "PPO"), score("ppo", "PPO and RLHF"))  # exact beats prefix
        self.assertGreater(score("ppo", "PPO and RLHF"), score("ppo", "Notes on PPO"))  # prefix beats substring
        self.assertGreater(score("ppo", "Notes on PPO"), score("ppo", "Proximal Policy Optimization"))  # substring beats scattered
        self.assertGreater(score("pg", "Policy Gradient"), score("pg", "upgrade"))  # word starts beat mid-word

    def test_pick_returns_each_note_once_with_its_best_field(self):
        vault = TempVaultCopy(SESSIONS["resolver_vault"])
        try:
            service = VaultService(vault.root)
            service.build()
            result = service.pick("proximal policy")
            paths = [r["path"] for r in result["results"]]
            self.assertEqual(len(paths), len(set(paths)))
            top = result["results"][0]
            self.assertEqual((top["path"], top["matched_field"]), ("Concepts/Proximal Policy Optimization.md", "title"))
            self.assertEqual(service.pick("PPO")["results"][0]["matched_field"], "alias")
            self.assertEqual(service.pick("o", limit=2)["total"], len(service.pick("o", limit=200)["results"]))
            service.close()
        finally:
            vault.cleanup()


class PickCacheTests(unittest.TestCase):
    def test_picker_sees_new_notes_after_rebuild_by_itself_or_another_client(self):
        vault = TempVaultCopy(SESSIONS["resolver_vault"])
        try:
            service = VaultService(vault.root)
            service.build()
            self.assertEqual(service.pick("grpo")["total"], 0)
            (vault.root / "GRPO.md").write_text("# GRPO")
            service.build()
            self.assertEqual([r["path"] for r in service.pick("grpo")["results"]], ["GRPO.md"])

            (vault.root / "Notes" / "Grpo results.md").write_text("# results")
            other = VaultService(vault.root)  # another client re-indexes the same database
            other.build()
            other.close()
            self.assertEqual(service.pick("grpo")["total"], 2)
            service.close()
        finally:
            vault.cleanup()

    def test_generation_only_changes_when_content_changes(self):
        vault = TempVaultCopy(SESSIONS["resolver_vault"])
        try:
            service = VaultService(vault.root)
            service.build()
            meta = service._built_index().meta
            first = meta()["generation"]
            service.build()
            self.assertEqual(meta()["generation"], first)
            (vault.root / "New.md").write_text("# new")
            service.build()
            self.assertEqual(int(meta()["generation"]), int(first) + 1)
            service.close()
        finally:
            vault.cleanup()


if __name__ == "__main__":
    unittest.main()


class ExploreSessionContentTests(unittest.TestCase):
    """What the recorded explore session (sample_vault_explore.jsonl, replayed through `vault serve` above) shows."""

    @classmethod
    def setUpClass(cls):
        _, exchanges = load_session("sample_vault_explore")
        cls.by_id = {e["request"]["id"]: e["response"] for e in exchanges}

    def result(self, number):
        return self.by_id[number]["result"]

    def targets(self, run):
        return [r["target"] for s in run["sections"] for r in s["results"]]

    def test_the_same_start_by_path_or_name_gives_the_same_answer(self):
        self.assertEqual(self.result(6), self.result(7))
        run = self.result(6)
        self.assertEqual((run["start"], run["start_kind"], run["numbers"]["source"]),
                         ("20_ZETTLEKASTEN/kl-penalty-in-rlhf.md", "note", "current"))
        for section in run["sections"]:
            for item in section["results"]:
                self.assertEqual(item["lens"], section["lens"])
                self.assertTrue(item["evidence"] and "\n" not in item["evidence"])

    def test_marks_kept_and_fresh(self):
        run = self.result(7)
        first, hidden, role = (next(r for s in run["sections"] for r in s["results"] if r["ref"] == ref) for ref in ("1.1", "6.1", "7.1"))
        self.assertEqual(self.result(8)["marked"][0]["target"], first["target"])
        self.assertEqual(self.result(10)["marked"][0]["verdict"], "dismiss")
        self.assertEqual(self.by_id[11]["error"]["code"], -32007)
        kept = {(k["lens"], k["target"]): k for k in self.result(12)["kept"]}
        self.assertEqual(set(kept), {("neighbourhood", first["target"]), ("hidden", hidden["target"])})
        self.assertEqual(kept[("hidden", hidden["target"])]["evidence"], hidden["evidence"])
        fresh = self.result(16)
        self.assertTrue(fresh["fresh"])
        self.assertFalse(set(self.targets(fresh)) & set(self.targets(run)))  # seen and dismissed: gone
        self.assertNotIn(role["target"], self.targets(fresh))
        order = [s["lens"] for s in fresh["sections"]]
        self.assertEqual(order[-1], "role")  # dismissed from: last
        self.assertEqual(fresh["numbers"]["source"], "reused")  # a link was written in between: 2 notes changed

    def test_link_writes_both_notes_map_since_sees_the_gap_closed_unlink_restores(self):
        dry, written = self.result(14), self.result(15)
        self.assertEqual((dry["status"], dry["written"]), ("dry-run", False))
        self.assertEqual((written["status"], written["kept"], len(written["files"])), ("written", True, 2))
        self.assertEqual([f["diff"] for f in dry["files"]], [f["diff"] for f in written["files"]])
        a, b = written["notes"]
        for change, other in zip(written["files"], (b, a)):
            self.assertIn(f"+- [[{Path(other).stem}]] — gap: ", change["diff"])
        since = self.result(17)
        closed = [g for g in since["gaps"] if g["status"] == "closed"]
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["added"], [sorted([a, b])])
        undone = self.result(19)
        self.assertEqual((undone["status"], undone["recorded"]), ("written", True))
        for old, new in zip(written["files"], undone["files"]):  # each line it added is removed again
            added = [l[1:] for l in old["diff"].splitlines() if l.startswith("+") and not l.startswith("+++")]
            removed = [l[1:] for l in new["diff"].splitlines() if l.startswith("-") and not l.startswith("---")]
            self.assertEqual(added, removed)

    def test_errors(self):
        self.assertEqual(self.by_id[1]["error"]["code"], -32001)
        self.assertEqual(self.by_id[25]["error"]["code"], -32002)
        self.assertEqual((self.by_id[26]["error"]["code"], self.by_id[26]["error"]["message"]), (-32007, "hidden only run from a note"))
        self.assertEqual(self.by_id[27]["error"]["code"], -32602)
