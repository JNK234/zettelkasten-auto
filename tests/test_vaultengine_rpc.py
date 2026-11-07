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
}
SESSION_FORMATS = {"sample_vault_format": "brain"}
REPO_ROOT = Path(__file__).parent.parent
BRAIN_PATH = REPO_ROOT / "formats" / "brain.yaml"
RECORD = os.environ.get("VAULTENGINE_RECORD") == "1"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def schema_validator(definition):
    return jsonschema.Draft202012Validator({"$defs": SCHEMA["$defs"], "$ref": f"#/$defs/{definition}"})


def normalize(value, vault_root):
    """Replace run-specific values (temp paths, timings, timestamps) with placeholders of the same type."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "seconds" and isinstance(item, (int, float)):
                out[key] = 0.0  # keeps the schema type (number)
            elif key == "last_indexed_at" and isinstance(item, str):
                out[key] = "<TIMESTAMP>"
            else:
                out[key] = normalize(item, vault_root)
        return out
    if isinstance(value, list):
        return [normalize(item, vault_root) for item in value]
    if isinstance(value, str):
        return value.replace(vault_root, "<VAULT>").replace(str(REPO_ROOT), "<REPO>")
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
        name = path.stem
        server = RpcServer(VaultService(vault.root, format_path=BRAIN_PATH if name in SESSION_FORMATS else None))
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
        self.assertEqual(codes, {-32700, -32600, -32601, -32602, -32001, -32002, -32003, -32004})


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
