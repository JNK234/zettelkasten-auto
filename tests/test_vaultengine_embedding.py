"""Engine settings, the embedding provider, the vector cache in the index, and the CLI/RPC around them.

The Model2Vec provider is tested twice: on tiny models written here by hand (a tokenizer and a
safetensors file), where every vector can be computed on paper, and on the real
models/potion-base-8M that config/config.yaml points at (skipped when it is not there).
"""

import io
import json
import os
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vaultengine.cli import main as cli_main
from vaultengine.embedding import EmbeddingError, load_provider
from vaultengine.embedding.explain import _centered, shared_words, text_bag
from vaultengine.embedding.model2vec import Model2VecProvider
from vaultengine.embedding.store import embed_notes, note_text, note_vectors, status
from vaultengine.index import VaultIndex, default_db_path
from vaultengine.settings import REPO_CONFIG, SettingsError, load_settings

from embedding_model import MODEL_DIR, requires_model
from test_vaultengine_search import TempVault

REPO_ROOT = Path(__file__).resolve().parent.parent
NO_ENGINE_ENV = {k: v for k, v in os.environ.items() if not k.startswith("VAULTENGINE_")}


# ----- a tiny Model2Vec model, written by hand ---------------------------------------------

WORDS = ["alpha", "beta", "gamma", "re", "##ward"]


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, blobs, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        data = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def make_model(folder: Path, matrix: np.ndarray | None = None, dtype: str = "F32", extra: dict | None = None,
               config: dict | None = None) -> np.ndarray:
    """tokenizer.json (WordPiece over WORDS, [UNK] = 1) + model.safetensors; returns the embeddings."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

    folder.mkdir(parents=True, exist_ok=True)
    vocab = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(WORDS)}}
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.save(str(folder / "tokenizer.json"))
    if matrix is None:
        matrix = np.arange(len(vocab) * 3, dtype=np.float32).reshape(len(vocab), 3) % 5 - 1.5
    np_dtype = {"F32": "<f4", "F16": "<f2", "F64": "<f8", "I8": "i1"}[dtype]
    write_safetensors(folder / "model.safetensors", {"embeddings": (dtype, matrix.astype(np_dtype)), **(extra or {})})
    (folder / "config.json").write_text(json.dumps(config or {"model_type": "model2vec", "normalize": True}))
    return matrix


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class SyntheticModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name) / "tiny-model"
        self.matrix = make_model(self.folder)
        self.provider = Model2VecProvider(self.folder)

    def tearDown(self):
        self.tmp.cleanup()

    def test_vector_is_the_normalised_mean_of_the_known_token_rows(self):
        ids = {w: i + 2 for i, w in enumerate(WORDS)}
        # "zzz" is unknown: dropped, not averaged in as [UNK]. "Reward" is re + ##ward. Case is folded.
        got = self.provider.embed(["Alpha beta ALPHA zzz reward"])[0]
        rows = [ids["alpha"], ids["beta"], ids["alpha"], ids["re"], ids["##ward"]]
        expected = unit(self.matrix[rows].mean(axis=0))
        np.testing.assert_allclose(got, expected, atol=1e-6)
        self.assertEqual(got.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(got)), 1.0, places=6)

    def test_empty_and_unknown_only_texts_are_zero_vectors(self):
        got = self.provider.embed(["", "zzz qqq", "alpha"])
        self.assertFalse(got[0].any())
        self.assertFalse(got[1].any())
        self.assertTrue(got[2].any())
        self.assertEqual(self.provider.embed([]).shape, (0, 3))

    def test_words_cover_exactly_the_averaged_tokens(self):
        words = self.provider.words("Alpha, reward zzz")
        self.assertEqual(words, [("alpha", [2]), ("reward", [5, 6])])  # "," is unknown here, zzz too
        ids = [i for _, pieces in words for i in pieces]
        np.testing.assert_allclose(unit(self.provider.token_vectors(ids).mean(axis=0)),
                                   self.provider.embed(["Alpha, reward zzz"])[0], atol=1e-6)

    def test_model_id_is_the_folder_name_unless_configured(self):
        self.assertEqual(self.provider.model_id, "tiny-model")
        self.assertEqual(Model2VecProvider(self.folder, "my-id").model_id, "my-id")
        self.assertEqual(self.provider.describe(), "tiny-model (model2vec, 3 dims)")

    def test_half_precision_weights(self):
        folder = Path(self.tmp.name) / "f16"
        matrix = make_model(folder, dtype="F16")
        got = Model2VecProvider(folder).embed(["alpha gamma"])[0]
        np.testing.assert_allclose(got, unit(matrix.astype(np.float16).astype(np.float64)[[2, 4]].mean(axis=0)), atol=1e-3)

    def test_any_model2vec_folder_is_a_settings_change(self):
        config = Path(self.tmp.name) / "config.yaml"
        config.write_text("engine:\n  embedding:\n    model_path: tiny-model\n")
        provider = load_settings(config_file=config, env={}).provider()
        self.assertEqual((provider.model_id, provider.dim), ("tiny-model", 3))
        env = {"VAULTENGINE_EMBEDDING__MODEL_PATH": str(Path(self.tmp.name) / "f16")}
        make_model(Path(env["VAULTENGINE_EMBEDDING__MODEL_PATH"]), dtype="F16")
        self.assertEqual(load_settings(config_file=config, env=env).provider().model_id, "f16")

    def test_unusable_models_are_refused_with_the_reason(self):
        base = Path(self.tmp.name)
        cases = {
            "missing": (None, "embedding model folder not found"),
            "no-tokenizer": ("no tokenizer", "is not a Model2Vec model: no tokenizer.json"),
            "wrong-name": ({"weights": ("F32", np.zeros((7, 3), np.float32))}, "expected exactly one tensor 'embeddings'"),
            "quantized": ({"mapping": ("F32", np.zeros(7, np.float32))}, "vocabulary-quantized Model2Vec models are not supported"),
            "int8": ("int8", "has dtype I8"),
            "short": ("short", "the tokenizer has 7 tokens but 'embeddings' has 4 rows"),
            "not-m2v": ("config", "is a 'bert' model, not model2vec"),
            "garbage": ("garbage", "not a safetensors file"),
        }
        for name, (setup, message) in cases.items():
            folder = base / name
            if setup is None:
                pass
            elif setup == "no tokenizer":
                make_model(folder)
                (folder / "tokenizer.json").unlink()
            elif name == "wrong-name":
                make_model(folder)
                write_safetensors(folder / "model.safetensors", setup)
            elif name == "quantized":
                make_model(folder, extra=setup)
            elif setup == "int8":
                make_model(folder, dtype="I8")
            elif setup == "short":
                make_model(folder, matrix=np.ones((4, 3), np.float32))
            elif setup == "config":
                make_model(folder, config={"model_type": "bert"})
            elif setup == "garbage":
                make_model(folder)
                (folder / "model.safetensors").write_bytes(b"\xff" * 64)
            with self.subTest(name), self.assertRaises(EmbeddingError) as caught:
                Model2VecProvider(folder)
            self.assertIn(message, str(caught.exception))

    def test_unknown_provider_and_no_model(self):
        with self.assertRaisesRegex(EmbeddingError, "unknown embedding provider 'nope' \\(known: model2vec\\)"):
            load_provider("nope", self.folder)
        with self.assertRaisesRegex(EmbeddingError, "no embedding model configured"):
            load_provider("model2vec", None)


@requires_model
class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider = load_provider("model2vec", MODEL_DIR)

    def cos(self, a, b):
        va, vb = self.provider.embed([a, b])
        return float(va @ vb)

    def test_the_model(self):
        self.assertEqual((self.provider.model_id, self.provider.dim), ("potion-base-8M", 256))
        self.assertEqual(self.provider.matrix.shape, (29528, 256))
        self.assertEqual(self.provider.unk_id, 1)

    def test_same_text_same_vector(self):
        text = "The KL penalty keeps the policy close to the reference model."
        first = self.provider.embed([text])[0]
        again = self.provider.embed(["other text first", text, "and after"])[1]  # in a batch, at another position
        fresh = load_provider("model2vec", MODEL_DIR).embed([text])[0]  # another load of the model
        self.assertEqual(first.tobytes(), again.tobytes())
        self.assertEqual(first.tobytes(), fresh.tobytes())
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=6)

    def test_a_synonym_pair_beats_an_unrelated_pair(self):
        cases = [
            ("car", "automobile", "banana"),
            ("physician", "doctor", "guitar"),
            ("The film was wonderful.", "The movie was great.", "The invoice is overdue."),
            ("reward hacking", "specification gaming", "sourdough bread"),
        ]
        for word, synonym, unrelated in cases:
            with self.subTest(word=word):
                self.assertGreater(self.cos(word, synonym), self.cos(word, unrelated) + 0.1)

    def test_similarity_splits_exactly_into_word_shares(self):
        # The centered cosine = the words' credit from one side + a term without words (explain.py).
        a = "The KL penalty keeps the policy close to the reference model during PPO."
        b = "A divergence penalty anchors the policy to the reference; PPO clips the update."
        mean = unit(self.provider.embed(["bananas and apples", "a note about gardening"]).mean(axis=0)) * 0.3
        bag_a, bag_b = text_bag(self.provider, a), text_bag(self.provider, b)
        (ua, scale_a), (ub, _) = _centered(bag_a, mean), _centered(bag_b, mean)
        raw_norm = np.linalg.norm(bag_a.raw(256))
        total = sum(bag_a.alpha[w] * float(bag_a.vectors[w] @ ub) for w in bag_a.alpha) / scale_a
        total -= float(mean @ ub) * raw_norm / scale_a
        va, vb = (v - mean for v in self.provider.embed([a, b]).astype(np.float64))
        self.assertAlmostEqual(total, float(va @ vb / np.linalg.norm(va) / np.linalg.norm(vb)), places=5)
        words = shared_words(bag_a, bag_b, mean)
        self.assertTrue(words)
        self.assertTrue(set(words) <= set(a.lower().replace(".", " ").split()) & set(b.lower().replace(";", " ").split()))


# ----- the vector cache in the index ---------------------------------------------------------

NOTES = {
    "kl.md": "Status: #baby\n\n# KL penalty\nThe KL penalty keeps the policy near the reference model. #rl\n",
    "dpo.md": "# DPO\nDirect preference optimization folds the KL constraint into the loss.\n",
    "bread.md": "# Bread\nSourdough needs a starter, flour, water and patience.\n",
    "a/Same.md": "Identical text in two folders.\n",
    "b/Same.md": "Identical text in two folders.\n",
}


@requires_model
class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider = load_provider("model2vec", MODEL_DIR)

    def setUp(self):
        self.vault = TempVault(NOTES)
        self.ix = VaultIndex(default_db_path(self.vault.root))
        self.ix.update(self.vault.root)

    def tearDown(self):
        self.ix.close()
        self.vault.cleanup()

    def embed(self, provider=None, **options):
        return embed_notes(self.ix.conn, self.vault.root, provider or self.provider, **options)

    def vector_rows(self):
        return self.ix.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def test_every_note_once_then_nothing(self):
        report = self.embed()
        self.assertEqual(report.embedded, sorted(NOTES))
        self.assertEqual((report.cached, report.notes), (0, 5))
        self.assertEqual(self.vector_rows(), 4)  # a/Same.md and b/Same.md have the same text: one vector
        again = self.embed()
        self.assertEqual((again.embedded, again.cached), ([], 5))

    def test_the_cached_vector_is_the_providers_vector_of_the_note_text(self):
        self.embed()
        vectors = note_vectors(self.ix.conn, self.vault.root, self.provider)
        for file_id, path in self.ix.conn.execute("SELECT id, path FROM files"):
            expected = self.provider.embed([note_text(self.vault.root, path)])[0]
            self.assertEqual(vectors.get(file_id).tobytes(), expected.tobytes(), path)

    def test_editing_one_note_re_embeds_only_that_note(self):
        self.embed()
        self.vault.write("dpo.md", NOTES["dpo.md"] + "It needs no reward model.\n")
        self.ix.update(self.vault.root)
        report = self.embed()
        self.assertEqual((report.embedded, report.cached), (["dpo.md"], 4))
        self.assertEqual(self.vector_rows(), 4)  # the old text's vector was dropped by the index update

    def test_edits_that_do_not_change_the_text_re_embed_nothing(self):
        self.embed()
        # Tags (#rl, and #baby as the Status value) and blank lines are not part of the embedded text.
        self.vault.write("kl.md", NOTES["kl.md"].replace("#rl", "#ml").replace("Status: #baby", "Status: #done") + "\n\n")
        self.assertEqual(self.ix.update(self.vault.root).changed, ["kl.md"])  # re-parsed ...
        self.assertEqual(self.embed().embedded, [])  # ... and not re-embedded

    def test_a_rename_re_embeds_the_note(self):
        self.embed()
        (self.vault.root / "bread.md").rename(self.vault.root / "sourdough.md")
        self.ix.update(self.vault.root)
        self.assertEqual(self.embed().embedded, ["sourdough.md"])  # the title is part of the text

    def test_changing_the_model_id_re_embeds_everything_and_keeps_the_old_vectors(self):
        self.embed()
        other = load_provider("model2vec", MODEL_DIR, "potion-base-8M-renamed")
        report = self.embed(other)
        self.assertEqual((len(report.embedded), report.cached), (5, 0))
        self.assertEqual(self.vector_rows(), 8)
        self.assertEqual(self.embed().embedded, [])  # back to the first id: all cached
        info = status(self.ix.conn, "potion-base-8M")
        self.assertEqual((info["with_vector"], info["missing"]), (5, 0))
        self.assertEqual(info["other_models"], [{"model_id": "potion-base-8M-renamed", "vectors": 4, "dim": 256}])
        self.assertEqual(self.embed(prune=True).pruned, 4)
        self.assertEqual(self.vector_rows(), 4)

    def test_full_re_embeds_and_status_counts(self):
        self.assertEqual(status(self.ix.conn, "potion-base-8M")["missing"], 5)
        self.embed()
        self.assertEqual(len(self.embed(full=True).embedded), 5)
        self.vault.write("new.md", "A new note.\n")
        self.ix.update(self.vault.root)
        info = status(self.ix.conn, "potion-base-8M")
        self.assertEqual((info["notes"], info["with_vector"], info["missing"], info["missing_paths"]), (6, 5, 1, ["new.md"]))

    def test_deleting_a_note_drops_its_vector(self):
        self.embed()
        (self.vault.root / "bread.md").unlink()
        self.ix.update(self.vault.root)
        self.assertEqual(self.vector_rows(), 3)

    def test_a_note_edited_after_the_index(self):
        bread = self.ix.conn.execute("SELECT id FROM files WHERE path = 'bread.md'").fetchone()[0]
        old = self.provider.embed([note_text(self.vault.root, "bread.md")])[0]
        self.vault.write("bread.md", "# Bread\nThe KL penalty keeps the policy near the reference model.\n")
        new = self.provider.embed([note_text(self.vault.root, "bread.md")])[0]
        # No vector for the indexed text, which is gone from disk: embedded as it is now, stored under that text's hash.
        report = self.embed()
        self.assertEqual(report.changed_since_index, ["bread.md"])
        self.assertEqual(note_vectors(self.ix.conn, self.vault.root, self.provider).get(bread).tobytes(), new.tobytes())
        # Once indexed and embedded, a later edit keeps the indexed vector until the next index, like the rest of the index.
        self.ix.update(self.vault.root)
        self.vault.write("bread.md", NOTES["bread.md"])
        self.assertEqual(note_vectors(self.ix.conn, self.vault.root, self.provider).get(bread).tobytes(), new.tobytes())
        self.ix.update(self.vault.root)
        self.assertEqual(note_vectors(self.ix.conn, self.vault.root, self.provider).get(bread).tobytes(), old.tobytes())


class CacheTextTests(unittest.TestCase):
    """What counts as a change of a note's text (no model needed)."""

    def test_what_changes_the_embedded_text(self):
        notes = {"tags.md": "Source: Smith 2024\n\n# T\nBody text. #rl\n", "code.md": "# C\nText.\n```\ncode\n```\n",
                 "url.md": "# U\nSee https://arxiv.org/abs/1 now.\n", "key.md": "Source: Smith\n\n# K\nBody.\n"}
        vault = TempVault(notes)
        try:
            with VaultIndex(default_db_path(vault.root)) as ix:
                ix.update(vault.root)
                hashes = lambda: dict(ix.conn.execute("SELECT f.path, n.embed_hash FROM notes n JOIN files f ON f.id = n.file_id"))
                before = hashes()
                vault.write("tags.md", notes["tags.md"].replace("#rl", "#ml") + "\n\n  \n")  # a tag, blank lines
                vault.write("code.md", notes["code.md"].replace("code", "other code"))  # inside a code block
                vault.write("url.md", notes["url.md"].replace("/abs/1", "/abs/2"))  # a URL
                vault.write("key.md", notes["key.md"].replace("Smith", "Jones"))  # a property value: text
                ix.update(vault.root)
                after = hashes()
            self.assertEqual([p for p in sorted(notes) if before[p] != after[p]], ["key.md"])
            self.assertEqual(note_text(vault.root, "key.md"), "key K Jones Body.")  # title, heading, value; no key
        finally:
            vault.cleanup()

    def test_math_is_left_out_and_prices_stay(self):
        from vaultengine.embedding.text import without_math

        cases = {
            "a $x$ b": "a   b",
            "the loss $\\frac{1}{n}\\sum_i \\ell_i$ here": "the loss   here",
            "shown\n$$\n\\mathbb{E}[x]\n$$\nafter": "shown\n \nafter",
            "costs $5 and $10 today": "costs $5 and $10 today",  # a space after the opening $ or before the closing one
            "from $x$2": "from $x$2",  # a digit after the closing $: a price
            "escaped \\$5 and \\$6": "escaped \\$5 and \\$6",
            "spans $a\nb$ lines": "spans $a\nb$ lines",  # inline math stays on one line
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(without_math(text), expected)
        vault = TempVault({"m.md": "# M\nVariance $\\sigma^2$ of the mean.\n$$\n\\frac{a}{b}\n$$\n"})
        try:
            with VaultIndex(default_db_path(vault.root)) as ix:
                ix.update(vault.root)
                before = ix.conn.execute("SELECT embed_hash FROM notes").fetchone()[0]
                self.assertEqual(note_text(vault.root, "m.md"), "m M Variance of the mean.")
                vault.write("m.md", "# M\nVariance $\\sigma_x^2$ of the mean.\n$$\n\\tilde{a}\n$$\n")  # other math only
                ix.update(vault.root)
                self.assertEqual(ix.conn.execute("SELECT embed_hash FROM notes").fetchone()[0], before)
        finally:
            vault.cleanup()

    def test_same_text_same_hash(self):
        vault = TempVault({"a/Same.md": "Identical.\n", "b/Same.md": "Identical.\n", "c/Other.md": "Identical.\n"})
        try:
            with VaultIndex(default_db_path(vault.root)) as ix:
                ix.update(vault.root)
                hashes = dict(ix.conn.execute("SELECT f.path, n.embed_hash FROM notes n JOIN files f ON f.id = n.file_id"))
            self.assertEqual(hashes["a/Same.md"], hashes["b/Same.md"])
            self.assertNotEqual(hashes["a/Same.md"], hashes["c/Other.md"])  # the title is part of the text
        finally:
            vault.cleanup()


# ----- settings --------------------------------------------------------------------------------


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.config = self.dir / "conf" / "config.yaml"
        self.config.parent.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, text=None, env=None, flags=()):
        if text is not None:
            self.config.write_text(text)
        return load_settings(flags, config_file=self.config if text is not None else None, env=env or {})

    def test_defaults(self):
        settings = self.load("llm: {provider: openai}\n")  # a file without an engine section
        e = settings.engine
        self.assertEqual((e.vault, e.index, e.format), (Path("."), None, None))
        self.assertEqual(e.index_path, Path(".vaultengine/index.sqlite"))
        self.assertEqual((e.embedding.provider, e.embedding.model_path, e.model_id), ("model2vec", None, None))
        self.assertEqual(settings.sources, {})

    def test_flag_over_env_over_file_over_default(self):
        text = "engine:\n  format: from-file\n  index: idx.sqlite\n  embedding:\n    model_id: file-id\n"
        env = {"VAULTENGINE_FORMAT": "from-env", "VAULTENGINE_EMBEDDING__MODEL_ID": "env-id"}
        flags = [("format", "from-flag", "--format")]
        s = self.load(text, env, flags)
        self.assertEqual(s.engine.format, "from-flag")
        self.assertEqual(s.sources["format"], "flag --format")
        self.assertEqual(s.engine.embedding.model_id, "env-id")
        self.assertEqual(s.sources["embedding.model_id"], "env VAULTENGINE_EMBEDDING__MODEL_ID")
        self.assertEqual(s.engine.index, self.config.parent / "idx.sqlite")
        self.assertEqual(s.sources["index"], f"file {self.config}")
        self.assertEqual(s.engine.embedding.provider, "model2vec")
        self.assertNotIn("embedding.provider", s.sources)  # the default
        self.assertEqual(self.load(text, env).engine.format, "from-env")
        self.assertEqual(self.load(text).engine.format, "from-file")
        self.assertEqual(self.load("engine: {}\n").engine.format, None)

    def test_relative_paths(self):
        text = "engine:\n  vault: ../notes\n  embedding:\n    model_path: ~/models/m\n"
        s = self.load(text)
        self.assertEqual(s.engine.vault, self.dir / "notes")  # from the file: next to the file
        self.assertEqual(s.engine.embedding.model_path, Path.home() / "models" / "m")
        s = self.load(text, {"VAULTENGINE_VAULT": "rel/vault"}, [("index", "my.sqlite", "--db")])
        self.assertEqual((s.engine.vault, s.engine.index), (Path("rel/vault"), Path("my.sqlite")))  # as given
        self.assertEqual(s.engine.index_path, Path("my.sqlite"))
        s = self.load("engine:\n  format: ../formats/brain.yaml\n")
        self.assertEqual(s.engine.format, str(self.dir / "formats" / "brain.yaml"))
        self.assertEqual(self.load("engine:\n  format: brain\n").engine.format, "brain")  # a name stays a name

    def test_which_config_file(self):
        other = self.dir / "other.yaml"
        other.write_text("engine: {format: other}\n")
        self.config.write_text("engine: {format: mine}\n")
        env = {"VAULTENGINE_CONFIG": str(other)}
        self.assertEqual(load_settings(env=env).engine.format, "other")
        self.assertEqual(load_settings(env=env).config_found_by, "$VAULTENGINE_CONFIG")
        self.assertEqual(load_settings(config_file=self.config, env=env).engine.format, "mine")  # --config wins
        self.assertEqual(load_settings(env={}).config_file, REPO_CONFIG)
        with self.assertRaisesRegex(SettingsError, "config file not found: nope.yaml \\(from --config\\)"):
            load_settings(config_file="nope.yaml", env={})
        with self.assertRaisesRegex(SettingsError, "from \\$VAULTENGINE_CONFIG"):
            load_settings(env={"VAULTENGINE_CONFIG": "/nope.yaml"})

    def test_invalid_settings_name_the_key_and_where_it_was_set(self):
        cases = [
            ("engine:\n  embeding: {}\n", {}, [], f"engine.embeding: not a setting  [from file {self.config}]"),
            ("engine: {}\n", {"VAULTENGINE_EMBEDDING__MODELPATH": "x"}, [],
             "engine.embedding.modelpath: not a setting  [from env VAULTENGINE_EMBEDDING__MODELPATH]"),
            ("engine: {}\n", {"VAULTENGINE_EMBEDDING__PROVIDER": "openai"}, [],
             "engine.embedding.provider: unknown provider 'openai'; known: model2vec  [from env VAULTENGINE_EMBEDDING__PROVIDER]"),
            ("engine: {}\n", {}, [("explore.k", "3", "--setting explore.k")],
             "engine.explore.k: not a setting  [from flag --setting explore.k]"),
            ("engine: {}\n", {"VAULTENGINE_EXPLORE__WEIGHTS__TEXT": "-1"}, [],
             "engine.explore.weights.text: Input should be greater than or equal to 0 (got '-1')  [from env VAULTENGINE_EXPLORE__WEIGHTS__TEXT]"),
            ("engine:\n  explore:\n    weights: {direct: 0, cocitation: 0, coupling: 0, tags: 0}\n", {},
             [("explore.weights.text", "0", "--setting explore.weights.text")],
             f"engine.explore.weights: at least one weight must be above 0  [from file {self.config}, flag --setting explore.weights.text]"),
            ("engine:\n  format: [a, b]\n", {}, [], "engine.format: Input should be a valid string"),
            ("engine:\n  embedding:\n    model_id: ' x'\n", {}, [], "engine.embedding.model_id: must be non-empty"),
            ("- a list\n", {}, [], "expected a mapping at the top level"),
            ("engine: 3\n", {}, [], "'engine' must be a mapping of settings"),
            ("engine: [\n", {}, [], "cannot read"),
        ]
        for text, env, flags, message in cases:
            with self.subTest(message), self.assertRaises(SettingsError) as caught:
                self.load(text, env, flags)
            self.assertIn(message, str(caught.exception))

    def test_other_vaultengine_variables_are_not_settings(self):
        env = {"VAULTENGINE_RECORD": "1", "VAULTENGINE_TEST_CRASH_AFTER_RENAMES": "1", "VAULTENGINE_FORMAT": "brain"}
        settings = self.load("engine: {}\n", env)
        self.assertEqual((settings.engine.format, sorted(settings.sources)), ("brain", ["format"]))
        with self.assertRaisesRegex(SettingsError, "engine.embedding.modelpath: not a setting"):
            self.load("engine: {}\n", {"VAULTENGINE_EMBEDDING__MODELPATH": "x"})

    def test_the_repository_config(self):
        s = load_settings(env={})
        self.assertEqual(s.config_file, REPO_CONFIG)
        self.assertEqual(s.engine.embedding.model_path, REPO_ROOT / "models" / "potion-base-8M")
        self.assertEqual((s.engine.embedding.provider, s.engine.model_id), ("model2vec", "potion-base-8M"))
        self.assertEqual(s.engine.format, None)
        sys.path.insert(0, str(REPO_ROOT))
        from src.main import load_config  # the zettel tool reads the same file and ignores the engine section

        self.assertIn("engine", load_config())

    def test_rows_for_config_show(self):
        s = self.load("engine:\n  embedding:\n    model_path: m\n", {"VAULTENGINE_FORMAT": "brain"})
        rows = {r["key"]: r for r in s.rows()}
        self.assertEqual(list(rows), ["vault", "index", "memory", "format", "embedding.provider", "embedding.model_path",
                                      "embedding.model_id", "related.weights.graph", "related.weights.cocitation",
                                      "related.weights.coupling", "related.weights.tags", "related.weights.text",
                                      "related.unlinked_weights.graph", "related.unlinked_weights.cocitation",
                                      "related.unlinked_weights.coupling", "related.unlinked_weights.tags",
                                      "related.unlinked_weights.text",
                                      "explore.weights.direct", "explore.weights.cocitation",
                                      "explore.weights.coupling", "explore.weights.tags", "explore.weights.text",
                                      "explore.percentile", "explore.topic_types", "explore.meaning",
                                      "explore.meaning_neighbours", "explore.meaning_percentile",
                                      "explore.seed", "explore.resolution", "explore.path_budget", "explore.lenses",
                                      "explore.start_lenses", "explore.per_lens", "explore.duplicate_percentile",
                                      "explore.stats_refresh", "explore.link_section"])
        self.assertEqual(rows["related.weights.text"]["value"], "0.3")
        self.assertEqual(rows["related.unlinked_weights.text"]["value"], "0.7")  # from vault experiment, 2026-11-10
        self.assertEqual(rows["explore.topic_types"]["value"], "['topic', 'page']")
        self.assertEqual((rows["explore.weights.text"]["value"], rows["explore.weights.text"]["source"]), ("0.3", "default"))
        self.assertEqual(rows["format"], {"key": "format", "value": "brain", "source": "env VAULTENGINE_FORMAT",
                                          "env": "VAULTENGINE_FORMAT"})
        self.assertEqual(rows["embedding.model_id"]["value"], "m")
        self.assertEqual(rows["embedding.model_id"]["source"], "default (the model folder's name)")
        self.assertEqual(rows["index"]["source"], "default (<vault>/.vaultengine/index.sqlite)")
        self.assertEqual(rows["vault"]["value"], str(Path.cwd()))


# ----- the CLI and the JSON-RPC server ---------------------------------------------------------


def run_cli(*args, env=None):
    out, err = io.StringIO(), io.StringIO()
    with patch.dict(os.environ, {**NO_ENGINE_ENV, **(env or {})}, clear=True), redirect_stdout(out), redirect_stderr(err):
        code = cli_main(list(args))
    return code, out.getvalue(), err.getvalue()


class ConfigCliTests(unittest.TestCase):
    def test_config_show(self):
        code, out, err = run_cli("config", "show", "/tmp", "--db", "x.sqlite", env={"VAULTENGINE_FORMAT": "brain"})
        self.assertEqual((code, err), (0, ""))
        lines = out.splitlines()
        self.assertEqual(lines[0], "Engine settings · precedence: flag > env > file > default")
        self.assertIn(f"  config file  {REPO_CONFIG}", out)
        self.assertTrue(next(line for line in lines if line.startswith("  vault ")).endswith("flag <vault>"))
        self.assertTrue(next(line for line in lines if line.startswith("  index ")).endswith("flag --db"))
        self.assertTrue(next(line for line in lines if line.startswith("  format ")).endswith("env VAULTENGINE_FORMAT"))
        model = next(line for line in lines if line.startswith("  embedding.model_path "))
        self.assertIn(str(REPO_ROOT / "models" / "potion-base-8M"), model)
        self.assertIn("  explore.weights.text ", out)
        code, out, _ = run_cli("config", "show", "--json", "--setting", "embedding.model_id=x")
        data = json.loads(out)
        self.assertEqual(data["precedence"], ["flag", "env", "file", "default"])
        row = next(r for r in data["settings"] if r["key"] == "embedding.model_id")
        self.assertEqual((row["value"], row["source"]), ("x", "flag --setting embedding.model_id"))

    def test_bad_settings_are_usage_errors(self):
        code, out, err = run_cli("config", "show", env={"VAULTENGINE_EMBEDDING__PROVIDER": "openai"})
        self.assertEqual((code, out), (2, ""))
        self.assertIn("error: invalid engine settings:", err)
        self.assertIn("unknown provider 'openai'", err)
        code, _, err = run_cli("stats", "--setting", "nonsense")
        self.assertEqual(code, 2)
        self.assertIn("expected KEY=VALUE", err)

    def test_the_format_can_come_from_the_settings(self):
        vault = REPO_ROOT / "tests" / "fixtures" / "vault"
        code, _, err = run_cli("formats", str(vault))
        self.assertEqual(code, 2)
        self.assertIn("error: no format: give --format NAME", err)
        code, out, err = run_cli("formats", str(vault), env={"VAULTENGINE_FORMAT": "brain"})
        self.assertEqual(code, 0, err)
        self.assertIn("notes      180   180 typed, 0 untyped", out)
        code, _, err = run_cli("formats", str(vault), "--format", "nope", env={"VAULTENGINE_FORMAT": "brain"})
        self.assertEqual(code, 2)  # the flag wins, and "nope" is no format
        self.assertIn("nope", err)


@requires_model
class EmbedCliTests(unittest.TestCase):
    def setUp(self):
        self.vault = TempVault(NOTES)
        self.root = str(self.vault.root)

    def tearDown(self):
        self.vault.cleanup()

    def test_embed_and_status(self):
        code, out, err = run_cli("embed", self.root, "--status")
        self.assertEqual(code, 2)
        self.assertIn("no index at", err)
        code, out, err = run_cli("embed", self.root)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("  model      potion-base-8M (model2vec, 256 dims) · " + str(MODEL_DIR), out)
        self.assertIn("  index      5 parsed (5 added, 0 changed, 0 deleted)", out)
        self.assertIn("  vectors    5 embedded, 0 cached (5 notes)", out)
        self.vault.write("dpo.md", NOTES["dpo.md"] + "Edited.\n")
        code, out, _ = run_cli("embed", self.root, "--status")
        self.assertIn("  vectors    5 of 5 indexed notes (256 dims) · every indexed note has one, but the index is out of date", out)
        self.assertIn("  pending    0 added, 1 changed, 0 deleted since the last index", out)
        code, out, _ = run_cli("embed", self.root)
        self.assertIn("  vectors    1 embedded, 4 cached (5 notes)\n    + dpo.md", out)
        code, out, _ = run_cli("embed", self.root, "--status", "--json")
        data = json.loads(out)
        self.assertEqual((data["model_id"], data["with_vector"], data["missing"], data["dim"]), ("potion-base-8M", 5, 0, 256))
        code, out, _ = run_cli("embed", self.root, "--setting", "embedding.model_id=other", "--json")
        self.assertEqual((len(json.loads(out)["embedded"]), json.loads(out)["cached"]), (5, 0))
        code, out, _ = run_cli("embed", self.root, "--status")
        self.assertIn("  other ids  other: 4 vectors (256 dims)", out)

    def test_a_missing_model_stops_embed_and_only_switches_off_text_elsewhere(self):
        env = {"VAULTENGINE_EMBEDDING__MODEL_PATH": "/no/such/model"}
        code, out, err = run_cli("embed", self.root, env=env)  # embedding is all `vault embed` does: an error
        self.assertEqual((code, out), (2, ""))
        self.assertIn("error: embedding model folder not found: /no/such/model", err)
        self.assertFalse(default_db_path(self.root).exists())  # the model is checked before the index is touched
        run_cli("index", self.root)
        for command in (["related", self.root, "kl"], ["suggest", self.root, "--links"], ["explain", self.root, "kl", "dpo"]):
            with self.subTest(command[0]):
                code, out, err = run_cli(*command, "--color", "never", env=env)
                self.assertEqual((code, err), (0, ""))
                self.assertIn("text signal unavailable: embedding model folder not found: /no/such/model", out)
        code, out, err = run_cli("suggest", self.root, "--notes", env=env)  # missing notes need no embeddings
        self.assertEqual(code, 0, err)

    def test_read_commands_never_write_vectors(self):
        run_cli("index", self.root)
        with VaultIndex(default_db_path(self.root)) as ix:
            before = "\n".join(ix.conn.iterdump())
        for command in (["related", self.root, "kl"], ["suggest", self.root], ["explain", self.root, "kl", "dpo"]):
            with self.subTest(command[0]):
                code, out, err = run_cli(*command, "--color", "never")
                self.assertEqual((code, err), (0, ""))
                self.assertIn(f"5 notes had no cached vector: embedded in memory for this run only; `vault embed {self.root}`"
                              " caches them", out)
        with VaultIndex(default_db_path(self.root)) as ix:
            self.assertEqual("\n".join(ix.conn.iterdump()), before)  # not a row changed
            self.assertEqual(status(ix.conn, "potion-base-8M")["missing"], 5)
        run_cli("embed", self.root)
        code, out, _ = run_cli("related", self.root, "kl", "--color", "never")
        self.assertNotIn("no cached vector", out)


class RpcEmbeddingTests(unittest.TestCase):
    def test_an_unavailable_model_switches_off_only_the_text_signal(self):
        from vaultengine.rpc.server import RpcServer
        from vaultengine.service import VaultService

        vault = TempVault(NOTES)
        try:
            settings = load_settings([("embedding.model_path", "/no/such/model", "test")], env={})
            server = RpcServer(VaultService(vault.root, settings=settings))
            call = lambda method, **params: json.loads(server.handle_text(json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": method, "params": params})))
            call("index.build")
            related = call("note.related", note="kl")["result"]
            off = {"available": False, "model_id": None, "unavailable": "embedding model folder not found: /no/such/model",
                   "uncached": 0}
            self.assertEqual(related["text_signal"], off)
            self.assertEqual(related["weights"], {"graph": 0.5 / 0.7, "cocitation": 0.0, "coupling": 0.0, "tags": 0.2 / 0.7, "text": 0.0})
            self.assertTrue(all(h["text"] == 0 for h in related["hits"]))
            links = call("links.suggest")["result"]
            self.assertEqual((links["text_signal"], links["weights"]["text"]), (off, 0.0))
            self.assertIn("result", call("notes.suggest"))  # no embeddings needed
            self.assertEqual(call("note.related", note="nope")["error"]["code"], -32002)
            server.service.close()
        finally:
            vault.cleanup()


if __name__ == "__main__":
    unittest.main()
