"""Static embeddings in the Model2Vec format (e.g. minishlab/potion-base-8M), with numpy and tokenizers only.

A Model2Vec model is a folder with

* ``tokenizer.json`` - a Hugging Face ``tokenizers`` tokenizer;
* ``model.safetensors`` - one 2-D tensor ``embeddings``: row i is the vector of token id i;
* ``config.json`` (optional) - ``"model_type": "model2vec"`` and the settings it was made with.

A text's vector, as Model2Vec computes it: tokenize without special tokens ([CLS]/[SEP]),
drop the unknown-token id, take the mean of those rows, and normalise to unit length.
Two differences from ``model2vec.StaticModel.encode``, both deliberate:

* no truncation: ``encode`` keeps the first 512 tokens by default; a note is embedded whole;
* the result is always normalised (``encode`` follows the model's ``normalize`` flag), because
  vectors are compared by dot product as cosines. A text with no known tokens is all zeros.

The safetensors file is read without the ``safetensors`` package: an 8-byte little-endian
header length, a JSON header (dtype, shape and byte range of each tensor), then the raw
data, which is memory-mapped, so only the rows that are used are read from disk.

Vocabulary-quantized Model2Vec models (extra ``mapping``/``weights`` tensors) are refused
with a clear error rather than computed wrongly.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Sequence

import numpy as np

from vaultengine.embedding.provider import EmbeddingError, StaticEmbeddingProvider, default_model_id, register

_DTYPES = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}
_MAX_HEADER = 100 * 1024 * 1024


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Every tensor in a safetensors file, memory-mapped (read-only)."""
    size = path.stat().st_size
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise EmbeddingError(f"{path}: not a safetensors file (shorter than 8 bytes)")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len > min(_MAX_HEADER, size - 8):
            raise EmbeddingError(f"{path}: not a safetensors file (header length {header_len} for a {size}-byte file)")
        try:
            header = json.loads(handle.read(header_len).decode("utf-8"))  # UTF-8 by the format's definition
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EmbeddingError(f"{path}: unreadable safetensors header ({exc})") from None
    if not isinstance(header, dict):
        raise EmbeddingError(f"{path}: unreadable safetensors header (not a JSON object)")
    base = 8 + header_len
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = _DTYPES.get(info.get("dtype"))
        if dtype is None:
            raise EmbeddingError(f"{path}: tensor {name!r} has dtype {info.get('dtype')}; supported: {', '.join(_DTYPES)}")
        shape = tuple(int(n) for n in info["shape"])
        start, end = (int(n) for n in info["data_offsets"])
        if end - start != int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize or base + end > size:
            raise EmbeddingError(f"{path}: tensor {name!r} byte range {start}..{end} does not match shape {list(shape)}")
        tensors[name] = np.memmap(path, dtype=dtype, mode="r", offset=base + start, shape=shape)
    return tensors


class Model2VecProvider(StaticEmbeddingProvider):
    name = "model2vec"

    def __init__(self, model_path: Path, model_id: str | None = None) -> None:
        from tokenizers import Tokenizer

        folder = Path(model_path).expanduser()
        if not folder.is_dir():
            raise EmbeddingError(f"embedding model folder not found: {folder}")
        missing = [name for name in ("tokenizer.json", "model.safetensors") if not (folder / name).is_file()]
        if missing:
            raise EmbeddingError(f"{folder} is not a Model2Vec model: no {' or '.join(missing)}")
        config_file = folder / "config.json"
        if config_file.is_file():
            try:
                model_type = json.loads(config_file.read_text(encoding="utf-8")).get("model_type")
            except (OSError, ValueError, AttributeError) as exc:
                raise EmbeddingError(f"{config_file}: unreadable ({exc})") from None
            if model_type not in (None, "model2vec"):
                raise EmbeddingError(f"{folder} is a {model_type!r} model, not model2vec")

        tensors = read_safetensors(folder / "model.safetensors")
        extra = sorted(set(tensors) - {"embeddings"})
        if "embeddings" not in tensors or extra:
            found = ", ".join(sorted(tensors)) or "none"
            raise EmbeddingError(
                f"{folder / 'model.safetensors'}: expected exactly one tensor 'embeddings' (found: {found}); "
                "vocabulary-quantized Model2Vec models are not supported"
            )
        self.matrix = tensors["embeddings"]
        if self.matrix.ndim != 2:
            raise EmbeddingError(f"{folder}: 'embeddings' has shape {list(self.matrix.shape)}; expected [vocabulary, dims]")

        try:
            self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
            spec = json.loads((folder / "tokenizer.json").read_text(encoding="utf-8"))
        except Exception as exc:  # tokenizers raises plain Exception subclasses
            raise EmbeddingError(f"{folder / 'tokenizer.json'}: cannot load ({exc})") from None
        # Whatever the file says, a note is tokenized whole and without padding.
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        model = spec.get("model", {})
        if isinstance(model.get("unk_token"), str):
            self.unk_id = self.tokenizer.token_to_id(model["unk_token"])
        elif isinstance(model.get("unk_id"), int):  # Unigram tokenizers store the id
            self.unk_id = model["unk_id"]
        else:
            self.unk_id = None
        vocab = self.tokenizer.get_vocab_size(with_added_tokens=True)
        if vocab > self.matrix.shape[0]:
            raise EmbeddingError(
                f"{folder}: the tokenizer has {vocab} tokens but 'embeddings' has {self.matrix.shape[0]} rows"
            )
        self.dim = int(self.matrix.shape[1])
        self.model_id = model_id or default_model_id(folder)
        self.source = str(folder.resolve())

    def _ids(self, ids: list[int]) -> list[int]:
        return [i for i in ids if i != self.unk_id] if self.unk_id is not None else list(ids)

    def words(self, text: str) -> list[tuple[str, list[int]]]:
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        found: list[tuple[str, list[int]]] = []
        current = None
        for token_id, word, (start, end) in zip(encoding.ids, encoding.word_ids, encoding.offsets):
            if word != current or not found:
                current = word
                found.append(("", []))
                span = [start, end]
            span[1] = end
            if self.unk_id is None or token_id != self.unk_id:
                found[-1][1].append(token_id)
            found[-1] = (text[span[0] : span[1]].lower(), found[-1][1])
        return [(w, ids) for w, ids in found if ids]

    def token_vectors(self, ids: Sequence[int]) -> np.ndarray:
        return np.asarray(self.matrix[np.asarray(ids, dtype=np.int64)], dtype=np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        if not texts:
            return out
        for row, encoding in enumerate(self.tokenizer.encode_batch(list(texts), add_special_tokens=False)):
            ids = self._ids(encoding.ids)
            if not ids:
                continue
            mean = self.matrix[np.asarray(ids, dtype=np.int64)].mean(axis=0, dtype=np.float64)
            norm = np.linalg.norm(mean)
            if norm > 0:
                out[row] = (mean / norm).astype(np.float32)
        return out


register("model2vec", Model2VecProvider)
