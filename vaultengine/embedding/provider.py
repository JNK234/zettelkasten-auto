"""The embedding provider interface, and the registry that turns settings into a provider.

A provider turns texts into vectors. Everything else (the cache in the index, ``related``,
``suggest``) talks to this interface only, so a different model is a settings change:
``engine.embedding.provider`` picks the implementation, ``engine.embedding.model_path`` the model.

Contract of ``embed``:

* one row per text, ``float32``, shape ``(len(texts), dim)``;
* every row has unit length, so a dot product is the cosine; a text the model has
  no tokens for (empty, or only unknown tokens) gives a row of zeros;
* deterministic: the same text gives the same vector, bit for bit.

``model_id`` names the vector space. Cached vectors are keyed by it: vectors of
two different model ids are never compared, and changing it re-embeds everything.

Static providers (a fixed vector per token, averaged) can also say *which* tokens make
two texts alike. They implement ``StaticEmbeddingProvider``; ``related`` and ``suggest``
use that to name the words behind a "similar text" reason. Other providers simply give
no words.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:  # numpy is only imported when a provider is actually loaded
    import numpy as np


class EmbeddingError(Exception):
    """The model cannot be loaded or used: missing files, an unsupported format, a bad setting."""


class EmbeddingProvider(ABC):
    name: str  # the provider's registry name, e.g. "model2vec"
    model_id: str  # the vector space: part of every cache key
    dim: int
    source: str  # where the model came from (a folder), for messages

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> "np.ndarray":
        """Unit-length float32 rows, one per text (zeros for a text with no known tokens)."""

    def describe(self) -> str:
        return f"{self.model_id} ({self.name}, {self.dim} dims)"


class StaticEmbeddingProvider(EmbeddingProvider):
    """A provider whose vector is the normalised mean of per-token vectors.

    Then a text's vector is a sum over its words, and a similarity can be split into what
    each word adds to it (``embedding/explain.py``).
    """

    @abstractmethod
    def words(self, text: str) -> list[tuple[str, list[int]]]:
        """The words of ``text`` in order, each (lower-cased word, its token ids; unknown tokens dropped).

        Every token that ``embed`` averages is in exactly one word here.
        """

    @abstractmethod
    def token_vectors(self, ids: Sequence[int]) -> "np.ndarray":
        """The (unnormalised) vectors of these token ids, float32, one row each."""


# name -> factory(model_path, model_id) ; filled by the modules that implement providers
_REGISTRY: dict[str, Callable[[Path, str | None], EmbeddingProvider]] = {}


def register(name: str, factory: Callable[[Path, str | None], EmbeddingProvider]) -> None:
    _REGISTRY[name] = factory


def provider_names() -> list[str]:
    _load_builtin()
    return sorted(_REGISTRY)


def _load_builtin() -> None:
    from vaultengine.embedding import model2vec  # noqa: F401  (registers "model2vec")


def default_model_id(model_path: Path) -> str:
    """The model id when none is configured: the model folder's name, e.g. ``potion-base-8M``."""
    return Path(model_path).resolve().name


def load_provider(name: str, model_path: Path | None, model_id: str | None = None) -> EmbeddingProvider:
    """The provider ``name`` for the model at ``model_path``. Raises ``EmbeddingError`` with what to fix."""
    _load_builtin()
    if name not in _REGISTRY:
        raise EmbeddingError(f"unknown embedding provider {name!r} (known: {', '.join(sorted(_REGISTRY))})")
    if model_path is None:
        raise EmbeddingError("no embedding model configured: set engine.embedding.model_path in config/config.yaml")
    return _REGISTRY[name](Path(model_path), model_id)
