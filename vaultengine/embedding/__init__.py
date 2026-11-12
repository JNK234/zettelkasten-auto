"""Embeddings for the vault engine: a provider interface, a Model2Vec implementation, and the vector cache.

* ``provider.py`` - ``EmbeddingProvider`` (texts -> unit vectors), the registry, ``load_provider``;
* ``model2vec.py`` - static Model2Vec models (tokenizer.json + model.safetensors), numpy and tokenizers only;
* ``store.py`` - what text a note is embedded as, and the vectors cached in the index by content hash and model id.

numpy and tokenizers are imported only when a provider is loaded or vectors are read.
"""

from vaultengine.embedding.provider import (
    EmbeddingError,
    EmbeddingProvider,
    StaticEmbeddingProvider,
    default_model_id,
    load_provider,
    provider_names,
)

__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "StaticEmbeddingProvider",
    "default_model_id",
    "load_provider",
    "provider_names",
]
