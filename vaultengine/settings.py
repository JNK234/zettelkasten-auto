"""Engine settings: the ``engine:`` section of config/config.yaml, environment variables and command-line flags.

Every setting of the vault engine lives here, and each is looked up in this order,
the first that sets it wins:

1. a flag:  the ``vault`` argument, ``--db``, ``--format``, or ``--setting KEY=VALUE`` for any key;
2. the environment:  ``VAULTENGINE_`` + the key in capitals, ``__`` between levels, e.g.
   ``VAULTENGINE_FORMAT=brain``, ``VAULTENGINE_EMBEDDING__MODEL_PATH=/models/potion-base-32M``.
   A variable is a setting when its first part names one (VAULT, INDEX, FORMAT, EMBEDDING,
   EXPLORE); below that every key is checked (a typo is an error). Other ``VAULTENGINE_``
   variables (``VAULTENGINE_CONFIG``, test switches like ``VAULTENGINE_RECORD``) are not settings;
3. the ``engine:`` section of the config file;
4. the default (the model below).

The config file is ``--config FILE``, else ``$VAULTENGINE_CONFIG``, else ``config/config.yaml``
of this repository (when it exists). Relative paths are resolved against the folder of the
config file when they come from the file; from a flag or the environment they are used as
given (relative to the current folder), ``~`` expanded. The merged values are validated with Pydantic: an unknown
key, a wrong type or an unknown provider is an error that names the key and where it came from.

``vault config show`` prints the result, with the source of every value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

ENV_PREFIX = "VAULTENGINE_"
CONFIG_ENV = "VAULTENGINE_CONFIG"
REPO_CONFIG = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
# Keys whose values are paths (resolved as described above).
PATH_KEYS = {("vault",), ("index",), ("memory",), ("embedding", "model_path")}
# Keys that are a path when they look like one (a format is a name or a file).
MAYBE_PATH_KEYS = {("format",)}
SECTIONS = {("embedding",), ("related",), ("related", "weights"), ("related", "unlinked_weights"), ("explore",),
            ("explore", "weights"), ("findings",)}


class SettingsError(Exception):
    """Settings that cannot be used; the message says which key, what is wrong and where it was set."""


# ----- the settings ----------------------------------------------------------


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class EmbeddingSettings(_Section):
    provider: str = "model2vec"
    """Which implementation turns text into vectors (see vaultengine/embedding/provider.py)."""
    model_path: Path | None = None
    """The model folder. For model2vec: tokenizer.json + model.safetensors."""
    model_id: str | None = None
    """Names the vector space; cached vectors are keyed by it. Default: the model folder's name."""

    @field_validator("provider")
    @classmethod
    def _known_provider(cls, value: str) -> str:
        from vaultengine.embedding import provider_names

        if value not in provider_names():
            raise ValueError(f"unknown provider {value!r}; known: {', '.join(provider_names())}")
        return value

    @field_validator("model_id")
    @classmethod
    def _usable_id(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("must be non-empty, without leading or trailing spaces")
        return value


class RelatedWeights(_Section):
    """How plain `vault related` (and deepen, drafts) mixes its signals, linked notes included. Rescaled to sum to 1."""

    graph: float = Field(0.5, ge=0)
    """Personalized PageRank over links and tags, divided by degree."""
    cocitation: float = Field(0.0, ge=0)
    """Notes that link to both."""
    coupling: float = Field(0.0, ge=0)
    """Notes (and missing notes) both link to."""
    tags: float = Field(0.2, ge=0)
    """Shared tags, weighted by rarity."""
    text: float = Field(0.3, ge=0)
    """Embedding similarity."""

    @model_validator(mode="after")
    def _some_weight(self) -> "RelatedWeights":
        if not any(getattr(self, name) > 0 for name in type(self).model_fields):
            raise ValueError("at least one weight must be above 0")
        return self


class UnlinkedWeights(RelatedWeights):
    """The mix for finding what is not linked yet: `vault related --unlinked`, `vault discover`, `vault explore`.
    Defaults from `vault experiment` (docs/experiments/2026-11-10.md), which measures exactly that: the reasons
    are in config/config.yaml."""

    graph: float = Field(0.1, ge=0)
    cocitation: float = Field(0.05, ge=0)
    coupling: float = Field(0.05, ge=0)
    tags: float = Field(0.1, ge=0)
    text: float = Field(0.7, ge=0)


class RelatedSettings(_Section):
    """`vault related`: which signals rank related notes, and how much each counts."""

    weights: RelatedWeights = RelatedWeights()
    unlinked_weights: UnlinkedWeights = UnlinkedWeights()


class ExploreWeights(_Section):
    """How much each signal between two notes counts in `vault explain` (relations.py). Rescaled to sum to 1."""

    direct: float = Field(0.3, ge=0)
    """A link between the two, either way."""
    cocitation: float = Field(0.15, ge=0)
    """Notes that link to both."""
    coupling: float = Field(0.15, ge=0)
    """Notes (and missing notes) both link to."""
    tags: float = Field(0.1, ge=0)
    """Shared tags, weighted by rarity."""
    text: float = Field(0.3, ge=0)
    """Embedding similarity."""

    @model_validator(mode="after")
    def _some_weight(self) -> "ExploreWeights":
        if not any(getattr(self, name) > 0 for name in type(self).model_fields):
            raise ValueError("at least one weight must be above 0")
        return self


class ExploreSettings(_Section):
    """Settings of the explore engine: how two notes relate (`vault explain`), the map, the lenses."""

    weights: ExploreWeights = ExploreWeights()
    percentile: float = Field(90.0, gt=0, lt=100)
    """What stands out: at or above this percentile of the vault (bridges, lenses). A share, not a score."""
    topic_types: list[str] = ["topic", "page"]
    """Note types (from the format) that are topic pages: `vault map` says which communities have none."""
    meaning: bool = False
    """`vault map`: add meaning edges for the communities (--meaning / --no-meaning override it)."""
    meaning_neighbours: int = Field(1, ge=1)
    """`vault map --meaning`: join two notes that are each among the other's this many nearest neighbours."""
    meaning_percentile: float = Field(50.0, gt=0, lt=100)
    """`vault map --meaning`: join two notes that are each other's nearest neighbour by embedding when their
    similarity is at or above this percentile of every note's nearest-neighbour similarity."""

    @field_validator("topic_types", mode="before")
    @classmethod
    def _comma_list(cls, value: Any) -> Any:
        return [part.strip() for part in value.split(",") if part.strip()] if isinstance(value, str) else value
    seed: int = 42
    """Seeds the community search and any sampling: the same seed gives the same result."""
    resolution: float = Field(1.0, gt=0)
    """Louvain's resolution: above 1 more and smaller communities, below 1 fewer and larger."""
    path_budget: int = Field(20_000_000, ge=1)
    """Edge visits betweenness may spend; above it, a seeded sample of source notes is used."""
    lenses: list[str] = ["neighbourhood", "cited", "foundations", "backbone", "cluster",
                         "hidden", "role", "gap", "surprise", "unwritten"]
    """`vault explore`: the lenses it runs, in this order (before the keep rates reorder them); see explore.py."""
    start_lenses: list[str] = ["gap", "surprise", "unwritten", "topicless"]
    """`vault explore` without a starting point: the vault-wide lenses whose findings it proposes as places to start."""
    per_lens: int = Field(4, ge=3, le=5)
    """`vault explore`: results per section (3 to 5); deepen and discover get the same space."""
    duplicate_percentile: float = Field(99.0, gt=0, lt=100)
    """`vault explore`: two notes count as near-duplicates at or above this percentile of note-pair similarity."""
    stats_refresh: float = Field(0.02, ge=0, le=1)
    """`vault explore`: keep using the stored vault-wide numbers (distributions, key terms) until more than this share
    of the notes changed since they were computed (stats.py); 0 recomputes after any change."""
    link_section: str = "See Also"
    """`vault link-it`: the section both notes get the link under (created at the end when missing)."""

    @field_validator("lenses", "start_lenses", mode="before")
    @classmethod
    def _lens_list(cls, value: Any) -> Any:
        return [part.strip() for part in value.split(",") if part.strip()] if isinstance(value, str) else value

    @field_validator("lenses", "start_lenses")
    @classmethod
    def _known_lenses(cls, value: list[str], info) -> list[str]:
        from vaultengine.explore import REGISTRY

        unknown = [name for name in value if name not in REGISTRY]
        if unknown:
            raise ValueError(f"unknown lens {', '.join(map(repr, unknown))}; known: {', '.join(REGISTRY)}")
        if not value or len(set(value)) != len(value):
            raise ValueError("give each lens once, at least one")
        if info.field_name == "start_lenses":
            local = [name for name in value if not REGISTRY[name].vault_wide]
            if local:
                raise ValueError(f"{', '.join(local)} need a starting point; vault-wide lenses: "
                                 + ", ".join(n for n, lens in REGISTRY.items() if lens.vault_wide))
        else:
            wide = [name for name in value if not REGISTRY[name].from_start]
            if wide:
                raise ValueError(f"{', '.join(wide)} only run without a starting point (engine.explore.start_lenses)")
        return value

    @field_validator("link_section")
    @classmethod
    def _one_line(cls, value: str) -> str:
        if not value.strip() or "\n" in value or value != value.strip():
            raise ValueError("must be one line of text, without leading or trailing spaces")
        return value


class FindingsSettings(_Section):
    """`vault findings` (and pipeline, next, ideas): how selective link findings are (findings.py)."""

    link_floor: float = Field(0.2, ge=0)
    """A link finding scores at least this, whatever the vault's 99th percentile of structural pairs comes to."""
    link_limit: int = Field(500, ge=1)
    """At most this many link findings, the best scores first (counted before notes the format ignores are left out)."""


class EngineSettings(_Section):
    vault: Path = Path(".")
    """The vault folder."""
    index: Path | None = None
    """The index file. Default: <vault>/.vaultengine/index.sqlite."""
    memory: Path | None = None
    """What `vault explore` remembers (marks, seen results, links it made, map snapshots); never the notes.
    Default: <vault>/.vaultengine/memory.sqlite."""
    format: str | None = None
    """The format (a name in ./formats/ or the bundled formats/, or a file) for check, fix, coverage and writes."""
    embedding: EmbeddingSettings = EmbeddingSettings()
    related: RelatedSettings = RelatedSettings()
    explore: ExploreSettings = ExploreSettings()
    findings: FindingsSettings = FindingsSettings()

    @property
    def index_path(self) -> Path:
        from vaultengine.index import default_db_path

        return self.index if self.index is not None else default_db_path(self.vault)

    @property
    def memory_path(self) -> Path:
        return self.memory if self.memory is not None else Path(self.vault) / ".vaultengine" / "memory.sqlite"

    @property
    def model_id(self) -> str | None:
        from vaultengine.embedding import default_model_id

        e = self.embedding
        return e.model_id or (default_model_id(e.model_path) if e.model_path is not None else None)


# ----- loading -------------------------------------------------------------------


@dataclass
class Settings:
    engine: EngineSettings
    sources: dict[str, str]  # dotted key -> where its value came from ("default", "file", "env NAME", "flag --x")
    config_file: Path | None
    config_found_by: str  # "--config", "$VAULTENGINE_CONFIG", "repository default" or "none"
    _provider: Any = field(default=None, repr=False)

    def provider(self):
        """The embedding provider, loaded on first use (raises EmbeddingError with what to fix)."""
        if self._provider is None:
            from vaultengine.embedding import load_provider

            e = self.engine.embedding
            self._provider = load_provider(e.provider, e.model_path, e.model_id)
        return self._provider

    def rows(self) -> list[dict[str, Any]]:
        """Every setting as {key, value, source, env}, for ``vault config show``."""
        out = []
        for key in all_keys():
            dotted = ".".join(key)
            value: Any = self.engine
            for part in key:
                value = getattr(value, part)
            source = self.sources.get(dotted, "default")
            if key == ("index",) and self.engine.index is None:
                value, source = self.engine.index_path, "default (<vault>/.vaultengine/index.sqlite)"
            elif key == ("memory",) and self.engine.memory is None:
                value, source = self.engine.memory_path, "default (<vault>/.vaultengine/memory.sqlite)"
            elif key == ("embedding", "model_id") and self.engine.embedding.model_id is None:
                value = self.engine.model_id
                source = "default (the model folder's name)"
            if isinstance(value, Path):
                value = value.resolve()  # as used: relative to the current folder
            out.append({"key": dotted, "value": None if value is None else str(value), "source": source, "env": env_name(key)})
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_file": str(self.config_file) if self.config_file else None,
            "config_found_by": self.config_found_by,
            "precedence": ["flag", "env", "file", "default"],
            "settings": self.rows(),
        }


def all_keys() -> list[tuple[str, ...]]:
    keys: list[tuple[str, ...]] = []

    def walk(model: type[BaseModel], prefix: tuple[str, ...]) -> None:
        for name, info in model.model_fields.items():
            sub = info.annotation
            if isinstance(sub, type) and issubclass(sub, BaseModel):
                if sub.model_fields:
                    walk(sub, prefix + (name,))
                else:
                    keys.append(prefix + (name,))
            else:
                keys.append(prefix + (name,))

    walk(EngineSettings, ())
    return keys


def env_name(key: Iterable[str]) -> str:
    return ENV_PREFIX + "__".join(part.upper() for part in key)


def _find_config(flag: str | Path | None, env: Mapping[str, str]) -> tuple[Path | None, str]:
    if flag:
        path = Path(flag).expanduser()
        if not path.is_file():
            raise SettingsError(f"config file not found: {flag} (from --config)")
        return path.resolve(), "--config"
    if env.get(CONFIG_ENV):
        path = Path(env[CONFIG_ENV]).expanduser()
        if not path.is_file():
            raise SettingsError(f"config file not found: {env[CONFIG_ENV]} (from ${CONFIG_ENV})")
        return path.resolve(), f"${CONFIG_ENV}"
    if REPO_CONFIG.is_file():
        return REPO_CONFIG, "repository default"
    return None, "none"


def _read_file(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SettingsError(f"{path}: cannot read ({exc})") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SettingsError(f"{path}: expected a mapping at the top level")
    engine = data.get("engine")
    if engine is None:
        return {}
    if not isinstance(engine, dict):
        raise SettingsError(f"{path}: 'engine' must be a mapping of settings")
    return engine


def _flatten(data: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    flat: dict[tuple[str, ...], Any] = {}
    for key, value in data.items():
        path = prefix + (str(key),)
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, path))
        elif not (isinstance(value, dict) and path in SECTIONS):  # an empty section is nothing
            flat[path] = value
    return flat


def _resolve(key: tuple[str, ...], value: Any, base: Path | None) -> Any:
    """Path values with ~ expanded and, given a ``base``, made absolute against it; anything else unchanged."""
    if not isinstance(value, str) or not value:
        return value
    looks_like_path = key in PATH_KEYS or (
        key in MAYBE_PATH_KEYS and ("/" in value or value.endswith((".yaml", ".yml")))
    )
    if not looks_like_path:
        return value
    path = Path(value).expanduser()
    if base is None or path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def load_settings(
    flags: Iterable[tuple[str, Any, str]] = (),
    *,
    config_file: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Merge default < file < env < flags and validate.

    ``flags``: (dotted key, value, how it was given, e.g. "--db"). A value of None is "not given".
    ``env`` defaults to ``os.environ``.
    """
    env = os.environ if env is None else env
    path, found_by = _find_config(config_file, env)

    layers: list[tuple[dict[tuple[str, ...], Any], dict[tuple[str, ...], str]]] = []
    if path is not None:
        values = {k: _resolve(k, v, path.parent) for k, v in _flatten(_read_file(path)).items() if v is not None}
        layers.append((values, {k: f"file {path}" for k in values}))
    env_values, env_sources = {}, {}
    for name, value in env.items():
        if not name.startswith(ENV_PREFIX) or name == CONFIG_ENV or value == "":
            continue
        key = tuple(part.lower() for part in name[len(ENV_PREFIX):].split("__"))
        if key[0] not in EngineSettings.model_fields:  # not a setting (see the module docstring)
            continue
        env_values[key] = _resolve(key, value, None)
        env_sources[key] = f"env {name}"
    layers.append((env_values, env_sources))
    flag_values, flag_sources = {}, {}
    for dotted, value, given_as in flags:
        if value is None or value == "":
            continue
        key = tuple(dotted.removeprefix("engine.").split("."))
        flag_values[key] = str(value) if isinstance(value, Path) else value
        flag_sources[key] = f"flag {given_as}"
    layers.append((flag_values, flag_sources))

    merged: dict[tuple[str, ...], Any] = {}
    sources: dict[tuple[str, ...], str] = {}
    for values, where in layers:
        for key, value in values.items():
            # A key set here replaces the key, its parents and its children set by a lower layer.
            for old in [k for k in merged if k[: len(key)] == key or key[: len(k)] == k]:
                del merged[old], sources[old]
            merged[key], sources[key] = value, where[key]

    nested: dict[str, Any] = {}
    for key, value in merged.items():
        node = nested
        for part in key[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                break
        else:
            node[key[-1]] = value
    try:
        engine = EngineSettings.model_validate(nested)
    except ValidationError as exc:
        raise SettingsError(_explain(exc, sources)) from None
    return Settings(engine, {".".join(k): v for k, v in sources.items()}, path, found_by)


def _explain(exc: ValidationError, sources: dict[tuple[str, ...], str]) -> str:
    lines = ["invalid engine settings:"]
    for error in exc.errors():
        key = tuple(str(part) for part in error["loc"])
        where = sources.get(key) or ", ".join(sorted({s for k, s in sources.items() if k[: len(key)] == key})) or "default"
        if error["type"] == "extra_forbidden":
            message = "not a setting"
        else:
            message = error["msg"].removeprefix("Value error, ")
            if error["type"] != "value_error" and "input" in error and not isinstance(error["input"], dict):
                message += f" (got {error['input']!r})"
        lines.append(f"  engine.{'.'.join(key)}: {message}  [from {where}]")
    known = ", ".join(".".join(k) for k in all_keys())
    lines.append(f"  settings: {known}")
    return "\n".join(lines)
