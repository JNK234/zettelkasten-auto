"""Formats: describe your own note conventions in YAML and check a vault against them.

    from vaultengine import Vault
    from vaultengine.formats import TypedVault, load_format

    typed = TypedVault(load_format("formats/brain.yaml"), Vault.load("/path/to/vault"))
    typed.type_of("20_ZETTLEKASTEN/reward.md")      # "zettel"
    typed.relations(path)                             # declared + mirrored relation edges
    typed.links(path)                                 # every link with its role
    typed.issues()                                    # departures from the format

The engine itself knows no conventions; they all come from the format file.
"""

from pathlib import Path

from vaultengine.formats.spec import (
    FORMAT_VERSION,
    FormatError,
    FormatSpec,
    Match,
    MetadataKey,
    NoteType,
    Relation,
    SectionPattern,
    load_format,
    parse_format,
    resolve_format_path,
)
from vaultengine.formats.typed import Issue, MetadataValue, MetaItem, RelationEdge, RoledLink, TypedVault

# formats/ at the repository root (where brain.yaml lives).
BUNDLED_FORMATS_DIR = Path(__file__).resolve().parent.parent.parent / "formats"

__all__ = [
    "BUNDLED_FORMATS_DIR",
    "FORMAT_VERSION",
    "FormatError",
    "FormatSpec",
    "Issue",
    "Match",
    "MetaItem",
    "MetadataKey",
    "MetadataValue",
    "NoteType",
    "Relation",
    "RelationEdge",
    "RoledLink",
    "SectionPattern",
    "TypedVault",
    "load_format",
    "parse_format",
    "resolve_format_path",
]
