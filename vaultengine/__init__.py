"""vaultengine: read an Obsidian vault the way Obsidian does.

Core package with no LLM, embedding or database dependencies.
"""

from vaultengine.model import BlockId, ExternalLink, Heading, Link, ParsedNote, Tag
from vaultengine.parser import is_valid_tag, parse_file, parse_note
from vaultengine.resolver import ResolvedLink, Resolver
from vaultengine.vault import Vault
from vaultengine.graph import LinkGraph

__all__ = [
    "BlockId",
    "ExternalLink",
    "Heading",
    "Link",
    "LinkGraph",
    "ParsedNote",
    "ResolvedLink",
    "Resolver",
    "Tag",
    "Vault",
    "is_valid_tag",
    "parse_file",
    "parse_note",
]
