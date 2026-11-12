"""vaultengine: read an Obsidian vault the way Obsidian does.

Core package with no LLM or database server dependencies. Embeddings (``vaultengine.embedding``) use a local
static model with numpy and tokenizers, imported only when vectors are needed.
"""

from vaultengine.model import BlockId, ExternalLink, HeaderField, Heading, InlineField, Link, ParsedNote, Tag
from vaultengine.parser import is_valid_tag, parse_file, parse_note
from vaultengine.resolver import ResolvedLink, Resolver
from vaultengine.vault import Vault
from vaultengine.graph import LinkGraph

__all__ = [
    "BlockId",
    "ExternalLink",
    "HeaderField",
    "Heading",
    "InlineField",
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
