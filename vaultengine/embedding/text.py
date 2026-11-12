r"""The text a note is embedded as, and its hash (the cache key, with the model id).

What goes in: the title (file name), aliases, headings and body, with the same masking as
the search index (frontmatter, code, %% comments, inline #tags, block ids and property keys
like ``Status:`` blanked; property values stay). Tags are left out because ``related`` has
a tags signal of its own. URLs are removed: their pieces (https, arxiv, org, abs) are
not what a note is about. So is TeX math, ``$$ ... $$`` and ``$ ... $``: its commands
(\frac, \mathbb, \tilde) are not words, and two math-heavy notes would look alike for
that alone. Inline math follows pandoc's rule, so prices stay text: the opening ``$`` has
a non-space right after it, the closing one a non-space right before it and no digit
after it ("costs $5 and $10" is not math; ``\$`` is never a delimiter). Whitespace is
collapsed.

It depends on the note alone (its path and its text), never on other notes, so editing one
note can only change that note's vector. The hash is SHA-256 of this text: an edit that does
not change it (whitespace, a tag, frontmatter other than aliases) does not re-embed, and a
rename does (the title is part of the text).

Variants (``vault experiment`` compares them; the index embeds ``whole``):

* ``whole``         - the above: title, aliases, headings, body;
* ``no_template``   - the same without template headings (the ones the format requires for the note's
                      type, or, without a format, a heading written the same way in 3 or more notes) and
                      without metadata (``Key: value`` header lines and ``key:: value`` fields, values too):
                      the parts every note of a template shares, which make notes look alike by style;
* ``first_section`` - the title and the first section that has text (metadata left out): what the
                      note says first, usually what it is.

Bump ``EMBED_TEXT_VERSION`` when what goes in changes; the index then re-parses every note
to store the new hashes, and notes whose text really changed are embedded again.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace

from vaultengine.model import ParsedNote
from vaultengine.parser import NoteLayout
from vaultengine.search import field_segments

EMBED_TEXT_VERSION = 2  # 2: TeX math left out
_URL = re.compile(r"\b(?:https?|ftp)://\S+|\bwww\.\S+", re.IGNORECASE)
_DISPLAY_MATH = re.compile(r"(?<!\\)\$\$.+?(?<!\\)\$\$", re.DOTALL)
_INLINE_MATH = re.compile(r"(?<![\\$])\$(?=[^\s$])(?:[^$\n\\]|\\.)+?(?<![\s\\])\$(?![\d$])")


def without_math(text: str) -> str:
    """``text`` with every ``$$ ... $$`` and ``$ ... $`` (pandoc's rule) blanked."""
    return _INLINE_MATH.sub(" ", _DISPLAY_MATH.sub(" ", text))


VARIANTS = ("whole", "no_template", "first_section")


def embedding_text(path: str, note: ParsedNote, layout: NoteLayout, variant: str = "whole",
                   is_template=None) -> str:
    """What the note is embedded as. ``is_template``: heading text -> is it one of the note's template
    headings (``no_template``; None: none is)."""
    if variant == "whole":
        segments = field_segments(path, note, layout)
        parts = [text for name in ("title", "aliases", "headings", "body") for text, _ in segments[name]]
    else:
        from vaultengine.properties import body_fields

        masked = list(layout.masked)
        for f in body_fields(layout.text, layout.masked):  # metadata: the whole field line, key and value
            start = f.key_span[0]
            end = layout.text.find("\n", start)
            end = len(masked) if end == -1 else end
            masked[start:end] = " " * (end - start)
        segments = field_segments(path, note, replace(layout, masked="".join(masked)))
        if variant == "no_template":
            headings = [text for text, _ in segments["headings"] if not (is_template and is_template(text.strip()))]
            parts = [segments["title"][0][0], *(t for t, _ in segments["aliases"]), *headings,
                     *(t for t, _ in segments["body"])]
        elif variant == "first_section":
            first = next((t for t, _ in segments["body"] if any(ch.isalnum() for ch in without_math(t))), "")
            parts = [segments["title"][0][0], first]
        else:
            raise ValueError(f"unknown embedding text variant {variant!r}; known: {', '.join(VARIANTS)}")
    return " ".join(without_math(_URL.sub(" ", "\n".join(parts))).split())


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
