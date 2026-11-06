"""Blank out regions of a note whose contents Obsidian does not index.

Masked characters are replaced with spaces and newlines are kept. The masked
text has the same length and line structure as the input, so offsets and line
numbers found in it are valid for the original text.

Masked regions:
* fenced code blocks (``` or ~~~, including the fence lines), also when the
  fence is indented inside a list item or prefixed by blockquote markers;
* inline code spans (a run of N backticks closed by a run of exactly N);
* Obsidian comments (%% ... %%), which may span lines.

Not masked: indented (4-space) code blocks, because in real notes 4-space
indentation is mostly list-item continuation, where links must still count.
"""

from __future__ import annotations

import re

_FENCE_OPEN = re.compile(r"^[ \t]*(?:>[ \t]?)*[ \t]*(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_FENCE_CLOSE = re.compile(r"^[ \t]*(?:>[ \t]?)*[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*$")
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_BACKTICK_RUN = re.compile(r"`+")


def _blank(chars: list[str], start: int, end: int) -> None:
    for index in range(start, end):
        if chars[index] != "\n":
            chars[index] = " "


def mask_region(text: str, start: int, end: int) -> str:
    chars = list(text)
    _blank(chars, start, end)
    return "".join(chars)


def mask_fenced_code(text: str) -> str:
    chars = list(text)
    offset = 0
    open_fence: str | None = None
    for line in text.split("\n"):
        line_end = offset + len(line)
        if open_fence is None:
            match = _FENCE_OPEN.match(line)
            if match and not (match["fence"][0] == "`" and "`" in match["info"]):
                open_fence = match["fence"]
                _blank(chars, offset, line_end)
        else:
            _blank(chars, offset, line_end)
            match = _FENCE_CLOSE.match(line)
            if (
                match
                and match["fence"][0] == open_fence[0]
                and len(match["fence"]) >= len(open_fence)
            ):
                open_fence = None
        offset = line_end + 1
    # An unclosed fence runs to the end of the document (CommonMark/Obsidian).
    return "".join(chars)


def mask_inline_code_and_comments(text: str) -> str:
    chars = list(text)
    position = 0
    length = len(text)
    while position < length:
        next_tick = text.find("`", position)
        next_comment = text.find("%%", position)
        if next_tick == -1 and next_comment == -1:
            break
        if next_comment != -1 and (next_tick == -1 or next_comment < next_tick):
            close = text.find("%%", next_comment + 2)
            end = length if close == -1 else close + 2
            _blank(chars, next_comment, end)
            position = end
            continue

        run = _BACKTICK_RUN.match(text, next_tick)
        assert run is not None
        width = len(run.group())
        paragraph_end = _BLANK_LINE.search(text, run.end())
        limit = paragraph_end.start() if paragraph_end else length
        closing = re.compile(r"(?<!`)`{%d}(?!`)" % width).search(text, run.end(), limit)
        if closing:
            _blank(chars, next_tick, closing.end())
            position = closing.end()
        else:
            position = run.end()  # unmatched backticks are literal text
    return "".join(chars)


def mask_code(text: str) -> str:
    """Mask fenced code first, then inline code spans and %% comments."""
    return mask_inline_code_and_comments(mask_fenced_code(text))
