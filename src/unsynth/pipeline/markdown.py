"""Markdown-aware segmentation. Code, tables, and URLs stay untouched."""

from __future__ import annotations

import re
from collections.abc import Iterator

from unsynth.types import Segment, SegmentKind

# Any fenced opener: ```python, ```c++, ```python:3.11, ~~~, indented-or-not.
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[^\n]*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
TABLE_RE = re.compile(r"^\s*\|.+\|\s*$")
HR_RE = re.compile(r"^\s*([-*_]\s*){3,}$")
FRONTMATTER_RE = re.compile(r"^---\s*$")
HTML_RE = re.compile(r"^</?[a-zA-Z][^>]*>")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")


def parse_markdown(
    text: str, *, protect_code: bool = True, protect_tables: bool = True
) -> list[Segment]:
    """Split *text* into rewriteable vs protected segments.

    This is a linear scan, not a full CommonMark parse. It is conservative:
    if we are unsure, we protect.
    """

    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    segments: list[Segment] = []
    buf: list[str] = []
    kind: SegmentKind = "prose"
    protected = False
    in_fence = False
    in_frontmatter = False
    offset = 0
    start = 0

    def flush(at: int) -> None:
        nonlocal buf, kind, protected, start
        if not buf:
            start = at
            return
        chunk = "".join(buf)
        segments.append(Segment(kind, chunk, protected, start, at))
        buf = []
        kind = "prose"
        protected = False
        start = at

    # Optional YAML front matter.
    if lines and FRONTMATTER_RE.match(lines[0].rstrip("\n")):
        in_frontmatter = True
        kind = "frontmatter"
        protected = True

    for line in lines:
        stripped = line.rstrip("\n")
        if in_frontmatter:
            buf.append(line)
            offset += len(line)
            if len(buf) > 1 and FRONTMATTER_RE.match(stripped):
                flush(offset)
                in_frontmatter = False
            continue
        if FENCE_RE.match(stripped):
            if in_fence:
                buf.append(line)
                offset += len(line)
                flush(offset)
                in_fence = False
                continue
            flush(offset)
            in_fence = True
            kind = "code"
            protected = protect_code
            buf.append(line)
            offset += len(line)
            continue
        if in_fence:
            buf.append(line)
            offset += len(line)
            continue
        line_kind, line_protected = _classify_line(stripped, protect_tables=protect_tables)
        if buf and (line_kind != kind or line_protected != protected):
            flush(offset)
        kind = line_kind
        protected = line_protected
        buf.append(line)
        offset += len(line)

    flush(offset)
    return _coalesce(segments)


def _classify_line(stripped: str, *, protect_tables: bool) -> tuple[SegmentKind, bool]:
    if not stripped.strip():
        return "prose", False
    if HEADING_RE.match(stripped):
        # Headings stay put: changing them wrecks anchors and usually
        # concatenates into the next paragraph if whitespace is lost.
        return "heading", True
    if LIST_RE.match(stripped):
        return "list_item", False
    if stripped.lstrip().startswith(">"):
        return "blockquote", False
    if protect_tables and (TABLE_RE.match(stripped) or HR_RE.match(stripped)):
        return "table", True
    if HTML_RE.match(stripped.lstrip()):
        return "html", True
    return "prose", False


def _coalesce(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []
    out = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        if prev.kind == seg.kind and prev.protected == seg.protected:
            out[-1] = Segment(
                prev.kind,
                prev.text + seg.text,
                prev.protected,
                prev.start,
                seg.end,
            )
        else:
            out.append(seg)
    return out


def render_segments(segments: list[Segment]) -> str:
    return "".join(seg.text for seg in segments)


def iter_rewriteable(segments: list[Segment]) -> Iterator[tuple[int, Segment]]:
    for i, seg in enumerate(segments):
        if (
            not seg.protected
            and seg.kind in {"prose", "list_item", "blockquote"}
            and seg.text.strip()
        ):
            yield i, seg
