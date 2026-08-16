"""Long-document helpers: chunking, streaming, batch paths."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from unsynth.pipeline.markdown import iter_rewriteable, parse_markdown, render_segments
from unsynth.safety import split_prose_chunks
from unsynth.text import paragraphs
from unsynth.types import Segment


@dataclass
class Document:
    text: str
    path: Path | None = None
    markdown: bool = True

    @classmethod
    def from_path(cls, path: str | Path) -> Document:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        md = p.suffix.lower() in {".md", ".markdown", ".mdx", ".rst", ".txt", ""}
        return cls(text=text, path=p, markdown=md or True)

    def segments(self, *, protect_code: bool = True, protect_tables: bool = True) -> list[Segment]:
        if self.markdown:
            return parse_markdown(
                self.text, protect_code=protect_code, protect_tables=protect_tables
            )
        return [Segment("prose", self.text, False, 0, len(self.text))]

    def prose_blocks(self) -> list[str]:
        return [seg.text for _, seg in iter_rewriteable(self.segments())]


def _preserve_whitespace(original: str, rewritten: str) -> str:
    """Keep the leading/trailing whitespace of a markdown segment."""

    if not original.strip():
        return original
    lead_n = len(original) - len(original.lstrip())
    trail_n = len(original) - len(original.rstrip())
    lead = original[:lead_n]
    trail = original[len(original) - trail_n :] if trail_n else ""
    return f"{lead}{rewritten.strip()}{trail}"


def iter_prose(text: str, *, markdown: bool = True) -> Iterator[str]:
    if markdown:
        for _, seg in iter_rewriteable(parse_markdown(text)):
            yield seg.text
    else:
        yield from paragraphs(text)


def apply_to_markdown(
    text: str,
    transform: Callable[[str], str],
    *,
    protect_code: bool = True,
    protect_tables: bool = True,
) -> str:
    segments = parse_markdown(text, protect_code=protect_code, protect_tables=protect_tables)
    updated: list[Segment] = []
    for i, seg in enumerate(segments):
        if (
            seg.protected
            or not seg.text.strip()
            or seg.kind not in {"prose", "list_item", "blockquote"}
        ):
            updated.append(seg)
            continue
        pieces: list[str] = []
        for chunk in split_prose_chunks(seg.text):
            if not chunk.strip():
                pieces.append(chunk)
                continue
            try:
                transformed = transform(chunk)
            except Exception:
                pieces.append(chunk)
                continue
            pieces.append(_preserve_whitespace(chunk, transformed))
        new_text = "".join(pieces)
        updated.append(Segment(seg.kind, new_text, seg.protected, seg.start, seg.end))
        del i
    return render_segments(updated)


def iter_files(paths: Iterable[str | Path], suffixes: set[str] | None = None) -> Iterator[Path]:
    allowed = suffixes or {".md", ".markdown", ".txt", ".html", ".htm", ".rst", ".org"}
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            yield path
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in allowed:
                    yield child
