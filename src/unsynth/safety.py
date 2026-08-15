"""Production guards: sanitize input, keep scores finite, protect spans."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from unsynth.exceptions import UnSynthError
from unsynth.text import EMAIL_RE, URL_RE

# Two million characters is ~300–400 pages. Above that we still run, but
# callers can opt into a hard cap via Settings later.
SOFT_MAX_CHARS = 2_000_000
PROSE_CHUNK_CHARS = 3_500

_NUL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class InputTooLargeError(UnSynthError):
    """Document exceeds the configured hard size cap."""


def sanitize_text(text: str, *, hard_max: int | None = None) -> str:
    """Make arbitrary user text safe to run through the pipeline.

    * Rejects non-str at the type boundary (callers should pass str).
    * Drops C0 control characters except ``\\t``, ``\\n``, ``\\r``.
    * Replaces lone surrogates so later UTF-8 writes cannot fail.
    * Optionally enforces a hard character cap.
    """

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if hard_max is not None and len(text) > hard_max:
        raise InputTooLargeError(f"document is {len(text)} chars; cap is {hard_max}")
    cleaned = _NUL_RE.sub("", text)
    return cleaned.encode("utf-8", errors="replace").decode("utf-8")


def finite_unit(value: float, default: float = 0.0) -> float:
    """Clamp to [0, 1] and replace NaN / Inf."""

    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def finite_number(value: float, default: float = 0.0, *, hi: float = 1e9) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return float(max(-hi, min(hi, value)))


def protected_spans(text: str) -> list[tuple[int, int]]:
    """Character spans that must not be rewritten (URLs, emails)."""

    spans: list[tuple[int, int]] = []
    for rx in (URL_RE, EMAIL_RE):
        for match in rx.finditer(text):
            spans.append((match.start(), match.end()))
    spans.sort()
    return _merge_spans(spans)


def overlaps_protected(start: int, end: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(not (end <= lo or start >= hi) for lo, hi in spans)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    out = [spans[0]]
    for start, end in spans[1:]:
        prev_s, prev_e = out[-1]
        if start <= prev_e:
            out[-1] = (prev_s, max(prev_e, end))
        else:
            out.append((start, end))
    return out


def split_prose_chunks(block: str, limit: int = PROSE_CHUNK_CHARS) -> list[str]:
    """Split a huge prose block on paragraph boundaries without losing bytes."""

    if len(block) <= limit:
        return [block]
    parts = re.split(r"(\n\s*\n)", block)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        if buf and len(buf) + len(part) > limit and not re.fullmatch(r"\n\s*\n", part or ""):
            chunks.append(buf)
            buf = part
        else:
            buf += part
    if buf:
        chunks.append(buf)
    return chunks or [block]
