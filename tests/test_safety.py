from __future__ import annotations

import pytest

from unsynth.exceptions import UnSynthError
from unsynth.safety import (
    InputTooLargeError,
    finite_number,
    finite_unit,
    overlaps_protected,
    protected_spans,
    sanitize_text,
    split_prose_chunks,
)
from unsynth.text import stable_hash32, tokenize, words


def test_sanitize_drops_nuls_and_controls() -> None:
    raw = "hello\x00world\x07\ttab\nline"
    out = sanitize_text(raw)
    assert "\x00" not in out
    assert "\x07" not in out
    assert "hello" in out and "world" in out
    assert "\t" in out and "\n" in out


def test_sanitize_rejects_non_str() -> None:
    with pytest.raises(TypeError):
        sanitize_text(123)  # type: ignore[arg-type]


def test_sanitize_hard_max() -> None:
    with pytest.raises(InputTooLargeError):
        sanitize_text("abcdef", hard_max=3)
    assert isinstance(InputTooLargeError("x"), UnSynthError)


def test_finite_guards() -> None:
    assert finite_unit(float("nan")) == 0.0
    assert finite_unit(float("inf")) == 0.0
    assert finite_unit(-4.0) == 0.0
    assert finite_unit(2.0) == 1.0
    assert finite_unit(0.4) == 0.4
    assert finite_number(float("nan"), default=1.5) == 1.5


def test_url_and_email_spans() -> None:
    text = "See https://example.com/a?x=1 and write ops@example.com please."
    spans = protected_spans(text)
    assert len(spans) == 2
    url = text[spans[0][0] : spans[0][1]]
    assert url.startswith("https://")
    assert overlaps_protected(spans[0][0], spans[0][0] + 3, spans)


def test_split_prose_roundtrip() -> None:
    block = ("para one. " * 80 + "\n\n") * 6
    chunks = split_prose_chunks(block, limit=200)
    assert "".join(chunks) == block
    assert len(chunks) > 1


def test_negative_seed_hashes() -> None:
    a = stable_hash32(("x",), seed=-13)
    b = stable_hash32(("x",), seed=-13)
    c = stable_hash32(("x",), seed=13)
    assert a == b
    assert a != c


def test_unicode_tokenizer() -> None:
    toks = words("Café naïve Москва 東京 hello")
    assert "Café" in toks or "café" in {t.lower() for t in tokenize("Café") if t.is_word}
    assert "naïve" in [t.lower for t in tokenize("naïve") if t.is_word]
    assert any(t.is_word for t in tokenize("Москва"))
    assert any(t.is_word for t in tokenize("東京"))
