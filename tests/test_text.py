from __future__ import annotations

from unsynth.text import (
    flesch_reading_ease,
    jaccard,
    reconstruct,
    sentences,
    tfidf_cosine,
    token_change_rate,
    tokenize,
    words,
)


def test_tokenize_keeps_contractions() -> None:
    toks = tokenize("I don't think so.")
    assert [t.text for t in toks if t.is_word] == ["I", "don't", "think", "so"]


def test_sentences_split() -> None:
    parts = sentences("Hello there. How are you? Fine!")
    assert len(parts) == 3


def test_flesch_nonzero() -> None:
    score = flesch_reading_ease("The cat sat on the mat. It looked happy enough.")
    assert 0.0 < score < 120.0


def test_reconstruct_last_first() -> None:
    text = "alpha beta gamma"
    out = reconstruct(text, [(6, 10, "BETA"), (0, 5, "ALPHA")])
    assert out == "ALPHA BETA gamma"


def test_similarity_identity() -> None:
    s = "A short sentence about nothing much at all."
    assert tfidf_cosine(s, s) > 0.99
    assert jaccard(words(s), words(s)) == 1.0
    assert token_change_rate(s, s) == 0.0
