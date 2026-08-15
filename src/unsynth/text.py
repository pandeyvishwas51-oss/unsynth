"""Tokenization, sentence splitting, entropy, and lightweight NLP helpers.

These stay dependency-free. Optional backends (sentence-transformers, a local
LM) plug in at the quality / paraphrase layers, not here.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

# Unicode letters (café, naïve, Москва) plus ASCII numbers and leftover punct.
WORD_RE = re.compile(
    r"[^\W\d_]+(?:'[^\W\d_]+)?|\d+(?:\.\d+)?|[^\s\w]",
    re.UNICODE,
)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(A-Z0-9])")
WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

FUNCTION_WORDS: frozenset[str] = frozenset(
    """
    a an the and or but if while although because as than that this these those
    i you he she it we they me him her us them my your his its our their mine
    yours ours theirs myself yourself himself herself itself ourselves
    is am are was were be been being do does did have has had having
    can could may might must shall should will would
    of in on at to for from by with without into onto over under about
    between among through during before after against toward towards
    not no nor never none nothing nobody
    what which who whom whose where when why how
    so such too very just only even also still yet already
    there here then than once
    """.split()
)

CONTENT_POS_HINT = frozenset("nn vb jj rb".split())


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: int
    end: int
    kind: str  # word | number | punct | other

    @property
    def lower(self) -> str:
        return self.text.lower()

    @property
    def is_word(self) -> bool:
        return self.kind == "word"

    @property
    def is_alpha(self) -> bool:
        return self.is_word and self.text.isalpha()


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    for match in WORD_RE.finditer(text):
        raw = match.group(0)
        if raw[0].isalpha():
            kind = "word"
        elif raw[0].isdigit():
            kind = "number"
        elif raw.strip() == "":
            kind = "other"
        else:
            kind = "punct"
        tokens.append(Token(raw, match.start(), match.end(), kind))
    return tokens


def words(text: str) -> list[str]:
    return [t.text for t in tokenize(text) if t.is_word]


def lower_words(text: str) -> list[str]:
    return [t.lower for t in tokenize(text) if t.is_word]


def sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = SENTENCE_RE.split(stripped)
    return [p.strip() for p in parts if p.strip()]


def paragraphs(text: str) -> list[str]:
    chunks = re.split(r"\n\s*\n", text)
    return [c.strip() for c in chunks if c.strip()]


def syllable_count(word: str) -> int:
    """Crude English syllable estimator (good enough for Flesch)."""

    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    w = re.sub(r"e$", "", w)
    groups = re.findall(r"[aeiouy]+", w)
    return max(1, len(groups))


def flesch_reading_ease(text: str) -> float:
    sents = sentences(text)
    wds = words(text)
    if not sents or not wds:
        return 0.0
    syl = sum(syllable_count(w) for w in wds)
    asl = len(wds) / len(sents)
    asw = syl / len(wds)
    return float(206.835 - 1.015 * asl - 84.6 * asw)


def type_token_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(t.lower() for t in tokens)) / len(tokens)


def hapax_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(t.lower() for t in tokens)
    hapax = sum(1 for c in counts.values() if c == 1)
    return hapax / len(tokens)


def yules_k(tokens: Sequence[str]) -> float:
    """Yule's K vocabulary richness. Higher ≈ more repetitive."""

    if len(tokens) < 8:
        return 0.0
    counts = Counter(t.lower() for t in tokens)
    n = len(tokens)
    freq_of_freq: Counter[int] = Counter(counts.values())
    inner = sum(freq * (count**2) for count, freq in freq_of_freq.items())
    return 10_000 * (inner - n) / (n * n)


def honores_r(tokens: Sequence[str]) -> float:
    if len(tokens) < 8:
        return 0.0
    counts = Counter(t.lower() for t in tokens)
    v = len(counts)
    v1 = sum(1 for c in counts.values() if c == 1)
    if v == 0 or v1 == v:
        return 0.0
    return 100.0 * math.log(len(tokens)) / (1.0 - v1 / v)


def burstiness(sentence_lengths: Sequence[int]) -> float:
    """Coefficient of variation of sentence length.

    Human writing is bursty (short + long sentences). Many classical
    detectors treat low burstiness as an AI tell.
    """

    if len(sentence_lengths) < 2:
        return 0.0
    mean = sum(sentence_lengths) / len(sentence_lengths)
    if mean <= 0:
        return 0.0
    var = sum((x - mean) ** 2 for x in sentence_lengths) / (len(sentence_lengths) - 1)
    return math.sqrt(var) / mean


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def logistic(x: float, midpoint: float, steepness: float = 6.0) -> float:
    """Map a real feature onto (0, 1) centered at *midpoint*."""

    try:
        return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))
    except OverflowError:
        return 0.0 if x < midpoint else 1.0


def inv_logistic(x: float, midpoint: float, steepness: float = 6.0) -> float:
    return 1.0 - logistic(x, midpoint, steepness)


def ngrams(items: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(items) < n:
        return []
    return [tuple(items[i : i + n]) for i in range(len(items) - n + 1)]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def char_ngrams(text: str, n: int = 3) -> Counter[str]:
    compact = re.sub(r"\s+", " ", text.lower())
    if len(compact) < n:
        return Counter([compact] if compact else [])
    return Counter(compact[i : i + n] for i in range(len(compact) - n + 1))


def cosine_counters(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def tfidf_cosine(a: str, b: str) -> float:
    """Tiny TF-IDF cosine on word unigrams + character 3-grams."""

    wa, wb = lower_words(a), lower_words(b)
    if not wa or not wb:
        return 0.0
    ca, cb = Counter(wa), Counter(wb)
    # Blend word cosine with char-ngram cosine for short texts.
    word_cos = cosine_counters(ca, cb)
    char_cos = cosine_counters(char_ngrams(a, 3), char_ngrams(b, 3))
    return 0.55 * word_cos + 0.45 * char_cos


def token_change_rate(original: str, rewritten: str) -> float:
    a = lower_words(original)
    b = lower_words(rewritten)
    if not a:
        return 0.0 if not b else 1.0
    # Alignment-free: 1 - |multiset intersection| / max(len)
    ca, cb = Counter(a), Counter(b)
    common = sum((ca & cb).values())
    return 1.0 - (common / max(len(a), len(b)))


def stable_hash32(parts: Sequence[str | int], seed: int = 0) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(int(seed).to_bytes(8, "little", signed=True))
    for part in parts:
        h.update(b"\x1f")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest(), "little")


def seeded_rng(*parts: str | int, seed: int = 0) -> random.Random:
    return random.Random(stable_hash32(parts, seed=seed))


def sliding_windows(
    tokens: Sequence[Token], window: int, stride: int
) -> Iterator[tuple[int, Sequence[Token]]]:
    if window <= 0:
        return
    if len(tokens) <= window:
        yield 0, tokens
        return
    start = 0
    while start < len(tokens):
        yield start, tokens[start : start + window]
        if start + window >= len(tokens):
            break
        start += max(1, stride)


def reconstruct(text: str, replacements: Sequence[tuple[int, int, str]]) -> str:
    """Apply non-overlapping ``(start, end, new)`` replacements, last first."""

    if not replacements:
        return text
    ordered = sorted(replacements, key=lambda r: r[0], reverse=True)
    out = text
    last_start = len(text) + 1
    for start, end, new in ordered:
        if start < 0 or end < start or end > last_start:
            continue
        out = out[:start] + new + out[end:]
        last_start = start
    return out


def match_casing(source: str, replacement: str) -> str:
    if not source or not replacement:
        return replacement
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper() and source[1:].islower():
        return replacement[:1].upper() + replacement[1:]
    if source[:1].isupper() and len(source) > 1:
        return replacement[:1].upper() + replacement[1:]
    return replacement


def safe_join_sentences(parts: Sequence[str]) -> str:
    cleaned = [p.strip() for p in parts if p.strip()]
    if not cleaned:
        return ""
    glued: list[str] = []
    for part in cleaned:
        if glued and not glued[-1].endswith((".", "!", "?", ":", ";")):
            glued[-1] = glued[-1] + "."
        glued.append(part)
    return " ".join(glued)
