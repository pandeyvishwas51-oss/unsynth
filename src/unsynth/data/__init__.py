"""Bundled lexical resources (frequency ranks, synonyms, AI tells)."""

from unsynth.data.lexicon import (
    AI_OPENERS,
    AI_PHRASES,
    CONTRACTIONS,
    EXPAND_CONTRACTIONS,
    FREQUENCY,
    FUNCTION_WORD_WEIGHTS,
    HUMAN_ASIDES,
    PUNCTUATION_SWAPS,
    STRUCTURE_TRANSITIONS,
    SYNONYMS,
    TRANSITION_SWAPS,
    frequency_rank,
    unigram_logprob,
    word_entropy_hint,
)

__all__ = [
    "AI_OPENERS",
    "AI_PHRASES",
    "CONTRACTIONS",
    "EXPAND_CONTRACTIONS",
    "FREQUENCY",
    "FUNCTION_WORD_WEIGHTS",
    "HUMAN_ASIDES",
    "PUNCTUATION_SWAPS",
    "STRUCTURE_TRANSITIONS",
    "SYNONYMS",
    "TRANSITION_SWAPS",
    "frequency_rank",
    "unigram_logprob",
    "word_entropy_hint",
]
