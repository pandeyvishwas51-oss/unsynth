from __future__ import annotations

from unsynth.rewriters.lexical import LexicalRewriter
from unsynth.rewriters.quality import QualityGate
from unsynth.rewriters.structural import StructuralRewriter
from unsynth.rewriters.style import StyleHumanizer


def test_lexical_kills_stock_phrases(ai_essay: str, settings) -> None:  # type: ignore[no-untyped-def]
    out = LexicalRewriter(settings).rewrite(ai_essay, strength=0.9)
    assert out.edits >= 1
    assert "it is important to note" not in out.rewritten.lower()
    assert out.quality.similarity > 0.35


def test_style_introduces_contractions(settings) -> None:  # type: ignore[no-untyped-def]
    text = (
        "They are not ready. It is not finished. We are going to the store. "
        "I am sure we will not be late. They have not seen the draft."
    )
    out = StyleHumanizer(settings).rewrite(text, strength=0.95)
    lowered = out.rewritten.lower()
    assert any(c in lowered for c in ("n't", "it's", "we're", "i'm", "i'll", "they've"))


def test_structural_changes_long_coordinated(settings) -> None:  # type: ignore[no-untyped-def]
    text = (
        "The committee reviewed the long proposal in detail, and the board "
        "voted to delay the launch until spring. The press team drafted a note."
    )
    out = StructuralRewriter(settings).rewrite(text, strength=0.9)
    assert out.rewritten
    assert out.quality.passed


def test_quality_rejects_unrelated(settings) -> None:  # type: ignore[no-untyped-def]
    gate = QualityGate(settings)
    report = gate.evaluate(
        "The history of tea in Britain is mostly a story of trade routes.",
        "Quantum chromodynamics describes the strong force between quarks.",
    )
    assert report.similarity < 0.5
    assert report.passed is False
