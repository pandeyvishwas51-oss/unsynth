from __future__ import annotations

from unsynth.config import Settings
from unsynth.detectors.anthropic import AnthropicDetectionAdapter
from unsynth.detectors.classical import ClassicalDetector
from unsynth.detectors.ensemble import EnsembleDetector
from unsynth.detectors.registry import DetectorRegistry
from unsynth.detectors.statistical import StatisticalWatermarkDetector
from unsynth.detectors.stylometric import StylometricDetector


def test_classical_ranks_ai_higher(ai_essay: str, human_note: str) -> None:
    det = ClassicalDetector()
    ai = det.detect(ai_essay)
    human = det.detect(human_note)
    assert ai.score > human.score
    assert ai.score > 0.45
    assert human.score < 0.55


def test_stylometric_runs(ai_essay: str) -> None:
    result = StylometricDetector().detect(ai_essay)
    assert 0.0 <= result.score <= 1.0
    assert result.signals
    assert result.details["tokens"] > 20


def test_statistical_short_text() -> None:
    result = StatisticalWatermarkDetector().detect("too short")
    assert result.label == "uncertain"
    assert result.confidence == 0.0


def test_statistical_long_text(ai_essay: str) -> None:
    result = StatisticalWatermarkDetector().detect(ai_essay * 3)
    assert 0.0 <= result.score <= 1.0
    assert "disclaimer" in result.details
    assert result.family.value == "watermark"


def test_anthropic_idle_without_keys() -> None:
    result = AnthropicDetectionAdapter().detect("anything at all " * 20)
    assert result.score == 0.0
    assert result.details["status"] == "unconfigured"


def test_ensemble_members(ai_essay: str, settings: Settings) -> None:
    ens = EnsembleDetector(settings)
    result = ens.detect(ai_essay)
    assert result.name == "ensemble"
    members = result.details["members"]
    names = {m["name"] for m in members}  # type: ignore[index]
    assert {"classical", "stylometric", "statistical"} <= names
    assert result.score > 0.3


def test_registry_unknown() -> None:
    reg = DetectorRegistry()
    assert "classical" in reg.available()
    import pytest

    from unsynth.exceptions import PluginError

    with pytest.raises(PluginError):
        reg.create("not-a-real-detector")
