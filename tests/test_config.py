from __future__ import annotations

from pathlib import Path

import pytest

from unsynth.config import Settings, load_settings
from unsynth.exceptions import ConfigError


def test_defaults_load() -> None:
    s = Settings()
    assert "classical" in s.detect.detectors
    assert s.backend.kind.value == "none"
    assert 0.0 < s.rewrite.min_similarity < 1.0


def test_yaml_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "unsynth.yaml"
    cfg.write_text("rewrite:\n  max_passes: 7\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    loaded = load_settings()
    assert loaded.rewrite.max_passes == 7


def test_missing_explicit_config() -> None:
    with pytest.raises(ConfigError):
        load_settings("/no/such/unsynth.yaml")


def test_label_for_thresholds() -> None:
    s = Settings()
    assert s.label_for(0.9) == "ai"
    assert s.label_for(0.1) == "human"
    assert s.label_for(0.9, family_watermark=True) == "watermarked"
