"""Isolation: one bad unit must not take down the product."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.helpers import AI_PARAGRAPH, assert_pipeline_sane, assert_result_sane
from unsynth.cli.main import app
from unsynth.config import Settings, load_settings
from unsynth.detectors.base import BaseDetector
from unsynth.detectors.classical import ClassicalDetector
from unsynth.detectors.ensemble import EnsembleDetector
from unsynth.detectors.registry import DetectorRegistry
from unsynth.exceptions import ConfigError, MetadataError
from unsynth.metadata.strip import strip_path
from unsynth.pipeline.document import apply_to_markdown
from unsynth.pipeline.orchestrator import UnSynthPipeline
from unsynth.plugins import load_directory_plugins
from unsynth.rewriters.base import BaseRewriter
from unsynth.rewriters.lexical import LexicalRewriter
from unsynth.rewriters.pipeline import RewriteStack
from unsynth.safety import atomic_write_text
from unsynth.types import (
    DetectionContext,
    DetectorFamily,
    DetectorResult,
    PipelineMode,
    RewriteResult,
)


class BoomDetector(BaseDetector):
    name = "boom"
    family = DetectorFamily.CLASSICAL

    def detect(self, text: str, *, context: DetectionContext | None = None) -> DetectorResult:
        raise RuntimeError("detector exploded")


class BoomRewriter(BaseRewriter):
    name = "boom"
    requires_backend = False

    def rewrite(self, text: str, *, strength: float = 0.45) -> RewriteResult:
        raise RuntimeError("rewriter exploded")


def test_ensemble_survives_exploding_child(settings: Settings) -> None:
    ens = EnsembleDetector(
        settings, detectors=[BoomDetector(settings), ClassicalDetector(settings)]
    )
    result = ens.detect(AI_PARAGRAPH)
    assert_result_sane(result)
    members = result.details["members"]
    assert any(m["name"] == "boom" and m["score"] == 0.0 for m in members)  # type: ignore[index]
    assert any(m["name"] == "classical" and float(m["score"]) > 0.4 for m in members)  # type: ignore[index]


def test_stack_survives_exploding_strategy(settings: Settings) -> None:
    stack = RewriteStack(settings, strategies=[BoomRewriter(settings), LexicalRewriter(settings)])
    out = stack.run(AI_PARAGRAPH, strength=0.8)
    assert isinstance(out.rewritten, str)
    assert any("boom:error" in n for n in out.notes)
    assert "it is important to note" not in out.rewritten.lower()


def test_transform_crash_keeps_original_chunk() -> None:
    src = "Please leverage robust tools.\n\nSecond paragraph stays too.\n"

    def _boom(chunk: str) -> str:
        raise RuntimeError("nope")

    assert apply_to_markdown(src, _boom) == src


def test_unknown_detector_is_skipped(settings: Settings) -> None:
    settings.detect.detectors = ["classical", "not-real", "stylometric"]
    dets = DetectorRegistry(settings).create_many()
    names = {d.name for d in dets}
    assert "classical" in names
    assert "not-real" not in names


def test_missing_plugin_dir_does_not_crash(tmp_path: Path) -> None:
    found = load_directory_plugins([tmp_path / "does-not-exist"])
    assert found == {}


def test_broken_plugin_file_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("raise SystemExit('nope')\n", encoding="utf-8")
    (tmp_path / "good.py").write_text(
        "from unsynth.detectors.base import BaseDetector\n"
        "from unsynth.types import DetectionContext, DetectorFamily, DetectorResult\n"
        "class Plugin(BaseDetector):\n"
        "    name = 'plugok'\n"
        "    family = DetectorFamily.CLASSICAL\n"
        "    def detect(self, text, *, context=None):\n"
        "        return self.empty_result(self.name, self.family, 'ok')\n",
        encoding="utf-8",
    )
    found = load_directory_plugins([tmp_path])
    assert "plugok" in found
    assert "bad" not in found


def test_pipeline_unknown_strategy_still_runs(settings: Settings) -> None:
    settings.rewrite.strategies = ["lexical", "does-not-exist"]
    result = UnSynthPipeline(settings).run(AI_PARAGRAPH, mode=PipelineMode.REWRITE)
    assert_pipeline_sane(result)


def test_config_rejects_bad_numbers(tmp_path: Path) -> None:
    bad = tmp_path / "unsynth.yaml"
    bad.write_text("rewrite:\n  max_passes: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(bad)
    bad.write_text("runtime:\n  workers: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(bad)


def test_config_rejects_broken_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "unsynth.toml"
    cfg.write_text("rewrite = [[[[", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(cfg)


def test_strip_refuses_directory_dest(tmp_path: Path) -> None:
    src = tmp_path / "note.txt"
    src.write_text("hello <?xpacket begin=''?><x><?xpacket end='w'?>", encoding="utf-8")
    dest = tmp_path / "outdir"
    dest.mkdir()
    with pytest.raises(MetadataError):
        strip_path(src, dest)


def test_strip_creates_parent_dirs(tmp_path: Path) -> None:
    src = tmp_path / "note.txt"
    src.write_text("hello <?xpacket begin=''?><x><?xpacket end='w'?> tail", encoding="utf-8")
    dest = tmp_path / "nested" / "deep" / "out.txt"
    report = strip_path(src, dest)
    assert dest.is_file()
    assert b"xpacket" not in dest.read_bytes()
    assert report.removed


def test_atomic_write_replaces(tmp_path: Path) -> None:
    dest = tmp_path / "out.txt"
    dest.write_text("old", encoding="utf-8")
    atomic_write_text(dest, "new-content")
    assert dest.read_text(encoding="utf-8") == "new-content"
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_cli_batch_continues_after_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir()
    good = src / "ok.md"
    good.write_text(AI_PARAGRAPH + "\n", encoding="utf-8")
    bad = src / "bad.md"
    bad.write_text("x\n", encoding="utf-8")
    real_run = UnSynthPipeline.run

    def _maybe_boom(self, text, *, mode=PipelineMode.CLEAN, context=None, progress=None):  # type: ignore[no-untyped-def]
        if text.strip() == "x":
            raise RuntimeError("simulated")
        return real_run(self, text, mode=mode, context=context, progress=progress)

    monkeypatch.setattr(UnSynthPipeline, "run", _maybe_boom)
    result = CliRunner().invoke(app, ["batch", str(src), "-o", str(out), "--mode", "detect"])
    assert result.exit_code != 0
    assert list(out.glob("*.md")) or list(out.rglob("*"))


def test_cli_bad_config_is_clean_exit(tmp_path: Path) -> None:
    cfg = tmp_path / "unsynth.yaml"
    cfg.write_text("detect:\n  ai_likely: 9\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["--config", str(cfg), "version"])
    # version does not load settings; doctor does
    result = CliRunner().invoke(app, ["--config", str(cfg), "doctor"])
    assert result.exit_code == 1
    assert "config" in result.output.lower() or "invalid" in result.output.lower()
