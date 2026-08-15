from __future__ import annotations

import json
from pathlib import Path

from tests.helpers import AI_PARAGRAPH, HUMAN_PARAGRAPH
from typer.testing import CliRunner

from unsynth.cli.main import app

runner = CliRunner()


def test_detect_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.md"
    p.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["detect", str(p), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "score" in payload
    assert 0.0 <= payload["score"] <= 1.0


def test_detect_stdin_pipe() -> None:
    result = runner.invoke(app, ["detect", "--json"], input=AI_PARAGRAPH)
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["score"] > 0.4


def test_detect_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["detect", str(tmp_path / "nope.md")])
    assert result.exit_code != 0


def test_clean_json_and_report(tmp_path: Path) -> None:
    src = tmp_path / "in.md"
    src.write_text(AI_PARAGRAPH + "\n", encoding="utf-8")
    report = tmp_path / "r.md"
    dest = tmp_path / "out.md"
    result = runner.invoke(
        app,
        [
            "clean",
            str(src),
            "-o",
            str(dest),
            "--report",
            str(report),
            "--max-passes",
            "1",
            "--quiet",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "text" in payload and "result" in payload
    assert dest.is_file()
    assert report.is_file()
    assert "UnSynth report" in report.read_text(encoding="utf-8")


def test_batch_skips_binaries(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    (src / "a.md").write_text(HUMAN_PARAGRAPH + "\n", encoding="utf-8")
    (src / "b.txt").write_text(AI_PARAGRAPH + "\n", encoding="utf-8")
    (src / "c.bin").write_bytes(b"\xff\x00\x01")
    result = runner.invoke(app, ["batch", str(src), "-o", str(out), "--mode", "detect"])
    assert result.exit_code == 0
    written = list(out.rglob("*"))
    assert written
    assert not any(p.suffix == ".bin" for p in written)


def test_rewrite_strength_bounds(tmp_path: Path) -> None:
    p = tmp_path / "a.md"
    p.write_text(AI_PARAGRAPH + "\n", encoding="utf-8")
    hi = runner.invoke(app, ["rewrite", str(p), "--strength", "1.5"])
    assert hi.exit_code != 0
    ok = runner.invoke(app, ["rewrite", str(p), "--strength", "0.2", "--dry-run"])
    assert ok.exit_code == 0


def test_config_invalid_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "unsynth.yaml"
    cfg.write_text("detect:\n  ai_likely: 4.0\n", encoding="utf-8")
    result = runner.invoke(app, ["--config", str(cfg), "doctor"])
    assert result.exit_code != 0


def test_nul_file_detect(tmp_path: Path) -> None:
    p = tmp_path / "nul.md"
    p.write_bytes(b"It is important to note that we leverage robust tools.\x00 extra words here.\n")
    result = runner.invoke(app, ["detect", str(p), "--json"])
    assert result.exit_code == 0
    json.loads(result.stdout)
