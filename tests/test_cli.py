from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from unsynth.cli.main import app

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_detect(tmp_path: Path) -> None:
    p = tmp_path / "a.md"
    p.write_text(
        "In today's digital age it is important to note that we leverage robust tools. " * 8,
        encoding="utf-8",
    )
    result = runner.invoke(app, ["detect", str(p)])
    assert result.exit_code == 0
    assert "classical" in result.output or "ensemble" in result.output


def test_cli_clean_stdout(tmp_path: Path) -> None:
    p = tmp_path / "a.md"
    p.write_text(
        "It is important to note that a wide range of teams can leverage robust systems. "
        "Furthermore, they should utilize the comprehensive landscape.\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["clean", str(p), "--quiet", "--max-passes", "1"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_cli_doctor() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "detectors" in result.output


def test_cli_config_show() -> None:
    result = runner.invoke(app, ["config-show"])
    assert result.exit_code == 0
    assert "detect" in result.stdout


def test_cli_rewrite_dry_run(tmp_path: Path) -> None:
    p = tmp_path / "a.md"
    p.write_text("It is important to note that we leverage robust tools.\n", encoding="utf-8")
    result = runner.invoke(app, ["rewrite", str(p), "--dry-run"])
    assert result.exit_code == 0
    assert "would rewrite" in result.output


def test_cli_strip_html(tmp_path: Path) -> None:
    p = tmp_path / "page.html"
    p.write_text("<html><head><meta name='generator' content='x'></head></html>", encoding="utf-8")
    dest = tmp_path / "out.html"
    result = runner.invoke(app, ["strip", str(p), "-o", str(dest)])
    assert result.exit_code == 0
    assert dest.is_file()


def test_cli_eval(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("The cat sat on the mat and looked around the room.\n", encoding="utf-8")
    b.write_text("The cat sat on the rug and glanced around the room.\n", encoding="utf-8")
    result = runner.invoke(app, ["eval", str(a), str(b)])
    assert result.exit_code == 0
    assert "similarity" in result.output
