from __future__ import annotations

from pathlib import Path

from unsynth.pipeline.document import Document, apply_to_markdown, iter_files, iter_prose


def test_document_from_path(tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    p.write_text("# Hi\n\nHello there.\n\n```\ncode\n```\n", encoding="utf-8")
    doc = Document.from_path(p)
    blocks = doc.prose_blocks()
    assert any("Hello" in b for b in blocks)
    assert all("code" not in b for b in blocks)


def test_apply_to_markdown_skips_code() -> None:
    src = "Please leverage this.\n\n```\nKEEP\n```\n"
    out = apply_to_markdown(src, lambda s: s.replace("leverage", "use"))
    assert "use" in out
    assert "KEEP" in out


def test_iter_prose_plain() -> None:
    chunks = list(iter_prose("one para\n\ntwo para", markdown=False))
    assert chunks == ["one para", "two para"]


def test_iter_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"\x00")
    found = [p.name for p in iter_files([tmp_path])]
    assert found == ["a.md"]
