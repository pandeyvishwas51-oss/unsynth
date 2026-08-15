from __future__ import annotations

from pathlib import Path

from unsynth.metadata.strip import strip_html, strip_path


def test_strip_html_meta() -> None:
    html = """
    <html><head>
    <meta name="generator" content="ChatGPT">
    <meta name="c2pa" content="claim">
    <meta property="og:title" content="keep">
    <link rel="c2pa" href="manifest.json">
    </head><body><p>Hello</p></body></html>
    """
    cleaned, hits = strip_html(html)
    assert "generator" not in cleaned
    assert "c2pa" not in cleaned.lower() or "manifest" not in cleaned
    assert "og:title" in cleaned
    assert hits


def test_strip_xmp_file(tmp_path: Path) -> None:
    blob = b"hello <?xpacket begin=''?><x:xmpmeta>secret</x:xmpmeta><?xpacket end='w'?> tail"
    src = tmp_path / "note.txt"
    src.write_bytes(blob)
    report = strip_path(src)
    out = Path(report.output).read_bytes()
    assert b"xpacket" not in out
    assert b"hello" in out
    assert report.removed
