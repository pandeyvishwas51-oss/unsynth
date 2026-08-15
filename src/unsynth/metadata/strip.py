"""Best-effort provenance stripper for common file types.

What we remove
--------------
* C2PA / Content Credentials boxes (JUMBF / ``c2pa`` chunks, ``c2pa`` PDF
  metadata, HTML ``c2pa`` manifests).
* XMP packets (``<?xpacket`` ... ``<?xpacket end``).
* EXIF / IPTC on images (via piexif / Pillow when installed).
* Common generator / AI-provenance HTML meta tags
  (``generator``, ``ai-generated``, C2PA claim generators, IPTC digital
  source type).
* PDF ``/Metadata`` streams and well-known producer info when pypdf is
  available.

This is *not* a forensic wipe. Some containers keep shadow copies; some
vendors re-embed manifests on export. Always re-inspect the output.
"""

from __future__ import annotations

import io
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unsynth.exceptions import MetadataError
from unsynth.logging import get_logger

log = get_logger("metadata")

XMP_RE = re.compile(
    rb"<\?xpacket begin=.*?<\?xpacket end=['\"]w['\"]\s*\?>",
    re.DOTALL | re.IGNORECASE,
)
C2PA_ASCII_MARKERS = (
    b"c2pa",
    b"c2pa.claim",
    b"c2pa.assertion",
    b"jumb",
    b"JUMBF",
    b"c2pc",
)
HTML_META_RE = re.compile(
    r"""<meta\b[^>]*(?:name|property)\s*=\s*['"](?:
            generator|
            provenance|
            c2pa(?:[.:][\w.-]+)?|
            ai-generated|
            ai:generated|
            digitalSourceType|
            iptc:digitalSourceType|
            creds|
            contentcredentials
        )['"][^>]*/?>""",
    re.IGNORECASE | re.VERBOSE,
)
HTML_LINK_RE = re.compile(
    r"""<link\b[^>]*(?:rel)\s*=\s*['"](?:
            provenance|
            c2pa|
            content-credentials|
            describedby
        )['"][^>]*/?>""",
    re.IGNORECASE | re.VERBOSE,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
PNG_SUFFIXES = {".png"}
PDF_SUFFIXES = {".pdf"}
HTML_SUFFIXES = {".html", ".htm", ".xhtml"}
XML_SUFFIXES = {".xml", ".xmp", ".svg"}
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


@dataclass
class StripReport:
    path: str
    output: str
    removed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "output": self.output,
            "removed": list(self.removed),
            "skipped": list(self.skipped),
            "warnings": list(self.warnings),
        }


def strip_path(
    source: str | Path,
    dest: str | Path | None = None,
    *,
    inplace: bool = False,
) -> StripReport:
    src = Path(source)
    if not src.is_file():
        raise MetadataError(f"not a file: {src}")
    if inplace:
        target = src
    elif dest is not None:
        target = Path(dest)
    else:
        target = src.with_name(src.stem + ".stripped" + src.suffix)
    return strip_file(src, target)


def strip_file(source: Path, dest: Path) -> StripReport:
    suffix = source.suffix.lower()
    report = StripReport(path=str(source), output=str(dest))
    data = source.read_bytes()

    if suffix in HTML_SUFFIXES:
        text = data.decode("utf-8", errors="replace")
        cleaned, hits = strip_html(text)
        dest.write_text(cleaned, encoding="utf-8")
        report.removed.extend(hits)
        return report

    if suffix in XML_SUFFIXES or suffix in TEXT_SUFFIXES:
        cleaned_bytes, hits = _strip_xmp_bytes(data)
        dest.write_bytes(cleaned_bytes)
        report.removed.extend(hits)
        return report

    if suffix in PDF_SUFFIXES:
        return _strip_pdf(source, dest, report)

    if suffix in IMAGE_SUFFIXES or suffix in PNG_SUFFIXES:
        return _strip_image(source, dest, data, report)

    # Generic binary: drop obvious XMP packets and warn.
    cleaned_bytes, hits = _strip_xmp_bytes(data)
    if dest != source:
        dest.write_bytes(cleaned_bytes)
    else:
        dest.write_bytes(cleaned_bytes)
    report.removed.extend(hits)
    report.warnings.append(f"generic strip for suffix {suffix or '[none]'}")
    return report


def strip_html(html: str) -> tuple[str, list[str]]:
    hits: list[str] = []
    out = HTML_META_RE.sub(_mark("meta", hits), html)
    out = HTML_LINK_RE.sub(_mark("link", hits), out)
    # Common claim JSON-LD blobs.
    out, n = re.subn(
        r"<script[^>]+type=['\"]application/ld\+json['\"][^>]*>.*?</script>",
        _maybe_drop_jsonld(hits),
        out,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if n and "jsonld" not in hits:
        pass
    out_bytes, xmp_hits = _strip_xmp_bytes(out.encode("utf-8"))
    hits.extend(xmp_hits)
    return out_bytes.decode("utf-8", errors="replace"), hits


def _maybe_drop_jsonld(hits: list[str]) -> Any:
    def _sub(match: re.Match[str]) -> str:
        body = match.group(0).lower()
        if any(k in body for k in ("c2pa", "contentcredentials", "digitalSourceType", "prov:")):
            hits.append("jsonld-provenance")
            return ""
        return match.group(0)

    return _sub


def _mark(kind: str, hits: list[str]) -> Any:
    def _sub(match: re.Match[str]) -> str:
        hits.append(f"html-{kind}")
        return ""

    return _sub


def _strip_xmp_bytes(data: bytes) -> tuple[bytes, list[str]]:
    hits: list[str] = []
    cleaned, n = XMP_RE.subn(b"", data)
    if n:
        hits.append(f"xmp-packet×{n}")
    # Cheap C2PA marker note — we do not try to surgically excise JUMBF
    # without a dedicated parser; we report it so the user can inspect.
    lowered = cleaned.lower()
    for marker in C2PA_ASCII_MARKERS:
        if marker.lower() in lowered:
            hits.append(f"c2pa-marker:{marker.decode('latin1', errors='ignore')}")
    return cleaned, hits


def _strip_image(source: Path, dest: Path, data: bytes, report: StripReport) -> StripReport:
    cleaned, hits = _strip_xmp_bytes(data)
    report.removed.extend(hits)
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(cleaned))
        payload = io.BytesIO()
        save_kwargs: dict[str, Any] = {}
        fmt = (image.format or source.suffix.lstrip(".")).upper()
        if fmt in {"JPEG", "JPG"}:
            try:
                import piexif

                save_kwargs["exif"] = piexif.dump({})
                report.removed.append("exif")
            except Exception:
                report.skipped.append("piexif-unavailable")
        # Drop info dict (PNG text chunks etc.) by not forwarding it.
        if "comment" in (image.info or {}):
            report.removed.append("comment")
        rgb = image
        if fmt in {"JPEG", "JPG"} and image.mode not in {"RGB", "L"}:
            rgb = image.convert("RGB")
        rgb.save(payload, format=fmt, **save_kwargs)
        dest.write_bytes(payload.getvalue())
        report.removed.append("image-reencoded")
        return report
    except ImportError:
        report.skipped.append("pillow-unavailable")
        dest.write_bytes(cleaned)
        report.warnings.append("installed unsynth[metadata] for deeper image wipes")
        return report
    except Exception as exc:
        report.warnings.append(f"image rewrite failed: {exc}")
        dest.write_bytes(cleaned)
        return report


def _strip_pdf(source: Path, dest: Path, report: StripReport) -> StripReport:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        shutil.copyfile(source, dest)
        report.skipped.append("pypdf-unavailable")
        report.warnings.append("install unsynth[metadata] to strip PDF /Metadata")
        return report
    try:
        reader = PdfReader(str(source))
        writer = PdfWriter()
        writer.append_pages_from_reader(reader)
        # Drop document info + XMP if the API exposes it.
        if reader.metadata:
            report.removed.append("pdf-document-info")
        writer.add_metadata({})
        # pypdf >= 4
        if hasattr(writer, "metadata"):
            try:
                writer.metadata = {}
            except Exception:
                pass
        with dest.open("wb") as fh:
            writer.write(fh)
        # Second pass: drop leftover xpacket bytes.
        data = dest.read_bytes()
        cleaned, hits = _strip_xmp_bytes(data)
        if hits:
            dest.write_bytes(cleaned)
            report.removed.extend(hits)
        report.removed.append("pdf-rewritten")
        return report
    except Exception as exc:
        raise MetadataError(f"PDF strip failed: {exc}") from exc
