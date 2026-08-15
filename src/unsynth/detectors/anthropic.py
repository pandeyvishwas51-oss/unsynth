"""Future Anthropic Detection API adapter.

Anthropic has discussed a detection path for Claude's generative watermark
(tournament-sampling / SynthID-Text family). A public, generally available
HTTP API was not a stable contract when UnSynth shipped, so this adapter:

* is a first-class detector in the registry (``anthropic``);
* stays silent unless ``UNSYNTH_ANTHROPIC_API_KEY`` *and*
  ``UNSYNTH_ANTHROPIC_DETECTION_URL`` (or the equivalent settings) are set;
* never invents a "Claude watermark found" score from local heuristics;
* is the only module that should grow official response parsing later.

This keeps the rest of the pipeline honest: local statistical heuristics
live in ``statistical.py``; vendor-keyed detection lives here.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from unsynth.detectors.base import BaseDetector
from unsynth.exceptions import DetectorError
from unsynth.logging import get_logger
from unsynth.types import DetectionContext, DetectorFamily, DetectorResult, Signal

log = get_logger("detectors.anthropic")


class AnthropicDetectionAdapter(BaseDetector):
    name = "anthropic"
    family = DetectorFamily.API
    version = "0.1.0-adapter"

    def detect(self, text: str, *, context: DetectionContext | None = None) -> DetectorResult:
        del context
        api_key = os.environ.get("UNSYNTH_ANTHROPIC_API_KEY", "").strip()
        url = os.environ.get("UNSYNTH_ANTHROPIC_DETECTION_URL", "").strip()
        extra = self.settings.backend.api_key  # unused unless callers stash it
        if not api_key and extra:
            api_key = extra
        if not api_key or not url:
            return self.result(
                0.0,
                confidence=0.0,
                details={
                    "status": "unconfigured",
                    "message": (
                        "Anthropic Detection API is not configured. Set "
                        "UNSYNTH_ANTHROPIC_API_KEY and UNSYNTH_ANTHROPIC_DETECTION_URL "
                        "to enable keyed Claude / SynthID-style detection. "
                        "Local statistical heuristics are the `statistical` detector."
                    ),
                },
                signals=[Signal("configured", 0.0, 1.0, "adapter idle")],
            )

        payload = {"text": text, "encoding": "utf-8"}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "unsynth/0.1.0",
        }
        try:
            response = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.settings.backend.timeout_s,
            )
            response.raise_for_status()
            body: Any = response.json()
        except httpx.HTTPError as exc:
            raise DetectorError(f"Anthropic detection request failed: {exc}") from exc

        score, confidence, details = _parse_vendor_body(body)
        return self.result(
            score,
            confidence=confidence,
            details=details,
            signals=[Signal("vendor_score", score, 1.0, "Anthropic Detection API")],
        )


def _parse_vendor_body(body: Any) -> tuple[float, float, dict[str, Any]]:
    """Best-effort parse of a not-yet-stable vendor payload."""

    if not isinstance(body, dict):
        return 0.0, 0.0, {"raw": body, "status": "unrecognized"}

    score = _first_float(body, ("score", "watermark_score", "detection_score", "p_watermark"))
    confidence = _first_float(body, ("confidence", "certainty"), default=0.5)
    label = body.get("label") or body.get("verdict")
    details: dict[str, Any] = {
        "status": "ok",
        "vendor_label": label,
        "raw_keys": sorted(str(k) for k in body),
    }
    if "watermarked" in body:
        details["watermarked"] = body["watermarked"]
        if score is None:
            score = 1.0 if body["watermarked"] else 0.0
    return float(score or 0.0), float(confidence or 0.0), details


def _first_float(
    body: dict[str, Any], keys: tuple[str, ...], default: float | None = None
) -> float | None:
    for key in keys:
        value = body.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return default
