"""UnSynth — detect and disrupt AI-text detectors and statistical LLM watermarks.

Honesty first: UnSynth does **not** cryptographically erase a private-key
generative watermark. Those schemes (SynthID-Text, Kirchenbauer-style
green/red lists, tournament sampling) can only be disrupted by changing
the token sequence and the local context that later tokens are seeded
from. UnSynth is the strongest *practical* disruption + humanization
toolkit we can ship without the secret key.
"""

from __future__ import annotations

from unsynth.config import Settings, load_settings
from unsynth.detectors.ensemble import EnsembleDetector
from unsynth.pipeline.orchestrator import UnSynthPipeline, run_pipeline
from unsynth.types import (
    DetectorResult,
    EvalReport,
    PipelineResult,
    RewriteResult,
)

__version__ = "0.1.0"
__all__ = [
    "DetectorResult",
    "EnsembleDetector",
    "EvalReport",
    "PipelineResult",
    "RewriteResult",
    "Settings",
    "UnSynthPipeline",
    "__version__",
    "load_settings",
    "run_pipeline",
]
