"""Pluggable detectors for classical AI-text cues and watermark heuristics."""

from unsynth.detectors.anthropic import AnthropicDetectionAdapter
from unsynth.detectors.base import BaseDetector
from unsynth.detectors.classical import ClassicalDetector
from unsynth.detectors.ensemble import EnsembleDetector
from unsynth.detectors.registry import DetectorRegistry
from unsynth.detectors.statistical import StatisticalWatermarkDetector
from unsynth.detectors.stylometric import StylometricDetector

__all__ = [
    "AnthropicDetectionAdapter",
    "BaseDetector",
    "ClassicalDetector",
    "DetectorRegistry",
    "EnsembleDetector",
    "StatisticalWatermarkDetector",
    "StylometricDetector",
]
