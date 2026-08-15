"""Typed errors raised by UnSynth."""

from __future__ import annotations


class UnSynthError(Exception):
    """Base class for all UnSynth errors."""


class ConfigError(UnSynthError):
    """Invalid configuration or missing required setting."""


class DetectorError(UnSynthError):
    """A detector failed in a non-recoverable way."""


class RewriterError(UnSynthError):
    """A rewrite strategy failed."""


class BackendError(UnSynthError):
    """Local or remote language-model backend is unavailable or failed."""


class QualityGateError(UnSynthError):
    """A rewritten candidate failed a quality gate."""


class MetadataError(UnSynthError):
    """Provenance / metadata stripping failed."""


class PluginError(UnSynthError):
    """A detector or rewriter plugin could not be loaded."""
