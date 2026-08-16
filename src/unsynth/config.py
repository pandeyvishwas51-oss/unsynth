"""YAML / TOML / environment configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from unsynth.exceptions import ConfigError
from unsynth.types import BackendKind

CONFIG_FILENAMES = (
    "unsynth.yaml",
    "unsynth.yml",
    "unsynth.toml",
    ".unsynth.yaml",
)


class DetectSettings(BaseModel):
    detectors: list[str] = Field(
        default_factory=lambda: ["classical", "stylometric", "statistical"]
    )
    ai_likely: float = 0.62
    watermark_likely: float = 0.70
    uncertain: float = 0.45
    window_tokens: int = 256
    window_stride: int = 128

    @field_validator("ai_likely", "watermark_likely", "uncertain")
    @classmethod
    def _unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("thresholds must be in [0, 1]")
        return value

    @field_validator("window_tokens", "window_stride")
    @classmethod
    def _positive_window(cls, value: int) -> int:
        if value < 1:
            raise ValueError("window sizes must be >= 1")
        return value


class RewriteSettings(BaseModel):
    max_passes: int = 4
    min_similarity: float = 0.82
    target_ai_score: float = 0.40
    target_watermark_score: float = 0.48
    initial_strength: float = 0.42
    strength_step: float = 0.16
    max_strength: float = 0.92
    strategies: list[str] = Field(
        default_factory=lambda: ["lexical", "structural", "style", "paraphrase"]
    )
    protect_markdown: bool = True
    protect_code: bool = True
    protect_tables: bool = True
    protect_urls: bool = True
    allow_backtranslate: bool = False

    @field_validator("max_passes")
    @classmethod
    def _passes(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_passes must be >= 1")
        return value

    @field_validator(
        "min_similarity",
        "target_ai_score",
        "target_watermark_score",
        "initial_strength",
        "strength_step",
        "max_strength",
    )
    @classmethod
    def _unit(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("must be in [0, 1]")
        return value


class BackendSettings(BaseModel):
    kind: BackendKind = BackendKind.NONE
    model: str = "llama3.1:8b"
    host: str = "http://127.0.0.1:11434"
    api_key: str = ""
    temperature: float = 0.85
    max_tokens: int = 2048
    timeout_s: float = 120.0


class QualitySettings(BaseModel):
    embeddings: Literal["none", "auto", "sentence-transformers"] = "auto"
    embedding_model: str = "all-MiniLM-L6-v2"
    min_readability: float = 20.0
    max_length_ratio: float = 1.45
    min_length_ratio: float = 0.62


class MetadataSettings(BaseModel):
    strip_c2pa: bool = True
    strip_xmp: bool = True
    strip_exif: bool = True
    strip_html_provenance: bool = True


class RuntimeSettings(BaseModel):
    log_level: str = "INFO"
    seed: int = 13
    workers: int = 1
    plugin_dirs: list[str] = Field(default_factory=list)

    @field_validator("workers")
    @classmethod
    def _workers(cls, value: int) -> int:
        if value < 1:
            raise ValueError("workers must be >= 1")
        return value


class Settings(BaseSettings):
    """Root settings object.

    Load order (later wins):
    1. Built-in defaults
    2. First found config file (cwd, then platform config dir)
    3. Explicit ``--config`` path
    4. Environment variables prefixed with ``UNSYNTH_``
    """

    model_config = SettingsConfigDict(
        env_prefix="UNSYNTH_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    detect: DetectSettings = Field(default_factory=DetectSettings)
    rewrite: RewriteSettings = Field(default_factory=RewriteSettings)
    backend: BackendSettings = Field(default_factory=BackendSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    metadata: MetadataSettings = Field(default_factory=MetadataSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    def label_for(self, score: float, *, family_watermark: bool = False) -> str:
        if family_watermark:
            if score >= self.detect.watermark_likely:
                return "watermarked"
            if score >= self.detect.uncertain:
                return "uncertain"
            return "human"
        if score >= self.detect.ai_likely:
            return "ai"
        if score >= self.detect.uncertain:
            return "uncertain"
        return "human"


def _platform_config_dir() -> Path:
    try:
        from platformdirs import user_config_dir

        return Path(user_config_dir("unsynth", "unsynth"))
    except Exception:  # pragma: no cover - extremely defensive
        return Path.home() / ".config" / "unsynth"


def discover_config_path(explicit: str | Path | None = None) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        return path
    env = os.environ.get("UNSYNTH_CONFIG")
    if env:
        path = Path(env).expanduser()
        if not path.is_file():
            raise ConfigError(f"UNSYNTH_CONFIG does not exist: {path}")
        return path
    search = [Path.cwd(), _platform_config_dir()]
    for directory in search:
        for name in CONFIG_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _read_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    if path.suffix.lower() == ".toml":
        import tomllib

        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML {path}: {exc}") from exc
    else:
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML {path}: {exc}") from exc
        data = loaded if isinstance(loaded, dict) else {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return data


def load_settings(explicit: str | Path | None = None) -> Settings:
    """Load settings from disk + environment."""

    path = discover_config_path(explicit)
    raw: dict[str, Any] = _read_file(path) if path is not None else {}
    try:
        return Settings.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(f"invalid configuration: {exc}") from exc
