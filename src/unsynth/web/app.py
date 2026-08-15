"""Tiny FastAPI UI for local detect + clean. Not exposed by default."""

from __future__ import annotations

from typing import Any

from unsynth.config import Settings
from unsynth.pipeline.orchestrator import UnSynthPipeline
from unsynth.types import PipelineMode


def create_app(settings: Settings | None = None) -> Any:
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install unsynth[web] to use the local server") from exc

    cfg = settings or Settings()
    pipeline = UnSynthPipeline(cfg)
    app = FastAPI(title="UnSynth", version="0.1.0")

    class TextIn(BaseModel):
        text: str = Field(..., min_length=1)
        mode: PipelineMode = PipelineMode.CLEAN

    @app.get("/health")  # type: ignore[untyped-decorator]
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/run")  # type: ignore[untyped-decorator]
    def run(body: TextIn) -> dict[str, Any]:
        result = pipeline.run(body.text, mode=body.mode)
        return {"result": result.as_dict(), "text": result.output}

    return app


def run_server(settings: Settings, *, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
