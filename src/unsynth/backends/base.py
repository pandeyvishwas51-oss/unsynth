"""Dispatch to Ollama, an OpenAI-compatible server, or Hugging Face.

Nothing here is imported at CLI startup except the dispatcher. Heavy
libraries stay optional extras.
"""

from __future__ import annotations

from unsynth.config import Settings
from unsynth.exceptions import BackendError
from unsynth.logging import get_logger
from unsynth.types import BackendKind

log = get_logger("backend")


def is_available(settings: Settings) -> bool:
    if settings.backend.kind is BackendKind.NONE:
        return False
    if settings.backend.kind is BackendKind.OLLAMA:
        return _ollama_alive(settings)
    if settings.backend.kind is BackendKind.OPENAI_COMPATIBLE:
        return bool(settings.backend.host)
    if settings.backend.kind is BackendKind.TRANSFORMERS:
        try:
            import transformers  # noqa: F401

            return True
        except Exception:
            return False
    return False


def complete(
    settings: Settings,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
) -> str:
    kind = settings.backend.kind
    temp = settings.backend.temperature if temperature is None else temperature
    if kind is BackendKind.NONE:
        raise BackendError("no LLM backend configured (backend.kind = none)")
    if kind is BackendKind.OLLAMA:
        return _ollama_complete(settings, prompt, system=system, temperature=temp)
    if kind is BackendKind.OPENAI_COMPATIBLE:
        return _openai_complete(settings, prompt, system=system, temperature=temp)
    if kind is BackendKind.TRANSFORMERS:
        return _hf_complete(settings, prompt, system=system, temperature=temp)
    raise BackendError(f"unknown backend kind: {kind}")


def _ollama_alive(settings: Settings) -> bool:
    try:
        import httpx

        url = settings.backend.host.rstrip("/") + "/api/tags"
        response = httpx.get(url, timeout=2.0)
        return response.status_code < 500
    except Exception:
        return False


def _ollama_complete(
    settings: Settings, prompt: str, *, system: str | None, temperature: float
) -> str:
    import httpx

    url = settings.backend.host.rstrip("/") + "/api/generate"
    payload = {
        "model": settings.backend.model,
        "prompt": prompt if not system else f"{system}\n\n{prompt}",
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": settings.backend.max_tokens,
        },
    }
    try:
        response = httpx.post(url, json=payload, timeout=settings.backend.timeout_s)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise BackendError(f"ollama request failed: {exc}") from exc
    text = data.get("response")
    if not isinstance(text, str):
        raise BackendError("ollama returned no 'response' field")
    return text


def _openai_complete(
    settings: Settings, prompt: str, *, system: str | None, temperature: float
) -> str:
    import os

    import httpx

    api_key = settings.backend.api_key or os.environ.get("UNSYNTH_OPENAI_API_KEY", "")
    base = settings.backend.host.rstrip("/")
    if not base.endswith("/v1"):
        # Accept either a bare host or a full /v1 URL.
        if not base.endswith("/chat/completions"):
            url = base + "/v1/chat/completions"
        else:
            url = base
    else:
        url = base + "/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": settings.backend.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": settings.backend.max_tokens,
    }
    try:
        response = httpx.post(
            url, json=payload, headers=headers, timeout=settings.backend.timeout_s
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])
    except Exception as exc:
        raise BackendError(f"openai-compatible request failed: {exc}") from exc


def _hf_complete(settings: Settings, prompt: str, *, system: str | None, temperature: float) -> str:
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover
        raise BackendError(
            "transformers extra is not installed; pip install 'unsynth[transformers]'"
        ) from exc
    text = prompt if not system else f"{system}\n\n{prompt}"
    try:
        gen = pipeline("text-generation", model=settings.backend.model)
        out = gen(
            text,
            max_new_tokens=min(settings.backend.max_tokens, 512),
            temperature=max(0.1, temperature),
            do_sample=True,
            return_full_text=False,
        )
    except Exception as exc:
        raise BackendError(f"transformers generation failed: {exc}") from exc
    if not out:
        raise BackendError("transformers returned an empty result")
    return str(out[0]["generated_text"])
