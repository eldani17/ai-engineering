"""Helpers para fabricar excepciones reales de los SDKs sin tocar la red."""

from __future__ import annotations

from typing import Any

import httpx2


def _response(status: int, headers: dict[str, str] | None = None) -> httpx2.Response:
    request = httpx2.Request("POST", "https://api.test/v1/x")
    return httpx2.Response(status, request=request, headers=headers or {})


def status_error(cls: type[Any], status: int, headers: dict[str, str] | None = None) -> Any:
    """Instancia una subclase de `APIStatusError` (de openai o anthropic)."""
    return cls("boom", response=_response(status, headers), body=None)


def timeout_error(cls: type[Any]) -> Any:
    return cls(request=httpx2.Request("POST", "https://api.test/v1/x"))


def connection_error(cls: type[Any]) -> Any:
    return cls(message="Connection error.", request=httpx2.Request("POST", "https://api.test/v1/x"))
