"""Fixtures compartidas: sin red y sin esperas reales en los reintentos."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from llm_client import base
from llm_client.schemas import ModelConfig


@pytest.fixture(autouse=True)
def sin_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Reemplaza `asyncio.sleep` del módulo base para que los backoffs no demoren."""
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(base.asyncio, "sleep", fake_sleep)
    return delays


@pytest.fixture
def api_key() -> SecretStr:
    return SecretStr("sk-test-dummy")


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(model="modelo-test", max_tokens=64, max_retries=2, timeout=5)


class FakeAsyncStream:
    """Iterador asíncrono que entrega chunks y, opcionalmente, falla en el medio."""

    def __init__(self, items: list[Any], fail_with: Exception | None = None, fail_at: int | None = None):
        self._items = items
        self._fail_with = fail_with
        self._fail_at = fail_at if fail_at is not None else len(items)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        for index, item in enumerate(self._items):
            if self._fail_with is not None and index == self._fail_at:
                raise self._fail_with
            yield item
        if self._fail_with is not None and self._fail_at >= len(self._items):
            raise self._fail_with


def async_mock(**kwargs: Any) -> AsyncMock:
    return AsyncMock(**kwargs)
