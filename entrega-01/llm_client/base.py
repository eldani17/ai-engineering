"""Interfaz común de los clientes y lógica compartida de reintentos."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import TypeVar

from .exceptions import LLMError
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider, Role

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Backoff exponencial: delay = min(BASE * 2**intento, MAX) + jitter.
RETRY_BASE_DELAY = 0.5
RETRY_MAX_DELAY = 8.0


class BaseLLMClient(ABC):
    """Contrato que cumplen todos los proveedores.

    Las subclases implementan `_generate` y `_stream` (un solo intento, lanzan
    `LLMError`). Esta clase agrega los reintentos y el manejo controlado:

    - `generate()` nunca lanza `LLMError`: devuelve un `ModelResponse` con `error`.
    - `stream()` reintenta solo antes del primer fragmento; si falla, lanza la
      excepción propia para que quien consume el generador la capture.
    """

    provider: Provider

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ API pública

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Pide una respuesta completa. Devuelve siempre un `ModelResponse`."""
        self._validate_messages(messages)
        started = time.perf_counter()
        try:
            return await self._run_with_retries(lambda: self._generate(messages))
        except LLMError as error:
            logger.info("Error controlado en generate(): %s", error)
            return ModelResponse(
                provider=self.provider,
                model=self.config.model,
                latency_ms=_elapsed_ms(started),
                error=error.to_info(),
            )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Generador asíncrono de fragmentos de texto.

        Reintenta mientras no haya salido ningún fragmento; una vez que empezó
        a emitir, un fallo se propaga como `LLMError` (no se puede "rebobinar").
        """
        self._validate_messages(messages)
        attempt = 0
        while True:
            emitted = False
            try:
                async for chunk in self._stream(messages):
                    emitted = True
                    yield chunk
                return
            except LLMError as error:
                error.attempts = attempt + 1
                if emitted or not error.retryable or attempt >= self.config.max_retries:
                    logger.info("Error controlado en stream(): %s", error)
                    raise
                await self._backoff(attempt, error)
                attempt += 1

    async def aclose(self) -> None:
        """Cierra recursos del SDK (conexiones HTTP). Las subclases lo sobreescriben."""

    async def __aenter__(self) -> BaseLLMClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------- para subclases

    @abstractmethod
    async def _generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Un único intento de generación. Debe lanzar `LLMError` ante fallos."""

    @abstractmethod
    def _stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Un único intento de streaming. Debe lanzar `LLMError` ante fallos."""

    @staticmethod
    def split_system(messages: Sequence[ChatMessage]) -> tuple[str | None, list[ChatMessage]]:
        """Separa los mensajes `system` (concatenados) del resto de la conversación."""
        system_parts = [m.content for m in messages if m.role is Role.SYSTEM]
        rest = [m for m in messages if m.role is not Role.SYSTEM]
        return ("\n\n".join(system_parts) if system_parts else None), rest

    # ------------------------------------------------------------------ internos

    async def _run_with_retries(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Ejecuta `operation` con backoff exponencial para errores reintentables."""
        attempt = 0
        while True:
            try:
                return await operation()
            except LLMError as error:
                error.attempts = attempt + 1
                if not error.retryable or attempt >= self.config.max_retries:
                    raise
                await self._backoff(attempt, error)
                attempt += 1

    async def _backoff(self, attempt: int, error: LLMError) -> None:
        delay = min(RETRY_BASE_DELAY * (2**attempt), RETRY_MAX_DELAY)
        retry_after = getattr(error, "retry_after", None)
        if retry_after:
            delay = max(delay, float(retry_after))
        delay += random.uniform(0, 0.25)  # jitter para no sincronizar reintentos
        logger.info(
            "Reintento %d/%d en %.2fs tras %s",
            attempt + 1,
            self.config.max_retries,
            delay,
            type(error).__name__,
        )
        await asyncio.sleep(delay)

    @staticmethod
    def _validate_messages(messages: Sequence[ChatMessage]) -> None:
        if not messages:
            raise ValueError("hace falta al menos un mensaje")
        if all(m.role is Role.SYSTEM for m in messages):
            raise ValueError("hace falta al menos un mensaje de usuario o asistente")


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
