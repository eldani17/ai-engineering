"""Cliente asíncrono para OpenAI (Chat Completions)."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import openai
from openai import AsyncOpenAI
from pydantic import SecretStr

from .base import BaseLLMClient, _elapsed_ms
from .exceptions import (
    LLMAPIError,
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider


class OpenAIClient(BaseLLMClient):
    """Implementación sobre `AsyncOpenAI` y `chat.completions`."""

    provider = Provider.OPENAI

    def __init__(
        self,
        api_key: SecretStr,
        config: ModelConfig,
        client: AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(config)
        # `max_retries=0`: los reintentos los maneja BaseLLMClient de forma explícita.
        self._client = client or AsyncOpenAI(
            api_key=api_key.get_secret_value(),
            timeout=config.timeout,
            max_retries=0,
        )

    async def aclose(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------ helpers

    def _request_kwargs(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        """OpenAI acepta el mensaje `system` dentro de la lista de mensajes."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        return kwargs

    # ------------------------------------------------------------- implementación

    async def _generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        started = time.perf_counter()
        try:
            completion = await self._client.chat.completions.create(**self._request_kwargs(messages))
        except openai.APIError as error:
            raise self._translate(error) from error

        choice = completion.choices[0] if completion.choices else None
        usage = completion.usage
        return ModelResponse(
            text=(choice.message.content or "") if choice else "",
            provider=self.provider,
            model=completion.model or self.config.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            latency_ms=_elapsed_ms(started),
            finish_reason=choice.finish_reason if choice else None,
        )

    async def _stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(
                **self._request_kwargs(messages),
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                # El último chunk (usage) trae `choices` vacío; el delta puede venir None.
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except openai.APIError as error:
            raise self._translate(error) from error

    # ---------------------------------------------------------------- errores

    def _translate(self, error: openai.APIError) -> LLMError:
        """Traduce excepciones del SDK respetando su herencia (específico antes que genérico)."""
        provider = self.provider
        # AuthenticationError y RateLimitError heredan de APIStatusError.
        if isinstance(error, openai.AuthenticationError):
            return LLMAuthError("API key de OpenAI inválida o ausente", provider=provider, status_code=401)
        if isinstance(error, openai.PermissionDeniedError):
            return LLMAuthError("la API key de OpenAI no tiene permisos", provider=provider, status_code=403)
        if isinstance(error, openai.RateLimitError):
            return LLMRateLimitError(
                "límite de uso de OpenAI superado",
                provider=provider,
                status_code=429,
                retry_after=_retry_after(error),
            )
        if isinstance(error, openai.APIStatusError):
            return LLMAPIError(
                f"OpenAI respondió con error: {error.message}",
                provider=provider,
                status_code=error.status_code,
            )
        # APITimeoutError hereda de APIConnectionError: va primero.
        if isinstance(error, openai.APITimeoutError):
            return LLMTimeoutError(
                f"OpenAI no respondió en {self.config.timeout:g}s", provider=provider
            )
        if isinstance(error, openai.APIConnectionError):
            return LLMConnectionError("no se pudo conectar con OpenAI", provider=provider)
        return LLMAPIError(f"error inesperado de OpenAI: {error}", provider=provider)


def _retry_after(error: openai.APIStatusError) -> float | None:
    """Lee el header `retry-after` (segundos) si el SDK lo expone."""
    try:
        value = error.response.headers.get("retry-after")
    except AttributeError:
        return None
    try:
        return float(value) if value else None
    except (TypeError, ValueError):
        return None
