"""Cliente asíncrono para Anthropic (Messages API)."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
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


class AnthropicClient(BaseLLMClient):
    """Implementación sobre `AsyncAnthropic` y `messages`."""

    provider = Provider.ANTHROPIC

    def __init__(
        self,
        api_key: SecretStr,
        config: ModelConfig,
        client: AsyncAnthropic | None = None,
    ) -> None:
        super().__init__(config)
        # `max_retries=0`: los reintentos los maneja BaseLLMClient de forma explícita.
        self._client = client or AsyncAnthropic(
            api_key=api_key.get_secret_value(),
            timeout=config.timeout,
            max_retries=0,
        )

    async def aclose(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------ helpers

    def _request_kwargs(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        """Anthropic recibe `system` como parámetro aparte y exige `max_tokens`."""
        system, conversation = self.split_system(messages)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": m.role.value, "content": m.content} for m in conversation],
        }
        if system is not None:
            kwargs["system"] = system
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        return kwargs

    # ------------------------------------------------------------- implementación

    async def _generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        started = time.perf_counter()
        try:
            message = await self._client.messages.create(**self._request_kwargs(messages))
        except anthropic.APIError as error:
            raise self._translate(error) from error

        # El contenido es una lista de bloques; solo nos interesan los de texto.
        text = "".join(block.text for block in message.content if block.type == "text")
        usage = message.usage
        return ModelResponse(
            text=text,
            provider=self.provider,
            model=message.model or self.config.model,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            latency_ms=_elapsed_ms(started),
            finish_reason=message.stop_reason,
        )

    async def _stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            async with self._client.messages.stream(**self._request_kwargs(messages)) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text
        except anthropic.APIError as error:
            raise self._translate(error) from error

    # ---------------------------------------------------------------- errores

    def _translate(self, error: anthropic.APIError) -> LLMError:
        """Traduce excepciones del SDK respetando su herencia (específico antes que genérico)."""
        provider = self.provider
        if isinstance(error, anthropic.AuthenticationError):
            return LLMAuthError("API key de Anthropic inválida o ausente", provider=provider, status_code=401)
        if isinstance(error, anthropic.PermissionDeniedError):
            return LLMAuthError("la API key de Anthropic no tiene permisos", provider=provider, status_code=403)
        if isinstance(error, anthropic.RateLimitError):
            return LLMRateLimitError(
                "límite de uso de Anthropic superado",
                provider=provider,
                status_code=429,
                retry_after=_retry_after(error),
            )
        if isinstance(error, anthropic.APIStatusError):
            # 529 (overloaded) también es >= 500, así que queda marcado como reintentable.
            return LLMAPIError(
                f"Anthropic respondió con error: {error.message}",
                provider=provider,
                status_code=error.status_code,
            )
        if isinstance(error, anthropic.APITimeoutError):
            return LLMTimeoutError(
                f"Anthropic no respondió en {self.config.timeout:g}s", provider=provider
            )
        if isinstance(error, anthropic.APIConnectionError):
            return LLMConnectionError("no se pudo conectar con Anthropic", provider=provider)
        return LLMAPIError(f"error inesperado de Anthropic: {error}", provider=provider)


def _retry_after(error: anthropic.APIStatusError) -> float | None:
    """Lee el header `retry-after` (segundos) si el SDK lo expone."""
    try:
        value = error.response.headers.get("retry-after")
    except AttributeError:
        return None
    try:
        return float(value) if value else None
    except (TypeError, ValueError):
        return None
