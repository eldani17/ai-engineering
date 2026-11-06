"""Cliente Anthropic con el SDK mockeado: system aparte, bloques de texto, streaming."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest
from pydantic import SecretStr

from llm_client.anthropic_client import AnthropicClient
from llm_client.exceptions import (
    LLMAPIError,
    LLMAuthError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from llm_client.schemas import ChatMessage, ModelConfig, Provider, Role

from .conftest import FakeAsyncStream
from .sdk_errors import connection_error, status_error, timeout_error

MENSAJES = [
    ChatMessage(role=Role.SYSTEM, content="Sos breve."),
    ChatMessage(role=Role.USER, content="¿Qué es la entropía?"),
]


def _message(text: str = "Medida del desorden.") -> SimpleNamespace:
    return SimpleNamespace(
        model="claude-test",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(input_tokens=20, output_tokens=9),
        stop_reason="end_turn",
    )


def _client(sdk: MagicMock, config: ModelConfig) -> AnthropicClient:
    return AnthropicClient(SecretStr("sk-ant-dummy"), config, client=sdk)


def _stream_cm(texts: list[str], fail_with: Exception | None = None, fail_at: int | None = None):
    """Simula `client.messages.stream(...)` como context manager asíncrono con `text_stream`."""

    @asynccontextmanager
    async def cm(**_: object):
        yield SimpleNamespace(text_stream=FakeAsyncStream(texts, fail_with=fail_with, fail_at=fail_at))

    return cm


@pytest.fixture
def sdk() -> MagicMock:
    mock = MagicMock()
    mock.messages.create = AsyncMock()
    mock.close = AsyncMock()
    return mock


async def test_generate_ok_y_system_aparte(sdk: MagicMock, config: ModelConfig) -> None:
    sdk.messages.create.return_value = _message()
    response = await _client(sdk, config).generate(MENSAJES)

    assert response.success
    assert response.text == "Medida del desorden."  # ignora bloques que no son text
    assert response.provider is Provider.ANTHROPIC
    assert (response.input_tokens, response.output_tokens) == (20, 9)
    assert response.finish_reason == "end_turn"

    kwargs = sdk.messages.create.call_args.kwargs
    assert kwargs["system"] == "Sos breve."
    assert kwargs["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]
    assert kwargs["max_tokens"] == 64
    assert "temperature" not in kwargs


async def test_sin_system_no_se_envia_el_parametro(sdk: MagicMock, config: ModelConfig) -> None:
    sdk.messages.create.return_value = _message()
    await _client(sdk, config).generate([ChatMessage(role=Role.USER, content="hola")])
    assert "system" not in sdk.messages.create.call_args.kwargs


async def test_auth_error_controlado(sdk: MagicMock, config: ModelConfig, sin_sleep: list[float]) -> None:
    sdk.messages.create.side_effect = status_error(anthropic.AuthenticationError, 401)
    response = await _client(sdk, config).generate(MENSAJES)

    assert not response.success
    assert response.error is not None
    assert response.error.type == "LLMAuthError"
    assert sdk.messages.create.await_count == 1
    assert sin_sleep == []


async def test_overloaded_529_se_reintenta(sdk: MagicMock, config: ModelConfig) -> None:
    sdk.messages.create.side_effect = [
        status_error(anthropic.InternalServerError, 529),
        _message("ok"),
    ]
    response = await _client(sdk, config).generate(MENSAJES)
    assert response.success and response.text == "ok"
    assert sdk.messages.create.await_count == 2


@pytest.mark.parametrize(
    ("sdk_error", "expected", "retryable"),
    [
        (lambda: status_error(anthropic.RateLimitError, 429), LLMRateLimitError, True),
        (lambda: timeout_error(anthropic.APITimeoutError), LLMTimeoutError, True),
        (lambda: connection_error(anthropic.APIConnectionError), LLMConnectionError, True),
        (lambda: status_error(anthropic.BadRequestError, 400), LLMAPIError, False),
        (lambda: status_error(anthropic.NotFoundError, 404), LLMAPIError, False),
    ],
)
def test_traduccion_de_errores(sdk: MagicMock, config: ModelConfig, sdk_error, expected, retryable) -> None:
    translated = _client(sdk, config)._translate(sdk_error())
    assert isinstance(translated, expected)
    assert translated.retryable is retryable
    assert translated.provider is Provider.ANTHROPIC


async def test_stream_produce_fragmentos(sdk: MagicMock, config: ModelConfig) -> None:
    sdk.messages.stream = _stream_cm(["La ", "", "entropía", " es..."])
    chunks = [c async for c in _client(sdk, config).stream(MENSAJES)]
    assert chunks == ["La ", "entropía", " es..."]


async def test_stream_falla_controlado(sdk: MagicMock, config: ModelConfig) -> None:
    sdk.messages.stream = _stream_cm(["a"], fail_with=status_error(anthropic.RateLimitError, 429), fail_at=1)
    received: list[str] = []
    with pytest.raises(LLMRateLimitError):
        async for chunk in _client(sdk, config).stream(MENSAJES):
            received.append(chunk)
    assert received == ["a"]
