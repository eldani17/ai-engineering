"""Cliente OpenAI con el SDK mockeado: parsing, traducción de errores y streaming."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from pydantic import SecretStr

from llm_client.exceptions import (
    LLMAPIError,
    LLMAuthError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from llm_client.openai_client import OpenAIClient
from llm_client.schemas import ChatMessage, ModelConfig, Provider, Role

from .conftest import FakeAsyncStream
from .sdk_errors import connection_error, status_error, timeout_error

MENSAJES = [
    ChatMessage(role=Role.SYSTEM, content="Sos breve."),
    ChatMessage(role=Role.USER, content="¿Qué es la entropía?"),
]


def _completion(text: str = "Es una medida del desorden.") -> SimpleNamespace:
    return SimpleNamespace(
        model="gpt-test",
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7),
    )


def _chunk(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def _client(sdk: MagicMock, config: ModelConfig) -> OpenAIClient:
    return OpenAIClient(SecretStr("sk-dummy"), config, client=sdk)


@pytest.fixture
def sdk() -> MagicMock:
    mock = MagicMock()
    mock.chat.completions.create = AsyncMock()
    mock.close = AsyncMock()
    return mock


async def test_generate_ok(sdk: MagicMock, config: ModelConfig) -> None:
    sdk.chat.completions.create.return_value = _completion()
    response = await _client(sdk, config).generate(MENSAJES)

    assert response.success
    assert response.text == "Es una medida del desorden."
    assert response.provider is Provider.OPENAI
    assert (response.input_tokens, response.output_tokens) == (12, 7)
    assert response.finish_reason == "stop"
    assert response.latency_ms >= 0

    kwargs = sdk.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "modelo-test"
    assert kwargs["max_completion_tokens"] == 64
    assert "temperature" not in kwargs  # no se envía si es None
    # OpenAI recibe el system dentro de la lista
    assert kwargs["messages"][0] == {"role": "system", "content": "Sos breve."}


async def test_generate_envia_temperature_si_esta_seteada(sdk: MagicMock) -> None:
    sdk.chat.completions.create.return_value = _completion()
    cfg = ModelConfig(model="m", temperature=0.2, max_retries=0)
    await _client(sdk, cfg).generate(MENSAJES)
    assert sdk.chat.completions.create.call_args.kwargs["temperature"] == 0.2


async def test_auth_error_no_reintenta_y_devuelve_error_controlado(
    sdk: MagicMock, config: ModelConfig, sin_sleep: list[float]
) -> None:
    sdk.chat.completions.create.side_effect = status_error(openai.AuthenticationError, 401)
    response = await _client(sdk, config).generate(MENSAJES)

    assert not response.success
    assert response.error is not None
    assert response.error.type == "LLMAuthError"
    assert response.error.status_code == 401
    assert response.error.retryable is False
    assert response.error.attempts == 1
    assert sdk.chat.completions.create.await_count == 1
    assert sin_sleep == []


async def test_rate_limit_reintenta_y_luego_funciona(
    sdk: MagicMock, config: ModelConfig, sin_sleep: list[float]
) -> None:
    sdk.chat.completions.create.side_effect = [
        status_error(openai.RateLimitError, 429, {"retry-after": "1"}),
        status_error(openai.RateLimitError, 429),
        _completion("ok"),
    ]
    response = await _client(sdk, config).generate(MENSAJES)

    assert response.success and response.text == "ok"
    assert sdk.chat.completions.create.await_count == 3
    assert len(sin_sleep) == 2
    assert sin_sleep[0] >= 1.0  # respeta retry-after
    assert sin_sleep[1] >= 1.0  # backoff exponencial: 0.5 * 2**1


async def test_rate_limit_agota_reintentos(sdk: MagicMock, config: ModelConfig) -> None:
    sdk.chat.completions.create.side_effect = status_error(openai.RateLimitError, 429)
    response = await _client(sdk, config).generate(MENSAJES)

    assert not response.success
    assert response.error is not None
    assert response.error.type == "LLMRateLimitError"
    assert response.error.attempts == config.max_retries + 1
    assert sdk.chat.completions.create.await_count == config.max_retries + 1


@pytest.mark.parametrize(
    ("sdk_error", "expected", "retryable"),
    [
        (lambda: timeout_error(openai.APITimeoutError), LLMTimeoutError, True),
        (lambda: connection_error(openai.APIConnectionError), LLMConnectionError, True),
        (lambda: status_error(openai.InternalServerError, 503), LLMAPIError, True),
        (lambda: status_error(openai.BadRequestError, 400), LLMAPIError, False),
        (lambda: status_error(openai.PermissionDeniedError, 403), LLMAuthError, False),
    ],
)
def test_traduccion_de_errores(sdk: MagicMock, config: ModelConfig, sdk_error, expected, retryable) -> None:
    translated = _client(sdk, config)._translate(sdk_error())
    assert isinstance(translated, expected)
    assert translated.retryable is retryable
    assert translated.provider is Provider.OPENAI


async def test_error_encadena_la_causa(sdk: MagicMock, config: ModelConfig) -> None:
    original = timeout_error(openai.APITimeoutError)
    sdk.chat.completions.create.side_effect = original
    with pytest.raises(LLMTimeoutError) as info:
        await _client(sdk, config)._generate(MENSAJES)
    assert info.value.__cause__ is original


async def test_stream_produce_fragmentos(sdk: MagicMock, config: ModelConfig) -> None:
    sdk.chat.completions.create.return_value = FakeAsyncStream(
        [
            _chunk("La "),
            _chunk(None),  # delta sin contenido
            _chunk("entropía"),
            SimpleNamespace(choices=[]),  # chunk final de usage, sin choices
        ]
    )
    chunks = [c async for c in _client(sdk, config).stream(MENSAJES)]

    assert chunks == ["La ", "entropía"]
    kwargs = sdk.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}


async def test_stream_reintenta_antes_del_primer_fragmento(sdk: MagicMock, config: ModelConfig) -> None:
    sdk.chat.completions.create.side_effect = [
        status_error(openai.RateLimitError, 429),
        FakeAsyncStream([_chunk("ok")]),
    ]
    chunks = [c async for c in _client(sdk, config).stream(MENSAJES)]
    assert chunks == ["ok"]
    assert sdk.chat.completions.create.await_count == 2


async def test_stream_falla_controlado_a_mitad(sdk: MagicMock, config: ModelConfig) -> None:
    sdk.chat.completions.create.return_value = FakeAsyncStream(
        [_chunk("a"), _chunk("b")],
        fail_with=connection_error(openai.APIConnectionError),
        fail_at=1,
    )
    received: list[str] = []
    with pytest.raises(LLMConnectionError):
        async for chunk in _client(sdk, config).stream(MENSAJES):
            received.append(chunk)
    assert received == ["a"]
    assert sdk.chat.completions.create.await_count == 1  # no se reintenta a mitad


async def test_stream_auth_error_no_reintenta(sdk: MagicMock, config: ModelConfig) -> None:
    sdk.chat.completions.create.side_effect = status_error(openai.AuthenticationError, 401)
    with pytest.raises(LLMAuthError):
        async for _ in _client(sdk, config).stream(MENSAJES):
            pass
    assert sdk.chat.completions.create.await_count == 1


async def test_sin_mensajes_de_usuario_falla(sdk: MagicMock, config: ModelConfig) -> None:
    with pytest.raises(ValueError):
        await _client(sdk, config).generate([])
