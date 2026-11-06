"""El factory crea el cliente correcto y falla claro si falta la key."""

import pytest

from llm_client.anthropic_client import AnthropicClient
from llm_client.exceptions import LLMConfigError
from llm_client.manager import AsyncLLMManager
from llm_client.openai_client import OpenAIClient
from llm_client.schemas import ModelConfig, Provider, Settings


def _settings(**env: str) -> Settings:
    return Settings.from_env(env)


def test_crea_cliente_openai_por_defecto() -> None:
    manager = AsyncLLMManager(_settings(LLM_PROVIDER="openai", OPENAI_API_KEY="sk-x"))
    client = manager.create_client()
    assert isinstance(client, OpenAIClient)
    assert client.config.model == "gpt-6.1-sol"


def test_crea_cliente_anthropic_segun_env() -> None:
    manager = AsyncLLMManager(_settings(LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="sk-ant"))
    client = manager.create_client()
    assert isinstance(client, AnthropicClient)
    assert client.config.model == "claude-opus-5-5"


def test_provider_explicito_pisa_el_env() -> None:
    manager = AsyncLLMManager(_settings(LLM_PROVIDER="openai", OPENAI_API_KEY="a", ANTHROPIC_API_KEY="b"))
    assert isinstance(manager.create_client("anthropic"), AnthropicClient)
    assert isinstance(manager.create_client(Provider.OPENAI), OpenAIClient)


def test_config_personalizada() -> None:
    manager = AsyncLLMManager(_settings(OPENAI_API_KEY="a"))
    client = manager.create_client(config=ModelConfig(model="otro", max_tokens=5))
    assert client.config.model == "otro"


def test_falta_key_error_claro() -> None:
    manager = AsyncLLMManager(_settings(LLM_PROVIDER="anthropic"))
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        manager.create_client()


def test_proveedor_desconocido_error_claro() -> None:
    manager = AsyncLLMManager(_settings(OPENAI_API_KEY="a"))
    with pytest.raises(LLMConfigError, match="desconocido"):
        manager.create_client("gemini")


async def test_get_client_cachea_y_cierra() -> None:
    manager = AsyncLLMManager(_settings(OPENAI_API_KEY="a"))
    async with manager:
        first = manager.get_client()
        assert manager.get_client() is first
    assert manager._clients == {}
