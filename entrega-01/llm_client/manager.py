"""`AsyncLLMManager`: factory que instancia el proveedor configurado."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import SecretStr

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .exceptions import LLMConfigError
from .openai_client import OpenAIClient
from .schemas import ModelConfig, Provider, Settings

ClientFactory = Callable[[SecretStr, ModelConfig], BaseLLMClient]

_FACTORIES: dict[Provider, ClientFactory] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
}

_ENV_VAR_FOR_KEY: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
}


class AsyncLLMManager:
    """Crea clientes según `Settings.llm_provider` (variable `LLM_PROVIDER` del `.env`).

    Uso típico::

        settings = Settings.from_env()
        async with AsyncLLMManager(settings) as manager:
            client = manager.get_client()
            response = await client.generate(messages)
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._clients: dict[Provider, BaseLLMClient] = {}

    @property
    def provider(self) -> Provider:
        """Proveedor por defecto (el del `.env`)."""
        return self.settings.llm_provider

    def create_client(
        self,
        provider: Provider | str | None = None,
        config: ModelConfig | None = None,
    ) -> BaseLLMClient:
        """Instancia un cliente nuevo. Falla con `LLMConfigError` si falta la API key."""
        resolved = self._resolve_provider(provider)
        api_key = self.settings.api_key_for(resolved)
        if api_key is None or not api_key.get_secret_value().strip():
            raise LLMConfigError(
                f"falta la variable {_ENV_VAR_FOR_KEY[resolved]} para usar el proveedor "
                f"'{resolved.value}'. Cargala en el .env o en el entorno.",
                provider=resolved,
            )
        model_config = config or ModelConfig(model=self.settings.model_for(resolved))
        return _FACTORIES[resolved](api_key, model_config)

    def get_client(self, provider: Provider | str | None = None) -> BaseLLMClient:
        """Devuelve un cliente cacheado por proveedor (lo crea la primera vez)."""
        resolved = self._resolve_provider(provider)
        if resolved not in self._clients:
            self._clients[resolved] = self.create_client(resolved)
        return self._clients[resolved]

    async def aclose(self) -> None:
        """Cierra todos los clientes creados por `get_client`."""
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    async def __aenter__(self) -> AsyncLLMManager:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _resolve_provider(self, provider: Provider | str | None) -> Provider:
        if provider is None:
            return self.settings.llm_provider
        try:
            return Provider(provider)
        except ValueError as error:
            valid = ", ".join(p.value for p in Provider)
            raise LLMConfigError(f"proveedor desconocido '{provider}'. Opciones: {valid}") from error
