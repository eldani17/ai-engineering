"""Modelos Pydantic: mensajes, configuración del modelo, respuesta y settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import Enum

from pydantic import BaseModel, Field, SecretStr, computed_field, field_validator, model_validator

# Defaults verificados en la documentación oficial de cada proveedor (nov. 2026):
# - OpenAI: https://developers.openai.com/api/docs/models  -> gpt-6.1-sol (balance costo/calidad)
# - Anthropic: skill oficial claude-api                    -> claude-opus-5-5 (Opus vigente)
DEFAULT_OPENAI_MODEL = "gpt-6.1-sol"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"


class Role(str, Enum):
    """Roles posibles de un mensaje de chat."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Provider(str, Enum):
    """Proveedores soportados."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatMessage(BaseModel):
    """Un mensaje de la conversación. El contenido no puede estar vacío."""

    role: Role
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _content_no_vacio(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("el contenido del mensaje no puede estar vacío")
        return value


class ModelConfig(BaseModel):
    """Parámetros de generación y de resiliencia para una llamada al modelo.

    `temperature` y `top_p` son opcionales: si quedan en `None` no se envían,
    porque los modelos de razonamiento actuales (p. ej. claude-opus-5-5)
    rechazan parámetros de muestreo.
    """

    model: str = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    timeout: float = Field(default=60.0, gt=0, description="Timeout por request, en segundos")
    max_retries: int = Field(default=3, ge=0, description="Reintentos ante errores transitorios")

    @model_validator(mode="after")
    def _temperature_o_top_p(self) -> ModelConfig:
        if self.temperature is not None and self.top_p is not None:
            raise ValueError("usá temperature o top_p, pero no los dos a la vez")
        return self


class LLMErrorInfo(BaseModel):
    """Error controlado, serializable, que viaja dentro de `ModelResponse`."""

    type: str = Field(description="Nombre de la excepción propia, ej. LLMRateLimitError")
    message: str
    provider: Provider | None = None
    status_code: int | None = None
    retryable: bool = False
    attempts: int = Field(default=1, ge=1, description="Intentos realizados antes de rendirse")


class ModelResponse(BaseModel):
    """Respuesta normalizada de cualquier proveedor."""

    text: str = ""
    provider: Provider
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = Field(default=0.0, ge=0)
    finish_reason: str | None = None
    error: LLMErrorInfo | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success(self) -> bool:
        """`True` si la llamada terminó sin error."""
        return self.error is None


class Settings(BaseModel):
    """Configuración leída del entorno (después de `load_dotenv()`).

    Las API keys son `SecretStr`: no aparecen en `repr`, logs ni dumps.
    """

    llm_provider: Provider = Provider.OPENAI
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = DEFAULT_OPENAI_MODEL
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Construye los settings desde variables de entorno (o un mapping de prueba)."""
        source = os.environ if env is None else env

        def _get(name: str) -> str | None:
            value = source.get(name)
            return value.strip() if value and value.strip() else None

        raw: dict[str, object] = {}
        if (provider := _get("LLM_PROVIDER")) is not None:
            raw["llm_provider"] = provider.lower()
        if (openai_key := _get("OPENAI_API_KEY")) is not None:
            raw["openai_api_key"] = openai_key
        if (anthropic_key := _get("ANTHROPIC_API_KEY")) is not None:
            raw["anthropic_api_key"] = anthropic_key
        if (openai_model := _get("OPENAI_MODEL")) is not None:
            raw["openai_model"] = openai_model
        if (anthropic_model := _get("ANTHROPIC_MODEL")) is not None:
            raw["anthropic_model"] = anthropic_model
        return cls.model_validate(raw)

    def api_key_for(self, provider: Provider) -> SecretStr | None:
        """Devuelve la key del proveedor pedido (o `None` si no está cargada)."""
        return self.openai_api_key if provider is Provider.OPENAI else self.anthropic_api_key

    def model_for(self, provider: Provider) -> str:
        """Devuelve el nombre de modelo configurado para el proveedor."""
        return self.openai_model if provider is Provider.OPENAI else self.anthropic_model
