"""Validación Pydantic de mensajes, configuración, respuesta y settings."""

import pytest
from pydantic import ValidationError

from llm_client.schemas import (
    ChatMessage,
    LLMErrorInfo,
    ModelConfig,
    ModelResponse,
    Provider,
    Role,
    Settings,
)


class TestModelConfig:
    def test_config_valida(self) -> None:
        cfg = ModelConfig(model="m", temperature=0.7, max_tokens=100)
        assert cfg.temperature == 0.7
        assert cfg.max_retries == 3  # default

    @pytest.mark.parametrize("temperature", [-0.1, 2.1, 5])
    def test_temperatura_fuera_de_rango_falla(self, temperature: float) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(model="m", temperature=temperature)

    @pytest.mark.parametrize("max_tokens", [0, -5])
    def test_max_tokens_debe_ser_positivo(self, max_tokens: int) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(model="m", max_tokens=max_tokens)

    def test_top_p_fuera_de_rango_falla(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(model="m", top_p=1.5)

    def test_temperature_y_top_p_juntos_falla(self) -> None:
        with pytest.raises(ValidationError, match="temperature o top_p"):
            ModelConfig(model="m", temperature=0.5, top_p=0.9)

    def test_modelo_vacio_falla(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(model="")


class TestChatMessage:
    def test_mensaje_valido(self) -> None:
        msg = ChatMessage(role="user", content="hola")
        assert msg.role is Role.USER

    @pytest.mark.parametrize("content", ["", "   "])
    def test_contenido_vacio_falla(self, content: str) -> None:
        with pytest.raises(ValidationError):
            ChatMessage(role=Role.USER, content=content)

    def test_rol_invalido_falla(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessage(role="tool", content="x")


class TestModelResponse:
    def test_success_derivado_del_error(self) -> None:
        ok = ModelResponse(text="hola", provider=Provider.OPENAI, model="m")
        assert ok.success is True
        fallo = ModelResponse(
            provider=Provider.OPENAI,
            model="m",
            error=LLMErrorInfo(type="LLMAuthError", message="key inválida"),
        )
        assert fallo.success is False
        assert fallo.model_dump()["success"] is False


class TestSettings:
    def test_lee_del_entorno(self) -> None:
        settings = Settings.from_env(
            {
                "LLM_PROVIDER": "Anthropic",
                "ANTHROPIC_API_KEY": "sk-ant-123",
                "ANTHROPIC_MODEL": "claude-x",
            }
        )
        assert settings.llm_provider is Provider.ANTHROPIC
        assert settings.anthropic_model == "claude-x"
        assert settings.openai_api_key is None
        assert settings.api_key_for(Provider.ANTHROPIC) is not None
        assert settings.api_key_for(Provider.ANTHROPIC).get_secret_value() == "sk-ant-123"

    def test_defaults_sin_entorno(self) -> None:
        settings = Settings.from_env({})
        assert settings.llm_provider is Provider.OPENAI
        assert settings.model_for(Provider.OPENAI) == "gpt-6.1-sol"
        assert settings.model_for(Provider.ANTHROPIC) == "claude-opus-5-5"

    def test_variables_vacias_se_ignoran(self) -> None:
        settings = Settings.from_env({"OPENAI_API_KEY": "   ", "OPENAI_MODEL": ""})
        assert settings.openai_api_key is None
        assert settings.openai_model == "gpt-6.1-sol"

    def test_proveedor_invalido_falla(self) -> None:
        with pytest.raises(ValidationError):
            Settings.from_env({"LLM_PROVIDER": "gemini"})

    def test_la_key_no_se_filtra_en_repr(self) -> None:
        settings = Settings.from_env({"OPENAI_API_KEY": "sk-super-secreta"})
        assert "sk-super-secreta" not in repr(settings)
        assert "sk-super-secreta" not in str(settings.model_dump())
