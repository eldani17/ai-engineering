"""Cliente asíncrono unificado para LLMs (OpenAI y Anthropic).

Expone una interfaz común (`BaseLLMClient`) y un `AsyncLLMManager` que
instancia el proveedor configurado en `LLM_PROVIDER`.
"""

from .base import BaseLLMClient
from .exceptions import (
    LLMAPIError,
    LLMAuthError,
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .manager import AsyncLLMManager
from .schemas import (
    ChatMessage,
    LLMErrorInfo,
    ModelConfig,
    ModelResponse,
    Provider,
    Role,
    Settings,
)

__all__ = [
    "AsyncLLMManager",
    "BaseLLMClient",
    "ChatMessage",
    "LLMAPIError",
    "LLMAuthError",
    "LLMConfigError",
    "LLMConnectionError",
    "LLMError",
    "LLMErrorInfo",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "ModelConfig",
    "ModelResponse",
    "Provider",
    "Role",
    "Settings",
]
