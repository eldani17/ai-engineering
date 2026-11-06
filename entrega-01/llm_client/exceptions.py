"""Jerarquía propia de errores. Los clientes traducen las excepciones de cada SDK a estas."""

from __future__ import annotations

from .schemas import LLMErrorInfo, Provider


class LLMError(Exception):
    """Base de todos los errores controlados del cliente."""

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: Provider | None = None,
        status_code: int | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        if retryable is not None:
            self.retryable = retryable
        self.attempts = 1

    def to_info(self) -> LLMErrorInfo:
        """Convierte la excepción en un `LLMErrorInfo` serializable."""
        return LLMErrorInfo(
            type=type(self).__name__,
            message=self.message,
            provider=self.provider,
            status_code=self.status_code,
            retryable=self.retryable,
            attempts=self.attempts,
        )

    def __str__(self) -> str:
        prefix = f"[{self.provider.value}] " if self.provider else ""
        suffix = f" (HTTP {self.status_code})" if self.status_code else ""
        return f"{prefix}{self.message}{suffix}"


class LLMConfigError(LLMError):
    """Configuración inválida o incompleta (p. ej. falta la API key)."""


class LLMAuthError(LLMError):
    """Credenciales inválidas o sin permisos. Nunca se reintenta."""

    retryable = False


class LLMRateLimitError(LLMError):
    """HTTP 429: límite de uso superado. Se reintenta con backoff."""

    retryable = True

    def __init__(self, message: str, *, retry_after: float | None = None, **kwargs: object) -> None:
        super().__init__(message, **kwargs)  # type: ignore[arg-type]
        self.retry_after = retry_after


class LLMConnectionError(LLMError):
    """No se pudo conectar con la API (DNS, red caída, etc.). Se reintenta."""

    retryable = True


class LLMTimeoutError(LLMConnectionError):
    """La request superó el timeout. Hereda de conexión, igual que en los SDKs."""


class LLMAPIError(LLMError):
    """Cualquier otra respuesta no exitosa de la API. Se reintenta solo si es 5xx."""

    def __init__(self, message: str, *, status_code: int | None = None, **kwargs: object) -> None:
        retryable = kwargs.pop("retryable", None)
        if retryable is None:
            retryable = status_code is not None and status_code >= 500
        super().__init__(message, status_code=status_code, retryable=retryable, **kwargs)  # type: ignore[arg-type]
