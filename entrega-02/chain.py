"""Cadena LCEL de extracción de entidades técnicas.

    prompt | model.with_structured_output(ExtraccionTecnica) | verificación   (con .with_retry())

Punto de entrada: `process_text(text)`, asíncrono.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import anthropic
import openai
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from schemas import ExtraccionTecnica

logger = logging.getLogger("pipeline")

# Misma convención de variables de entorno que el cliente del Módulo 1.
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-6.1-sol"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_MAX_TOKENS = 1024
MAX_INTENTOS = 3

# Valores de finish_reason / stop_reason que indican que la respuesta quedó cortada.
_CORTES_POR_TOKENS = {"length", "max_tokens"}


# --------------------------------------------------------------------------- errores


class SalidaIncompletaError(Exception):
    """El modelo cortó la respuesta (se quedó sin tokens): el objeto no es confiable."""


class SalidaInvalidaError(Exception):
    """El modelo respondió, pero la salida no cumple el esquema Pydantic."""


class ConfiguracionError(Exception):
    """Falta la API key o el proveedor configurado no existe."""


_ERRORES_DE_SALIDA: tuple[type[BaseException], ...] = (
    SalidaIncompletaError,
    SalidaInvalidaError,
    ValidationError,
    OutputParserException,
    openai.LengthFinishReasonError,
)
_ERRORES_DE_TRANSPORTE: tuple[type[BaseException], ...] = (
    openai.APIConnectionError,  # incluye APITimeoutError
    openai.RateLimitError,
    openai.InternalServerError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


def _es_reintentable(error: BaseException) -> bool:
    """Decide si `.with_retry()` vuelve a intentar, y deja registro del motivo.

    Se reintenta la salida mal formada o incompleta y los errores transitorios de
    red. Un 401 o un 400 no se arreglan reintentando: se propagan de una.
    """
    nombre = type(error).__name__
    if isinstance(error, _ERRORES_DE_SALIDA):
        logger.warning("Salida mal formada o incompleta (%s): %s", nombre, _corto(error))
        return True
    if isinstance(error, _ERRORES_DE_TRANSPORTE):
        logger.warning("Error transitorio de API/red (%s): %s", nombre, _corto(error))
        return True
    logger.debug("Error no reintentable (%s)", nombre)  # lo informa process_text
    return False


def _corto(error: BaseException, limite: int = 300) -> str:
    texto = " ".join(str(error).split())
    return texto if len(texto) <= limite else texto[:limite] + "…"


# --------------------------------------------------------------------------- prompt


def _instrucciones_de_formato() -> str:
    """Arma las instrucciones de formato a partir del esquema, para que no se desincronicen."""
    lineas = [
        f"- {nombre}: {campo.description}"
        for nombre, campo in ExtraccionTecnica.model_fields.items()
    ]
    return "\n".join(lineas)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un analista técnico senior. Recibís un texto sin procesar (descripción de "
            "arquitectura, log de error o ticket de incidente) y extraés sus entidades técnicas.\n\n"
            "Reglas:\n"
            "- Basate solo en lo que dice el texto. No inventes tecnologías ni causas.\n"
            "- Si el texto es ambiguo, quedate con la lectura más conservadora.\n\n"
            "Formato de la respuesta (un único objeto con exactamente estos campos):\n"
            "{instrucciones_formato}",
        ),
        ("human", "Texto a analizar:\n\n{texto}"),
    ]
)


# --------------------------------------------------------------------------- modelo


def _env(nombre: str) -> str | None:
    valor = os.environ.get(nombre, "").strip()
    return valor or None


def _crear_modelo(proveedor: str) -> BaseChatModel:
    """Instancia `ChatOpenAI` o `ChatAnthropic` según el proveedor.

    `max_retries=0` apaga los reintentos internos del SDK: el único que reintenta
    es `.with_retry()`, así cada intento queda a la vista en los logs.
    """
    max_tokens = int(_env("LLM_MAX_TOKENS") or DEFAULT_MAX_TOKENS)
    extra: dict[str, Any] = {}
    if (temperatura := _env("LLM_TEMPERATURE")) is not None:
        extra["temperature"] = float(temperatura)  # opcional: hay modelos que no la aceptan

    if proveedor == "openai":
        if _env("OPENAI_API_KEY") is None:
            raise ConfiguracionError("Falta OPENAI_API_KEY en el entorno (.env).")
        return ChatOpenAI(
            model=_env("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
            max_tokens=max_tokens,
            max_retries=0,
            **extra,
        )
    if proveedor == "anthropic":
        if _env("ANTHROPIC_API_KEY") is None:
            raise ConfiguracionError("Falta ANTHROPIC_API_KEY en el entorno (.env).")
        return ChatAnthropic(
            model=_env("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            max_retries=0,
            **extra,
        )
    raise ConfiguracionError(f"LLM_PROVIDER inválido: {proveedor!r}. Usá 'openai' o 'anthropic'.")


# --------------------------------------------------------------------------- verificación


def _verificar_salida(salida: dict[str, Any]) -> ExtraccionTecnica:
    """Revisa la respuesta cruda antes de dar por bueno el objeto.

    `with_structured_output(..., include_raw=True)` devuelve
    `{"raw": AIMessage, "parsed": objeto | None, "parsing_error": error | None}`.
    Primero se mira el `finish_reason`; recién después el resultado del parseo.
    """
    raw = salida.get("raw")
    metadata = getattr(raw, "response_metadata", None) or {}
    motivo = metadata.get("finish_reason") or metadata.get("stop_reason")
    if motivo in _CORTES_POR_TOKENS:
        raise SalidaIncompletaError(
            f"el modelo cortó la respuesta por límite de tokens (finish_reason={motivo})"
        )

    error = salida.get("parsing_error")
    if error is not None:
        raise SalidaInvalidaError(f"{type(error).__name__}: {error}") from error

    parsed = salida.get("parsed")
    if not isinstance(parsed, ExtraccionTecnica):
        raise SalidaInvalidaError("el modelo no devolvió un objeto con el esquema pedido")

    logger.info(
        "Validación Pydantic OK (finish_reason=%s): %d tecnologías, criticidad=%s",
        motivo,
        len(parsed.tecnologias),
        parsed.nivel_de_criticidad.value,
    )
    return parsed


class LogDeIntentos(AsyncCallbackHandler):
    """Loguea cada llamada real al modelo, con su número de intento y el uso de tokens.

    Se crea una instancia por `process_text`, así que el contador es por texto.
    """

    def __init__(self) -> None:
        self._modelo: str | None = None
        self._intento = 0

    async def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[Any], **kwargs: Any
    ) -> None:
        modelo = (kwargs.get("metadata") or {}).get("ls_model_name", "?")
        if modelo != self._modelo:  # cambió el modelo: entró el fallback
            if self._modelo is not None:
                logger.warning("Fallback: %s agotó sus intentos, se pasa a %s", self._modelo, modelo)
            self._modelo, self._intento = modelo, 0
        self._intento += 1
        logger.info("Intento %d/%d: llamando a %s", self._intento, MAX_INTENTOS, modelo)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            mensaje = response.generations[0][0].message  # type: ignore[attr-defined]
        except (IndexError, AttributeError):
            return
        metadata = mensaje.response_metadata or {}
        uso = mensaje.usage_metadata or {}
        logger.info(
            "Respuesta recibida: finish_reason=%s | tokens entrada=%s salida=%s",
            metadata.get("finish_reason") or metadata.get("stop_reason"),
            uso.get("input_tokens"),
            uso.get("output_tokens"),
        )


# --------------------------------------------------------------------------- cadena


def _extractor(modelo: BaseChatModel) -> Runnable[Any, ExtraccionTecnica]:
    """Salida estructurada del modelo, con verificación y reintentos."""
    # Anthropic usa por defecto tool calling forzado, que los modelos con thinking no
    # admiten; "json_schema" es la salida estructurada nativa en los dos proveedores.
    return _con_reintentos(
        modelo.with_structured_output(ExtraccionTecnica, method="json_schema", include_raw=True)
    )


def _con_reintentos(estructurado: Runnable[Any, Any]) -> Runnable[Any, ExtraccionTecnica]:
    """Agrega la verificación y envuelve todo en `.with_retry()`.

    La verificación va adentro del bloque reintentado: si la salida está mal formada
    o incompleta, se vuelve a llamar al modelo.
    """
    return (estructurado | RunnableLambda(_verificar_salida)).with_retry(
        retry_if_exception_type=_es_reintentable,
        wait_exponential_jitter=True,
        stop_after_attempt=MAX_INTENTOS,
    )


def build_chain() -> Runnable[dict[str, str], ExtraccionTecnica]:
    """Arma la cadena: prompt | model.with_structured_output(Schema), con reintentos.

    Si además está cargada la key del otro proveedor, queda como fallback para
    cuando el principal agota sus intentos.
    """
    proveedor = (_env("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    extractor = _extractor(_crear_modelo(proveedor))

    alternativo = "anthropic" if proveedor == "openai" else "openai"
    if _env(f"{alternativo.upper()}_API_KEY") is not None:
        extractor = extractor.with_fallbacks([_extractor(_crear_modelo(alternativo))])
        logger.debug("Fallback habilitado: %s -> %s", proveedor, alternativo)

    prompt = PROMPT.partial(instrucciones_formato=_instrucciones_de_formato())
    return prompt | extractor


_chain: Runnable[dict[str, str], ExtraccionTecnica] | None = None


async def process_text(text: str) -> ExtraccionTecnica:
    """Ejecuta el pipeline sobre un texto y devuelve el objeto validado.

    Lanza `SalidaInvalidaError` / `SalidaIncompletaError` si el modelo no logró una
    salida válida después de los reintentos, o el error de API correspondiente.
    """
    global _chain
    if _chain is None:
        _chain = build_chain()

    logger.info("Procesando texto de %d caracteres", len(text))
    try:
        resultado = await _chain.ainvoke({"texto": text}, config={"callbacks": [LogDeIntentos()]})
    except _ERRORES_DE_SALIDA as error:
        logger.error(
            "Sin salida válida después de %d intentos (%s): %s", MAX_INTENTOS, type(error).__name__, _corto(error)
        )
        raise
    except Exception as error:
        logger.error(
            "Falló la llamada al modelo, no se reintenta (%s): %s", type(error).__name__, _corto(error)
        )
        raise
    logger.info("Extracción completada")
    return resultado
