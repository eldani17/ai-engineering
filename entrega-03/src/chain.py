"""Generación grounded: cadena LCEL asíncrona que responde solo con el contexto recuperado.

    pregunta ──► retriever (k=4) ──► format_docs ──► prompt ──► LLM ──► PydanticOutputParser ──► RAGAnswer
                                                              └──── .with_retry(): hasta 3 intentos ────┘

Después del parser se contrastan las fuentes que citó el modelo con los archivos que
realmente devolvió el retriever, para que no pueda citar un archivo que no vio.
Puntos de entrada: `ask(pregunta)` y `ask_many(preguntas)`, ambos asíncronos.
"""

from __future__ import annotations

import asyncio
import logging
import os
from operator import itemgetter
from typing import Any

import anthropic
import openai
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from src.config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PROVIDER,
    MAX_INTENTOS,
)
from src.prompts import PROMPT, parser
from src.schemas import RAGAnswer

logger = logging.getLogger("rag")

_CORTES_POR_TOKENS = {"length", "max_tokens"}


# --------------------------------------------------------------------------- errores


class ConfiguracionError(Exception):
    """Falta la API key o el proveedor configurado no existe."""


class SalidaIncompletaError(Exception):
    """El modelo cortó la respuesta por límite de tokens: el JSON no es confiable."""


_ERRORES_DE_SALIDA: tuple[type[BaseException], ...] = (
    OutputParserException,  # JSON inválido o que no cumple RAGAnswer
    ValidationError,
    SalidaIncompletaError,
)
_ERRORES_DE_TRANSPORTE: tuple[type[BaseException], ...] = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


def _es_reintentable(error: BaseException) -> bool:
    """Se reintenta la salida mal formada y los errores transitorios; un 401/400 no."""
    if isinstance(error, _ERRORES_DE_SALIDA):
        logger.warning("Salida inválida, se reintenta (%s): %s", type(error).__name__, _corto(error))
        return True
    if isinstance(error, _ERRORES_DE_TRANSPORTE):
        logger.warning("Error transitorio de API, se reintenta (%s): %s", type(error).__name__, _corto(error))
        return True
    return False


def _corto(error: BaseException, limite: int = 250) -> str:
    texto = " ".join(str(error).split())
    return texto if len(texto) <= limite else texto[:limite] + "…"


# --------------------------------------------------------------------------- modelo


def _env(nombre: str) -> str | None:
    valor = os.environ.get(nombre, "").strip()
    return valor or None


def crear_modelo(proveedor: str) -> BaseChatModel:
    """`ChatOpenAI` o `ChatAnthropic` según el proveedor (misma convención que PE1 y PE2).

    `max_retries=0`: el único que reintenta es `.with_retry()`, así cada intento se ve en los logs.
    """
    max_tokens = int(_env("LLM_MAX_TOKENS") or DEFAULT_MAX_TOKENS)
    extra: dict[str, Any] = {}
    if (temperatura := _env("LLM_TEMPERATURE")) is not None:
        extra["temperature"] = float(temperatura)  # opcional: hay modelos que no la aceptan

    if proveedor == "openai":
        if _env("OPENAI_API_KEY") is None:
            raise ConfiguracionError("Falta OPENAI_API_KEY en el entorno (.env).")
        return ChatOpenAI(
            model=_env("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL, max_tokens=max_tokens, max_retries=0, **extra
        )
    if proveedor == "anthropic":
        if _env("ANTHROPIC_API_KEY") is None:
            raise ConfiguracionError("Falta ANTHROPIC_API_KEY en el entorno (.env).")
        return ChatAnthropic(
            model=_env("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL, max_tokens=max_tokens, max_retries=0, **extra
        )
    raise ConfiguracionError(f"LLM_PROVIDER inválido: {proveedor!r}. Usá 'openai' o 'anthropic'.")


# --------------------------------------------------------------------------- pasos de la cadena


def format_docs(documentos: list[Document]) -> str:
    """Arma el bloque CONTEXTO: cada fragmento con su archivo y sección, para poder citarlo."""
    bloques = []
    for numero, doc in enumerate(documentos, start=1):
        fuente = doc.metadata.get("source", "desconocida")
        seccion = doc.metadata.get("seccion") or "-"
        bloques.append(f"[Fragmento {numero}] fuente: {fuente} | sección: {seccion}\n{doc.page_content}")
    return "\n\n---\n\n".join(bloques) if bloques else "(no se recuperó ningún fragmento)"


def _verificar_corte(mensaje: AIMessage) -> AIMessage:
    """Si el modelo cortó por tokens, el JSON está incompleto: se descarta y se reintenta."""
    metadata = mensaje.response_metadata or {}
    motivo = metadata.get("finish_reason") or metadata.get("stop_reason")
    if motivo in _CORTES_POR_TOKENS:
        raise SalidaIncompletaError(f"el modelo cortó la respuesta por límite de tokens ({motivo})")
    return mensaje


def _generador(llm: Runnable) -> Runnable[dict[str, str], RAGAnswer]:
    """prompt | llm | PydanticOutputParser, con reintentos ante salida inválida o errores de red.

    La verificación de corte y el parser van adentro del bloque reintentado: si el JSON
    llega mal formado o no cumple `RAGAnswer`, se vuelve a llamar al modelo.
    """
    return (PROMPT | llm | RunnableLambda(_verificar_corte) | parser).with_retry(
        retry_if_exception_type=_es_reintentable,  # type: ignore[arg-type]
        wait_exponential_jitter=True,
        stop_after_attempt=MAX_INTENTOS,
    )


def _reconciliar_fuentes(entrada: dict[str, Any]) -> RAGAnswer:
    """Deja en `sources` solo archivos que el retriever efectivamente devolvió."""
    respuesta: RAGAnswer = entrada["respuesta"]
    documentos: list[Document] = entrada["documentos"]
    recuperadas = list(dict.fromkeys(d.metadata.get("source", "") for d in documentos if d.metadata.get("source")))

    if not respuesta.found_in_context:
        return respuesta

    validas = [fuente for fuente in respuesta.sources if fuente in recuperadas]
    inventadas = [fuente for fuente in respuesta.sources if fuente not in recuperadas]
    if inventadas:
        logger.warning("El modelo citó fuentes que no estaban en el contexto, se descartan: %s", inventadas)
    if not validas:
        logger.warning("Ninguna fuente citada era válida: se usan las recuperadas %s", recuperadas)
        validas = recuperadas
    return respuesta.model_copy(update={"sources": validas})


def _log_recuperacion(entrada: dict[str, Any]) -> dict[str, Any]:
    documentos: list[Document] = entrada["documentos"]
    logger.info(
        "Recuperados %d fragmentos: %s",
        len(documentos),
        ", ".join(f"{d.metadata.get('source')}#{d.metadata.get('chunk_index')}" for d in documentos),
    )
    return entrada


# --------------------------------------------------------------------------- cadena


def build_rag_chain(
    retriever: BaseRetriever | Runnable[str, list[Document]],
    llm: Runnable | None = None,
    fallback: Runnable | None = None,
) -> Runnable[str, RAGAnswer]:
    """Arma la cadena LCEL completa. `retriever` y `llm` se inyectan para poder testear offline.

    Sin `llm`, se crea el del proveedor de `LLM_PROVIDER`; si además está la key del otro
    proveedor, queda como fallback cuando el principal agota sus intentos.
    """
    if llm is None:
        proveedor = (_env("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
        llm = crear_modelo(proveedor)
        alternativo = "openai" if proveedor == "anthropic" else "anthropic"
        if fallback is None and _env(f"{alternativo.upper()}_API_KEY") is not None:
            fallback = crear_modelo(alternativo)

    generacion = _generador(llm)
    if fallback is not None:
        generacion = generacion.with_fallbacks([_generador(fallback)])

    contexto_y_pregunta = {
        "context": itemgetter("documentos") | RunnableLambda(format_docs),
        "question": itemgetter("pregunta"),
    }
    return (
        RunnableParallel(documentos=retriever, pregunta=RunnablePassthrough())
        | RunnableLambda(_log_recuperacion)
        | RunnablePassthrough.assign(respuesta=contexto_y_pregunta | generacion)
        | RunnableLambda(_reconciliar_fuentes)
    ).with_config(run_name="rag_chain")


class LogDeLlamadas(AsyncCallbackHandler):
    """Loguea cada llamada real al modelo (número de intento, motivo de fin y tokens)."""

    def __init__(self) -> None:
        self._intento = 0

    async def on_chat_model_start(self, serialized: dict[str, Any], messages: list[Any], **kwargs: Any) -> None:
        self._intento += 1
        modelo = (kwargs.get("metadata") or {}).get("ls_model_name", "?")
        logger.info("Llamada %d al modelo %s", self._intento, modelo)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            mensaje = response.generations[0][0].message  # type: ignore[attr-defined]
        except (IndexError, AttributeError):
            return
        metadata = mensaje.response_metadata or {}
        uso = mensaje.usage_metadata or {}
        logger.info(
            "Respuesta del modelo: fin=%s | tokens entrada=%s salida=%s",
            metadata.get("finish_reason") or metadata.get("stop_reason"),
            uso.get("input_tokens"),
            uso.get("output_tokens"),
        )


# --------------------------------------------------------------------------- API asíncrona

_chain: Runnable[str, RAGAnswer] | None = None


def get_chain() -> Runnable[str, RAGAnswer]:
    """Cadena por defecto (retriever sobre ./vectorstore + LLM del .env), creada una sola vez."""
    global _chain
    if _chain is None:
        from src.retriever import get_retriever  # import diferido: carga el modelo de embeddings

        _chain = build_rag_chain(get_retriever())
    return _chain


async def ask(question: str, chain: Runnable[str, RAGAnswer] | None = None) -> RAGAnswer:
    """Responde una pregunta con `ainvoke` y devuelve el objeto validado por el parser."""
    question = question.strip()
    if not question:
        raise ValueError("La pregunta está vacía.")
    logger.info("Pregunta: %s", question)
    respuesta = await (chain or get_chain()).ainvoke(question, config={"callbacks": [LogDeLlamadas()]})
    logger.info("Respuesta validada: found_in_context=%s, fuentes=%s", respuesta.found_in_context, respuesta.sources)
    return respuesta


async def ask_many(
    questions: list[str], chain: Runnable[str, RAGAnswer] | None = None
) -> list[RAGAnswer | BaseException]:
    """Varias preguntas en paralelo con `asyncio.gather`; un fallo no frena a las demás."""
    cadena = chain or get_chain()
    return await asyncio.gather(*(ask(q, cadena) for q in questions), return_exceptions=True)
