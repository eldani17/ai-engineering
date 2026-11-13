"""Tests offline de la resiliencia: el modelo se reemplaza por respuestas guionadas."""

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

import chain
from chain import PROMPT, SalidaIncompletaError, SalidaInvalidaError, _con_reintentos
from schemas import ExtraccionTecnica

OBJETO = ExtraccionTecnica(
    tecnologias=["Kubernetes", "PostgreSQL"],
    nivel_de_criticidad="alta",
    resumen_tecnico="Errores 502 en pagos tras un deploy; se sospecha del pool de conexiones.",
)


def _ok(finish_reason="stop"):
    raw = AIMessage(content="", response_metadata={"finish_reason": finish_reason})
    return {"raw": raw, "parsed": OBJETO, "parsing_error": None}


def _mal_formada():
    try:
        ExtraccionTecnica.model_validate({"tecnologias": [], "nivel_de_criticidad": "urgente"})
    except ValidationError as error:
        raw = AIMessage(content="", response_metadata={"finish_reason": "stop"})
        return {"raw": raw, "parsed": None, "parsing_error": error}


def _cortada():
    raw = AIMessage(content="", response_metadata={"stop_reason": "max_tokens"})
    return {"raw": raw, "parsed": None, "parsing_error": None}


def _guion(*pasos):
    """Runnable que devuelve (o lanza) un paso por llamada, y cuenta las llamadas."""
    llamadas = []

    def responder(_entrada):
        paso = pasos[len(llamadas)]
        llamadas.append(paso)
        if isinstance(paso, BaseException):
            raise paso
        return paso

    return _con_reintentos(RunnableLambda(responder)), llamadas


@pytest.fixture(autouse=True)
def _sin_espera(monkeypatch):
    """Anula el backoff para que los tests no duerman."""
    import langchain_core.runnables.retry as retry

    monkeypatch.setattr(retry, "wait_exponential_jitter", lambda **_: (lambda _state: 0))


async def test_salida_valida_no_reintenta():
    extractor, llamadas = _guion(_ok())
    assert await extractor.ainvoke({}) == OBJETO
    assert len(llamadas) == 1


async def test_salida_mal_formada_se_reintenta_y_se_recupera():
    extractor, llamadas = _guion(_mal_formada(), _ok())
    assert await extractor.ainvoke({}) == OBJETO
    assert len(llamadas) == 2


async def test_respuesta_cortada_por_tokens_se_reintenta():
    extractor, llamadas = _guion(_cortada(), _ok())
    assert await extractor.ainvoke({}) == OBJETO
    assert len(llamadas) == 2


async def test_finish_reason_length_se_detecta_aunque_haya_objeto():
    extractor, llamadas = _guion(_ok(finish_reason="length"), _ok())
    assert await extractor.ainvoke({}) == OBJETO
    assert len(llamadas) == 2


async def test_salida_siempre_invalida_agota_los_intentos():
    extractor, llamadas = _guion(*[_mal_formada()] * chain.MAX_INTENTOS)
    with pytest.raises(SalidaInvalidaError):
        await extractor.ainvoke({})
    assert len(llamadas) == chain.MAX_INTENTOS


async def test_siempre_cortada_termina_en_salida_incompleta():
    extractor, _ = _guion(*[_cortada()] * chain.MAX_INTENTOS)
    with pytest.raises(SalidaIncompletaError):
        await extractor.ainvoke({})


async def test_error_de_red_se_reintenta():
    red = openai.APIConnectionError(request=httpx.Request("POST", "https://api.example"))
    extractor, llamadas = _guion(red, _ok())
    assert await extractor.ainvoke({}) == OBJETO
    assert len(llamadas) == 2


async def test_error_de_autenticacion_no_se_reintenta():
    respuesta = httpx.Response(401, request=httpx.Request("POST", "https://api.example"))
    auth = openai.AuthenticationError("key inválida", response=respuesta, body=None)
    extractor, llamadas = _guion(auth, _ok())
    with pytest.raises(openai.AuthenticationError):
        await extractor.ainvoke({})
    assert len(llamadas) == 1


def test_prompt_recibe_texto_e_instrucciones_de_formato():
    assert set(PROMPT.input_variables) == {"texto", "instrucciones_formato"}
    mensajes = PROMPT.format_messages(
        texto="Redis no responde", instrucciones_formato=chain._instrucciones_de_formato()
    )
    assert "nivel_de_criticidad" in mensajes[0].content
    assert "Redis no responde" in mensajes[1].content
