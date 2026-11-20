"""Tests offline de la cadena RAG: retriever y LLM se reemplazan por respuestas guionadas."""

import json

import httpx
import anthropic
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src import chain
from src.chain import SalidaIncompletaError, ask, ask_many, build_rag_chain, format_docs
from src.prompts import PROMPT
from src.schemas import RAGAnswer

DOCS = [
    Document(page_content="Se usa canary: 10 % del tráfico.", metadata={"source": "02_guia.md", "chunk_index": 3, "seccion": "Despliegue progresivo"}),
    Document(page_content="SEV1: 10 minutos.", metadata={"source": "03_guardias.md", "chunk_index": 1, "seccion": "Niveles"}),
]


def _json(**campos):
    base = {"answer": "Canary, empezando con el 10 %.", "found_in_context": True, "sources": ["02_guia.md"], "confidence": "alta"}
    return AIMessage(content=json.dumps({**base, **campos}, ensure_ascii=False))


NO_SE = {"answer": "No lo sé. No tengo acceso a esa información.", "found_in_context": False, "sources": [], "confidence": "baja"}


def _llm(*pasos):
    """LLM falso: devuelve (o lanza) un paso por llamada y guarda los prompts que recibió."""
    prompts = []

    def responder(prompt_value):
        paso = pasos[len(prompts)]
        prompts.append(prompt_value)
        if isinstance(paso, BaseException):
            raise paso
        return paso

    return RunnableLambda(responder), prompts


def _retriever(docs=DOCS):
    preguntas = []

    def buscar(pregunta):
        preguntas.append(pregunta)
        return docs

    return RunnableLambda(buscar), preguntas


@pytest.fixture(autouse=True)
def _sin_espera(monkeypatch):
    """Anula el backoff para que los tests no duerman."""
    import langchain_core.runnables.retry as retry

    monkeypatch.setattr(retry, "wait_exponential_jitter", lambda **_: (lambda _state: 0))


async def test_flujo_completo_pasa_contexto_y_pregunta_al_modelo():
    retriever, preguntas = _retriever()
    llm, prompts = _llm(_json())
    respuesta = await build_rag_chain(retriever, llm).ainvoke("¿Qué estrategia de despliegue se usa?")

    assert isinstance(respuesta, RAGAnswer)
    assert respuesta.sources == ["02_guia.md"]
    assert preguntas == ["¿Qué estrategia de despliegue se usa?"]
    sistema, humano = prompts[0].to_messages()
    assert "No lo sé" in sistema.content and "found_in_context" in sistema.content
    assert "Se usa canary" in humano.content and "fuente: 02_guia.md" in humano.content
    assert "¿Qué estrategia de despliegue se usa?" in humano.content


async def test_respuesta_fuera_de_contexto():
    llm, _ = _llm(AIMessage(content=json.dumps(NO_SE, ensure_ascii=False)))
    respuesta = await build_rag_chain(_retriever()[0], llm).ainvoke("¿Capital de Australia?")
    assert not respuesta.found_in_context and respuesta.sources == []
    assert respuesta.answer.startswith("No lo sé")


async def test_json_en_bloque_markdown_tambien_se_parsea():
    llm, _ = _llm(AIMessage(content="```json\n" + _json().content + "\n```"))
    assert (await build_rag_chain(_retriever()[0], llm).ainvoke("x")).found_in_context


async def test_json_invalido_se_reintenta():
    llm, prompts = _llm(AIMessage(content="Claro, la respuesta es canary."), _json())
    respuesta = await build_rag_chain(_retriever()[0], llm).ainvoke("x")
    assert respuesta.found_in_context and len(prompts) == 2


async def test_json_que_no_cumple_el_esquema_se_reintenta():
    llm, prompts = _llm(_json(found_in_context=False), _json())  # incoherente: sin "No lo sé"
    await build_rag_chain(_retriever()[0], llm).ainvoke("x")
    assert len(prompts) == 2


async def test_salida_cortada_por_tokens_se_reintenta_y_despues_falla():
    cortada = AIMessage(content='{"answer": "Can', response_metadata={"stop_reason": "max_tokens"})
    llm, prompts = _llm(*[cortada] * chain.MAX_INTENTOS)
    with pytest.raises(SalidaIncompletaError):
        await build_rag_chain(_retriever()[0], llm).ainvoke("x")
    assert len(prompts) == chain.MAX_INTENTOS


async def test_error_de_autenticacion_no_se_reintenta():
    respuesta = httpx.Response(401, request=httpx.Request("POST", "https://api.example"))
    auth = anthropic.AuthenticationError("key inválida", response=respuesta, body=None)
    llm, prompts = _llm(auth, _json())
    with pytest.raises(anthropic.AuthenticationError):
        await build_rag_chain(_retriever()[0], llm).ainvoke("x")
    assert len(prompts) == 1


async def test_fallback_al_otro_proveedor():
    respuesta = httpx.Response(401, request=httpx.Request("POST", "https://api.example"))
    principal, _ = _llm(anthropic.AuthenticationError("key inválida", response=respuesta, body=None))
    alternativo, prompts = _llm(_json())
    resultado = await build_rag_chain(_retriever()[0], principal, fallback=alternativo).ainvoke("x")
    assert resultado.found_in_context and len(prompts) == 1


async def test_fuentes_inventadas_se_descartan():
    llm, _ = _llm(_json(sources=["02_guia.md", "inventado.md"]))
    respuesta = await build_rag_chain(_retriever()[0], llm).ainvoke("x")
    assert respuesta.sources == ["02_guia.md"]


async def test_si_ninguna_fuente_es_valida_se_usan_las_recuperadas():
    llm, _ = _llm(_json(sources=["inventado.md"]))
    respuesta = await build_rag_chain(_retriever()[0], llm).ainvoke("x")
    assert respuesta.sources == ["02_guia.md", "03_guardias.md"]


async def test_ask_rechaza_pregunta_vacia():
    with pytest.raises(ValueError):
        await ask("   ", chain=build_rag_chain(_retriever()[0], _llm()[0]))


async def test_ask_many_corre_en_paralelo_y_aisla_errores():
    cadena = build_rag_chain(_retriever()[0], RunnableLambda(lambda _: _json()))
    resultados = await ask_many(["uno", "  ", "tres"], chain=cadena)
    assert isinstance(resultados[0], RAGAnswer) and isinstance(resultados[2], RAGAnswer)
    assert isinstance(resultados[1], ValueError)


def test_format_docs_cita_fuente_y_seccion():
    texto = format_docs(DOCS)
    assert "[Fragmento 1] fuente: 02_guia.md | sección: Despliegue progresivo" in texto
    assert "[Fragmento 2] fuente: 03_guardias.md" in texto
    assert format_docs([]) == "(no se recuperó ningún fragmento)"


def test_prompt_tiene_contexto_pregunta_y_formato():
    assert set(PROMPT.input_variables) == {"context", "question"}
