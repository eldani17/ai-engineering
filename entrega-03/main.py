"""CLI del sistema RAG local.

Uso:
    python main.py ingest [--force]      # indexa data/ en ./vectorstore (se omite si ya existe)
    python main.py buscar "consulta"     # solo recuperación: muestra los k fragmentos y su distancia
    python main.py ask "pregunta"        # RAG completo: recupera, genera y valida con Pydantic
    python main.py chat                  # modo interactivo
    python main.py demo                  # preguntas de ejemplo en paralelo (incluye fuera de contexto)

Agregá --quiet antes del subcomando para ocultar los logs: `python main.py --quiet demo`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from dotenv import load_dotenv

from src.chain import ConfiguracionError, ask, ask_many, get_chain
from src.config import TOP_K
from src.ingest import build_vectorstore
from src.retriever import BaseNoIndexadaError, EmbeddingsNoCoincidentesError, get_vectorstore
from src.schemas import RAGAnswer

# 4 preguntas que el dataset responde (una por archivo) y 2 que no.
PREGUNTAS_DEMO: list[tuple[str, str]] = [
    ("dentro", "¿Qué estrategia de despliegue se usa en producción y qué porcentaje del tráfico recibe primero la versión nueva?"),
    ("dentro", "¿En cuánto tiempo hay que reconocer una alerta SEV1 y a quién se escala si nadie responde?"),
    ("dentro", "¿Cómo se calcula el peso facturable de un envío en la API de cotizaciones?"),
    ("dentro", "¿Por qué Kairo usa NATS JetStream en lugar de Kafka?"),
    ("fuera", "¿Cuál es la capital de Australia?"),
    ("fuera", "¿Qué CRM usa el equipo comercial de Transportes Altiplano para gestionar clientes?"),
]


def imprimir_respuesta(pregunta: str, respuesta: RAGAnswer) -> None:
    marca = "EN CONTEXTO" if respuesta.found_in_context else "FUERA DE CONTEXTO"
    print(f"\nPregunta: {pregunta}")
    print(f"[{marca} | confianza {respuesta.confidence}]")
    print(f"Respuesta: {respuesta.answer}")
    print(f"Fuentes: {', '.join(respuesta.sources) if respuesta.sources else '(ninguna)'}")


def asegurar_base() -> None:
    """`ask`, `chat` y `demo` no re-indexan: si la base ya existe, la ingesta se omite."""
    build_vectorstore(force=False)


# --------------------------------------------------------------------------- subcomandos


def cmd_ingest(force: bool) -> int:
    resultado = build_vectorstore(force=force)
    if resultado.indexada:
        print(f"Indexados {resultado.documentos} documentos en {resultado.fragmentos} fragmentos.")
    else:
        print(f"La base ya existía ({resultado.fragmentos} fragmentos): no se re-indexó. Usá --force para reconstruirla.")
    return 0


def cmd_buscar(consulta: str) -> int:
    """Prueba de la capa de recuperación sola, sin LLM."""
    asegurar_base()
    resultados = get_vectorstore().similarity_search_with_score(consulta, k=TOP_K)
    print(f"\nConsulta: {consulta}\nTop {TOP_K} fragmentos (distancia coseno: más baja = más parecido):")
    for posicion, (doc, distancia) in enumerate(resultados, start=1):
        extracto = " ".join(doc.page_content.split())[:160]
        print(
            f"\n{posicion}. {doc.metadata['source']}#{doc.metadata['chunk_index']} "
            f"| sección: {doc.metadata.get('seccion') or '-'} | distancia={distancia:.3f}\n   {extracto}…"
        )
    return 0


async def cmd_ask(pregunta: str) -> int:
    asegurar_base()
    imprimir_respuesta(pregunta, await ask(pregunta))
    return 0


async def cmd_chat() -> int:
    asegurar_base()
    print("Modo interactivo. Escribí tu pregunta (o 'salir' para terminar).")
    while True:
        try:
            pregunta = (await asyncio.to_thread(input, "\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if pregunta.lower() in {"salir", "exit", "quit"}:
            return 0
        if not pregunta:
            continue
        try:
            imprimir_respuesta(pregunta, await ask(pregunta))
        except ConfiguracionError:
            raise
        except Exception as error:  # un error en una pregunta no corta la sesión
            print(f"[ERROR] {type(error).__name__}: {error}")


async def cmd_demo() -> int:
    asegurar_base()
    preguntas = [p for _, p in PREGUNTAS_DEMO]
    get_chain()  # carga el modelo de embeddings antes de medir el tiempo de las consultas
    inicio = time.perf_counter()
    resultados = await ask_many(preguntas)
    duracion = time.perf_counter() - inicio

    aciertos = 0
    for (esperado, pregunta), resultado in zip(PREGUNTAS_DEMO, resultados):
        if isinstance(resultado, ConfiguracionError):
            raise resultado
        if isinstance(resultado, BaseException):
            print(f"\nPregunta: {pregunta}\n[ERROR] {type(resultado).__name__}: {resultado}")
            continue
        imprimir_respuesta(pregunta, resultado)
        if resultado.found_in_context == (esperado == "dentro"):
            aciertos += 1

    print(
        f"\n{aciertos} de {len(PREGUNTAS_DEMO)} preguntas se clasificaron como se esperaba "
        f"(dentro/fuera de contexto). {len(preguntas)} preguntas en paralelo en {duracion:.1f} s."
    )
    return 0 if aciertos == len(PREGUNTAS_DEMO) else 1


# --------------------------------------------------------------------------- entrada


def main() -> int:
    parser = argparse.ArgumentParser(description="Sistema de recuperación semántica local (RAG).")
    parser.add_argument("--quiet", action="store_true", help="oculta los logs")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_ingest = sub.add_parser("ingest", help="indexa data/ en ChromaDB")
    p_ingest.add_argument("--force", action="store_true", help="borra la colección y la reconstruye")
    sub.add_parser("buscar", help="muestra los fragmentos recuperados, sin LLM").add_argument("consulta")
    sub.add_parser("ask", help="responde una pregunta").add_argument("pregunta")
    sub.add_parser("chat", help="modo interactivo")
    sub.add_parser("demo", help="preguntas de ejemplo en paralelo")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("rag").setLevel(logging.WARNING if args.quiet else logging.INFO)

    try:
        if args.comando == "ingest":
            return cmd_ingest(args.force)
        if args.comando == "buscar":
            return cmd_buscar(args.consulta)
        if args.comando == "ask":
            return asyncio.run(cmd_ask(args.pregunta))
        if args.comando == "chat":
            return asyncio.run(cmd_chat())
        return asyncio.run(cmd_demo())
    except (ConfiguracionError, BaseNoIndexadaError, EmbeddingsNoCoincidentesError, FileNotFoundError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
