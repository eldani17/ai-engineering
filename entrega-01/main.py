"""Demo del cliente: pregunta "¿Qué es la entropía?" en modo normal y en streaming.

Uso:
    python main.py                       # usa LLM_PROVIDER del .env
    python main.py --provider anthropic  # pisa el .env
    python main.py --mode stream         # solo streaming
    python main.py --verbose             # muestra logs de reintentos
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_client import (
    AsyncLLMManager,
    BaseLLMClient,
    ChatMessage,
    LLMError,
    Provider,
    Role,
    Settings,
)

PREGUNTA = "¿Qué es la entropía?"
SYSTEM_PROMPT = "Sos un docente de física. Respondé en español rioplatense, en no más de tres párrafos."


def _mensajes() -> list[ChatMessage]:
    return [
        ChatMessage(role=Role.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=Role.USER, content=PREGUNTA),
    ]


async def modo_normal(client: BaseLLMClient) -> bool:
    """Pide la respuesta completa e imprime texto, tokens y latencia."""
    print(f"\n=== Modo normal ({client.provider.value} / {client.config.model}) ===")
    response = await client.generate(_mensajes())
    if not response.success:
        assert response.error is not None
        print(f"\n[ERROR] {response.error.type}: {response.error.message}")
        print(f"        intentos: {response.error.attempts} | reintentable: {response.error.retryable}")
        return False
    print(response.text)
    print(
        f"\n-- tokens entrada: {response.input_tokens} | tokens salida: {response.output_tokens}"
        f" | latencia: {response.latency_ms:.0f} ms | finish_reason: {response.finish_reason}"
    )
    return True


async def modo_streaming(client: BaseLLMClient) -> bool:
    """Imprime los fragmentos a medida que llegan y mide TTFT y tiempo total."""
    print(f"\n=== Modo streaming ({client.provider.value} / {client.config.model}) ===")
    started = time.perf_counter()
    ttft_ms: float | None = None
    chunks = 0
    try:
        async for chunk in client.stream(_mensajes()):
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - started) * 1000
            chunks += 1
            print(chunk, end="", flush=True)
    except LLMError as error:
        print(f"\n[ERROR] {type(error).__name__}: {error.message}")
        print(f"        intentos: {error.attempts} | reintentable: {error.retryable}")
        return False
    total_ms = (time.perf_counter() - started) * 1000
    ttft_text = f"{ttft_ms:.0f} ms" if ttft_ms is not None else "n/a"
    print(f"\n\n-- TTFT: {ttft_text} | total: {total_ms:.0f} ms | fragmentos: {chunks}")
    return True


async def run(provider: str | None, mode: str) -> int:
    load_dotenv()
    try:
        settings = Settings.from_env()
    except ValidationError as error:
        print(f"[ERROR] Configuración inválida en el .env:\n{error}")
        return 1

    try:
        async with AsyncLLMManager(settings) as manager:
            client = manager.get_client(provider)
            ok = True
            if mode in ("normal", "both"):
                ok = await modo_normal(client) and ok
            if mode in ("stream", "both"):
                ok = await modo_streaming(client) and ok
            return 0 if ok else 1
    except LLMError as error:
        # Errores de configuración (falta la key, proveedor desconocido, etc.).
        print(f"[ERROR] {type(error).__name__}: {error.message}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Cliente LLM asíncrono: demo normal + streaming.")
    parser.add_argument(
        "--provider",
        choices=[p.value for p in Provider],
        default=None,
        help="Proveedor a usar; pisa LLM_PROVIDER del .env",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra los logs de reintentos y errores del cliente",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "stream", "both"],
        default="both",
        help="Qué modos ejecutar (default: both)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        exit_code = asyncio.run(run(args.provider, args.mode))
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
