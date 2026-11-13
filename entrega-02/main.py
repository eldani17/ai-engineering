"""Script de prueba: corre el pipeline sobre varios textos en paralelo.

Uso:
    python main.py            # todos los casos
    python main.py --quiet    # sin logs, solo los resultados
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

from chain import ConfiguracionError, process_text

CASOS: list[tuple[str, str]] = [
    (
        "Incidente crítico",
        "El servicio de pagos en Kubernetes tira 502 desde el deploy de ayer. Sospechamos "
        "del pool de conexiones de PostgreSQL; Redis responde bien.",
    ),
    (
        "Descripción de arquitectura",
        "La API está hecha en FastAPI y corre en contenedores Docker detrás de un Nginx. "
        "Cachea las consultas frecuentes en Redis y persiste en PostgreSQL. Los eventos de "
        "facturación se publican en Kafka y los consume un worker en Go.",
    ),
    (
        "Mejora menor",
        "Estaría bueno migrar los tests del frontend de Jest a Vitest para que corran más "
        "rápido en GitHub Actions. No es urgente, lo vemos el próximo sprint.",
    ),
    (
        "Texto ambiguo (prueba de estrés)",
        "Che, ayer anduvo todo medio raro y hoy sigue igual. Alguien que se fije qué pasa "
        "porque el cliente ya preguntó dos veces.",
    ),
]


async def main() -> int:
    try:
        resultados = await asyncio.gather(
            *(process_text(texto) for _, texto in CASOS), return_exceptions=True
        )
    except ConfiguracionError as error:  # por si falla antes de crear las tareas
        print(f"[CONFIG] {error}")
        return 1

    fallos = 0
    for (titulo, texto), resultado in zip(CASOS, resultados):
        print(f"\n=== {titulo} ===")
        print(f"Entrada: {texto}")
        if isinstance(resultado, ConfiguracionError):
            print(f"[CONFIG] {resultado}")
            return 1
        if isinstance(resultado, BaseException):
            fallos += 1
            print(f"[SIN SALIDA VÁLIDA] {type(resultado).__name__}: {resultado}")
        else:
            print(resultado.model_dump_json(indent=2))

    print(f"\n{len(CASOS) - fallos} de {len(CASOS)} textos devolvieron un objeto validado.")
    return 0 if fallos < len(CASOS) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba del pipeline de extracción técnica.")
    parser.add_argument("--quiet", action="store_true", help="oculta los logs del pipeline")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("pipeline").setLevel(logging.ERROR if args.quiet else logging.INFO)
    sys.exit(asyncio.run(main()))
