"""Configuración central del sistema RAG: una sola fuente de verdad para rutas y parámetros.

Todo lo que tiene que coincidir entre la ingesta y la consulta (modelo de embeddings,
carpeta de la base, nombre de la colección) vive acá y en ningún otro lado.
"""

from __future__ import annotations

from pathlib import Path

# Rutas absolutas a partir de la raíz del proyecto, para que funcionen igual desde la CLI,
# desde los tests y desde el notebook (que corre con otro directorio de trabajo).
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
VECTORSTORE_DIR = PROJECT_DIR / "vectorstore"
COLLECTION_NAME = "pre_entrega_3"

# Extensiones que acepta la ingesta.
EXTENSIONES_VALIDAS = (".md", ".txt")

# --- Embeddings -------------------------------------------------------------------------
# Modelo local de sentence-transformers, multilingüe (el dataset está en español) y con
# ventana de 512 tokens: un chunk de 800 caracteres (~200 tokens) entra completo, sin
# truncarse. Los modelos E5 se entrenaron con prefijos distintos para documentos y
# consultas; los aplica `embeddings.py`.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
PREFIJO_DOCUMENTO = "passage: "
PREFIJO_CONSULTA = "query: "

# --- Chunking ---------------------------------------------------------------------------
CHUNK_SIZE = 800  # caracteres: alcanza para una sección corta o una tabla entera
CHUNK_OVERLAP = 100  # ~12 %: evita cortar una idea justo en el borde entre dos chunks

# --- Recuperación -----------------------------------------------------------------------
# Entre 3 y 5 fragmentos: con más, el modelo pierde la información que queda en el medio
# del contexto ("Lost in the Middle") y el costo por consulta crece sin mejorar la respuesta.
TOP_K = 4
TOP_K_MIN, TOP_K_MAX = 3, 5

# --- LLM (misma convención de variables de entorno que las pre-entregas 1 y 2) ---------
DEFAULT_PROVIDER = "anthropic"
DEFAULT_OPENAI_MODEL = "gpt-6.1-sol"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_MAX_TOKENS = 1024
MAX_INTENTOS = 3


def validar_top_k(k: int) -> int:
    """Rechaza un `k` fuera del rango 3-5 en lugar de mandar contexto de más al modelo."""
    if not TOP_K_MIN <= k <= TOP_K_MAX:
        raise ValueError(f"top_k debe estar entre {TOP_K_MIN} y {TOP_K_MAX} (se recibió {k}).")
    return k
