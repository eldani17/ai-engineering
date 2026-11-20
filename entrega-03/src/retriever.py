"""Capa de recuperación: abre la colección persistida y devuelve los k fragmentos más cercanos.

La pregunta se convierte en embedding con la MISMA instancia que se usó para indexar
(`get_embeddings()`), y antes de abrir la base se verifica que el modelo coincida con el
que quedó registrado en la metadata de la colección.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, TOP_K, VECTORSTORE_DIR, validar_top_k
from src.embeddings import get_embeddings
from src.ingest import modelo_indexado, vectorstore_exists

logger = logging.getLogger("rag")


class BaseNoIndexadaError(RuntimeError):
    """No hay base vectorial poblada: hay que correr la ingesta primero."""


class EmbeddingsNoCoincidentesError(RuntimeError):
    """La colección se indexó con un modelo de embeddings distinto al configurado."""


def get_vectorstore(
    *,
    persist_dir: Path = VECTORSTORE_DIR,
    collection_name: str = COLLECTION_NAME,
    embeddings: Embeddings | None = None,
    embedding_model: str = EMBEDDING_MODEL,
) -> Chroma:
    """Abre la colección existente. Falla si no existe o si el modelo de embeddings no coincide."""
    if not vectorstore_exists(persist_dir, collection_name):
        raise BaseNoIndexadaError(
            f"No hay una colección {collection_name!r} poblada en {persist_dir}. Corré `python main.py ingest`."
        )
    indexado = modelo_indexado(persist_dir, collection_name)
    if indexado != embedding_model:
        raise EmbeddingsNoCoincidentesError(
            f"La base se indexó con {indexado!r} pero la consulta usaría {embedding_model!r}. "
            "Reindexá con `python main.py ingest --force`."
        )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(persist_dir),
    )


def get_retriever(k: int = TOP_K, **kwargs) -> VectorStoreRetriever:
    """Retriever de similitud que devuelve exactamente `k` fragmentos (3 <= k <= 5)."""
    validar_top_k(k)
    return get_vectorstore(**kwargs).as_retriever(search_type="similarity", search_kwargs={"k": k})
