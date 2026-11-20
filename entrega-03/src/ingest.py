"""Módulo de ingesta: carga los documentos, los fragmenta y los persiste en ChromaDB.

    data/*.md|*.txt ──► Document ──► RecursiveCharacterTextSplitter ──► Chroma (./vectorstore)

Si la colección ya existe y tiene fragmentos, la ingesta se omite (salvo `force=True`).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    EXTENSIONES_VALIDAS,
    PROJECT_DIR,
    VECTORSTORE_DIR,
)
from src.embeddings import get_embeddings

logger = logging.getLogger("rag")

# Separadores pensados para Markdown: primero se intenta cortar entre secciones, después
# entre párrafos, después entre líneas y recién al final entre palabras.
SEPARADORES = ["\n## ", "\n### ", "\n\n", "\n", " ", ""]

_TITULO = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class ResultadoIngesta:
    """Qué hizo `build_vectorstore`: sirve para loguear y para los tests."""

    indexada: bool  # False si se omitió porque la base ya existía
    documentos: int
    fragmentos: int


# --------------------------------------------------------------------------- carga


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Lee todos los .md / .txt de `data_dir` y les agrega la metadata `source`."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de datos: {data_dir}")

    archivos = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in EXTENSIONES_VALIDAS)
    if not archivos:
        raise FileNotFoundError(f"No hay archivos {', '.join(EXTENSIONES_VALIDAS)} en {data_dir}")

    documentos = []
    for archivo in archivos:
        texto = archivo.read_text(encoding="utf-8").strip()
        if not texto:
            logger.warning("Se ignora %s: está vacío", archivo.name)
            continue
        documentos.append(Document(page_content=texto, metadata={"source": archivo.name}))
    logger.info("Documentos cargados: %d (%s)", len(documentos), ", ".join(d.metadata["source"] for d in documentos))
    return documentos


# --------------------------------------------------------------------------- chunking


def _seccion_en(texto: str, posicion: int) -> str:
    """Último título de Markdown que aparece antes de `posicion` (la sección del chunk)."""
    seccion = ""
    for match in _TITULO.finditer(texto):
        if match.start() > posicion:
            break
        seccion = match.group(2).strip()
    return seccion


def split_documents(
    documentos: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Fragmenta con `RecursiveCharacterTextSplitter` y enriquece la metadata de cada chunk.

    Cada fragmento conserva `source` y suma `chunk_index` (posición dentro del archivo) y
    `seccion` (el título bajo el que cae), que después se usan para citar la fuente.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARADORES,
        add_start_index=True,
    )
    fragmentos: list[Document] = []
    for documento in documentos:
        partes = splitter.split_documents([documento])
        for indice, parte in enumerate(partes):
            parte.metadata["chunk_index"] = indice
            parte.metadata["seccion"] = _seccion_en(documento.page_content, parte.metadata["start_index"])
        fragmentos.extend(partes)
    logger.info(
        "Chunking: %d documentos -> %d fragmentos (chunk_size=%d, chunk_overlap=%d)",
        len(documentos),
        len(fragmentos),
        chunk_size,
        chunk_overlap,
    )
    return fragmentos


def _ruta(path: Path) -> str:
    """Ruta relativa al proyecto para los logs (sin exponer la ruta absoluta de la máquina)."""
    try:
        return "./" + str(Path(path).resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def _id_de(fragmento: Document) -> str:
    """ID determinístico: re-indexar el mismo archivo no duplica fragmentos."""
    return f"{fragmento.metadata['source']}#{fragmento.metadata['chunk_index']}"


# --------------------------------------------------------------------------- persistencia


def _abrir_coleccion(persist_dir: Path, collection_name: str) -> chromadb.Collection | None:
    """Devuelve la colección si existe, sin crear nada en disco si no existe."""
    if not Path(persist_dir).is_dir():
        return None
    cliente = chromadb.PersistentClient(path=str(persist_dir))
    try:
        return cliente.get_collection(collection_name)
    except Exception:  # NotFoundError o ValueError según la versión de chromadb
        return None


def vectorstore_exists(persist_dir: Path = VECTORSTORE_DIR, collection_name: str = COLLECTION_NAME) -> bool:
    """True si la carpeta existe Y la colección tiene al menos un fragmento."""
    coleccion = _abrir_coleccion(persist_dir, collection_name)
    return coleccion is not None and coleccion.count() > 0


def modelo_indexado(persist_dir: Path = VECTORSTORE_DIR, collection_name: str = COLLECTION_NAME) -> str | None:
    """Nombre del modelo de embeddings con el que se indexó la colección (o None)."""
    coleccion = _abrir_coleccion(persist_dir, collection_name)
    if coleccion is None:
        return None
    return (coleccion.metadata or {}).get("embedding_model")


def build_vectorstore(
    force: bool = False,
    *,
    data_dir: Path = DATA_DIR,
    persist_dir: Path = VECTORSTORE_DIR,
    collection_name: str = COLLECTION_NAME,
    embeddings: Embeddings | None = None,
    embedding_model: str = EMBEDDING_MODEL,
) -> ResultadoIngesta:
    """Indexa `data_dir` en ChromaDB, salvo que la base ya esté poblada.

    - Sin `force`: si la colección existe y tiene fragmentos, no se toca (la ingesta es
      cara: recalcula todos los embeddings).
    - Con `force`: borra la colección y la vuelve a crear desde cero.

    El nombre del modelo de embeddings queda guardado en la metadata de la colección, así
    el retriever puede detectar si se intenta consultar con un modelo distinto.
    """
    if vectorstore_exists(persist_dir, collection_name) and not force:
        cantidad = _abrir_coleccion(persist_dir, collection_name).count()  # type: ignore[union-attr]
        logger.info("Base ya indexada en %s (%d fragmentos): se omite la ingesta", _ruta(persist_dir), cantidad)
        return ResultadoIngesta(indexada=False, documentos=0, fragmentos=cantidad)

    if force and _abrir_coleccion(persist_dir, collection_name) is not None:
        chromadb.PersistentClient(path=str(persist_dir)).delete_collection(collection_name)
        logger.info("--force: se borró la colección %r para reconstruirla", collection_name)

    documentos = load_documents(data_dir)
    fragmentos = split_documents(documentos)

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings or get_embeddings(),
        ids=[_id_de(f) for f in fragmentos],
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
    )
    logger.info(
        "Ingesta completa: %d documentos, %d fragmentos persistidos en %s (colección %r)",
        len(documentos),
        len(fragmentos),
        _ruta(persist_dir),
        collection_name,
    )
    return ResultadoIngesta(indexada=True, documentos=len(documentos), fragmentos=len(fragmentos))
