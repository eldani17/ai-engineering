"""Tests offline de la ingesta y la recuperación (embeddings falsos, ChromaDB en tmp_path)."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.config import CHUNK_SIZE
from src.ingest import build_vectorstore, load_documents, split_documents, vectorstore_exists
from src.retriever import BaseNoIndexadaError, EmbeddingsNoCoincidentesError, get_retriever, get_vectorstore

MODELO = "fake-embeddings"


@pytest.fixture
def datos(tmp_path):
    carpeta = tmp_path / "data"
    carpeta.mkdir()
    secciones = "\n\n".join(f"## Sección {i}\n\n" + ("Texto de la sección %d. " % i) * 40 for i in range(1, 5))
    (carpeta / "a.md").write_text("# Documento A\n\n" + secciones, encoding="utf-8")
    (carpeta / "b.txt").write_text("Un archivo de texto plano corto.", encoding="utf-8")
    (carpeta / "vacio.md").write_text("   ", encoding="utf-8")
    (carpeta / "ignorar.pdf").write_text("no se indexa", encoding="utf-8")
    return carpeta


@pytest.fixture
def opciones(datos, tmp_path):
    """Parámetros comunes: todo en tmp_path y con embeddings falsos (sin descargar el modelo)."""
    return {
        "persist_dir": tmp_path / "vectorstore",
        "collection_name": "test",
        "embeddings": DeterministicFakeEmbedding(size=32),
        "embedding_model": MODELO,
    }


def _ingestar(datos, opciones, force=False):
    return build_vectorstore(force, data_dir=datos, **opciones)


def test_load_documents_lee_md_y_txt_con_source(datos):
    documentos = load_documents(datos)
    assert [d.metadata["source"] for d in documentos] == ["a.md", "b.txt"]  # sin vacíos ni .pdf


def test_load_documents_falla_si_no_hay_carpeta(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "no-existe")


def test_split_respeta_el_tamano_y_agrega_metadata(datos):
    fragmentos = split_documents(load_documents(datos))
    del_a = [f for f in fragmentos if f.metadata["source"] == "a.md"]
    assert len(del_a) > 1
    assert all(len(f.page_content) <= CHUNK_SIZE for f in fragmentos)
    assert [f.metadata["chunk_index"] for f in del_a] == list(range(len(del_a)))
    assert del_a[-1].metadata["seccion"] == "Sección 4"


def test_vectorstore_exists_no_crea_la_carpeta(tmp_path):
    destino = tmp_path / "vectorstore"
    assert vectorstore_exists(destino, "test") is False
    assert not destino.exists()


def test_la_segunda_ingesta_se_omite(datos, opciones):
    primera = _ingestar(datos, opciones)
    segunda = _ingestar(datos, opciones)
    assert primera.indexada and primera.documentos == 2
    assert not segunda.indexada
    assert segunda.fragmentos == primera.fragmentos
    assert vectorstore_exists(opciones["persist_dir"], "test")


def test_force_reconstruye_sin_duplicar(datos, opciones):
    primera = _ingestar(datos, opciones)
    forzada = _ingestar(datos, opciones, force=True)
    assert forzada.indexada
    assert get_vectorstore(**opciones)._collection.count() == primera.fragmentos


def test_retriever_devuelve_k_fragmentos(datos, opciones):
    _ingestar(datos, opciones)
    documentos = get_retriever(k=3, **opciones).invoke("sección 2")
    assert len(documentos) == 3
    assert all(isinstance(d, Document) and "source" in d.metadata for d in documentos)


@pytest.mark.parametrize("k", [2, 6, 50])
def test_retriever_rechaza_k_fuera_de_rango(k, datos, opciones):
    _ingestar(datos, opciones)
    with pytest.raises(ValueError):
        get_retriever(k=k, **opciones)


def test_consultar_sin_base_avisa(opciones):
    with pytest.raises(BaseNoIndexadaError):
        get_vectorstore(**opciones)


def test_consultar_con_otro_modelo_de_embeddings_falla(datos, opciones):
    _ingestar(datos, opciones)
    with pytest.raises(EmbeddingsNoCoincidentesError):
        get_vectorstore(**{**opciones, "embedding_model": "otro-modelo"})
