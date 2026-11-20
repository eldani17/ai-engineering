"""Instancia única de embeddings, compartida por la ingesta y la consulta.

Indexar con un modelo y consultar con otro deja los vectores en espacios distintos y la
búsqueda devuelve cualquier cosa. Para que eso no pueda pasar, `ingest.py` y `retriever.py`
obtienen los embeddings solo a través de `get_embeddings()`.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

# Menos ruido en la consola: sin barras de progreso ni avisos de HuggingFace al cargar el modelo.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
# torch avisa en macOS que no soporta redirects (irrelevante acá); el filtro sobrevive a su setLevel.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").addFilter(lambda _registro: False)

from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from src.config import EMBEDDING_MODEL, PREFIJO_CONSULTA, PREFIJO_DOCUMENTO

logger = logging.getLogger("rag")


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Devuelve siempre la misma instancia (el modelo se carga en memoria una sola vez).

    - `normalize_embeddings=True` deja los vectores con norma 1, así la distancia coseno de
      ChromaDB es directamente comparable entre consultas.
    - E5 espera el prefijo "passage: " en los documentos y "query: " en las preguntas.
    """
    logger.info("Cargando el modelo de embeddings %s", EMBEDDING_MODEL)
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"prompt": PREFIJO_DOCUMENTO, "normalize_embeddings": True},
        query_encode_kwargs={"prompt": PREFIJO_CONSULTA, "normalize_embeddings": True},
    )
