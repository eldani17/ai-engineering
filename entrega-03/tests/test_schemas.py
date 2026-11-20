"""Tests del contrato RAGAnswer: solo hay dos formas válidas de responder."""

import pytest
from pydantic import ValidationError

from src.schemas import RAGAnswer


def test_respuesta_con_contexto_y_fuentes():
    r = RAGAnswer(answer="Canary con 10 %.", found_in_context=True, sources=["a.md", "a.md", " "], confidence="alta")
    assert r.sources == ["a.md"]  # sin duplicados ni vacíos


def test_no_lo_se_sin_fuentes():
    r = RAGAnswer(answer="No lo sé. No tengo acceso a esa información.", found_in_context=False, confidence="baja")
    assert r.sources == []


@pytest.mark.parametrize(
    "datos",
    [
        {"answer": "Canary.", "found_in_context": True, "sources": [], "confidence": "alta"},
        {"answer": "No lo sé.", "found_in_context": True, "sources": ["a.md"], "confidence": "alta"},
        {"answer": "No lo sé.", "found_in_context": False, "sources": ["a.md"], "confidence": "baja"},
        {"answer": "Canberra.", "found_in_context": False, "sources": [], "confidence": "baja"},
        {"answer": "Canary.", "found_in_context": True, "sources": ["a.md"], "confidence": "altísima"},
        {"answer": "Canary.", "found_in_context": True, "sources": ["a.md"], "confidence": "alta", "extra": 1},
    ],
    ids=["contexto-sin-fuentes", "contexto-pero-no-se", "no-se-con-fuentes", "fuera-sin-no-se", "confianza-invalida", "campo-extra"],
)
def test_respuestas_incoherentes_se_rechazan(datos):
    with pytest.raises(ValidationError):
        RAGAnswer(**datos)
