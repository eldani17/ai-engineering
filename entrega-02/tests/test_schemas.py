import pytest
from pydantic import ValidationError

from schemas import ExtraccionTecnica, NivelCriticidad

VALIDO = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL.",
}


def test_objeto_valido_se_serializa_con_los_tres_campos():
    obj = ExtraccionTecnica.model_validate(VALIDO)
    assert obj.nivel_de_criticidad is NivelCriticidad.ALTA
    assert obj.model_dump(mode="json") == VALIDO


def test_tecnologias_vacia_falla():
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({**VALIDO, "tecnologias": []})


def test_tecnologias_solo_strings_vacios_falla():
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({**VALIDO, "tecnologias": ["", "   "]})


def test_tecnologias_se_normalizan():
    obj = ExtraccionTecnica.model_validate(
        {**VALIDO, "tecnologias": [" Redis ", "redis", "", "PostgreSQL"]}
    )
    assert obj.tecnologias == ["Redis", "PostgreSQL"]


def test_criticidad_fuera_del_enum_falla():
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({**VALIDO, "nivel_de_criticidad": "urgente"})


def test_resumen_demasiado_corto_falla():
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({**VALIDO, "resumen_tecnico": "Se cayó."})


def test_campo_faltante_falla():
    incompleto = {k: v for k, v in VALIDO.items() if k != "resumen_tecnico"}
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate(incompleto)


def test_campo_extra_falla():
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({**VALIDO, "autor": "alguien"})
