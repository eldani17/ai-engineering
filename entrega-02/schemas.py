"""Contrato de datos del pipeline: qué forma tiene que tener la salida del LLM.

Las `description` de cada campo viajan al modelo dentro del schema, así que
además de documentar le dicen qué completar.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NivelCriticidad(str, Enum):
    """Qué tan grave es lo que describe el texto."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Entidades técnicas extraídas de un texto libre (ticket, log, arquitectura)."""

    model_config = ConfigDict(extra="forbid")

    tecnologias: list[str] = Field(
        min_length=1,
        description=(
            "Tecnologías mencionadas explícitamente en el texto (lenguajes, frameworks, "
            "bases de datos, servicios cloud, herramientas), con su nombre canónico. "
            "Sin duplicados y sin inventar ninguna que no aparezca."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description=(
            "Gravedad del escenario: 'alta' si hay caída o degradación en producción, "
            "pérdida de datos o un problema de seguridad; 'media' si hay un problema real "
            "pero acotado o con workaround; 'baja' si es una mejora, una consulta o una "
            "descripción sin incidente."
        ),
    )
    resumen_tecnico: str = Field(
        min_length=20,
        description=(
            "Resumen técnico de una o dos oraciones, en español, con el problema o la "
            "arquitectura descripta y la causa probable si se menciona."
        ),
    )

    @field_validator("tecnologias", mode="before")
    @classmethod
    def _limpiar_tecnologias(cls, valor: object) -> object:
        """Saca strings vacíos y duplicados (sin distinguir mayúsculas), conservando el orden.

        Corre antes del `min_length=1`, así que una lista como `["", " "]` queda vacía
        y falla la validación en vez de pasar como si tuviera contenido.
        """
        if not isinstance(valor, list):
            return valor
        vistas: set[str] = set()
        limpias: list[object] = []
        for item in valor:
            if not isinstance(item, str):
                limpias.append(item)  # que Pydantic reporte el tipo incorrecto
                continue
            nombre = item.strip()
            if nombre and nombre.casefold() not in vistas:
                vistas.add(nombre.casefold())
                limpias.append(nombre)
        return limpias

    @field_validator("resumen_tecnico", mode="before")
    @classmethod
    def _recortar_resumen(cls, valor: object) -> object:
        return valor.strip() if isinstance(valor, str) else valor
