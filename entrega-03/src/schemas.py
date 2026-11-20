"""Contrato de la respuesta del sistema RAG (lo que valida el `PydanticOutputParser`)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RESPUESTA_SIN_CONTEXTO = "No lo sé"


class RAGAnswer(BaseModel):
    """Respuesta grounded: o sale del contexto con sus fuentes, o es un "No lo sé" sin fuentes."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        min_length=1,
        description=(
            "Respuesta a la pregunta, redactada en español usando SOLO el contexto. "
            f"Si el contexto no alcanza para responder, debe empezar exactamente con '{RESPUESTA_SIN_CONTEXTO}'."
        ),
    )
    found_in_context: bool = Field(
        description="true si la respuesta sale del contexto; false si el contexto no la contiene."
    )
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "Nombres de archivo (campo 'fuente' del contexto) de donde salió la información, "
            "por ejemplo '02_guia_despliegue.md'. Lista vacía si found_in_context es false."
        ),
    )
    confidence: Literal["alta", "media", "baja"] = Field(
        description=(
            "alta: el contexto lo dice explícitamente; media: hubo que combinar fragmentos; "
            "baja: el contexto lo cubre solo en parte. Si found_in_context es false, usar 'baja'."
        )
    )

    @field_validator("sources")
    @classmethod
    def _sin_duplicados(cls, fuentes: list[str]) -> list[str]:
        return list(dict.fromkeys(f.strip() for f in fuentes if f.strip()))

    @model_validator(mode="after")
    def _coherencia(self) -> RAGAnswer:
        """Las dos formas válidas de responder: con contexto y fuentes, o "No lo sé" sin fuentes."""
        dice_no_se = self.answer.strip().lower().startswith(RESPUESTA_SIN_CONTEXTO.lower())
        if self.found_in_context:
            if not self.sources:
                raise ValueError("found_in_context=true requiere al menos una fuente en 'sources'")
            if dice_no_se:
                raise ValueError(f"found_in_context=true no puede responder '{RESPUESTA_SIN_CONTEXTO}'")
        else:
            if self.sources:
                raise ValueError("found_in_context=false no puede citar fuentes")
            if not dice_no_se:
                raise ValueError(f"found_in_context=false requiere que 'answer' empiece con '{RESPUESTA_SIN_CONTEXTO}'")
        return self
