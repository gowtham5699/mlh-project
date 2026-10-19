"""Pydantic request and response contracts for the MindShift API."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    # "strength" is accepted as an alias for "alpha" (original API spec).
    model_config = ConfigDict(populate_by_name=True)

    prompt: str = Field(min_length=1, max_length=8_000)
    concept: str = Field(default="pirate", min_length=1)
    alpha: float = Field(
        default=0.6,
        ge=0.0,
        le=4.0,
        validation_alias=AliasChoices("alpha", "strength"),
    )
    layer: int = Field(default=11, ge=0)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    seed: int = Field(default=42, ge=0)
    include_prompted: bool = Field(
        default=False,
        description="Also generate an unsteered, prompt-engineered answer for comparison.",
    )


class GenerateResponse(BaseModel):
    baseline: str
    steered: str
    prompted: str | None = None
    concept: str
    alpha: float
    strength: float | None = None
    layer: int
    latency_ms: float = Field(ge=0.0)

    def model_post_init(self, __context) -> None:  # noqa: ANN001
        if self.strength is None:
            self.strength = self.alpha


class HealthResponse(BaseModel):
    status: str
    model_id: str
    device: str
    loaded_concepts: list[str] = Field(default_factory=list)


class ConceptInfo(BaseModel):
    name: str
    layers: list[int] = Field(default_factory=list)
    description: str | None = None


class ConceptsResponse(BaseModel):
    concepts: list[ConceptInfo] = Field(default_factory=list)
