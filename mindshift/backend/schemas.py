"""Pydantic request and response contracts for the MindShift API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8_000)
    concept: str = Field(default="pirate", min_length=1)
    alpha: float = Field(default=0.8, ge=0.0, le=8.0)
    layer: int = Field(default=12, ge=0)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    seed: int = Field(default=42, ge=0)


class GenerateResponse(BaseModel):
    baseline: str
    steered: str
    concept: str
    alpha: float
    layer: int
    latency_ms: float = Field(ge=0.0)


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
