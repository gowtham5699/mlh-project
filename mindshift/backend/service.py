"""Boundary between the HTTP API and the model/steering implementation."""

from __future__ import annotations

from typing import Protocol

from .schemas import ConceptInfo, GenerateRequest, GenerateResponse


class GenerationService(Protocol):
    """Interface the Phase 1-3 implementation plugs into."""

    @property
    def model_id(self) -> str: ...

    @property
    def device(self) -> str: ...

    def list_concepts(self) -> list[ConceptInfo]: ...

    def generate(self, request: GenerateRequest) -> GenerateResponse: ...


class GenerationServiceError(Exception):
    """Base class for expected generation failures."""


class UnknownConceptError(GenerationServiceError):
    pass


class UnknownLayerError(GenerationServiceError):
    pass


class VectorNotFoundError(GenerationServiceError):
    pass
