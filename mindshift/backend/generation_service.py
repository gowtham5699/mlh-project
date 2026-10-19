"""Real GenerationService: model runtime + steering vectors behind the HTTP API."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from .model_runtime import ModelRuntime, SteeringSpec
from .schemas import ConceptInfo, GenerateRequest, GenerateResponse
from .service import UnknownConceptError, UnknownLayerError, VectorNotFoundError
from .vector_registry import VectorRegistry

logger = logging.getLogger("mindshift.service")

DEFAULT_VECTOR_DIR = Path(__file__).resolve().parents[1] / "vectors"

# Instruction used ONLY for the optional "prompted" comparison column
# (prompt engineering, no activation steering).
PROMPT_STYLES = {
    "pirate": "Answer the following in the speech style of a pirate.\n\n",
}


class ModelGenerationService:
    """Runs baseline and steered generations sequentially with identical settings."""

    def __init__(self, runtime: Any, registry: VectorRegistry) -> None:
        self.runtime = runtime
        self.registry = registry

    @property
    def model_id(self) -> str:
        return self.runtime.model_id

    @property
    def device(self) -> str:
        return str(self.runtime.device)

    def list_concepts(self) -> list[ConceptInfo]:
        return self.registry.concepts()

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        concept_names = {c.name for c in self.registry.concepts()}
        if request.concept not in concept_names:
            raise UnknownConceptError(
                f"Unknown concept {request.concept!r}; available: {sorted(concept_names)}"
            )
        if request.layer >= self.runtime.num_layers:
            raise UnknownLayerError(
                f"Layer {request.layer} is out of range for a model with "
                f"{self.runtime.num_layers} layers"
            )
        if not self.registry.has(request.concept, request.layer):
            available = next(c.layers for c in self.registry.concepts() if c.name == request.concept)
            raise VectorNotFoundError(
                f"No {request.concept!r} vector for layer {request.layer}; "
                f"available layers: {available}"
            )

        vector = self.registry.load(request.concept, request.layer)
        spec = SteeringSpec(layer=request.layer, vector=vector, alpha=request.alpha)
        settings = dict(max_new_tokens=request.max_new_tokens, seed=request.seed)

        start = time.perf_counter()
        baseline = self.runtime.generate(request.prompt, **settings)
        steered = self.runtime.generate(request.prompt, steering=spec, **settings)
        prompted = None
        if request.include_prompted:
            prefix = PROMPT_STYLES.get(request.concept, "")
            prompted = self.runtime.generate(prefix + request.prompt, **settings)
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "generate concept=%s layer=%d alpha=%.2f latency=%.0fms",
            request.concept, request.layer, request.alpha, latency_ms,
        )
        return GenerateResponse(
            baseline=baseline,
            steered=steered,
            prompted=prompted,
            concept=request.concept,
            alpha=request.alpha,
            layer=request.layer,
            latency_ms=latency_ms,
        )


def build_generation_service() -> ModelGenerationService:
    """Service factory used at FastAPI startup (loads the model once)."""
    runtime = ModelRuntime()
    registry = VectorRegistry(
        os.getenv("VECTOR_DIR") or str(DEFAULT_VECTOR_DIR),
        normalize=os.getenv("NORMALIZE_VECTORS", "true").lower() != "false",
    )
    logger.info("Loaded concepts: %s", [c.name for c in registry.concepts()])
    return ModelGenerationService(runtime, registry)
