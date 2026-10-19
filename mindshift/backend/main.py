"""FastAPI application for MindShift."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request

from .schemas import (
    ConceptsResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from .service import (
    GenerationService,
    UnknownConceptError,
    UnknownLayerError,
    VectorNotFoundError,
)


logger = logging.getLogger(__name__)
ServiceFactory = Callable[[], GenerationService]


def create_app(
    service: GenerationService | None = None,
    service_factory: ServiceFactory | None = None,
) -> FastAPI:
    """Create the API with either a test service or a load-on-start factory."""

    if service is not None and service_factory is not None:
        raise ValueError("Provide service or service_factory, not both")

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        if service_factory is not None:
            api.state.generation_service = service_factory()
        try:
            yield
        finally:
            if service_factory is not None:
                api.state.generation_service = None

    api = FastAPI(
        title="MindShift API",
        version="0.1.0",
        lifespan=lifespan,
    )
    api.state.generation_service = service

    def require_service(request: Request) -> GenerationService:
        current_service = request.app.state.generation_service
        if current_service is None:
            raise HTTPException(
                status_code=503,
                detail="Model runtime is not loaded",
            )
        return current_service

    @api.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        current_service = require_service(request)
        concepts = current_service.list_concepts()
        return HealthResponse(
            status="ok",
            model_id=current_service.model_id,
            device=current_service.device,
            loaded_concepts=[concept.name for concept in concepts],
        )

    @api.get("/concepts", response_model=ConceptsResponse)
    def concepts(request: Request) -> ConceptsResponse:
        current_service = require_service(request)
        return ConceptsResponse(concepts=current_service.list_concepts())

    @api.post("/generate", response_model=GenerateResponse)
    def generate(
        payload: GenerateRequest,
        request: Request,
    ) -> GenerateResponse:
        current_service = require_service(request)
        try:
            return current_service.generate(payload)
        except (UnknownConceptError, UnknownLayerError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except VectorNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected generation failure")
            raise HTTPException(
                status_code=500,
                detail="Generation failed",
            ) from exc

    return api


def _load_service() -> GenerationService:
    # Imported lazily so importing this module (e.g. in tests) does not pull in torch.
    from .generation_service import build_generation_service

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return build_generation_service()


# `uvicorn mindshift.backend.main:app` loads the model once at startup.
app = create_app(service_factory=_load_service)
