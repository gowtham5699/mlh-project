from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from mindshift.backend.main import create_app
from mindshift.backend.schemas import (
    ConceptInfo,
    GenerateRequest,
    GenerateResponse,
)
from mindshift.backend.service import (
    UnknownConceptError,
    UnknownLayerError,
    VectorNotFoundError,
)


class FakeGenerationService:
    model_id = "test/model"
    device = "cpu"

    def __init__(self) -> None:
        self.failure: Exception | None = None
        self.last_request: GenerateRequest | None = None

    def list_concepts(self) -> list[ConceptInfo]:
        return [
            ConceptInfo(
                name="pirate",
                layers=[8, 12, 16],
                description="Pirate speaking style",
            )
        ]

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.last_request = request
        if self.failure is not None:
            raise self.failure
        return GenerateResponse(
            baseline="Plain response",
            steered="Arrr, steered response!",
            concept=request.concept,
            alpha=request.alpha,
            layer=request.layer,
            latency_ms=12.5,
        )


@pytest.fixture
def service() -> FakeGenerationService:
    return FakeGenerationService()


@pytest.fixture
def client(service: FakeGenerationService) -> TestClient:
    return TestClient(create_app(service=service))


def test_health_reports_runtime_and_loaded_concepts(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_id": "test/model",
        "device": "cpu",
        "loaded_concepts": ["pirate"],
    }


def test_concepts_returns_available_layers(client: TestClient) -> None:
    response = client.get("/concepts")

    assert response.status_code == 200
    assert response.json() == {
        "concepts": [
            {
                "name": "pirate",
                "layers": [8, 12, 16],
                "description": "Pirate speaking style",
            }
        ]
    }


def test_generate_passes_validated_request_to_service(
    client: TestClient,
    service: FakeGenerationService,
) -> None:
    response = client.post(
        "/generate",
        json={
            "prompt": "Explain gravity.",
            "concept": "pirate",
            "alpha": 0.8,
            "layer": 12,
            "max_new_tokens": 64,
            "seed": 7,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "baseline": "Plain response",
        "steered": "Arrr, steered response!",
        "concept": "pirate",
        "alpha": 0.8,
        "strength": 0.8,
        "prompted": None,
        "layer": 12,
        "latency_ms": 12.5,
    }
    assert service.last_request is not None
    assert service.last_request.prompt == "Explain gravity."
    assert service.last_request.seed == 7


def test_generate_accepts_strength_alias(
    client: TestClient,
    service: FakeGenerationService,
) -> None:
    response = client.post(
        "/generate",
        json={"prompt": "Explain gravity", "concept": "pirate", "strength": 2.0, "layer": 12},
    )

    assert response.status_code == 200
    assert service.last_request is not None
    assert service.last_request.alpha == 2.0
    assert response.json()["strength"] == 2.0


@pytest.mark.parametrize(
    ("failure", "status_code"),
    [
        (UnknownConceptError("Unknown concept: robot"), 400),
        (UnknownLayerError("Layer 99 is out of range"), 400),
        (VectorNotFoundError("No pirate vector for layer 8"), 404),
    ],
)
def test_generate_maps_expected_errors(
    client: TestClient,
    service: FakeGenerationService,
    failure: Exception,
    status_code: int,
) -> None:
    service.failure = failure

    response = client.post(
        "/generate",
        json={"prompt": "Hello", "concept": "pirate"},
    )

    assert response.status_code == status_code
    assert response.json() == {"detail": str(failure)}


def test_generate_rejects_invalid_payload(client: TestClient) -> None:
    response = client.post(
        "/generate",
        json={"prompt": "", "alpha": -1, "layer": -1},
    )

    assert response.status_code == 422


def test_routes_return_503_without_runtime() -> None:
    client = TestClient(create_app())

    for method, path in [
        ("get", "/health"),
        ("get", "/concepts"),
        ("post", "/generate"),
    ]:
        response = getattr(client, method)(
            path,
            **({"json": {"prompt": "Hello"}} if method == "post" else {}),
        )
        assert response.status_code == 503


def test_service_factory_loads_during_lifespan(
    service: FakeGenerationService,
) -> None:
    app = create_app(service_factory=lambda: service)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert app.state.generation_service is None
