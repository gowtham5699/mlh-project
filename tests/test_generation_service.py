"""Orchestration tests for ModelGenerationService.

Uses a recording runtime stand-in (test-only) to check how the service drives
baseline/steered/prompted generations; real model behavior is covered by
tests/test_steering.py (hook logic) and scripts/model_smoke_test.py.
"""
from __future__ import annotations

import json

import pytest
import torch

from mindshift.backend.generation_service import ModelGenerationService
from mindshift.backend.schemas import GenerateRequest
from mindshift.backend.service import (
    UnknownConceptError,
    UnknownLayerError,
    VectorNotFoundError,
)
from mindshift.backend.vector_registry import VectorRegistry


class RecordingRuntime:
    model_id = "test/model"
    device = "cpu"
    num_layers = 28

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompt, max_new_tokens=64, seed=0, steering=None, **kwargs):
        self.calls.append(
            {"prompt": prompt, "max_new_tokens": max_new_tokens, "seed": seed, "steering": steering}
        )
        return "steered" if steering is not None else "plain"


@pytest.fixture
def service(tmp_path):
    torch.save(torch.ones(4), tmp_path / "pirate_L12.pt")
    (tmp_path / "pirate_L12.json").write_text(
        json.dumps({"concept": "pirate", "layer": 12, "shape": [4], "mean_hidden_norm": 10.0})
    )
    runtime = RecordingRuntime()
    return ModelGenerationService(runtime, VectorRegistry(tmp_path)), runtime


def test_runs_baseline_and_steered_with_identical_settings(service):
    svc, runtime = service
    response = svc.generate(
        GenerateRequest(prompt="Hi", concept="pirate", alpha=0.5, layer=12, seed=7, max_new_tokens=16)
    )
    assert (response.baseline, response.steered) == ("plain", "steered")
    base, steered = runtime.calls
    assert base["steering"] is None
    assert steered["steering"].alpha == 0.5 and steered["steering"].layer == 12
    assert steered["steering"].vector.norm().item() == pytest.approx(10.0)  # normalized
    assert (base["seed"], base["max_new_tokens"]) == (steered["seed"], steered["max_new_tokens"]) == (7, 16)
    assert response.latency_ms >= 0 and response.strength == 0.5


def test_prompted_column_is_optional(service):
    svc, runtime = service
    svc.generate(GenerateRequest(prompt="Hi", alpha=1.0, layer=12))
    assert len(runtime.calls) == 2
    response = svc.generate(GenerateRequest(prompt="Hi", alpha=1.0, layer=12, include_prompted=True))
    assert len(runtime.calls) == 5
    assert runtime.calls[-1]["steering"] is None
    assert runtime.calls[-1]["prompt"].endswith("Hi") and runtime.calls[-1]["prompt"] != "Hi"
    assert response.prompted == "plain"


def test_error_mapping(service):
    svc, _ = service
    with pytest.raises(UnknownConceptError):
        svc.generate(GenerateRequest(prompt="Hi", concept="robot"))
    with pytest.raises(UnknownLayerError):
        svc.generate(GenerateRequest(prompt="Hi", layer=99))
    with pytest.raises(VectorNotFoundError):
        svc.generate(GenerateRequest(prompt="Hi", layer=8))
