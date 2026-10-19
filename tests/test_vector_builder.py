from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mindshift.backend.vector_builder import (
    ActivationVectorBuilder,
    PromptPair,
    save_built_vector,
)
from mindshift.backend.vector_registry import VectorRegistry


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        response = messages[-1]["content"]
        final_token = 2 if "pirate" in response else 1
        input_ids = torch.tensor([[1, final_token]])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, **kwargs):
        embedding = input_ids.unsqueeze(-1).repeat(1, 1, 3).float()
        layer_zero = embedding + self.anchor
        return SimpleNamespace(hidden_states=(embedding, layer_zero))


def test_build_save_and_load_normalized_vector(tmp_path) -> None:
    builder = ActivationVectorBuilder(FakeModel(), FakeTokenizer(), "fake/model")
    built = builder.build(
        concept="pirate",
        pairs=[
            PromptPair(
                prompt="Greet someone",
                positive="pirate greeting",
                negative="plain greeting",
            )
        ],
        layer=0,
    )

    assert torch.equal(built.vector, torch.ones(3))
    assert built.metadata["shape"] == [3]
    assert built.metadata["n_pairs"] == 1

    vector_path, metadata_path = save_built_vector(built, tmp_path)
    assert vector_path.is_file()
    assert metadata_path.is_file()

    raw_registry = VectorRegistry(tmp_path, normalize=False)
    assert torch.equal(raw_registry.load("pirate", 0), torch.ones(3))
    assert raw_registry.concepts()[0].layers == [0]

    normalized_registry = VectorRegistry(tmp_path, normalize=True)
    normalized = normalized_registry.load("pirate", 0)
    assert normalized.norm().item() == pytest.approx(
        built.metadata["mean_hidden_norm"]
    )


def test_builder_rejects_invalid_layer() -> None:
    builder = ActivationVectorBuilder(FakeModel(), FakeTokenizer(), "fake/model")

    with pytest.raises(ValueError, match="out of range"):
        builder.build(
            concept="pirate",
            pairs=[PromptPair("Prompt", "pirate response", "plain response")],
            layer=2,
        )
