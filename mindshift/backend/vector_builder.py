"""Build mean-difference activation steering vectors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

import torch


Pooling = Literal["last", "mean"]


@dataclass(frozen=True)
class PromptPair:
    prompt: str
    positive: str
    negative: str


@dataclass(frozen=True)
class BuiltVector:
    vector: torch.Tensor
    metadata: dict[str, Any]


def load_prompt_pairs(path: str | Path) -> tuple[str, list[PromptPair]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    concept = payload.get("concept")
    raw_pairs = payload.get("pairs")
    if not isinstance(concept, str) or not concept.strip():
        raise ValueError("Prompt-pair file must contain a non-empty concept")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("Prompt-pair file must contain at least one pair")

    pairs: list[PromptPair] = []
    for index, item in enumerate(raw_pairs):
        if not isinstance(item, dict):
            raise ValueError(f"Pair {index} must be an object")
        values = [item.get(key) for key in ("prompt", "positive", "negative")]
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError(
                f"Pair {index} needs non-empty prompt, positive, and negative text"
            )
        pairs.append(PromptPair(*values))
    return concept, pairs


class ActivationVectorBuilder:
    def __init__(self, model: Any, tokenizer: Any, model_id: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id

    def _encode_conversation(
        self,
        prompt: str,
        response: str,
    ) -> dict[str, torch.Tensor]:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )
        if isinstance(encoded, torch.Tensor):
            encoded = {
                "input_ids": encoded,
                "attention_mask": torch.ones_like(encoded),
            }
        if not isinstance(encoded, dict) or "input_ids" not in encoded:
            raise TypeError("Tokenizer chat template returned unsupported inputs")

        device = next(self.model.parameters()).device
        return {
            key: value.to(device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }

    def _pooled_activation(
        self,
        prompt: str,
        response: str,
        layer: int,
        pooling: Pooling,
    ) -> torch.Tensor:
        inputs = self._encode_conversation(prompt, response)
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states")
        if layer < 0 or layer + 1 >= len(hidden_states):
            raise ValueError(
                f"Layer {layer} is out of range for "
                f"{len(hidden_states) - 1} decoder layers"
            )

        # hidden_states[0] is the embedding output. hidden_states[L+1] is
        # decoder layer L's output, matching a forward hook on that layer.
        hidden = hidden_states[layer + 1]
        attention_mask = inputs.get("attention_mask")

        if pooling == "last":
            if attention_mask is None:
                pooled = hidden[:, -1, :]
            else:
                last_indices = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
                pooled = hidden[batch_indices, last_indices]
        elif pooling == "mean":
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")

        return pooled.squeeze(0).detach().float().cpu()

    def build(
        self,
        concept: str,
        pairs: list[PromptPair],
        layer: int,
        pooling: Pooling = "last",
    ) -> BuiltVector:
        if not pairs:
            raise ValueError("At least one prompt pair is required")

        positive = [
            self._pooled_activation(pair.prompt, pair.positive, layer, pooling)
            for pair in pairs
        ]
        negative = [
            self._pooled_activation(pair.prompt, pair.negative, layer, pooling)
            for pair in pairs
        ]
        positive_stack = torch.stack(positive)
        negative_stack = torch.stack(negative)
        vector = positive_stack.mean(dim=0) - negative_stack.mean(dim=0)
        all_activations = torch.cat([positive_stack, negative_stack], dim=0)

        metadata = {
            "concept": concept,
            "model_id": self.model_id,
            "layer": layer,
            "shape": list(vector.shape),
            "method": "mean_difference",
            "pooling": pooling,
            "n_pairs": len(pairs),
            "raw_norm": float(vector.norm().item()),
            "mean_hidden_norm": float(
                all_activations.norm(dim=1).mean().item()
            ),
            "normalization": "unit_vector_scaled_by_mean_hidden_norm_at_load",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return BuiltVector(vector=vector, metadata=metadata)


def save_built_vector(
    built: BuiltVector,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    concept = str(built.metadata["concept"])
    safe_concept = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in concept
    )
    layer = int(built.metadata["layer"])
    vector_path = output_directory / f"{safe_concept}_L{layer}.pt"
    metadata_path = output_directory / f"{safe_concept}_L{layer}.json"

    torch.save(built.vector.cpu(), vector_path)
    metadata_path.write_text(
        json.dumps(built.metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return vector_path, metadata_path
