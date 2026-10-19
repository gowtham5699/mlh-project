"""Build mean-difference activation steering vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

import torch


# "last":     hidden state of the final token.
# "mean":     mean over every token (prompt + response, incl. the attention sink).
# "response": mean over the assistant response tokens only. The shared prompt
#             cancels in pos - neg anyway, and excluding it (and the massive
#             first-token "attention sink" activation) gives a cleaner direction
#             and a realistic per-token hidden norm.
Pooling = Literal["last", "mean", "response"]


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


def _as_input_dict(encoded: Any) -> Mapping:
    if isinstance(encoded, torch.Tensor):
        return {
            "input_ids": encoded,
            "attention_mask": torch.ones_like(encoded),
        }
    # transformers >= 5 returns a BatchEncoding (a UserDict), not a dict.
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError("Tokenizer chat template returned unsupported inputs")
    return encoded


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
        encoded = _as_input_dict(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True,
            )
        )

        device = next(self.model.parameters()).device
        return {
            key: value.to(device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }

    def _prompt_length(self, prompt: str) -> int:
        """Token count of the generation prompt, as used at inference time."""
        encoded = _as_input_dict(
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,  # matches ModelRuntime.build_inputs
            )
        )
        return int(encoded["input_ids"].shape[1])

    def _pooled_activations(
        self,
        prompt: str,
        response: str,
        layers: Sequence[int],
        pooling: Pooling,
    ) -> dict[int, tuple[torch.Tensor, float]]:
        """Return {layer: (pooled activation, mean per-token norm)}."""
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
        for layer in layers:
            if layer < 0 or layer + 1 >= len(hidden_states):
                raise ValueError(
                    f"Layer {layer} is out of range for "
                    f"{len(hidden_states) - 1} decoder layers"
                )

        attention_mask = inputs.get("attention_mask")
        if pooling not in ("last", "mean", "response"):
            raise ValueError(f"Unsupported pooling method: {pooling}")
        if pooling == "response":
            seq_len = inputs["input_ids"].shape[1]
            start = min(self._prompt_length(prompt), seq_len - 1)

        result: dict[int, tuple[torch.Tensor, float]] = {}
        for layer in layers:
            # hidden_states[0] is the embedding output. hidden_states[L+1] is
            # decoder layer L's output, matching a forward hook on that layer.
            hidden = hidden_states[layer + 1]

            if pooling == "last":
                if attention_mask is None:
                    pooled = hidden[:, -1, :]
                else:
                    last_indices = attention_mask.sum(dim=1) - 1
                    batch_indices = torch.arange(
                        hidden.shape[0], device=hidden.device
                    )
                    pooled = hidden[batch_indices, last_indices]
                token_norm = pooled.float().norm(dim=-1).mean().item()
            elif pooling == "mean":
                if attention_mask is None:
                    pooled = hidden.mean(dim=1)
                else:
                    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                token_norm = hidden.float().norm(dim=-1).mean().item()
            else:  # response
                tokens = hidden[:, start:, :]
                pooled = tokens.mean(dim=1)
                token_norm = tokens.float().norm(dim=-1).mean().item()

            result[layer] = (pooled.squeeze(0).detach().float().cpu(), token_norm)
        return result

    def _pooled_activation(
        self,
        prompt: str,
        response: str,
        layer: int,
        pooling: Pooling,
    ) -> torch.Tensor:
        return self._pooled_activations(prompt, response, [layer], pooling)[layer][0]

    def build_layers(
        self,
        concept: str,
        pairs: list[PromptPair],
        layers: Sequence[int],
        pooling: Pooling = "last",
    ) -> dict[int, BuiltVector]:
        """Build one vector per layer using a single forward pass per example."""
        if not pairs:
            raise ValueError("At least one prompt pair is required")
        layers = list(layers)
        if not layers:
            raise ValueError("At least one layer is required")

        positive: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
        negative: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
        norms: dict[int, list[float]] = {l: [] for l in layers}
        for pair in pairs:
            for text, store in ((pair.positive, positive), (pair.negative, negative)):
                acts = self._pooled_activations(pair.prompt, text, layers, pooling)
                for layer, (pooled, token_norm) in acts.items():
                    store[layer].append(pooled)
                    norms[layer].append(token_norm)

        built: dict[int, BuiltVector] = {}
        for layer in layers:
            vector = (
                torch.stack(positive[layer]).mean(dim=0)
                - torch.stack(negative[layer]).mean(dim=0)
            )
            metadata = {
                "concept": concept,
                "model_id": self.model_id,
                "layer": layer,
                "shape": list(vector.shape),
                "method": "mean_difference",
                "pooling": pooling,
                "n_pairs": len(pairs),
                "raw_norm": float(vector.norm().item()),
                # Mean per-token hidden norm over the pooled tokens. The
                # registry rescales the unit direction to this value so that
                # alpha=1 adds roughly one typical token's worth of norm.
                "mean_hidden_norm": float(sum(norms[layer]) / len(norms[layer])),
                "normalization": "unit_vector_scaled_by_mean_hidden_norm_at_load",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            built[layer] = BuiltVector(vector=vector, metadata=metadata)
        return built

    def build(
        self,
        concept: str,
        pairs: list[PromptPair],
        layer: int,
        pooling: Pooling = "last",
    ) -> BuiltVector:
        return self.build_layers(concept, pairs, [layer], pooling)[layer]


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
