"""Discover and load saved activation vectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .schemas import ConceptInfo


class VectorRegistry:
    def __init__(
        self,
        directory: str | Path,
        normalize: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.normalize = normalize
        self._entries: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
        self.refresh()

    def refresh(self) -> None:
        entries: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
        if not self.directory.exists():
            self._entries = entries
            return

        for metadata_path in sorted(self.directory.glob("*_L*.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            concept = metadata.get("concept")
            layer = metadata.get("layer")
            if not isinstance(concept, str) or not isinstance(layer, int):
                raise ValueError(f"Invalid vector metadata: {metadata_path}")

            vector_path = metadata_path.with_suffix(".pt")
            if not vector_path.is_file():
                raise FileNotFoundError(
                    f"Vector file for {metadata_path.name} is missing"
                )
            entries[(concept, layer)] = (vector_path, metadata)

        self._entries = entries

    def concepts(self) -> list[ConceptInfo]:
        layers_by_concept: dict[str, list[int]] = {}
        for concept, layer in self._entries:
            layers_by_concept.setdefault(concept, []).append(layer)
        return [
            ConceptInfo(name=concept, layers=sorted(layers))
            for concept, layers in sorted(layers_by_concept.items())
        ]

    def has(self, concept: str, layer: int) -> bool:
        return (concept, layer) in self._entries

    def metadata(self, concept: str, layer: int) -> dict[str, Any]:
        try:
            return dict(self._entries[(concept, layer)][1])
        except KeyError as exc:
            raise KeyError(f"No vector for concept={concept!r}, layer={layer}") from exc

    def load(self, concept: str, layer: int) -> torch.Tensor:
        try:
            vector_path, metadata = self._entries[(concept, layer)]
        except KeyError as exc:
            raise KeyError(f"No vector for concept={concept!r}, layer={layer}") from exc

        try:
            vector = torch.load(
                vector_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:  # Compatibility with older supported PyTorch versions.
            vector = torch.load(vector_path, map_location="cpu")

        if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
            raise ValueError(f"Invalid vector tensor: {vector_path}")
        if not torch.isfinite(vector).all():
            raise ValueError(f"Vector contains non-finite values: {vector_path}")

        expected_shape = metadata.get("shape")
        if expected_shape != list(vector.shape):
            raise ValueError(f"Vector shape does not match metadata: {vector_path}")

        if not self.normalize:
            return vector

        raw_norm = vector.norm()
        if raw_norm.item() == 0:
            raise ValueError(f"Cannot normalize a zero vector: {vector_path}")
        mean_hidden_norm = metadata.get("mean_hidden_norm")
        if not isinstance(mean_hidden_norm, (int, float)) or mean_hidden_norm <= 0:
            raise ValueError(
                f"Metadata lacks a positive mean_hidden_norm: {vector_path}"
            )
        return vector / raw_norm * float(mean_hidden_norm)
