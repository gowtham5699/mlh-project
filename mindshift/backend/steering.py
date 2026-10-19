"""Activation steering via PyTorch forward hooks.

Formula applied to the residual stream output of one decoder layer:

    modified_hidden = hidden + alpha * steering_vector

No weights are modified; the hook is registered for the duration of a single
generation and always removed afterwards.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import nn

logger = logging.getLogger("mindshift.steering")

MODE_ALL = "all"  # add to every position (prompt + generated tokens)
MODE_GENERATED = "generated"  # add only to positions that produce new tokens


class SteeringError(ValueError):
    """Raised for invalid steering configuration."""


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the list of transformer decoder layers (Qwen/Llama style)."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise SteeringError("Could not locate decoder layers at model.model.layers")
    return layers


def apply_steering(hidden: torch.Tensor, vector: torch.Tensor, alpha: float,
                   mode: str = MODE_ALL) -> torch.Tensor:
    """Return hidden + alpha * vector, preserving shape, dtype and device.

    hidden: (batch, seq, hidden_size); vector: (hidden_size,).
    """
    if hidden.dim() != 3:
        raise SteeringError(f"Expected hidden of shape (batch, seq, hidden); got {tuple(hidden.shape)}")
    if vector.dim() != 1 or vector.shape[0] != hidden.shape[-1]:
        raise SteeringError(
            f"Steering vector shape {tuple(vector.shape)} incompatible with hidden size {hidden.shape[-1]}"
        )
    delta = (alpha * vector).to(device=hidden.device, dtype=hidden.dtype)
    if mode == MODE_GENERATED and hidden.shape[1] > 1:
        # Prefill: only the last prompt position produces the first new token.
        out = hidden.clone()
        out[:, -1, :] = out[:, -1, :] + delta
        return out
    return hidden + delta


class SteeringHook:
    """Context manager that steers one decoder layer during generation.

    Usage:
        with SteeringHook(model, layer=12, vector=v, alpha=1.0):
            model.generate(...)
    """

    def __init__(self, model: nn.Module, layer: int, vector: torch.Tensor,
                 alpha: float, mode: str = MODE_ALL):
        self.layers = get_decoder_layers(model)
        if not (0 <= layer < len(self.layers)):
            raise SteeringError(f"layer {layer} out of range [0, {len(self.layers) - 1}]")
        if mode not in (MODE_ALL, MODE_GENERATED):
            raise SteeringError(f"unknown mode {mode!r}")
        if not isinstance(vector, torch.Tensor):
            vector = torch.as_tensor(vector)
        hidden_size = model.config.hidden_size
        if vector.dim() != 1 or vector.shape[0] != hidden_size:
            raise SteeringError(
                f"vector must have shape ({hidden_size},); got {tuple(vector.shape)}"
            )
        if not torch.isfinite(vector).all():
            raise SteeringError("vector contains NaN or Inf")
        if not isinstance(alpha, (int, float)) or alpha != alpha or abs(alpha) == float("inf"):
            raise SteeringError("alpha must be a finite number")
        self.layer = layer
        self.alpha = float(alpha)
        self.mode = mode
        self.vector = vector.detach()
        self._handle: Optional[Any] = None
        self.calls = 0

    def _hook(self, module: nn.Module, inputs: tuple, output: Any) -> Any:
        if self.alpha == 0.0:
            return None  # leave output untouched: exact baseline behavior
        self.calls += 1
        if isinstance(output, tuple):
            steered = apply_steering(output[0], self.vector, self.alpha, self.mode)
            return (steered,) + tuple(output[1:])
        if isinstance(output, torch.Tensor):
            return apply_steering(output, self.vector, self.alpha, self.mode)
        raise SteeringError(f"Unsupported layer output type: {type(output)}")

    def __enter__(self) -> "SteeringHook":
        if self._handle is not None:
            raise SteeringError("hook already registered")
        self._handle = self.layers[self.layer].register_forward_hook(self._hook)
        logger.debug("steering hook registered layer=%d alpha=%.3f mode=%s",
                     self.layer, self.alpha, self.mode)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        logger.debug("steering hook removed layer=%d (calls=%d)", self.layer, self.calls)


def count_hooks(model: nn.Module) -> int:
    """Number of SteeringHook hooks still registered on decoder layers.

    Transformers registers its own hooks (e.g. for output_hidden_states), so only
    hooks owned by SteeringHook are counted.
    """
    return sum(
        1
        for layer in get_decoder_layers(model)
        for hook in layer._forward_hooks.values()
        if isinstance(getattr(hook, "__self__", None), SteeringHook)
    )


class LayerOutputCapture:
    """Context manager recording a layer's output hidden states.

    Register it AFTER entering SteeringHook to observe the steered activations
    (forward hooks run in registration order; transformers' own
    output_hidden_states capture runs before steering and shows unsteered values).
    """

    def __init__(self, model: nn.Module, layer: int):
        self.module = get_decoder_layers(model)[layer]
        self.outputs: list = []
        self._handle = None

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.outputs.append(hidden.detach().clone())

    def __enter__(self) -> "LayerOutputCapture":
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
