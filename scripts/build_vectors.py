"""Build activation vectors from paired examples."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindshift.backend.vector_builder import (
    ActivationVectorBuilder,
    load_prompt_pairs,
    save_built_vector,
)


def parse_layers(value: str) -> list[int]:
    try:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Layers must be comma-separated integers") from exc
    if not layers or any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError("At least one non-negative layer is required")
    return layers


def choose_device(requested: str | None) -> tuple[torch.device, torch.dtype]:
    if requested:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device.type == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=os.getenv("MODEL_ID", "Qwen/Qwen3-1.7B"),
    )
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("12"))
    parser.add_argument(
        "--pairs",
        type=Path,
        default=ROOT / "mindshift" / "data" / "pirate_pairs.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "mindshift" / "vectors",
    )
    parser.add_argument("--pooling", choices=["last", "mean", "response"], default="response")
    parser.add_argument("--device", default=os.getenv("DEVICE"))
    args = parser.parse_args()

    concept, pairs = load_prompt_pairs(args.pairs)
    device, dtype = choose_device(args.device)
    print(f"Loading {args.model_id} on {device} ({dtype})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    try:  # transformers >= 4.56 / 5.x
        model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    builder = ActivationVectorBuilder(model, tokenizer, args.model_id)
    print(f"Building {concept!r} vectors for layers {args.layers} ({args.pooling} pooling)")
    built_by_layer = builder.build_layers(
        concept=concept,
        pairs=pairs,
        layers=args.layers,
        pooling=args.pooling,
    )
    for layer, built in built_by_layer.items():
        vector_path, metadata_path = save_built_vector(built, args.output)
        print(
            f"Saved {vector_path.name} (layer {layer}, raw norm="
            f"{built.metadata['raw_norm']:.2f}, mean token norm="
            f"{built.metadata['mean_hidden_norm']:.2f})"
        )

if __name__ == "__main__":
    main()
