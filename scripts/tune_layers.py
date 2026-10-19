"""Sweep (layer, alpha) to find where steering reliably takes effect.

Builds vectors for every candidate layer in memory (response-token pooling),
then generates steered text for held-out prompts at each alpha. A generation
"succeeds" when it contains pirate-style markers AND is not degenerate.

    python -m scripts.tune_layers
    python -m scripts.tune_layers --layers 4-24:2 --alphas 0.25,0.5,1 --output tune.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindshift.backend.model_runtime import ModelRuntime, SteeringSpec  # noqa: E402
from mindshift.backend.vector_builder import (  # noqa: E402
    ActivationVectorBuilder,
    load_prompt_pairs,
)
from mindshift.backend.vector_registry import VectorRegistry  # noqa: E402,F401
from mindshift.evaluation import non_ascii_ratio, repetition_ratio  # noqa: E402

import torch  # noqa: E402

# Held-out prompts (none appear in mindshift/data/pirate_pairs.json).
PROMPTS = [
    "Explain how photosynthesis works.",
    "Give me three tips for studying effectively.",
    "What is the capital of France and why is it famous?",
    "Describe how to bake a loaf of bread.",
    "Explain how a compass works.",
    "How do I stay motivated when learning to code?",
]

# Strict list: words that essentially only appear in pirate-speak. Generic
# words such as "be", "ship", "sea" or "captain" are deliberately excluded.
STRICT = re.compile(
    r"\b(arr+|ahoy|matey|mateys|me hearties|hearties|ye|yer|aye|avast|"
    r"landlubbers?|scallywags?|booty|plunder|doubloons|buccaneers?|"
    r"shiver me timbers|yo-ho|yarr+|cap'n|ol'|'tis|thar|lass|lads?)\b",
    re.IGNORECASE,
)


def parse_layers(value: str) -> list[int]:
    """'8,12,16' or a range 'start-end[:step]' such as '4-24:2'."""
    if "-" in value:
        span, _, step = value.partition(":")
        start, end = (int(x) for x in span.split("-"))
        return list(range(start, end + 1, int(step or 1)))
    return [int(x) for x in value.split(",") if x.strip()]


def is_fluent(text: str) -> bool:
    words = re.findall(r"\w+", text)
    return (
        len(words) >= 10
        and repetition_ratio(text) < 0.6
        and non_ascii_ratio(text) < 0.05
    )


def score(text: str) -> dict:
    hits = len(STRICT.findall(text))
    fluent = is_fluent(text)
    return {"hits": hits, "fluent": fluent, "success": fluent and hits >= 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("4-24:2"))
    parser.add_argument(
        "--alphas",
        type=lambda v: [float(x) for x in v.split(",")],
        default=[0.25, 0.5, 0.75, 1.0, 1.5],
    )
    parser.add_argument("--pooling", default="response")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=ROOT / "tune_results.json")
    args = parser.parse_args()

    runtime = ModelRuntime()
    concept, pairs = load_prompt_pairs(ROOT / "mindshift" / "data" / "pirate_pairs.json")
    builder = ActivationVectorBuilder(runtime.model, runtime.tokenizer, runtime.model_id)
    built = builder.build_layers(concept, pairs, args.layers, args.pooling)

    rows = []
    samples = {}
    t0 = time.time()
    for layer in args.layers:
        meta = built[layer].metadata
        unit = built[layer].vector / built[layer].vector.norm()
        vector = unit * meta["mean_hidden_norm"]  # same scaling as VectorRegistry
        for alpha in args.alphas:
            results = []
            for prompt in PROMPTS:
                text = runtime.generate(
                    prompt, args.max_new_tokens, 42, SteeringSpec(layer, vector, alpha)
                )
                results.append(score(text))
                samples[f"L{layer}_a{alpha}_{len(results) - 1}"] = text
            n = len(results)
            row = {
                "layer": layer,
                "alpha": alpha,
                "success_rate": sum(r["success"] for r in results) / n,
                "fluent_rate": sum(r["fluent"] for r in results) / n,
                "mean_hits": sum(r["hits"] for r in results) / n,
                "token_norm": round(meta["mean_hidden_norm"], 1),
            }
            rows.append(row)
            print(
                f"L{layer:>2} a={alpha:<4} success={row['success_rate']:.2f} "
                f"fluent={row['fluent_rate']:.2f} hits={row['mean_hits']:.1f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    args.output.write_text(
        json.dumps({"rows": rows, "samples": samples}, indent=1), encoding="utf-8"
    )
    best = max(rows, key=lambda r: (r["success_rate"], r["fluent_rate"], -r["alpha"]))
    print("\nTop configurations:")
    for r in sorted(rows, key=lambda r: (-r["success_rate"], -r["fluent_rate"], r["alpha"]))[:8]:
        print(f"  layer={r['layer']} alpha={r['alpha']} success={r['success_rate']:.2f} fluent={r['fluent_rate']:.2f}")
    print(f"\nBest: layer={best['layer']} alpha={best['alpha']}  (saved to {args.output})")


if __name__ == "__main__":
    main()
