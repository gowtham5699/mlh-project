"""Evaluate baseline and steered outputs through the MindShift API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindshift.evaluation import non_ascii_ratio, repetition_ratio


PROMPTS = [
    "Explain gravity to a ten-year-old.",
    "Write a short email asking to reschedule a meeting.",
    "Describe how a lighthouse helps ships at night.",
]


def parse_alphas(value: str) -> list[float]:
    try:
        alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Alphas must be comma-separated numbers") from exc
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise argparse.ArgumentTypeError("At least one non-negative alpha is required")
    return alphas


def generate(
    backend_url: str,
    prompt: str,
    concept: str,
    alpha: float,
    layer: int,
    seed: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    response = requests.post(
        f"{backend_url.rstrip('/')}/generate",
        json={
            "prompt": prompt,
            "concept": concept,
            "alpha": alpha,
            "layer": layer,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def safe_fence(text: str) -> str:
    return text.replace("```", "'''")


def render_report(
    results: list[dict[str, Any]],
    *,
    backend_url: str,
    concept: str,
    layer: int,
    seed: int,
) -> str:
    alpha_zero_results = [result for result in results if result["alpha"] == 0]
    alpha_zero_equal = bool(alpha_zero_results) and all(
        result["baseline"] == result["steered"]
        for result in alpha_zero_results
    )

    lines = [
        "# MindShift evaluation results",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Backend: `{backend_url}`",
        f"- Concept: `{concept}`",
        f"- Layer: `{layer}`",
        f"- Seed: `{seed}`",
        f"- Alpha=0 exact equality: **{'PASS' if alpha_zero_equal else 'FAIL'}**",
        "",
        "## Fluency-harm summary",
        "",
        "| Prompt | Alpha | Repetition ratio | Non-ASCII ratio | Latency (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        prompt_label = result["prompt"].replace("|", "\\|")
        lines.append(
            f"| {prompt_label} | {result['alpha']:g} | "
            f"{result['repetition_ratio']:.3f} | "
            f"{result['non_ascii_ratio']:.3f} | "
            f"{result['latency_ms']:.1f} |"
        )

    lines.extend(["", "## Outputs", ""])
    for result in results:
        lines.extend(
            [
                f"### Alpha {result['alpha']:g}: {result['prompt']}",
                "",
                "**Baseline**",
                "",
                "```text",
                safe_fence(result["baseline"]),
                "```",
                "",
                "**Steered**",
                "",
                "```text",
                safe_fence(result["steered"]),
                "```",
                "",
                "**Manual fluency note:** _Add after review._",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend-url",
        default=os.getenv("BACKEND_URL", "http://localhost:8000"),
    )
    parser.add_argument("--concept", default="pirate")
    parser.add_argument(
        "--layer",
        type=int,
        default=int(os.getenv("DEFAULT_LAYER", "12")),
    )
    parser.add_argument("--alphas", type=parse_alphas, default=parse_alphas("0,.5,1,1.5"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval_results.md",
    )
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    for prompt in PROMPTS:
        for alpha in args.alphas:
            print(f"Evaluating alpha={alpha:g}: {prompt}")
            payload = generate(
                backend_url=args.backend_url,
                prompt=prompt,
                concept=args.concept,
                alpha=alpha,
                layer=args.layer,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
            )
            steered = str(payload["steered"])
            results.append(
                {
                    "prompt": prompt,
                    "alpha": alpha,
                    "baseline": str(payload["baseline"]),
                    "steered": steered,
                    "repetition_ratio": repetition_ratio(steered),
                    "non_ascii_ratio": non_ascii_ratio(steered),
                    "latency_ms": float(payload["latency_ms"]),
                }
            )

    report = render_report(
        results,
        backend_url=args.backend_url,
        concept=args.concept,
        layer=args.layer,
        seed=args.seed,
    )
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
