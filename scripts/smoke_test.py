"""Smoke-test a running MindShift backend and optional frontend."""

from __future__ import annotations

import argparse
import os
from typing import Callable

import requests


def check(name: str, action: Callable[[], None]) -> bool:
    try:
        action()
    except Exception as exc:
        print(f"FAIL  {name}: {exc}")
        return False
    print(f"PASS  {name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend-url",
        default=os.getenv("BACKEND_URL", "http://localhost:8000"),
    )
    parser.add_argument("--frontend-url")
    parser.add_argument("--concept", default="pirate")
    parser.add_argument(
        "--layer",
        type=int,
        default=int(os.getenv("DEFAULT_LAYER", "12")),
    )
    args = parser.parse_args()
    backend_url = args.backend_url.rstrip("/")

    def health() -> None:
        response = requests.get(f"{backend_url}/health", timeout=10)
        response.raise_for_status()
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["model_id"]
        assert payload["device"]

    def concepts() -> None:
        response = requests.get(f"{backend_url}/concepts", timeout=10)
        response.raise_for_status()
        available = response.json()["concepts"]
        assert any(
            item["name"] == args.concept and args.layer in item["layers"]
            for item in available
        )

    def generation() -> None:
        response = requests.post(
            f"{backend_url}/generate",
            json={
                "prompt": "Explain gravity in two sentences.",
                "concept": args.concept,
                "alpha": 0,
                "layer": args.layer,
                "max_new_tokens": 64,
                "seed": 42,
            },
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        assert payload["baseline"]
        assert payload["baseline"] == payload["steered"]

    checks = [
        check("Phase 4 /health", health),
        check("Phase 4 /concepts", concepts),
        check("Phase 4 /generate alpha=0", generation),
    ]

    if args.frontend_url:
        def frontend() -> None:
            response = requests.get(args.frontend_url, timeout=10)
            response.raise_for_status()
            assert "streamlit" in response.text.casefold()

        checks.append(check("Phase 5 frontend", frontend))

    passed = sum(checks)
    print(f"\n{passed}/{len(checks)} smoke checks passed")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
