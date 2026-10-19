"""Minimal Streamlit UI for the MindShift activation-steering demo."""

from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
DEFAULT_LAYER = int(os.getenv("DEFAULT_LAYER", "11"))
DEFAULT_MOCK_MODE = os.getenv("USE_MOCK_API", "false").lower() in {
    "1",
    "true",
    "yes",
}

MOCK_CONCEPTS = [{"name": "pirate", "layers": [9, 11, 12, 13]}]


def _concept_names(payload: Any) -> list[str]:
    """Accept either a bare concept list or {"concepts": [...]}."""
    concepts = payload.get("concepts", []) if isinstance(payload, dict) else payload
    if not isinstance(concepts, list):
        return []

    names: list[str] = []
    for concept in concepts:
        if isinstance(concept, str):
            names.append(concept)
        elif isinstance(concept, dict) and isinstance(concept.get("name"), str):
            names.append(concept["name"])
    return names


def load_concepts(use_mock_api: bool) -> list[str]:
    if use_mock_api:
        return _concept_names(MOCK_CONCEPTS)

    response = requests.get(f"{BACKEND_URL}/concepts", timeout=5)
    response.raise_for_status()
    return _concept_names(response.json())


def generate(
    prompt: str,
    concept: str,
    strength: float,
    layer: int,
    use_mock_api: bool,
    include_prompted: bool = False,
) -> dict[str, Any]:
    if use_mock_api:
        started = time.perf_counter()
        return {
            "baseline": (
                "A compass points toward magnetic north and helps a traveler "
                "keep a steady direction."
            ),
            "steered": (
                "Arrr! A compass points toward magnetic north, helping any "
                "sailor keep a true course across the seas."
            ),
            "concept": concept,
            "layer": layer,
            "alpha": strength,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    response = requests.post(
        f"{BACKEND_URL}/generate",
        json={
            "prompt": prompt,
            "concept": concept,
            "alpha": strength,
            "layer": layer,
            "include_prompted": include_prompted,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="MindShift", page_icon="🧠", layout="wide")
st.title("MindShift")
st.caption("Compare a model's normal response with an activation-steered response.")

with st.sidebar:
    st.header("Connection")
    use_mock_api = st.toggle("Demo mode", value=DEFAULT_MOCK_MODE)
    if use_mock_api:
        st.warning("Demo mode: canned sample text, NOT model output.")
    else:
        st.caption(BACKEND_URL)

try:
    concepts = load_concepts(use_mock_api)
except (requests.RequestException, ValueError) as exc:
    st.error(f"Could not load concepts from the backend: {exc}")
    concepts = []

EXAMPLE_PROMPTS = [
    "Explain gravity to a ten-year-old.",
    "Give me three tips for preparing for a job interview.",
    "Describe how a lighthouse helps ships at night.",
]

example = st.selectbox(
    "Example prompts", ["(custom)"] + EXAMPLE_PROMPTS, key="example-prompt"
)

with st.form("generation-form"):
    prompt = st.text_area(
        "Prompt",
        value=example if example != "(custom)" else "Explain how a compass works.",
        height=140,
    )
    concept = st.selectbox(
        "Steering concept",
        concepts,
        disabled=not concepts,
    )

    control_left, control_right = st.columns(2)
    with control_left:
        strength = st.slider(
            "Strength (alpha)",
            min_value=0.0,
            max_value=2.0,
            value=0.6,
            step=0.1,
            help="Relative to typical activation magnitude at the layer. 0 = baseline.",
        )
    with control_right:
        layer = st.number_input(
            "Decoder layer",
            min_value=0,
            value=DEFAULT_LAYER,
            step=1,
        )

    include_prompted = st.checkbox(
        "Also show a prompt-engineered answer (no steering) for comparison",
        value=False,
    )

    submitted = st.form_submit_button(
        "Generate comparison",
        type="primary",
        disabled=not concepts,
    )

if submitted:
    if not prompt.strip():
        st.warning("Enter a prompt before generating.")
    else:
        try:
            with st.spinner("Generating baseline and steered responses…"):
                result = generate(
                    prompt=prompt.strip(),
                    concept=str(concept),
                    strength=strength,
                    layer=int(layer),
                    use_mock_api=use_mock_api,
                    include_prompted=include_prompted,
                )

            columns = st.columns(3 if result.get("prompted") else 2)
            with columns[0]:
                st.subheader("Baseline")
                st.write(result.get("baseline", "No baseline response returned."))
            with columns[1]:
                st.subheader("Steered")
                st.write(result.get("steered", "No steered response returned."))
            if result.get("prompted"):
                with columns[2]:
                    st.subheader("Prompted (no steering)")
                    st.write(result["prompted"])

            metric_left, metric_middle, metric_right = st.columns(3)
            metric_left.metric("Layer", result.get("layer", layer))
            metric_middle.metric("Strength", result.get("alpha", strength))
            metric_right.metric(
                "Latency",
                f"{result.get('latency_ms', 0):,.1f} ms",
            )
        except (requests.RequestException, ValueError) as exc:
            st.error(f"Generation failed: {exc}")

with st.expander("How activation steering works"):
    st.write(
        "MindShift adds a learned direction vector to the model's hidden "
        "activations at one decoder layer. The baseline and steered runs use "
        "the same prompt and decoding settings, so their difference shows the "
        "effect of that intervention."
    )
