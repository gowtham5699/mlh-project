# MindShift

MindShift is an activation-steering lab for comparing normal and steered
responses from Qwen. A forward hook adds a learned concept vector to one
decoder layer during generation.

## Architecture

1. `scripts/build_vectors.py` collects hidden activations from matched
   pirate/plain examples.
2. `mean(pirate) - mean(plain)` is saved with model, layer, norm, and pooling
   metadata.
3. The backend loads the model and vectors once, generates baseline and
   steered responses sequentially, and exposes them through FastAPI.
4. The Streamlit frontend displays both responses side by side.

## Setup

Use Python 3.11 or 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For an NVIDIA demo machine, install the matching CUDA build of PyTorch first.
For example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The default demo model is `Qwen/Qwen3-1.7B`. For CPU smoke tests:

```bash
export MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
export DEVICE=cpu
```

## Build vectors

Build the default pirate vector at layer 12:

```bash
python scripts/build_vectors.py --layers 12
```

Sweep several layers:

```bash
python scripts/build_vectors.py --layers 8,12,16
```

Raw vectors and JSON metadata are written to `mindshift/vectors/`. At load
time, normalized mode scales the unit direction to the mean hidden-state norm
recorded in the metadata.

## Run

Start the backend after the Phase 1–2 runtime is connected to the service
factory in `mindshift/backend/main.py`:

```bash
uvicorn mindshift.backend.main:app --host 0.0.0.0 --port 8000
```

Start the frontend:

```bash
streamlit run mindshift/frontend/app.py
```

Set `USE_MOCK_API=true` to preview the UI without a loaded backend.

## Verify

Run the automated tests:

```bash
pytest -q
```

Check a running backend:

```bash
python scripts/smoke_test.py
```

Include the frontend:

```bash
python scripts/smoke_test.py --frontend-url http://localhost:8501
```

## Evaluate

The evaluation uses three fixed prompts, deterministic settings, and alphas
`0, 0.5, 1, 1.5`. It records outputs, latency, repetition ratio, non-ASCII
ratio, and exact baseline equality at alpha zero.

```bash
python scripts/evaluate.py --layer 12 --output eval_results.md
```

Review each output and replace its manual fluency-note placeholder before
presenting results.

## Current integration boundary

The HTTP layer depends only on the `GenerationService` protocol. The model
runtime should be wrapped in an implementation of this protocol and supplied
with:

```python
app = create_app(service_factory=build_generation_service)
```

This keeps model loading, steering hooks, and vector construction separate
from API validation and error handling.

## Limitations

- Raw activation directions are model- and layer-specific.
- Large alpha values can harm fluency or overpower prompt instructions.
- Inference is intentionally sequential for deterministic comparisons.
- The simple text heuristics do not replace manual quality review.
