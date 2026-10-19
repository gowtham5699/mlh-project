"""Real-model smoke test for phases 1-2 (loads the actual model).

Run from the repo root:
    python -m scripts.model_smoke_test                  # uses MODEL_ID or Qwen3-1.7B
    MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct python -m scripts.model_smoke_test
Prints PASS/FAIL per check so results are easy to paste back.
"""
import logging
import sys

import torch

from mindshift.backend.model_runtime import ModelRuntime, SteeringSpec
from mindshift.backend.steering import LayerOutputCapture, SteeringHook, count_hooks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
PROMPT = "Explain gravity in two sentences."
MAX_NEW = 48
SEED = 1234
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main() -> int:
    rt = ModelRuntime()
    print(f"model={rt.model_id} device={rt.device} dtype={rt.dtype} layers={rt.num_layers}")

    # Phase 1: baseline
    base = rt.generate(PROMPT, MAX_NEW, SEED)
    print("BASELINE:", base)
    check("baseline non-empty and readable", len(base.split()) >= 5)
    check("baseline deterministic", base == rt.generate(PROMPT, MAX_NEW, SEED))

    # Phase 2: steering
    layer = min(12, rt.num_layers - 1)
    g = torch.Generator().manual_seed(0)
    v = torch.randn(rt.hidden_size, generator=g)
    v = v / v.norm()

    zero = rt.generate(PROMPT, MAX_NEW, SEED, SteeringSpec(layer, v, 0.0))
    check("alpha=0 matches baseline", zero == base)

    # Scale to the layer's typical hidden norm so alpha is meaningful.
    inputs = rt.build_inputs(PROMPT)
    with torch.inference_mode():
        with LayerOutputCapture(rt.model, layer) as base_cap:
            rt.model(**inputs)
        hnorm = base_cap.outputs[0][0, -1].float().norm().item()
        with SteeringHook(rt.model, layer, v * hnorm, 1.0):
            with LayerOutputCapture(rt.model, layer) as steer_cap:
                rt.model(**inputs)
    delta = (steer_cap.outputs[0] - base_cap.outputs[0]).float()
    expected = hnorm
    check("hidden states changed by ~alpha*|v|", abs(delta[0, -1].norm().item() - expected) / expected < 0.05,
          f"(delta norm {delta[0, -1].norm().item():.2f}, expected {expected:.2f})")

    steered = rt.generate(PROMPT, MAX_NEW, SEED, SteeringSpec(layer, v * hnorm, 1.0))
    print("STEERED (random direction, alpha=1):", steered)
    check("steered output differs from baseline", steered != base)
    check("no hooks leaked", count_hooks(rt.model) == 0)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
