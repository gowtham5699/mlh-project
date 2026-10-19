"""Hook-logic tests on a tiny randomly initialised Qwen2 model (no download)."""
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from backend.steering import (MODE_GENERATED, LayerOutputCapture, SteeringError, SteeringHook,
                              apply_steering, count_hooks, get_decoder_layers)

HIDDEN = 32
LAYERS = 4


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=100, hidden_size=HIDDEN, intermediate_size=64,
                      num_hidden_layers=LAYERS, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    m = Qwen2ForCausalLM(cfg).eval()
    return m


@pytest.fixture
def ids():
    return torch.tensor([[5, 7, 11, 13, 17]])


def gen(model, ids, n=8):
    with torch.inference_mode():
        return model.generate(ids, max_new_tokens=n, do_sample=False,
                              pad_token_id=0, attention_mask=torch.ones_like(ids))


def vec(scale=1.0):
    g = torch.Generator().manual_seed(1)
    return torch.randn(HIDDEN, generator=g) * scale


def test_alpha_zero_matches_baseline(model, ids):
    base = gen(model, ids)
    with SteeringHook(model, 2, vec(), alpha=0.0):
        steered = gen(model, ids)
    assert torch.equal(base, steered)


def test_alpha_positive_changes_hidden_states(model, ids):
    with torch.inference_mode():
        with LayerOutputCapture(model, 1) as cap_base:
            base = model(ids, output_hidden_states=True).hidden_states
        with SteeringHook(model, 1, vec(), alpha=3.0) as h:
            # Capture registered after the steering hook sees steered output.
            with LayerOutputCapture(model, 1) as cap_steered:
                steered = model(ids, output_hidden_states=True).hidden_states
    assert h.calls == 1
    delta = cap_steered.outputs[0] - cap_base.outputs[0]
    assert torch.allclose(delta, (3.0 * vec()).expand_as(delta), atol=1e-5)
    # Layers before the hook are unchanged; downstream layers change.
    assert torch.equal(steered[1], base[1])
    assert not torch.allclose(steered[3], base[3])


def test_generation_changes_with_large_alpha(model, ids):
    base = gen(model, ids)
    with SteeringHook(model, 1, vec(), alpha=50.0):
        steered = gen(model, ids)
    assert not torch.equal(base, steered)


def test_hook_applied_every_decode_step(model, ids):
    with SteeringHook(model, 1, vec(), alpha=1.0) as h:
        gen(model, ids, n=6)
    assert h.calls == 6  # 1 prefill + 5 cached decode steps


def test_hooks_removed_after_use_and_on_error(model, ids):
    assert count_hooks(model) == 0
    with SteeringHook(model, 1, vec(), alpha=1.0):
        assert count_hooks(model) == 1
    assert count_hooks(model) == 0
    with pytest.raises(RuntimeError):
        with SteeringHook(model, 1, vec(), alpha=1.0):
            raise RuntimeError("boom")
    assert count_hooks(model) == 0


def test_invalid_configs_raise(model):
    with pytest.raises(SteeringError):
        SteeringHook(model, 1, torch.zeros(HIDDEN + 1), 1.0)
    with pytest.raises(SteeringError):
        SteeringHook(model, 1, torch.zeros(2, HIDDEN), 1.0)
    with pytest.raises(SteeringError):
        SteeringHook(model, LAYERS, vec(), 1.0)
    with pytest.raises(SteeringError):
        SteeringHook(model, 1, torch.full((HIDDEN,), float("nan")), 1.0)
    with pytest.raises(SteeringError):
        SteeringHook(model, 1, vec(), float("inf"))


def test_apply_steering_preserves_dtype_device_and_shape():
    h = torch.zeros(2, 3, HIDDEN, dtype=torch.bfloat16)
    out = apply_steering(h, vec(), 2.0)
    assert out.dtype == torch.bfloat16 and out.shape == h.shape and out.device == h.device


def test_tuple_and_tensor_outputs_handled():
    class TupleLayer(torch.nn.Module):
        def forward(self, x):
            return (x, "extra")

    class TensorLayer(torch.nn.Module):
        def forward(self, x):
            return x

    class Dummy(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([layer])
            self.config = type("C", (), {"hidden_size": HIDDEN})()

    x = torch.zeros(1, 2, HIDDEN)
    for layer in (TupleLayer(), TensorLayer()):
        m = Dummy(layer)
        with SteeringHook(m, 0, torch.ones(HIDDEN), 2.0):
            out = m.model.layers[0](x)
        if isinstance(out, tuple):
            assert out[1] == "extra"
            out = out[0]
        assert torch.allclose(out, torch.full_like(x, 2.0))
        assert count_hooks(m) == 0


def test_generated_mode_only_steers_last_prefill_position():
    h = torch.zeros(1, 4, HIDDEN)
    out = apply_steering(h, torch.ones(HIDDEN), 1.0, MODE_GENERATED)
    assert torch.all(out[:, :3] == 0) and torch.all(out[:, 3] == 1)
    one = apply_steering(torch.zeros(1, 1, HIDDEN), torch.ones(HIDDEN), 1.0, MODE_GENERATED)
    assert torch.all(one == 1)


def test_get_decoder_layers(model):
    assert len(get_decoder_layers(model)) == LAYERS
