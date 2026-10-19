"""Model runtime: loads model/tokenizer once and generates text, optionally steered."""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .steering import MODE_ALL, SteeringHook, count_hooks

logger = logging.getLogger("mindshift.runtime")

DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"


@dataclass
class SteeringSpec:
    layer: int
    vector: torch.Tensor
    alpha: float
    mode: str = MODE_ALL


def resolve_device(device: Optional[str] = None) -> str:
    device = device or os.getenv("DEVICE") or ""
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class ModelRuntime:
    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        self.model_id = model_id or os.getenv("MODEL_ID") or DEFAULT_MODEL_ID
        self.device = resolve_device(device)
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._lock = threading.Lock()  # inference is strictly sequential
        logger.info("Loading %s on %s (%s)", self.model_id, self.device, self.dtype)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        logger.info("Model loaded in %.1fs", time.time() - t0)

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def build_inputs(self, prompt: str) -> dict:
        """Apply the chat template (non-thinking mode where supported)."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,  # ignored by templates that do not support it
        )
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 64, seed: int = 0,
                 steering: Optional[SteeringSpec] = None, do_sample: bool = False,
                 temperature: float = 0.7) -> str:
        """Generate a completion. Greedy by default; seeded sampling if do_sample."""
        with self._lock:
            inputs = self.build_inputs(prompt)
            torch.manual_seed(seed)
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                              pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            if do_sample:
                gen_kwargs.update(temperature=temperature, top_p=0.9)
            else:
                # Avoid warnings and make greedy settings explicit and identical.
                gen_kwargs.update(temperature=None, top_p=None, top_k=None)
            try:
                with torch.inference_mode():
                    if steering is not None:
                        with SteeringHook(self.model, steering.layer, steering.vector,
                                          steering.alpha, steering.mode):
                            out = self.model.generate(**inputs, **gen_kwargs)
                    else:
                        out = self.model.generate(**inputs, **gen_kwargs)
            finally:
                leaked = count_hooks(self.model)
                if leaked:
                    logger.error("%d steering hooks leaked after generation", leaked)
            new_tokens = out[0, inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
