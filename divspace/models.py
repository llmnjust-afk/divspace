"""Model loading and a thin wrapper that hides tokenizer and device bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn

from .utils import find_layers, find_final_norm, parse_dtype, resolve_device


def _dtype_kwarg(dtype: torch.dtype) -> dict:
    """transformers renamed ``torch_dtype`` to ``dtype`` in 4.56; support both."""
    import transformers

    try:
        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    except (AttributeError, ValueError):
        return {"torch_dtype": dtype}
    if (major, minor) >= (4, 56):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def _select_backbone(model: nn.Module) -> nn.Module:
    """Return the transformer trunk so callers can skip the output projection.

    Running a 7B causal LM without ``logits_to_keep`` materialises a
    batch x sequence x vocabulary tensor, which dominates memory during activation
    extraction. The trunk produces the same hidden states without that cost.
    """
    base = getattr(model, "base_model", None)
    if isinstance(base, nn.Module) and base is not model:
        return base
    return model


@dataclass
class LM:
    name: str
    model: nn.Module
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    layers: nn.ModuleList
    hidden_size: int
    final_norm: nn.Module | None = None
    backbone: nn.Module | None = None

    def run_backbone(self, **inputs):
        """Forward through the trunk only: hidden states, no vocabulary logits."""
        trunk = self.backbone if self.backbone is not None else self.model
        return trunk(**inputs)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def device_index(self) -> int:
        return self.device.index if self.device.index is not None else 0

    @classmethod
    def load(
        cls,
        name: str,
        device: str | None = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        attn_implementation: str = "sdpa",
        use_fast_tokenizer: bool = True,
        local_files_only: bool = False,
    ) -> "LM":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = parse_dtype(dtype)
        dev = resolve_device(device)
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
            local_files_only=local_files_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map={"": dev.index if dev.type == "cuda" else "cpu"},
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
            **_dtype_kwarg(torch_dtype),
        )
        model.eval()
        model.requires_grad_(False)
        layers = find_layers(model)
        hidden = getattr(model.config, "hidden_size", None) or getattr(
            model.config, "n_embd", None
        )
        if hidden is None:
            raise AttributeError("cannot infer hidden size from model config")
        return cls(
            name=name,
            model=model,
            tokenizer=tokenizer,
            device=dev,
            dtype=torch_dtype,
            layers=layers,
            hidden_size=int(hidden),
            final_norm=find_final_norm(model),
            backbone=_select_backbone(model),
        )

    def encode(
        self,
        texts: Sequence[str],
        max_length: int | None = None,
        padding_side: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if padding_side is not None:
            self.tokenizer.padding_side = padding_side
        return self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=True,
        ).to(self.device)

    def lm_head_device(self) -> torch.device:
        head = self.model.get_output_embeddings()
        return head.weight.device if head is not None else self.device

    def unembed(self, hidden: torch.Tensor) -> torch.Tensor:
        head = self.model.get_output_embeddings()
        if head is None:
            raise AttributeError(f"{self.name} exposes no output embedding")
        if self.final_norm is not None:
            hidden = self.final_norm(hidden)
        return head(hidden)

    def memory_report(self) -> dict:
        if self.device.type != "cuda":
            return {}
        return {
            "allocated_gb": round(torch.cuda.memory_allocated(self.device) / 1024**3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved(self.device) / 1024**3, 2),
        }


def build_tiny(dtype: str = "float32", n_layers: int = 4) -> LM:
    """A randomly initialised Llama used by the test suite; downloads nothing."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    model = LlamaForCausalLM(config)
    model.eval()

    class _Tok:
        pad_token = "<pad>"
        eos_token = "<eos>"
        padding_side = "right"
        pad_token_id = 0
        eos_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            if torch.is_tensor(ids):
                ids = ids.tolist()
            return " ".join(f"w{int(i)}" for i in ids)

        def __call__(self, texts, return_tensors="pt", padding=True, **kwargs):
            import torch as _t

            if isinstance(texts, str):
                texts = [texts]
            ids = [[(hash(w) % 200) + 1 for w in t.split()][:16] or [1] for t in texts]
            width = max(len(x) for x in ids)
            if self.padding_side == "left":
                padded = [[0] * (width - len(x)) + x for x in ids]
            else:
                padded = [x + [0] * (width - len(x)) for x in ids]
            mask = [[1] * len(x) + [0] * (width - len(x)) for x in ids]
            return {
                "input_ids": _t.tensor(padded),
                "attention_mask": _t.tensor(mask),
            }

    return LM(
        name="tiny-random-llama",
        model=model,
        tokenizer=_Tok(),
        device=torch.device("cpu"),
        dtype=parse_dtype(dtype),
        layers=find_layers(model),
        hidden_size=config.hidden_size,
        final_norm=find_final_norm(model),
        backbone=_select_backbone(model),
    )
