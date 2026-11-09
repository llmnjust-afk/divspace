"""Logit lens diagnostics: what the residual stream already predicts at each depth."""

from __future__ import annotations

from typing import Sequence

import torch

from ..hooks import capture
from ..models import LM


@torch.inference_mode()
def logit_lens(
    lm: LM,
    text: str,
    layers: Sequence[int],
    top_k: int = 5,
    position: int = -1,
    max_length: int = 512,
) -> dict[int, dict]:
    """Decode the next-token distribution read out of each layer's hidden state."""
    enc = lm.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    enc = {k: v.to(lm.device) for k, v in enc.items()}
    store: dict[int, torch.Tensor] = {}
    with capture(lm.layers, list(layers), store, positions="all", pool=None):
        lm.model(**enc)

    report: dict[int, dict] = {}
    for layer in layers:
        hidden = store[layer][:, position, :]
        logits = lm.unembed(hidden.to(lm.dtype)).float()
        logprobs = torch.log_softmax(logits, dim=-1)[0]
        probs = logprobs.exp()
        entropy = float(-(probs * logprobs).sum())
        top = torch.topk(logprobs, k=top_k)
        report[layer] = {
            "entropy": entropy,
            "top_tokens": [
                {
                    "token": lm.tokenizer.decode([int(idx)]),
                    "logprob": float(value),
                }
                for value, idx in zip(top.values, top.indices)
            ],
        }
    return report


def mean_entropy_profile(
    lm: LM,
    texts: Sequence[str],
    layers: Sequence[int],
    position: int = -1,
    max_length: int = 512,
) -> dict[int, float]:
    """Average next-token entropy per layer across texts; a cheap collapse probe."""
    totals = {layer: 0.0 for layer in layers}
    for text in texts:
        report = logit_lens(lm, text, layers, top_k=1, position=position, max_length=max_length)
        for layer in layers:
            totals[layer] += report[layer]["entropy"]
    return {layer: value / max(len(texts), 1) for layer, value in totals.items()}
