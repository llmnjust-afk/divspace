"""Attribution patching over decoder layers.

We compare a high-dispersion response (clean) against a low-dispersion response
(corrupt) for the same prompt and approximate the causal contribution of every
layer's residual stream to the difference in response log-likelihood.

The approximation is the first-order one from Kramar et al. (2024): the gradient
is taken at the corrupt run and multiplied by the clean-minus-corrupt activation
difference, which costs two forward passes and a single backward pass instead of
one forward pass per layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from ..hooks import capture_with_grad
from ..models import LM


def _token_ids(tokenizer, text: str, add_special_tokens: bool) -> list[int]:
    out = tokenizer(text, add_special_tokens=add_special_tokens)
    ids = out["input_ids"] if isinstance(out, dict) else out.input_ids
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


@dataclass
class RunRecord:
    metric: float
    activations: dict[int, torch.Tensor] = field(default_factory=dict)
    response_span: tuple[int, int] = (0, 0)


def _build_inputs(lm: LM, prompt: str, response: str, max_length: int):
    prompt_ids = _token_ids(lm.tokenizer, prompt, True)
    response_ids = _token_ids(lm.tokenizer, response, False)
    if not response_ids:
        response_ids = [lm.tokenizer.eos_token_id or 0]
    seq = (prompt_ids + response_ids)[:max_length]
    return seq, len(prompt_ids), len(seq)


def _forward(lm: LM, ids: list[int], layers: Sequence[int], with_grad: bool):
    input_ids = torch.tensor([ids], dtype=torch.long, device=lm.device)
    attention = torch.ones_like(input_ids)
    embedding_layer = lm.model.get_input_embeddings()

    graph = torch.enable_grad() if with_grad else torch.no_grad()
    store: dict[int, torch.Tensor] = {}
    with graph:
        embeds = embedding_layer(input_ids)
        if with_grad:
            embeds = embeds.detach().requires_grad_(True)
        with capture_with_grad(lm.layers, list(layers), store):
            output = lm.model(inputs_embeds=embeds, attention_mask=attention)
        logits = output.logits

    return RunRecord(metric=0.0, activations=store), logits


@torch.enable_grad()
def attribution_patching(
    lm: LM,
    prompt: str,
    clean_response: str,
    corrupt_response: str,
    layers: Sequence[int],
    max_length: int = 512,
) -> dict:
    """Return a per-layer scalar attribution and a per-dimension attribution vector.

    The metric is the mean teacher-forced log-likelihood of the response tokens.
    """
    layers = list(layers)
    clean_ids, clean_start, clean_end = _build_inputs(lm, prompt, clean_response, max_length)
    corrupt_ids, corrupt_start, corrupt_end = _build_inputs(
        lm, prompt, corrupt_response, max_length
    )

    clean_rec, clean_logits = _forward(lm, clean_ids, layers, with_grad=False)
    corrupt_rec, corrupt_logits = _forward(lm, corrupt_ids, layers, with_grad=True)

    def metric_and_slice(logits: torch.Tensor, ids: list[int], start: int, end: int):
        targets = torch.tensor(ids[1:], dtype=torch.long, device=logits.device).unsqueeze(0)
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        picked = logprobs[:, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        lo, hi = max(start - 1, 0), max(end - 1, 1)
        return picked[:, lo:hi].mean(), (lo, hi)

    clean_metric, _ = metric_and_slice(clean_logits, clean_ids, clean_start, clean_end)
    corrupt_metric, (lo, hi) = metric_and_slice(
        corrupt_logits, corrupt_ids, corrupt_start, corrupt_end
    )

    corrupt_metric.backward()

    scalars: dict[int, float] = {}
    vectors: dict[int, torch.Tensor] = {}
    for layer in layers:
        grad = corrupt_rec.activations[layer].grad
        if grad is None:
            raise RuntimeError(
                f"no gradient reached layer {layer}; the model may not support inputs_embeds"
            )
        clean_acts = clean_rec.activations[layer]
        corrupt_acts = corrupt_rec.activations[layer].detach()
        width = min(clean_acts.shape[1], corrupt_acts.shape[1], grad.shape[1])
        delta = clean_acts[:, :width] - corrupt_acts[:, :width]
        pair = grad[:, :width] * delta
        hi_aligned = min(hi, width)
        span = pair[:, lo:hi_aligned, :] if hi_aligned > lo else pair
        vectors[layer] = span.sum(dim=(0, 1)).float().cpu()
        scalars[layer] = float(vectors[layer].sum())

    return {
        "layers": layers,
        "scalars": scalars,
        "vectors": vectors,
        "clean_metric": float(clean_metric),
        "corrupt_metric": float(corrupt_metric.detach()),
        "delta_metric": float(clean_metric - corrupt_metric.detach()),
        "clean_tokens": clean_end - clean_start,
        "corrupt_tokens": corrupt_end - corrupt_start,
    }


def top_dimensions(vector: torch.Tensor, k: int = 20) -> dict:
    values, indices = torch.topk(vector.abs(), k=min(k, vector.numel()))
    return {
        "indices": indices.tolist(),
        "values": vector[indices].tolist(),
        "abs_total": float(vector.abs().sum()),
        "mass_top_k": float(values.sum() / vector.abs().sum().clamp_min(1e-8)),
    }
