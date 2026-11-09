"""Split-half stability of a direction fitted from a single prompt.

The pooled estimator is unstable on instruction models because it averages
directions that point in unrelated places. This module asks the sharper
question: is the direction even well defined *inside* one prompt? Two disjoint
halves of that prompt's own samples are fitted independently and compared.
"""

from __future__ import annotations

import torch

from .config import SamplingConfig
from .extract.directions import mean_diff, unit
from .extract.pipeline import response_activations
from .extract.select import dispersion
from .generate.sampler import prepare_prompt, sample
from .models import LM


def _direction_from(
    lm: LM,
    pairs: list[tuple[str, str]],
    layer: int,
    embed_layer: int,
    hi: float,
    lo: float,
    batch_size: int,
    max_length: int,
    pool: str,
) -> torch.Tensor | None:
    if len(pairs) < 4:
        return None
    embeddings = response_activations(
        lm, pairs, [embed_layer], batch_size=batch_size, max_length=max_length
    )[embed_layer]
    scores = dispersion(embeddings)
    order = torch.argsort(scores)
    n = scores.shape[0]
    n_hi = max(1, int(round((1.0 - hi) * n)))
    n_lo = max(1, int(round(lo * n)))
    highs = order[-n_hi:].tolist()
    lows = order[:n_lo].tolist()
    keep = sorted(set(highs) | set(lows))
    split = len([i for i in keep if i in set(highs)])
    acts = response_activations(
        lm,
        [pairs[i] for i in keep],
        [layer],
        batch_size=batch_size,
        max_length=max_length,
        pool=pool,
    )[layer]
    if split < 1 or acts.shape[0] - split < 1:
        return None
    return unit(mean_diff(acts[:split], acts[split:]))


def within_prompt_stability(
    lm: LM,
    prompts: list[str],
    layer: int,
    embed_layer: int,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
    max_length: int = 768,
    pool: str = "mean",
    use_chat_template: bool = True,
    seed: int = 0,
) -> list[dict]:
    """Cosine between directions fitted on two disjoint halves of one prompt's samples."""
    cfg = SamplingConfig(
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        seed=seed,
    )
    rows: list[dict] = []
    for index, prompt in enumerate(prompts):
        prepared = prepare_prompt(lm.tokenizer, prompt, use_chat_template)
        generated = sample(lm, [prompt], cfg, use_chat_template=use_chat_template, condition="probe")
        pairs = [(prepared, row["text"]) for row in generated.rows]
        if len(pairs) < 8:
            continue
        generator = torch.Generator().manual_seed(seed + index)
        order = torch.randperm(len(pairs), generator=generator).tolist()
        half = len(order) // 2
        halves = ([pairs[i] for i in order[:half]], [pairs[i] for i in order[half:]])
        fits = [
            _direction_from(
                lm, part, layer, embed_layer, 0.8, 0.2, batch_size, max_length, pool
            )
            for part in halves
        ]
        if any(fit is None for fit in fits):
            continue
        cosine = float(torch.dot(fits[0], fits[1]).clamp(-1.0, 1.0))
        rows.append(
            {
                "prompt_id": index,
                "prompt": prompt,
                "layer": layer,
                "n_samples": len(pairs),
                "cosine": cosine,
                "half_a_norm": float(fits[0].norm()),
                "half_b_norm": float(fits[1].norm()),
            }
        )
    return rows
