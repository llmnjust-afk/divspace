"""Prompt-local diversity steering.

A pooled direction averages responses to different prompts and therefore need not
correspond to anything the model uses for a single request. This module fits one
direction per prompt from that prompt's own samples and steers with it, against a
random direction and a direction borrowed from a different prompt.
"""

from __future__ import annotations

import torch

from .config import SamplingConfig
from .extract.directions import mean_diff, unit
from .extract.pipeline import response_activations
from .extract.select import dispersion, select_within_prompt
from .generate.sampler import prepare_prompt, sample
from .intervene.steer import random_direction
from .models import LM


def fit_one(
    lm: LM,
    prompt: str,
    layer: int,
    embed_layer: int,
    cfg: SamplingConfig,
    pool: str,
    use_chat_template: bool,
    max_length: int,
) -> torch.Tensor:
    """Direction from a single prompt's own high- versus low-dispersion samples."""
    prepared = prepare_prompt(lm.tokenizer, prompt, use_chat_template)
    generated = sample(lm, [prompt], cfg, use_chat_template=use_chat_template, condition="probe")
    pairs = [(prepared, row["text"]) for row in generated.rows]
    embeddings = response_activations(
        lm, pairs, [embed_layer], batch_size=cfg.batch_size, max_length=max_length
    )[embed_layer]
    scores = dispersion(embeddings)
    highs, lows = select_within_prompt(
        [scores], min_samples=cfg.n_samples, hi_quantile=0.8, lo_quantile=0.2
    )
    acts = response_activations(
        lm,
        [pairs[i] for i in highs + lows],
        [layer],
        batch_size=cfg.batch_size,
        max_length=max_length,
        pool=pool,
    )[layer]
    split = len(highs)
    return unit(mean_diff(acts[:split], acts[split:]))


def run_local(
    lm: LM,
    prompts: list[str],
    layer: int,
    fit_samples: int,
    eval_samples: int,
    coefs: list[float],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
    embed_layer: int,
    pool: str = "mean",
    use_chat_template: bool = True,
    max_length: int = 768,
    seed: int = 0,
) -> list[dict]:
    """Steer every prompt with its own direction, plus random and mismatched controls."""
    rows: list[dict] = []
    fit_cfg = SamplingConfig(
        n_samples=fit_samples,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        seed=seed,
    )
    eval_cfg = SamplingConfig(
        n_samples=eval_samples,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        seed=seed + 1,
    )

    previous: torch.Tensor | None = None
    for position, prompt in enumerate(prompts):
        local = fit_one(
            lm, prompt, layer, embed_layer, fit_cfg, pool, use_chat_template, max_length
        )
        conditions = {
            "local": local,
            "localrandom": random_direction(lm.hidden_size, seed=seed + 7 * layer + position),
        }
        if previous is not None:
            conditions["localother"] = previous
        previous = local

        for name, vector in conditions.items():
            for coef in coefs:
                steered = sample(
                    lm,
                    [prompt],
                    eval_cfg,
                    direction=vector,
                    layers=[layer],
                    coef=coef,
                    use_chat_template=use_chat_template,
                    condition=name,
                )
                for row in steered.rows:
                    row["cell"] = f"{name}_L{layer}_c{coef:+.2f}"
                    row["prompt"] = prompt
                    row["prompt_id"] = position
                    row["model"] = lm.name
                    rows.append(row)

        baseline = sample(
            lm, [prompt], eval_cfg, use_chat_template=use_chat_template, condition="baseline"
        )
        for row in baseline.rows:
            row["cell"] = "baseline"
            row["prompt"] = prompt
            row["prompt_id"] = position
            row["model"] = lm.name
            rows.append(row)
    return rows
