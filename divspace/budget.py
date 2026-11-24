"""Does the pooled contrast measure dispersion, or merely which prompt it came from?

The pooled split-half cosine is measured on far more samples per class than a
within-prompt fit can use, so a low within-prompt cosine is only informative if the
pooled statistic stays high at the same budget. This module samples every prompt
once and then answers three questions from that single set of generations:

* ``within``: cosine between two disjoint halves of one prompt's own samples.
* ``budget``: cosine between two halves of ``m`` prompts at a fixed per-arm budget,
  swept over ``m``. ``m = 1`` reproduces the within-prompt test, so a rise along the
  sweep isolates prompt mixing from sample size.
* ``between``: cosine between the pooled direction and a direction fitted from
  prompt-level mean dispersion, which is the identity component the pooled fit is
  hypothesised to recover.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import SamplingConfig
from .extract.directions import mean_diff, unit
from .extract.pipeline import response_activations
from .extract.select import dispersion
from .generate.sampler import prepare_prompt, sample
from .models import LM


@dataclass
class PromptSamples:
    """One prompt's responses plus their activations and dispersion scores."""

    prompt: str
    activations: dict[int, torch.Tensor]
    scores: torch.Tensor


def collect(
    lm: LM,
    prompts: list[str],
    layers: list[int],
    embed_layer: int,
    cfg: SamplingConfig,
    pool: str = "mean",
    use_chat_template: bool = True,
    max_length: int = 768,
) -> list[PromptSamples]:
    """Generate each prompt's samples once and cache activations and scores."""
    collected: list[PromptSamples] = []
    for prompt in prompts:
        prepared = prepare_prompt(lm.tokenizer, prompt, use_chat_template)
        generated = sample(lm, [prompt], cfg, use_chat_template=use_chat_template, condition="probe")
        pairs = [(prepared, row["text"]) for row in generated.rows]
        if len(pairs) < 8:
            continue
        embed = response_activations(
            lm, pairs, [embed_layer], batch_size=cfg.batch_size, max_length=max_length
        )[embed_layer]
        scores = dispersion(embed)
        acts = response_activations(
            lm,
            pairs,
            layers,
            batch_size=cfg.batch_size,
            max_length=max_length,
            pool=pool,
        )
        collected.append(
            PromptSamples(
                prompt=prompt,
                activations={layer: tensor.detach().cpu() for layer, tensor in acts.items()},
                scores=scores.detach().cpu(),
            )
        )
    return collected


def _class_split(scores: torch.Tensor, k: int) -> tuple[list[int], list[int]]:
    """Indices of the ``k`` most and ``k`` least dispersed responses."""
    order = torch.argsort(scores)
    k = max(1, min(k, scores.shape[0] // 2))
    return order[-k:].tolist(), order[:k].tolist()


def _direction(acts: torch.Tensor, highs: list[int], lows: list[int]) -> torch.Tensor | None:
    if not highs or not lows:
        return None
    return unit(mean_diff(acts[highs], acts[lows]))


def within_prompt_cosines(
    samples: list[PromptSamples], layer: int, seed: int = 0, k: int = 3
) -> list[float]:
    """Cosine between directions fitted on two disjoint halves of one prompt."""
    out: list[float] = []
    for index, item in enumerate(samples):
        acts = item.activations[layer]
        generator = torch.Generator().manual_seed(seed + index)
        order = torch.randperm(acts.shape[0], generator=generator).tolist()
        half = len(order) // 2
        halves = (order[:half], order[half:])
        fits = []
        for part in halves:
            part = torch.tensor(part)
            highs, lows = _class_split(item.scores[part], k)
            highs = [int(part[i]) for i in highs]
            lows = [int(part[i]) for i in lows]
            fits.append(_direction(acts, highs, lows))
        if any(fit is None for fit in fits):
            continue
        out.append(float(torch.dot(fits[0], fits[1]).clamp(-1.0, 1.0)))
    return out


def pooled_budget_cosines(
    samples: list[PromptSamples],
    layer: int,
    sizes: list[int],
    reps: int = 20,
    seed: int = 0,
    k: int = 3,
) -> dict[int, list[float]]:
    """Pooled split-half cosine using ``m`` prompts at ``k`` responses per arm each."""
    curve: dict[int, list[float]] = {}
    for size in sizes:
        m = min(size, len(samples))
        values: list[float] = []
        for rep in range(reps):
            generator = torch.Generator().manual_seed(seed + 1000 * rep + m)
            chosen = torch.randperm(len(samples), generator=generator)[:m].tolist()
            halves: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = [([], []), ([], [])]
            for prompt_index in chosen:
                item = samples[prompt_index]
                acts = item.activations[layer]
                order = torch.randperm(acts.shape[0], generator=generator).tolist()
                half = len(order) // 2
                for side, part in enumerate((order[:half], order[half:])):
                    part_tensor = torch.tensor(part)
                    highs, lows = _class_split(item.scores[part_tensor], k)
                    for target, group in ((0, highs), (1, lows)):
                        index = [int(part_tensor[i]) for i in group]
                        halves[side][target].append(acts[index])
            if any(not side[0] or not side[1] for side in halves):
                continue
            fits = [
                unit(mean_diff(torch.cat(side[0], 0), torch.cat(side[1], 0)))
                for side in halves
            ]
            values.append(float(torch.dot(fits[0], fits[1]).clamp(-1.0, 1.0)))
        curve[m] = values
    return curve


def between_prompt_direction(
    samples: list[PromptSamples], layer: int, quantile: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Direction separating prompts by their mean dispersion, and the pooled fit."""
    means = torch.tensor([float(item.scores.mean()) for item in samples])
    n = max(1, int(round(quantile * len(samples))))
    order = torch.argsort(means)
    high_prompts, low_prompts = order[-n:].tolist(), order[:n].tolist()

    def stack(indices: list[int]) -> torch.Tensor:
        return torch.cat([samples[i].activations[layer] for i in indices], 0)

    between = unit(stack(high_prompts).mean(0) - stack(low_prompts).mean(0))

    highs: list[torch.Tensor] = []
    lows: list[torch.Tensor] = []
    for item in samples:
        acts = item.activations[layer]
        hi, lo = _class_split(item.scores, 3)
        highs.append(acts[hi])
        lows.append(acts[lo])
    pooled = unit(mean_diff(torch.cat(highs, 0), torch.cat(lows, 0)))
    return between, pooled


def prompt_directions(
    samples: list[PromptSamples], layer: int, k: int = 6
) -> list[list[float]]:
    """One direction per prompt, for the pairwise-agreement diagnostic."""
    out: list[list[float]] = []
    for item in samples:
        highs, lows = _class_split(item.scores, k)
        vector = _direction(item.activations[layer], highs, lows)
        if vector is not None:
            out.append(vector.tolist())
    return out
