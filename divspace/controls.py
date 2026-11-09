"""Directions fitted from labelled contrasts, used to validate the pipeline itself.

A null result on diversity only means something if the same machinery can recover a
direction that is known to work. These helpers build such a direction from paired
prompt/target completions and hand it to the ordinary steering code.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .extract.directions import unit
from .extract.pipeline import DirectionArtifact
from .models import LM
from .extract.pipeline import response_activations


def load_pairs(path: str | Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    positive = [tuple(item) for item in payload["positive"]]
    negative = [tuple(item) for item in payload["negative"]]
    if len(positive) != len(negative):
        raise ValueError("positive and negative lists must be the same length")
    return positive, negative


def fit_contrast_direction(
    lm: LM,
    positive: list[tuple[str, str]],
    negative: list[tuple[str, str]],
    layers: list[int],
    batch_size: int = 8,
    max_length: int = 256,
    normalize: bool = True,
) -> DirectionArtifact:
    pos_acts = response_activations(lm, positive, layers, batch_size=batch_size, max_length=max_length)
    neg_acts = response_activations(lm, negative, layers, batch_size=batch_size, max_length=max_length)

    directions = {}
    norms = {}
    for layer in layers:
        raw = pos_acts[layer].mean(dim=0) - neg_acts[layer].mean(dim=0)
        norms[layer] = float(raw.norm())
        directions[layer] = unit(raw) if normalize else raw

    return DirectionArtifact(
        model=lm.name,
        layers=list(layers),
        directions=directions,
        norms=norms,
        embed_layer=layers[0],
        positive_pairs=[list(pair) for pair in positive],
        negative_pairs=[list(pair) for pair in negative],
        meta={"kind": "contrast", "n_pairs": len(positive), "targets": "labelled"},
    )


def split_half_agreement(
    lm: LM,
    positive: list[tuple[str, str]],
    negative: list[tuple[str, str]],
    layers: list[int],
    batch_size: int = 8,
    max_length: int = 256,
) -> dict[int, float]:
    """Same reliability check used for the diversity directions, for comparison."""
    half = len(positive) // 2
    first = fit_contrast_direction(lm, positive[:half], negative[:half], layers, batch_size, max_length)
    second = fit_contrast_direction(lm, positive[half:], negative[half:], layers, batch_size, max_length)
    return {
        layer: float(torch.dot(first.directions[layer], second.directions[layer]))
        for layer in layers
    }
