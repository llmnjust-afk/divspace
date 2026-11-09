"""Selecting high- versus low-dispersion generations without any human labels.

The selection signal is intrinsic: for each prompt we ask how far a sample sits
from the other samples drawn for that same prompt. Samples in the top quantile of
that distance are treated as diversity-supporting, the bottom quantile as
diversity-suppressing. Nothing outside the model's own outputs is consulted.
"""

from __future__ import annotations

import torch


def normalise(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(eps)


def dispersion(embeddings: torch.Tensor, exclude_self: bool = True) -> torch.Tensor:
    """Mean cosine distance from each row to the other rows; shape (N,)."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be (N, hidden)")
    if embeddings.shape[0] < 2:
        return torch.zeros(embeddings.shape[0])
    sim = normalise(embeddings) @ normalise(embeddings).T
    dist = (1.0 - sim).clamp_min(0.0)
    n = dist.shape[0]
    if exclude_self:
        dist = dist - torch.eye(n, device=dist.device, dtype=dist.dtype) * dist.diagonal()
        return dist.sum(dim=1) / (n - 1)
    return dist.mean(dim=1)


def select_within_prompt(
    scores_by_prompt: list[torch.Tensor],
    hi_quantile: float = 0.8,
    lo_quantile: float = 0.2,
    min_samples: int = 8,
) -> tuple[list[int], list[int]]:
    """Return global sample indices for the high and low dispersion groups.

    Selection happens per prompt so that prompt difficulty cannot leak into the
    contrast, then indices are remapped to the flat sample list.
    """
    if not 0.0 <= lo_quantile < hi_quantile <= 1.0:
        raise ValueError("require 0 <= lo_quantile < hi_quantile <= 1")

    highs: list[int] = []
    lows: list[int] = []
    offset = 0
    for scores in scores_by_prompt:
        n = scores.shape[0]
        if n < max(min_samples, 4):
            offset += n
            continue
        order = torch.argsort(scores)
        n_hi = max(1, int(round((1.0 - hi_quantile) * n)))
        n_lo = max(1, int(round(lo_quantile * n)))
        n_hi = max(1, min(n_hi, n - 1))
        n_lo = max(1, min(n_lo, n - 1))
        highs.extend((offset + order[-n_hi:]).tolist())
        lows.extend((offset + order[:n_lo]).tolist())
        offset += n
    if not highs or not lows:
        raise RuntimeError("no prompt produced enough samples to build a contrast")
    return highs, lows


def balance(highs: list[int], lows: list[int]) -> tuple[list[int], list[int]]:
    """Trim the larger group so both classes are the same size."""
    size = min(len(highs), len(lows))
    return highs[:size], lows[:size]
