"""Building direction vectors and low-rank subspaces from paired activations."""

from __future__ import annotations

import torch


def unit(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / vector.norm().clamp_min(eps)


def mean_diff(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Difference of class means; shape (hidden,) from (N, hidden) and (M, hidden)."""
    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("expected 2-D activation matrices")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError("positive and negative activations disagree on hidden size")
    return positive.mean(dim=0) - negative.mean(dim=0)


def pca_basis(matrix: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k principal directions of the contrastive differences, shape (k, hidden)."""
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    k = max(1, min(k, min(centered.shape) - 1))
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:k]


def orthogonalize(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove the components of vector that lie in the span of basis (k, hidden)."""
    if basis.ndim == 1:
        basis = basis.unsqueeze(0)
    projection = (vector @ basis.T) @ basis
    return vector - projection


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(eps))


def split_half_stability(
    positive: torch.Tensor,
    negative: torch.Tensor,
    seed: int = 0,
) -> dict[str, float]:
    """Cosine agreement between directions fitted on two disjoint halves.

    This is the cheapest honest reliability check for a steering direction: if the
    two halves disagree, the direction is noise rather than a stable axis.
    """
    generator = torch.Generator().manual_seed(seed)
    pos_perm = torch.randperm(positive.shape[0], generator=generator)
    neg_perm = torch.randperm(negative.shape[0], generator=generator)
    pos_a, pos_b = positive[pos_perm].chunk(2)
    neg_a, neg_b = negative[neg_perm].chunk(2)
    if min(pos_a.shape[0], pos_b.shape[0], neg_a.shape[0], neg_b.shape[0]) < 2:
        return {"cosine": float("nan"), "n_positive": int(positive.shape[0])}
    dir_a = mean_diff(pos_a, neg_a)
    dir_b = mean_diff(pos_b, neg_b)
    return {
        "cosine": cosine(unit(dir_a), unit(dir_b)),
        "n_positive": int(positive.shape[0]),
        "n_negative": int(negative.shape[0]),
    }
