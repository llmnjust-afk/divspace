"""Distributional diversity metrics computed from response embeddings."""

from __future__ import annotations

import torch


def normalise(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(eps)


def mean_pairwise_distance(embeddings: torch.Tensor) -> float:
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return 0.0
    similarity = normalise(embeddings) @ normalise(embeddings).T
    n = similarity.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    return float((1.0 - similarity)[mask].mean())


def vendi_score(embeddings: torch.Tensor, eps: float = 1e-12) -> float:
    """Effective number of distinct samples (Friedman & Dieng, 2023).

    Equals exp(entropy of the eigenvalues of the normalised similarity kernel), so
    a set of n identical responses scores 1 and n orthogonal responses score n.
    """
    n = embeddings.shape[0]
    if n < 2:
        return float(n)
    kernel = normalise(embeddings) @ normalise(embeddings).T
    eigenvalues = torch.linalg.eigvalsh(kernel.double()) / n
    eigenvalues = eigenvalues.clamp_min(eps)
    entropy = -(eigenvalues * eigenvalues.log()).sum()
    return float(torch.exp(entropy))


def effective_rank(embeddings: torch.Tensor, eps: float = 1e-12) -> float:
    """Entropy-based effective rank of the centred embedding matrix."""
    if embeddings.shape[0] < 2:
        return 0.0
    centred = embeddings - embeddings.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centred.double())
    total = singular.sum().clamp_min(eps)
    probs = (singular / total).clamp_min(eps)
    return float(torch.exp(-(probs * probs.log()).sum()))


def anisotropy(embeddings: torch.Tensor) -> float:
    """Mean cosine of each sample to the dataset mean: high values signal a narrow cone."""
    if embeddings.shape[0] < 2:
        return 0.0
    centre = embeddings.mean(dim=0, keepdim=True)
    return float((normalise(embeddings) @ normalise(centre).T).mean())


def pc1_ratio(embeddings: torch.Tensor, eps: float = 1e-12) -> float:
    """Share of the total variance carried by the first principal component.

    A diversity collapse concentrates variation along a single direction, so this
    ratio rises towards one even when pairwise distances stay roughly constant.
    """
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    centred = (embeddings - embeddings.mean(dim=0, keepdim=True)).double()
    covariance = centred.T @ centred / max(n - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum().clamp_min(eps)
    return float(eigenvalues[-1] / total)


def participation_ratio(embeddings: torch.Tensor, eps: float = 1e-12) -> float:
    """Effective number of dimensions that carry the variance (isotropic data scores d)."""
    if embeddings.shape[0] < 2:
        return 0.0
    centred = (embeddings - embeddings.mean(dim=0, keepdim=True)).double()
    singular = torch.linalg.svdvals(centred)
    squared = singular.pow(2)
    return float(squared.sum().pow(2) / squared.pow(2).sum().clamp_min(eps))
