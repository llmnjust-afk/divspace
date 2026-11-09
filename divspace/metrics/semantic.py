"""Semantic diversity: cluster the samples, then report the entropy of cluster sizes.

Two clustering back-ends are provided. The default is embedding agglomeration,
which needs no extra model and is therefore rerunnable offline. A bidirectional
entailment back-end is available when a local NLI checkpoint is supplied; it
follows the semantic-entropy formulation of Kuhn et al. (2023) but never calls an
external service.
"""

from __future__ import annotations

from typing import Sequence

import torch


def entropy_from_labels(labels: Sequence[int], eps: float = 1e-12) -> float:
    if not labels:
        return 0.0
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = float(len(labels))
    return float(-sum((c / total) * torch.log(torch.tensor(c / total + eps)) for c in counts.values()))


def normalised_entropy(labels: Sequence[int]) -> float:
    import math

    if len(labels) < 2:
        return 0.0
    return entropy_from_labels(labels) / math.log(len(labels))


def cluster_by_embedding(
    embeddings: torch.Tensor, threshold: float = 0.35
) -> list[int]:
    """Single-linkage clustering on centred cosine distance; deterministic.

    Hidden-state means share a large common component, so raw cosine similarity
    sits near one and every pair looks identical. Removing the dataset mean before
    comparing is what makes a distance threshold meaningful here.
    """
    n = embeddings.shape[0]
    if n == 0:
        return []
    centred = embeddings - embeddings.mean(dim=0, keepdim=True)
    norms = centred.norm(dim=-1, keepdim=True)
    fallback = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    unit = torch.where(norms > 1e-6, centred / norms.clamp_min(1e-8), fallback)
    distance = 1.0 - (unit @ unit.T)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if float(distance[i, j]) <= threshold:
                union(i, j)
    roots: dict[int, int] = {}
    labels = []
    for i in range(n):
        root = find(i)
        roots.setdefault(root, len(roots))
        labels.append(roots[root])
    return labels


class EntailmentClusterer:
    """Bidirectional-entailment clustering using a locally hosted NLI checkpoint."""

    def __init__(self, model_name: str = "microsoft/deberta-large-mnli", device: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if device:
            self.model.to(device)
        self.model.eval()
        self.entailment_index = 2

    @torch.inference_mode()
    def entails(self, premise: str, hypothesis: str) -> bool:
        enc = self.tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=256
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        logits = self.model(**enc).logits[0]
        return int(logits.argmax()) == self.entailment_index

    def cluster(self, texts: Sequence[str]) -> list[int]:
        n = len(texts)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if self.entails(texts[i], texts[j]) and self.entails(texts[j], texts[i]):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
        roots: dict[int, int] = {}
        labels = []
        for i in range(n):
            root = find(i)
            roots.setdefault(root, len(roots))
            labels.append(roots[root])
        return labels


def semantic_diversity(
    texts: Sequence[str],
    embeddings: torch.Tensor | None = None,
    clusterer: EntailmentClusterer | None = None,
    threshold: float = 0.35,
) -> dict:
    if clusterer is not None:
        labels = clusterer.cluster(texts)
        backend = "nli"
    elif embeddings is not None:
        labels = cluster_by_embedding(embeddings, threshold=threshold)
        backend = "embedding"
    else:
        raise ValueError("provide either embeddings or an EntailmentClusterer")
    n_clusters = len(set(labels))
    return {
        "backend": backend,
        "n_clusters": n_clusters,
        "cluster_ratio": n_clusters / max(len(texts), 1),
        "cluster_entropy": entropy_from_labels(labels),
        "cluster_entropy_norm": normalised_entropy(labels),
        "labels": labels,
    }
