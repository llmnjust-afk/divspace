"""Aggregating per-prompt metrics into per-condition scores."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Sequence

import torch

from . import embedding as emb_metrics
from . import lexical, quality, semantic


def score_group(
    texts: Sequence[str],
    embeddings: torch.Tensor,
    cluster_threshold: float = 0.35,
) -> dict:
    """All diversity and quality signals for the samples drawn for a single prompt."""
    texts = list(texts)
    scores = {
        "n": len(texts),
        "distinct_1": lexical.distinct_n(texts, 1),
        "distinct_2": lexical.distinct_n(texts, 2),
        "distinct_3": lexical.distinct_n(texts, 3),
        "rep_3": lexical.rep_n(texts, 3),
        "type_token_ratio": lexical.type_token_ratio(texts),
        "self_bleu": lexical.self_bleu(texts),
        "mpd": emb_metrics.mean_pairwise_distance(embeddings),
        "vendi": emb_metrics.vendi_score(embeddings),
        "effective_rank": emb_metrics.effective_rank(embeddings),
        "anisotropy": emb_metrics.anisotropy(embeddings),
    }
    semantic_scores = semantic.semantic_diversity(
        texts, embeddings=embeddings, threshold=cluster_threshold
    )
    scores["n_clusters"] = semantic_scores["n_clusters"]
    scores["cluster_ratio"] = semantic_scores["cluster_ratio"]
    scores["cluster_entropy_norm"] = semantic_scores["cluster_entropy_norm"]
    scores.update(quality.quality_report(texts))
    return scores


def aggregate_by_cell(
    rows: Sequence[dict],
    embeddings: dict[int, torch.Tensor],
    cluster_threshold: float = 0.35,
) -> dict[str, dict]:
    """Average per-prompt scores within each experimental cell and report dispersion."""
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        groups[(row.get("cell", "unknown"), row["prompt_id"])].append(position)

    per_cell: dict[str, list[dict]] = defaultdict(list)
    for (cell, _prompt), positions in groups.items():
        texts = [rows[i]["text"] for i in positions]
        matrix = torch.stack([embeddings[i] for i in positions])
        per_cell[cell].append(
            score_group(texts, matrix, cluster_threshold=cluster_threshold)
        )

    summary: dict[str, dict] = {}
    for cell, group_scores in per_cell.items():
        keys = group_scores[0].keys()
        entry: dict = {"n_prompts": len(group_scores)}
        for key in keys:
            values = [float(s[key]) for s in group_scores if isinstance(s[key], (int, float))]
            if not values:
                continue
            entry[key] = statistics.fmean(values)
            entry[f"{key}_stderr"] = (
                statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
            )
        summary[cell] = entry
    return summary


def mauve(candidate_texts: Sequence[str], reference_texts: Sequence[str], featuriser: str = "bert-base-uncased") -> dict:
    """Thin wrapper around mauve-text; returns a dict so it can be merged into results."""
    try:
        from mauve import compute_mauve

        value = compute_mauve(
            p_texts=list(candidate_texts),
            q_texts=list(reference_texts),
            featurize_model_name=featuriser,
            device_id=-1,
            max_text_length=256,
            verbose=False,
        )
        return {"mauve": float(value.mauve)}
    except ImportError:
        return {"mauve": None, "mauve_error": "mauve-text not installed"}
