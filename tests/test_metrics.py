import math

import torch

from divspace.metrics.embedding import (
    anisotropy,
    effective_rank,
    mean_pairwise_distance,
    vendi_score,
)
from divspace.metrics.lexical import distinct_n, rep_n, self_bleu, sentence_bleu
from divspace.metrics.quality import completion_rate, degeneracy
from divspace.metrics.semantic import cluster_by_embedding, semantic_diversity


SAMPLES = [
    "The sea was calm and the boat drifted slowly toward the harbour.",
    "A storm broke over the bay and the boat turned for home.",
    "Nobody expected the lighthouse to go dark that night.",
]


def test_identical_samples_have_zero_diversity():
    same = ["the same sentence repeated"] * 6
    assert distinct_n(same, 2) < 0.2
    assert self_bleu(same) > 0.99
    assert degeneracy(same) > 0.5


def test_distinct_n_is_bounded():
    value = distinct_n(SAMPLES, 1)
    assert 0.0 <= value <= 1.0


def test_self_bleu_identical_is_one():
    assert self_bleu(["alpha beta gamma delta"] * 3) > 0.99


def test_sentence_bleu_self_is_one():
    text = "a b c d e f"
    assert abs(sentence_bleu(text, text) - 1.0) < 1e-6


def test_vendi_identical_is_one():
    embeddings = torch.ones(8, 16)
    assert abs(vendi_score(embeddings) - 1.0) < 1e-3


def test_vendi_orthogonal_is_n():
    embeddings = torch.eye(6)
    assert abs(vendi_score(embeddings) - 6.0) < 1e-2


def test_mean_pairwise_distance_in_range():
    embeddings = torch.randn(10, 8)
    value = mean_pairwise_distance(embeddings)
    assert 0.0 <= value <= 2.0


def test_effective_rank_bounded_by_n():
    embeddings = torch.randn(10, 32)
    value = effective_rank(embeddings)
    assert 1.0 <= value <= 10.0 + 1e-6


def test_anisotropy_of_single_cluster_is_high():
    base = torch.randn(1, 64)
    embeddings = base + 0.01 * torch.randn(12, 64)
    assert anisotropy(embeddings) > 0.9


def test_clustering_groups_identical_items():
    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.01], [0.0, 1.0], [0.01, 1.0]]
    )
    labels = cluster_by_embedding(embeddings, threshold=0.2)
    assert len(set(labels)) == 2


def test_semantic_diversity_payload():
    embeddings = torch.eye(4)
    payload = semantic_diversity(["a", "b", "c", "d"], embeddings=embeddings)
    assert payload["n_clusters"] == 4
    assert set(payload) >= {"backend", "cluster_entropy_norm"}


def test_completion_rate():
    assert completion_rate(["done.", "also done!"]) == 1.0
    assert completion_rate(["unfinished", "done."]) == 0.5


def test_rep_n_empty_text():
    assert rep_n([""], 2) == 0.0
    assert math.isclose(distinct_n([""], 2), 0.0)
