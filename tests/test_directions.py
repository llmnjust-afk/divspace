import torch

from divspace.extract.directions import (
    cosine,
    mean_diff,
    orthogonalize,
    pca_basis,
    split_half_stability,
    unit,
)
from divspace.extract.select import dispersion, select_within_prompt


def test_unit_normalises():
    vector = torch.tensor([3.0, 4.0])
    assert abs(float(unit(vector).norm()) - 1.0) < 1e-6


def test_mean_diff_is_difference_of_means():
    positive = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    negative = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    result = mean_diff(positive, negative)
    assert torch.allclose(result, torch.tensor([2.0, -1.0]))


def test_pca_basis_is_orthonormal():
    matrix = torch.randn(64, 16)
    basis = pca_basis(matrix, 4)
    assert basis.shape == (4, 16)
    gram = basis @ basis.T
    assert torch.allclose(gram, torch.eye(4), atol=1e-4)


def test_orthogonalize_removes_component():
    basis = torch.tensor([[1.0, 0.0, 0.0]])
    vector = torch.tensor([2.0, 3.0, 0.0])
    residual = orthogonalize(vector, basis)
    assert abs(float(residual[0])) < 1e-6
    assert abs(float(residual[1]) - 3.0) < 1e-6


def test_split_half_stability_high_for_consistent_signal():
    positive = torch.randn(200, 32) + 1.0
    negative = torch.randn(200, 32) - 1.0
    report = split_half_stability(positive, negative)
    assert report["cosine"] > 0.8


def test_dispersion_orders_identical_samples_lowest():
    cluster = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    scores = dispersion(cluster)
    assert float(scores[0]) >= 0.0
    assert float(scores.sum()) > 0.0


def test_select_within_prompt_returns_balanced_groups():
    torch.manual_seed(0)
    groups = [torch.rand(10) for _ in range(4)]
    highs, lows = select_within_prompt(groups, 0.8, 0.2, min_samples=8)
    assert len(highs) == len(lows) == 8
    assert len(set(highs) & set(lows)) == 0


def test_cosine_bounds():
    a = torch.randn(16)
    b = torch.randn(16)
    value = cosine(a, a)
    assert abs(value - 1.0) < 1e-5
    assert -1.01 <= cosine(a, b) <= 1.01
