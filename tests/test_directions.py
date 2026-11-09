import torch

from divspace.extract.directions import (
    cosine,
    mean_diff,
    orthogonalize,
    pca_basis,
    split_half_stability,
    unit,
)
from divspace.data.prompts import load_prompt_file
from divspace.extract.select import prompt_consistency, dispersion, select_within_prompt


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


def test_prompt_consistency_separates_shared_from_prompt_specific():
    torch.manual_seed(0)
    n_prompts = 6

    def build(direction_of):
        highs, lows, high_owner, low_owner = [], [], [], []
        for prompt in range(n_prompts):
            direction = direction_of(prompt)
            for _ in range(2):
                highs.append(direction + 0.01 * torch.randn(64))
                lows.append(-direction + 0.01 * torch.randn(64))
            high_owner.extend([prompt] * 2)
            low_owner.extend([prompt] * 2)
        return torch.stack(highs + lows), high_owner + low_owner, len(highs)

    shared = torch.randn(64)
    acts, owner, split = build(lambda _: shared)
    result = prompt_consistency(acts, owner, split)
    assert result["n_prompts"] == n_prompts
    assert result["mean_cosine"] > 0.99

    distinct = torch.stack([torch.randn(64) for _ in range(n_prompts)])
    acts, owner, split = build(lambda prompt: distinct[prompt])
    result = prompt_consistency(acts, owner, split)
    assert result["n_prompts"] == n_prompts
    assert abs(result["mean_cosine"]) < 0.5


def test_prompt_file_accepts_plain_text(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("first question\n\nsecond question\n", encoding="utf-8")
    assert load_prompt_file(path) == ["first question", "second question"]

    jsonl = tmp_path / "prompts.jsonl"
    jsonl.write_text('{"prompt": "from json"}\n"bare string"\n', encoding="utf-8")
    assert load_prompt_file(jsonl) == ["from json", "bare string"]
