"""Regression tests for defects found in review."""

import math

import pytest
import torch

from divspace.config import SamplingConfig, load_models, model_by_short
from divspace.extract.directions import unit
from divspace.extract.pipeline import response_activations
from divspace.extract.select import dispersion
from divspace.hooks import capture
from divspace.intervene.steer import orthogonalised
from divspace.controls import fit_contrast_direction
from divspace.local import run_local
from divspace.models import build_tiny


def test_dispersion_has_no_constant_offset():
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    scores = dispersion(embeddings)
    assert math.isclose(float(scores[0]), 0.5, rel_tol=1e-5)
    assert math.isclose(float(scores[2]), 1.0, rel_tol=1e-5)


def test_dispersion_two_samples_equals_the_pair_distance():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    scores = dispersion(embeddings)
    assert math.isclose(float(scores[0]), 1.0, rel_tol=1e-5)
    assert math.isclose(float(scores[1]), 1.0, rel_tol=1e-5)


def test_dispersion_never_negative():
    embeddings = torch.randn(12, 16)
    assert float(dispersion(embeddings).min()) >= -1e-6


def test_orthogonal_control_excludes_the_direction_itself():
    a = unit(torch.tensor([1.0, 0.0, 0.0]))
    b = unit(torch.tensor([0.0, 1.0, 0.0]))
    c = unit(torch.tensor([0.0, 0.0, 1.0]))
    residual = orthogonalised(c, torch.stack([a, b]))
    assert torch.allclose(residual, c, atol=1e-6)


def test_model_lookup_accepts_the_yaml_key(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n"
        "  tiny-key:\n"
        "    name: EleutherAI/pythia-410m\n"
        "    short: tiny\n"
        "    n_layers: 24\n"
        "    hidden_size: 1024\n",
        encoding="utf-8",
    )
    models = load_models(config)
    assert model_by_short(models, "tiny-key").name == "EleutherAI/pythia-410m"
    assert model_by_short(models, "tiny").name == "EleutherAI/pythia-410m"


def test_greedy_sampling_rejects_multiple_samples():
    with pytest.raises(ValueError, match="greedy"):
        SamplingConfig(n_samples=4, temperature=0.0)


def test_response_activations_survive_a_prompt_longer_than_the_budget():
    lm = build_tiny(n_layers=2)
    long_prompt = " ".join(["word"] * 400)
    activations = response_activations(
        lm, [(long_prompt, "short answer")], [1], batch_size=1, max_length=8
    )[1]
    assert activations.shape == (1, lm.hidden_size)
    assert torch.isfinite(activations).all()


def test_response_activations_ignore_padding_in_the_pooled_mean():
    lm = build_tiny(n_layers=2)
    pairs = [("a b c", "d e"), ("a b c", "d e f g h i j k")]
    activations = response_activations(lm, pairs, [1], batch_size=2, max_length=64)[1]
    assert activations.shape == (2, lm.hidden_size)
    assert torch.isfinite(activations).all()


def test_backbone_forward_returns_no_logits():
    lm = build_tiny(n_layers=2)
    ids = torch.tensor([[1, 2, 3]])
    output = lm.run_backbone(input_ids=ids)
    assert hasattr(output, "last_hidden_state")
    assert not hasattr(output, "logits")


def test_capture_last_position_shape():
    lm = build_tiny(n_layers=2)
    ids = torch.tensor([[1, 2, 3, 4]])
    store: dict[int, torch.Tensor] = {}
    with capture(lm.layers, [1], store, positions="last", pool=None):
        lm.run_backbone(input_ids=ids)
    assert store[1].shape == (1, 1, lm.hidden_size)


def test_contrast_direction_is_unit_and_reproducible(tmp_path):
    lm = build_tiny(n_layers=4)
    pairs = [[f"q{i}", "Yes"] for i in range(6)]
    negative = [[f"q{i}", "No"] for i in range(6)]
    first = fit_contrast_direction(lm, pairs, negative, [2])
    second = fit_contrast_direction(lm, pairs, negative, [2])
    assert first.directions[2].shape == (lm.hidden_size,)
    assert abs(float(first.directions[2].norm()) - 1.0) < 1e-5
    assert torch.allclose(first.directions[2], second.directions[2], atol=1e-6)


def test_local_run_emits_one_cell_per_condition():
    lm = build_tiny(n_layers=4)
    rows = run_local(
        lm,
        ["prompt a", "prompt b"],
        layer=2,
        fit_samples=8,
        eval_samples=2,
        coefs=[-1.0, 1.0],
        temperature=1.0,
        top_p=0.95,
        max_new_tokens=4,
        batch_size=2,
        embed_layer=2,
        seed=0,
    )
    cells = {row["cell"] for row in rows}
    assert "baseline" in cells
    assert "local_L2_c+1.00" in cells and "local_L2_c-1.00" in cells
    assert "localrandom_L2_c+1.00" in cells
    assert "localother_L2_c+1.00" in cells
    assert {row["prompt_id"] for row in rows} == {0, 1}
