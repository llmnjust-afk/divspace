"""Tests for the budget-matched stability suite."""

import math

import torch

from divspace.budget import (
    PromptSamples,
    between_prompt_direction,
    pooled_budget_cosines,
    prompt_directions,
    within_prompt_cosines,
)


def planted(n_prompts=6, n_samples=12, hidden=24, seed=0):
    """Prompts differ in spread, and the spread lives on the same axis as the responses."""
    generator = torch.Generator().manual_seed(seed)
    axis = torch.zeros(hidden)
    axis[0] = 1.0
    samples = []
    for index in range(n_prompts):
        spread = 0.5 + 1.5 * index / max(1, n_prompts - 1)
        scores = spread * (
            0.6
            + torch.linspace(-1.0, 1.0, n_samples)[
                torch.randperm(n_samples, generator=generator)
            ]
        )
        acts = (spread + scores).unsqueeze(1) * axis + 0.01 * torch.randn(
            n_samples, hidden, generator=generator
        )
        samples.append(PromptSamples(prompt=str(index), activations={4: acts}, scores=scores))
    return samples


def test_within_prompt_cosines_recover_a_planted_axis():
    cosines = within_prompt_cosines(planted(), layer=4, seed=0, k=2)
    assert cosines
    assert min(cosines) > 0.9


def test_pooled_budget_curve_stays_high_when_the_axis_is_shared():
    curve = pooled_budget_cosines(planted(), layer=4, sizes=[1, 2, 4], reps=5, seed=0, k=2)
    assert set(curve) == {1, 2, 4}
    for size, values in curve.items():
        assert values, f"no draws for m={size}"
        assert sum(values) / len(values) > 0.9


def test_between_prompt_cosine_is_high_when_prompts_carry_the_axis():
    pair = between_prompt_direction(planted(n_prompts=10), layer=4, quantile=0.2)
    assert pair is not None
    between, pooled = pair
    assert math.isclose(float(between.norm()), 1.0, rel_tol=1e-4)
    assert float(between @ pooled) > 0.2


def test_prompt_directions_are_unit_vectors():
    directions = prompt_directions(planted(), layer=4, k=2)
    assert len(directions) == 6
    for vector in directions:
        norm = math.sqrt(sum(value * value for value in vector))
        assert math.isclose(norm, 1.0, rel_tol=1e-4)
