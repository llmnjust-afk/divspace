"""Steering sweeps plus the placebo controls a reliability claim needs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from ..config import SamplingConfig
from ..generate.sampler import GenerationResult, sample
from ..models import LM


def random_direction(hidden: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(hidden, generator=generator)
    return vector / vector.norm().clamp_min(1e-8)


def shuffled_direction(
    positive: torch.Tensor, negative: torch.Tensor, seed: int = 0
) -> torch.Tensor:
    """Same activations, destroyed pairing: isolates the role of the contrast itself."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(negative.shape[0], generator=generator)
    shuffled = negative[perm]
    size = min(positive.shape[0], shuffled.shape[0])
    vector = positive[:size].mean(0) - shuffled[:size].mean(0)
    return vector / vector.norm().clamp_min(1e-8)


def orthogonalised(direction: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    projection = (direction @ basis.T) @ basis
    residual = direction - projection
    return residual / residual.norm().clamp_min(1e-8)


def control_directions(
    lm,
    artifact,
    directions: dict[int, torch.Tensor],
    control: str | None,
    seed: int = 0,
    activations=None,
) -> dict[int, torch.Tensor]:
    """Build the placebo arm: random, pairing-shuffled, orthogonalised, or centroid."""
    if control in (None, "none"):
        return directions
    if control == "random":
        return {
            layer: random_direction(lm.hidden_size, seed=seed + layer) for layer in directions
        }
    if not artifact.positive_pairs:
        raise SystemExit(f"{control} control needs an artifact with stored pairs")
    if activations is None:
        raise SystemExit("activations are required for this control")
    layer_list = list(directions)
    positive = activations([tuple(pair) for pair in artifact.positive_pairs[:32]], layer_list)
    negative = activations([tuple(pair) for pair in artifact.negative_pairs[:32]], layer_list)
    if control == "shuffled":
        return {
            layer: shuffled_direction(positive[layer], negative[layer], seed=seed)
            for layer in layer_list
        }
    if control == "centroid":
        return {
            layer: torch.cat([positive[layer], negative[layer]], 0).mean(0)
            for layer in layer_list
        }
    if control == "orthogonal":
        keys = list(directions)
        if len(keys) < 2:
            raise SystemExit("the orthogonal control needs at least two layers")
        return {
            layer: orthogonalised(
                directions[layer],
                torch.stack([directions[other] for other in keys if other != layer]),
            )
            for layer in keys
        }
    raise SystemExit(f"unknown control {control}")


@dataclass
class Cell:
    layer: int
    coef: float
    condition: str

    @property
    def name(self) -> str:
        return f"{self.condition}_L{self.layer}_c{self.coef:+.2f}"


def _direction_for(direction, layer: int) -> torch.Tensor:
    if isinstance(direction, dict):
        if layer not in direction:
            raise KeyError(f"no direction for layer {layer}; have {sorted(direction)}")
        return direction[layer]
    return direction


def run_grid(
    lm: LM,
    prompts: Sequence[str],
    direction,
    layers: Iterable[int],
    coefs: Iterable[float],
    cfg: SamplingConfig,
    positions: str | int = "all",
    use_chat_template: bool = True,
    condition: str = "steer",
    mode: str = "inject",
    verbose: bool = False,
) -> tuple[GenerationResult, list[Cell]]:
    """Run the full layer x coefficient grid, one cell per (layer, coef) pair."""
    merged = GenerationResult()
    cells: list[Cell] = []
    for layer in layers:
        for coef in coefs:
            if coef == 0.0 and mode == "inject":
                tag = "sweep_identity"
            else:
                tag = condition
            result = sample(
                lm,
                prompts,
                cfg,
                direction=_direction_for(direction, layer),
                layers=[layer],
                coef=coef,
                positions=positions,
                use_chat_template=use_chat_template,
                condition=tag,
                mode=mode,
            )
            for row in result.rows:
                row["cell"] = Cell(layer, float(coef), tag).name
            merged.extend(result)
            cells.append(Cell(layer, float(coef), tag))
            if verbose:
                print(f"    {tag} layer={layer} coef={coef:+.2f} -> {len(result.rows)} samples")
    return merged, cells


def run_baseline(
    lm: LM,
    prompts: Sequence[str],
    cfg: SamplingConfig,
    use_chat_template: bool = True,
    condition: str = "baseline",
) -> GenerationResult:
    result = sample(
        lm, prompts, cfg, use_chat_template=use_chat_template, condition=condition
    )
    for row in result.rows:
        row["cell"] = condition
    return result
