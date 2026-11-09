"""Forward hooks for capturing and perturbing decoder-layer activations.

Every decoder layer in the architectures we target returns a tuple whose first
element is the hidden state, so both helpers normalise on that contract.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, Sequence

import torch
from torch import nn


def _split(output):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, ()


def _join(hidden, rest):
    if rest:
        return (hidden,) + rest
    return hidden


def select_positions(hidden: torch.Tensor, positions: str | int) -> slice | list[int]:
    width = hidden.shape[1]
    if positions in ("all", None):
        return slice(0, width)
    if positions == "last":
        return [width - 1]
    if positions == "first":
        return [0]
    if isinstance(positions, int):
        return list(range(max(0, width - positions), width))
    raise ValueError(f"unsupported positions spec {positions!r}")


@contextmanager
def capture(
    layers: nn.ModuleList,
    indices: Sequence[int],
    store: dict[int, torch.Tensor],
    positions: str | int = "all",
    pool: str | None = None,
    retain_grad: bool = False,
    detach: bool = True,
) -> Iterator[dict[int, torch.Tensor]]:
    """Store hidden states for the requested layers.

    pool="mean" collapses the token axis, pool="last" keeps the final token only,
    and pool=None keeps the full (batch, seq, hidden) tensor.
    """
    handles = []

    if pool not in (None, "mean", "last"):
        raise ValueError(f"pool must be None, 'mean' or 'last', got {pool!r}")

    def make(index: int) -> Callable:
        def hook(module, args, output):
            hidden, _ = _split(output)
            picked = hidden[:, select_positions(hidden, positions), :]
            if pool == "mean":
                picked = picked.mean(dim=1)
            elif pool == "last":
                picked = picked[:, -1, :]
            if detach:
                picked = picked.detach()
            if retain_grad and picked.requires_grad:
                picked.retain_grad()
            store[index] = picked

        return hook

    for index in indices:
        handles.append(layers[index].register_forward_hook(make(index)))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_with_grad(
    layers: nn.ModuleList,
    indices: Sequence[int],
    store: dict[int, torch.Tensor],
) -> Iterator[dict[int, torch.Tensor]]:
    """Store the raw layer outputs and keep their gradients for attribution.

    The stored tensor must be the very node that sits on the forward path; a
    sliced copy of it receives no gradient from the loss and would report None.
    """
    handles = []

    def make(index: int) -> Callable:
        def hook(module, args, output):
            hidden, _ = _split(output)
            if hidden.requires_grad:
                hidden.retain_grad()
            store[index] = hidden

        return hook

    for index in indices:
        handles.append(layers[index].register_forward_hook(make(index)))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def inject(
    layers: nn.ModuleList,
    indices: Iterable[int],
    direction: torch.Tensor,
    coef: float,
    positions: str | int = "all",
    relative: bool = False,
) -> Iterator[None]:
    """Add coef * direction to the residual stream leaving each layer.

    With relative=True the addition has norm |coef| * ||h||, i.e. coef is a
    fraction of the residual-stream norm. Without it coef multiplies the raw
    direction, whose norm differs by three orders of magnitude across models.
    """
    if direction.ndim != 1:
        raise ValueError(f"direction must be 1-D, got shape {tuple(direction.shape)}")
    handles = []

    def make(index: int) -> Callable:
        def hook(module, args, output):
            hidden, rest = _split(output)
            shaped = hidden.clone()
            idx = select_positions(shaped, positions)
            block = shaped[:, idx, :]
            vector = direction.to(dtype=hidden.dtype, device=hidden.device)
            if relative:
                scale = block.norm(dim=-1, keepdim=True).mean()
                vector = vector / vector.norm().clamp_min(1e-8) * scale
            shaped[:, idx, :] = block + coef * vector
            return _join(shaped, rest)

        return hook

    for index in indices:
        handles.append(layers[index].register_forward_hook(make(index)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def ablate(
    layers: nn.ModuleList,
    indices: Iterable[int],
    direction: torch.Tensor,
    positions: str | int = "all",
) -> Iterator[None]:
    """Project the direction out of the residual stream (causal necessity test)."""
    unit = direction / direction.norm().clamp_min(1e-8)
    handles = []

    def make(index: int) -> Callable:
        def hook(module, args, output):
            hidden, rest = _split(output)
            delta = unit.to(dtype=hidden.dtype, device=hidden.device)
            shaped = hidden.clone()
            idx = select_positions(shaped, positions)
            block = shaped[:, idx, :]
            shaped[:, idx, :] = block - (block @ delta).unsqueeze(-1) * delta
            return _join(shaped, rest)

        return hook

    for index in indices:
        handles.append(layers[index].register_forward_hook(make(index)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
