"""Typed configuration objects and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelSpec:
    name: str
    short: str
    family: str
    n_layers: int
    hidden_size: int

    @property
    def slug(self) -> str:
        return self.short


@dataclass
class ExtractionConfig:
    layers: Any = "middle"
    pool: str = "mean"
    n_samples: int = 16
    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 96
    hi_quantile: float = 0.8
    lo_quantile: float = 0.2
    min_samples: int = 8
    embed_layer_frac: float = 0.5
    normalize: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.lo_quantile < self.hi_quantile <= 1.0:
            raise ValueError("require 0 <= lo_quantile < hi_quantile <= 1")
        if self.pool not in {"mean", "last"}:
            raise ValueError("pool must be 'mean' or 'last'")


@dataclass
class SteeringConfig:
    layers: Any = "middle"
    coefs: list[float] = field(default_factory=lambda: [-2.0, -1.0, 0.0, 1.0, 2.0])
    positions: str = "all"
    normalize_direction: bool = True


@dataclass
class SamplingConfig:
    n_samples: int = 8
    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 96
    batch_size: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if self.temperature <= 0 and self.n_samples > 1:
            raise ValueError(
                "greedy decoding (temperature <= 0) cannot draw multiple samples; "
                "either raise the temperature or set n_samples to 1"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


def resolve_layers(spec: Any, n_layers: int) -> list[int]:
    """Turn a layer specification into an explicit sorted list of indices.

    Accepts an int, a list of ints, "all", "middle", "last", or a slice string
    such as "4-32:4".
    """
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, (list, tuple)):
        return sorted(int(x) for x in spec)
    if not isinstance(spec, str):
        raise TypeError(f"unsupported layer spec: {spec!r}")

    spec = spec.strip()
    if spec == "all":
        return list(range(n_layers))
    if spec == "middle":
        return sorted({n_layers // 4, n_layers // 2, (3 * n_layers) // 4})
    if spec == "last":
        return [n_layers - 1]
    if spec.isdigit():
        return [int(spec)]
    if "," in spec:
        return sorted({int(part) for part in spec.split(",") if part.strip().isdigit()})
    if ":" in spec and "-" in spec:
        span, step = spec.split(":")
        low, high = (int(x) for x in span.split("-"))
        return list(range(low, min(high, n_layers), int(step)))
    raise ValueError(f"cannot parse layer spec {spec!r}")


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_models(path: str | Path) -> dict[str, ModelSpec]:
    payload = load_yaml(path)
    models = {}
    for key, raw in payload["models"].items():
        models[key] = ModelSpec(
            name=raw["name"],
            short=raw.get("short", key),
            family=raw.get("family", "unknown"),
            n_layers=int(raw["n_layers"]),
            hidden_size=int(raw["hidden_size"]),
        )
    return models


def model_by_short(models: dict[str, ModelSpec], short: str) -> ModelSpec:
    """Look up a model by YAML key, short name, or Hugging Face id."""
    if short in models:
        return models[short]
    for spec in models.values():
        if spec.short == short or spec.name == short:
            return spec
    raise KeyError(
        f"unknown model {short!r}; available keys: {sorted(models)} "
        f"or ids: {[m.name for m in models.values()]}"
    )
