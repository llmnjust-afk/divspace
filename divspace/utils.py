"""Shared helpers: seeding, devices, layer lookup, JSONL IO, GPU accounting."""

from __future__ import annotations

import json
import os
import platform
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

_LAYER_PATHS = (
    "model.layers",
    "model.model.layers",
    "model.decoder.layers",
    "gpt_neox.layers",
    "transformer.h",
    "transformer.blocks",
)

_NORM_PATHS = (
    "model.norm",
    "model.model.norm",
    "model.decoder.final_layer_norm",
    "gpt_neox.final_layer_norm",
    "transformer.ln_f",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str | None) -> torch.device:
    if spec is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {spec!r} requested but CUDA is unavailable")
    return torch.device(spec)


def parse_dtype(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"unknown dtype {name!r}; choose from {sorted(table)}")
    return table[name]


def find_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    for path in _LAYER_PATHS:
        try:
            module = model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            return module
    raise AttributeError(
        f"could not locate the decoder layer list on {type(model).__name__}; "
        f"tried {_LAYER_PATHS}"
    )


def find_final_norm(model: torch.nn.Module) -> torch.nn.Module | None:
    for path in _NORM_PATHS:
        try:
            module = model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(module, torch.nn.Module):
            return module
    return None


def write_jsonl(path: str | os.PathLike, rows: Any, append: bool = True) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | os.PathLike, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: str | os.PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def shard(items: list, shard_id: int, num_shards: int) -> list:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    return items[shard_id::num_shards]


def shard_with_ids(items: list, shard_id: int, num_shards: int) -> tuple[list, list[int]]:
    """Shard by stride and return the global ids so shards stay joinable later."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    ids = list(range(len(items)))[shard_id::num_shards]
    return [items[i] for i in ids], ids


def remap_prompt_ids(rows: list[dict], mapping: list[int]) -> list[dict]:
    for row in rows:
        local = row.get("prompt_id", 0)
        if 0 <= local < len(mapping):
            row["prompt_id"] = mapping[local]
    return rows


def provenance(device: torch.device | None = None) -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None
    if torch.cuda.is_available() and (device is None or device.type == "cuda"):
        index = device.index if device is not None and device.index is not None else 0
        info["gpu"] = torch.cuda.get_device_name(index)
        info["gpu_total_gb"] = round(
            torch.cuda.get_device_properties(index).total_memory / 1024**3, 1
        )
    return info


@contextmanager
def gpu_clock(device: torch.device | None = None) -> Iterator[dict]:
    """Yield a dict that is populated with wall time and peak VRAM on exit."""
    stats: dict = {}
    start = time.time()
    if device is not None and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    try:
        yield stats
    finally:
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
            stats["peak_vram_gb"] = round(
                torch.cuda.max_memory_allocated(device) / 1024**3, 2
            )
        stats["seconds"] = round(time.time() - start, 2)
        stats["gpu_hours"] = round(stats["seconds"] / 3600.0, 4)


def humanise(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"
