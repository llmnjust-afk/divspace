"""End-to-end direction extraction: sample, rank by dispersion, contrast, fit."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch

from ..config import ExtractionConfig, SamplingConfig, resolve_layers
from ..generate.sampler import prepare_prompt, sample
from ..hooks import capture
from ..models import LM
from ..utils import provenance, seed_everything
from .directions import mean_diff, split_half_stability, unit
from .select import balance, dispersion, prompt_consistency, select_within_prompt


def _token_ids(tokenizer, text: str, add_special_tokens: bool = True) -> list[int]:
    out = tokenizer(text, add_special_tokens=add_special_tokens)
    ids = out["input_ids"] if isinstance(out, dict) else out.input_ids
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


@torch.inference_mode()
def response_activations(
    lm: LM,
    pairs: Sequence[tuple[str, str]],
    layers: Sequence[int],
    batch_size: int = 8,
    max_length: int = 768,
    pool: str = "mean",
) -> dict[int, torch.Tensor]:
    """Teacher-forced hidden states averaged (or last-token picked) over the response span."""
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    out: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        encoded = []
        for prompt, response in chunk:
            prompt_ids = _token_ids(lm.tokenizer, prompt, True)
            response_ids = _token_ids(lm.tokenizer, response, False)
            if not response_ids:
                response_ids = [lm.tokenizer.eos_token_id or 0]
            if len(prompt_ids) + len(response_ids) > max_length:
                response_ids = response_ids[: max(1, max_length - len(prompt_ids))]
            if len(prompt_ids) + len(response_ids) > max_length:
                prompt_ids = prompt_ids[: max(1, max_length - len(response_ids))]
            encoded.append((prompt_ids, response_ids))

        width = max(len(p) + len(r) for p, r in encoded)
        ids = torch.zeros(len(encoded), width, dtype=torch.long)
        mask = torch.zeros(len(encoded), width, dtype=torch.long)
        spans = []
        for row, (prompt_ids, response_ids) in enumerate(encoded):
            seq = prompt_ids + response_ids
            ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[row, : len(seq)] = 1
            spans.append((len(prompt_ids), len(seq)))

        store: dict[int, torch.Tensor] = {}
        with capture(lm.layers, list(layers), store, positions="all", pool=None):
            lm.run_backbone(
                input_ids=ids.to(lm.device), attention_mask=mask.to(lm.device)
            )

        for layer in layers:
            hidden = store[layer]
            for row, (prompt_len, total) in enumerate(spans):
                segment = hidden[row, prompt_len:total, :].float()
                if segment.shape[0] == 0:
                    segment = hidden[row, total - 1 : total, :].float()
                value = segment.mean(dim=0) if pool == "mean" else segment[-1]
                out[layer].append(value.cpu())

    return {layer: torch.stack(values) for layer, values in out.items()}


@dataclass
class DirectionArtifact:
    model: str
    layers: list[int]
    directions: dict[int, torch.Tensor]
    stability: dict[int, dict] = field(default_factory=dict)
    consistency: dict[int, dict] = field(default_factory=dict)
    norms: dict[int, float] = field(default_factory=dict)
    embed_layer: int = 0
    meta: dict = field(default_factory=dict)
    positive_pairs: list[list[str]] = field(default_factory=list)
    negative_pairs: list[list[str]] = field(default_factory=list)

    def layer(self, index: int) -> torch.Tensor:
        if index not in self.directions:
            raise KeyError(f"layer {index} not in artifact; have {sorted(self.directions)}")
        return self.directions[index]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model,
                "layers": self.layers,
                "directions": {k: v.cpu() for k, v in self.directions.items()},
                "stability": self.stability,
                "consistency": self.consistency,
                "norms": self.norms,
                "embed_layer": self.embed_layer,
                "meta": self.meta,
                "positive_pairs": self.positive_pairs,
                "negative_pairs": self.negative_pairs,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DirectionArtifact":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(**payload)


def extract_directions(
    lm: LM,
    prompts: Sequence[str],
    cfg: ExtractionConfig,
    seed: int = 0,
    embed_layer: int | None = None,
    use_chat_template: bool = True,
    batch_size: int = 8,
    max_length: int = 768,
    verbose: bool = False,
) -> DirectionArtifact:
    if not prompts:
        raise ValueError("need at least one prompt")

    seed_everything(seed)
    layers = resolve_layers(cfg.layers, lm.n_layers)
    embed = embed_layer
    if embed is None:
        embed = int(round(cfg.embed_layer_frac * (lm.n_layers - 1)))
    embed = max(0, min(embed, lm.n_layers - 1))

    prepared = [prepare_prompt(lm.tokenizer, p, use_chat_template) for p in prompts]
    sampling = SamplingConfig(
        n_samples=cfg.n_samples,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens,
        batch_size=batch_size,
        seed=seed,
    )
    generated = sample(
        lm, prompts, sampling, use_chat_template=use_chat_template, condition="probe"
    )
    rows = generated.rows
    pairs = [(prepared[row["prompt_id"]], row["text"]) for row in rows]

    embeddings = response_activations(
        lm, pairs, [embed], batch_size=batch_size, max_length=max_length
    )[embed]

    grouped: dict[int, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        grouped[row["prompt_id"]].append(position)

    scores_by_prompt = []
    for prompt_id in sorted(grouped):
        index = grouped[prompt_id]
        scores_by_prompt.append(dispersion(embeddings[index]))

    highs, lows = select_within_prompt(
        scores_by_prompt,
        hi_quantile=cfg.hi_quantile,
        lo_quantile=cfg.lo_quantile,
        min_samples=cfg.min_samples,
    )
    highs, lows = balance(highs, lows)

    selected = [pairs[i] for i in highs + lows]
    acts = response_activations(
        lm, selected, layers, batch_size=batch_size, max_length=max_length, pool=cfg.pool
    )

    # Which prompt each selected sample came from, in the order used for `acts`.
    owner = [rows[i]["prompt_id"] for i in highs + lows]
    split = len(highs)

    directions: dict[int, torch.Tensor] = {}
    stability: dict[int, dict] = {}
    norms: dict[int, float] = {}
    consistency: dict[int, dict] = {}
    for layer in layers:
        positive, negative = acts[layer][:split], acts[layer][split:]
        raw = mean_diff(positive, negative)
        norms[layer] = float(raw.norm())
        directions[layer] = unit(raw) if cfg.normalize else raw
        stability[layer] = split_half_stability(positive, negative, seed=seed)
        consistency[layer] = prompt_consistency(acts[layer], owner, split)

    if verbose:
        for layer in layers:
            print(
                f"  layer {layer:>2d}  ||d||={norms[layer]:7.3f}  "
                f"split-half cos={stability[layer]['cosine']:.3f}  "
                f"prompt-consistency={consistency[layer]['mean_cosine']:+.3f}"
            )

    return DirectionArtifact(
        model=lm.name,
        layers=layers,
        directions=directions,
        stability=stability,
        consistency=consistency,
        norms=norms,
        embed_layer=embed,
        meta={
            "extraction": cfg.__dict__,
            "seed": seed,
            "n_prompts": len(prompts),
            "n_generations": len(rows),
            "n_positive": split,
            "n_negative": len(lows),
            "max_length": max_length,
            "provenance": provenance(lm.device),
        },
        positive_pairs=[[pairs[i][0], pairs[i][1]] for i in highs],
        negative_pairs=[[pairs[i][0], pairs[i][1]] for i in lows],
    )
