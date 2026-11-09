"""Batched sampling with optional inference-time steering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from ..config import SamplingConfig
from ..hooks import ablate, inject
from ..models import LM
from ..utils import seed_everything


@dataclass
class GenerationResult:
    rows: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    generated_tokens: int = 0

    def extend(self, other: "GenerationResult") -> None:
        self.rows.extend(other.rows)
        self.prompt_tokens += other.prompt_tokens
        self.generated_tokens += other.generated_tokens


def prepare_prompt(tokenizer, prompt: str, use_chat_template: bool = True) -> str:
    if not use_chat_template:
        return prompt
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        return prompt
    try:
        return apply(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def _encode(lm: LM, texts: Sequence[str], max_length: int | None) -> dict:
    lm.tokenizer.padding_side = "left"
    enc = lm.tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
        add_special_tokens=True,
    )
    return {k: v.to(lm.device) for k, v in enc.items()}


def sample(
    lm: LM,
    prompts: Sequence[str],
    cfg: SamplingConfig,
    direction: torch.Tensor | None = None,
    layers: Sequence[int] | None = None,
    coef: float = 0.0,
    positions: str | int = "all",
    use_chat_template: bool = True,
    condition: str = "base",
    prompt_offset: int = 0,
    max_prompt_tokens: int = 512,
    mode: str = "inject",
    relative: bool = False,
) -> GenerationResult:
    """Draw cfg.n_samples continuations per prompt.

    Steering is applied as a residual-stream addition at every configured layer
    for the whole of generation, including cached steps.
    """
    seed_everything(cfg.seed)
    prepared = [prepare_prompt(lm.tokenizer, p, use_chat_template) for p in prompts]
    result = GenerationResult()

    for start in range(0, len(prepared), cfg.batch_size):
        batch = prepared[start : start + cfg.batch_size]
        enc = _encode(lm, batch, max_prompt_tokens)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

        generate_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=cfg.temperature > 0,
            temperature=max(cfg.temperature, 1e-5),
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            num_return_sequences=cfg.n_samples,
            pad_token_id=getattr(lm.tokenizer, "pad_token_id", None)
            or getattr(lm.tokenizer, "eos_token_id", None),
            eos_token_id=getattr(lm.tokenizer, "eos_token_id", None),
            use_cache=True,
        )

        context = _nullcontext()
        if direction is not None and (coef != 0.0 or mode == "ablate"):
            if mode == "ablate":
                context = ablate(lm.layers, layers, direction, positions)
            elif mode == "inject":
                context = inject(lm.layers, layers, direction, coef, positions, relative)
            else:
                raise ValueError(f"unknown intervention mode {mode!r}")

        with torch.inference_mode(), context:
            output = lm.model.generate(**generate_kwargs)

        width = input_ids.shape[1]
        result.prompt_tokens += int(input_ids.numel()) * cfg.n_samples
        for local, _ in enumerate(batch):
            block = output[local * cfg.n_samples : (local + 1) * cfg.n_samples]
            for k in range(block.shape[0]):
                new_ids = block[k, width:]
                text = lm.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                result.rows.append(
                    {
                        "prompt_id": prompt_offset + start + local,
                        "prompt": prompts[start + local],
                        "sample_id": k,
                        "text": text,
                        "generated_tokens": int(
                            (new_ids != getattr(lm.tokenizer, "pad_token_id", None)).sum()
                        ),
                        "condition": condition,
                        "mode": mode,
                        "coef": float(coef),
                        "layers": list(layers) if layers is not None else None,
                        "seed": cfg.seed,
                    }
                )
                result.generated_tokens += int(new_ids.shape[0])

    return result


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
