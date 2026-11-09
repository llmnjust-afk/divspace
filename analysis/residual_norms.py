"""Mean residual-stream norm per layer, to calibrate absolute steering coefficients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from divspace.config import load_models
from divspace.data.prompts import resolve_prompts
from divspace.models import LM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--prompt-set", default="all")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    models = load_models(args.models_config)
    name = models[args.model].name if args.model in models else args.model
    lm = LM.load(name, device="cuda:0", dtype=args.dtype)
    prompts = resolve_prompts(args.prompt_set)[: args.limit]
    batch = lm.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    batch = {key: value.to(lm.device) for key, value in batch.items()}

    captured: dict[int, float] = {}
    handles = []
    for index, layer in enumerate(lm.layers):
        def make(idx):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[idx] = float(hidden.norm(dim=-1).mean())
            return hook
        handles.append(layer.register_forward_hook(make(index)))

    with torch.inference_mode():
        lm.model(**batch, output_hidden_states=False)
    for handle in handles:
        handle.remove()

    norms = {index: round(captured[index], 2) for index in sorted(captured)}
    print(f"{lm.name}: residual norms {norms}")
    if args.out:
        Path(args.out).write_text(json.dumps(norms, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
