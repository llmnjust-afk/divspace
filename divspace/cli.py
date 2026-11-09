"""Command line entry points for every experimental stage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

from .attribute.attribution import attribution_patching, top_dimensions
from .attribute.logit_lens import logit_lens
from .config import (
    ExtractionConfig,
    SamplingConfig,
    load_models,
    model_by_short,
    resolve_layers,
)
from .data.prompts import resolve_prompts
from .extract.pipeline import DirectionArtifact, extract_directions, response_activations
from .intervene.steer import (
    orthogonalised,
    random_direction,
    run_baseline,
    run_grid,
    shuffled_direction,
)
from .metrics.aggregate import aggregate_by_cell, mauve
from .models import LM
from .utils import (
    gpu_clock,
    provenance,
    read_jsonl,
    remap_prompt_ids,
    shard_with_ids,
    write_json,
    write_jsonl,
)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="HF id or key from models.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")


def _add_shard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)


def _add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt-set", default="open")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-chat-template", action="store_true")


def _resolve_model_id(args) -> str:
    path = Path(args.models_config)
    if path.exists():
        models = load_models(path)
        try:
            return model_by_short(models, args.model).name
        except KeyError:
            return args.model
    return args.model


def _load_model(args) -> LM:
    return LM.load(
        _resolve_model_id(args),
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )


def cmd_probe(args) -> None:
    lm = _load_model(args)
    rows = []
    for batch_size in args.batch_sizes:
        prompts = [args.probe_prompt] * batch_size
        cfg = SamplingConfig(
            n_samples=1,
            temperature=0.8,
            top_p=0.9,
            max_new_tokens=args.max_new_tokens,
            batch_size=batch_size,
            seed=0,
        )
        try:
            with gpu_clock(lm.device) as stats:
                result = run_baseline(
                    lm, prompts, cfg, use_chat_template=not args.no_chat_template
                )
            rows.append(
                {
                    "batch_size": batch_size,
                    "status": "ok",
                    "tokens_per_second": round(result.generated_tokens / max(stats["seconds"], 1e-6), 1),
                    **stats,
                }
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append({"batch_size": batch_size, "status": "oom"})
            break
        print(json.dumps(rows[-1]))
    write_json(args.out, {"model": lm.name, "provenance": provenance(lm.device), "rows": rows})


def cmd_sample(args) -> None:
    lm = _load_model(args)
    prompts, prompt_ids = shard_with_ids(
        resolve_prompts(args.prompt_set, args.prompt_file, args.limit),
        args.shard_id,
        args.num_shards,
    )
    cfg = SamplingConfig(
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    with gpu_clock(lm.device) as stats:
        result = run_baseline(
            lm, prompts, cfg, use_chat_template=not args.no_chat_template
        )
    remap_prompt_ids(result.rows, prompt_ids)
    for row in result.rows:
        row["model"] = lm.name
        row["shard"] = args.shard_id
    write_jsonl(args.out, result.rows)
    print(f"wrote {len(result.rows)} samples in {stats['seconds']}s -> {args.out}")


def cmd_extract(args) -> None:
    lm = _load_model(args)
    prompts, _ = shard_with_ids(
        resolve_prompts(args.prompt_set, args.prompt_file, args.limit),
        args.shard_id,
        args.num_shards,
    )
    cfg = ExtractionConfig(
        layers=args.layers,
        pool=args.pool,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        hi_quantile=args.hi_quantile,
        lo_quantile=args.lo_quantile,
        min_samples=args.min_samples,
        embed_layer_frac=args.embed_layer_frac,
        normalize=not args.raw_direction,
    )
    with gpu_clock(lm.device) as stats:
        artifact = extract_directions(
            lm,
            prompts,
            cfg,
            seed=args.seed,
            embed_layer=args.embed_layer,
            use_chat_template=not args.no_chat_template,
            batch_size=args.batch_size,
            max_length=args.max_length,
            verbose=True,
        )
    artifact.meta["gpu_stats"] = stats
    artifact.save(args.out)
    print(
        f"{lm.name}: {len(artifact.layers)} layers, "
        f"{len(artifact.positive_pairs)} positive / {len(artifact.negative_pairs)} negative pairs"
        f" in {stats['seconds']}s -> {args.out}"
    )


def cmd_steer(args) -> None:
    lm = _load_model(args)
    artifact = DirectionArtifact.load(args.artifact)
    prompts, prompt_ids = shard_with_ids(
        resolve_prompts(args.prompt_set, args.prompt_file, args.limit),
        args.shard_id,
        args.num_shards,
    )
    layers = (
        resolve_layers(args.layers, lm.n_layers)
        if args.layers
        else [layer for layer in artifact.layers if layer < lm.n_layers]
    )
    direction = {layer: artifact.directions[layer] for layer in layers if layer in artifact.directions}
    if not direction:
        raise SystemExit("artifact contains none of the requested layers")

    if args.control == "random":
        direction = {
            layer: random_direction(lm.hidden_size, seed=args.seed + layer)
            for layer in direction
        }
    elif args.control == "shuffled":
        if not artifact.positive_pairs:
            raise SystemExit("shuffled control needs an artifact with stored pairs")
        layer_list = list(direction)
        positive_acts = response_activations(
            lm, [tuple(pair) for pair in artifact.positive_pairs[:32]], layer_list
        )
        negative_acts = response_activations(
            lm, [tuple(pair) for pair in artifact.negative_pairs[:32]], layer_list
        )
        direction = {
            layer: shuffled_direction(
                positive_acts[layer], negative_acts[layer], seed=args.seed
            )
            for layer in layer_list
        }
    elif args.control == "orthogonal":
        keys = list(direction)
        if len(keys) < 2:
            raise SystemExit("the orthogonal control needs at least two layers")
        direction = {
            layer: orthogonalised(
                direction[layer],
                torch.stack([direction[other] for other in keys if other != layer]),
            )
            for layer in keys
        }

    cfg = SamplingConfig(
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    rows = []
    if args.include_baseline:
        baseline = run_baseline(
            lm, prompts, cfg, use_chat_template=not args.no_chat_template
        )
        for row in baseline.rows:
            row["model"] = lm.name
            row["shard"] = args.shard_id
        rows.extend(baseline.rows)

    coefs = [0.0] if args.mode == "ablate" else args.coefs
    with gpu_clock(lm.device) as stats:
        grid, cells = run_grid(
            lm,
            prompts,
            direction,
            layers=list(direction),
            coefs=coefs,
            cfg=cfg,
            positions=args.positions,
            use_chat_template=not args.no_chat_template,
            condition=args.condition,
            mode=args.mode,
            verbose=args.verbose,
        )
    for row in grid.rows:
        row["model"] = lm.name
        row["shard"] = args.shard_id
        row["control"] = args.control
    rows.extend(grid.rows)
    remap_prompt_ids(rows, prompt_ids)
    write_jsonl(args.out, rows)
    print(f"{len(cells)} cells, {len(rows)} samples in {stats['seconds']}s -> {args.out}")


def cmd_attribute(args) -> None:
    lm = _load_model(args)
    artifact = DirectionArtifact.load(args.artifact)
    if not artifact.positive_pairs or not artifact.negative_pairs:
        raise SystemExit("artifact has no stored pairs; re-run extract")
    layers = (
        resolve_layers(args.layers, lm.n_layers) if args.layers else artifact.layers
    )

    positive_by_prompt: dict[str, list[str]] = defaultdict(list)
    for prompt, response in artifact.positive_pairs:
        positive_by_prompt[prompt].append(response)
    negative_by_prompt: dict[str, list[str]] = defaultdict(list)
    for prompt, response in artifact.negative_pairs:
        negative_by_prompt[prompt].append(response)

    trials: list[tuple[str, str, str]] = []
    for prompt in positive_by_prompt:
        if prompt not in negative_by_prompt:
            continue
        positives = positive_by_prompt[prompt]
        negatives = negative_by_prompt[prompt]
        for i in range(min(len(positives), len(negatives))):
            trials.append((prompt, positives[i], negatives[i]))
    if not trials:
        raise SystemExit("no prompt has both a positive and a negative response")
    trials = trials[: args.n_pairs]

    n = len(trials)
    totals = {layer: 0.0 for layer in layers}
    vectors = {layer: torch.zeros(lm.hidden_size) for layer in layers}
    deltas = []
    for prompt, positive, negative in trials:
        report = attribution_patching(
            lm, prompt, positive, negative, layers, max_length=args.max_length
        )
        deltas.append(report["delta_metric"])
        for layer in layers:
            totals[layer] += report["scalars"][layer]
            vectors[layer] += report["vectors"][layer]
    averaged = {layer: totals[layer] / n for layer in layers}
    payload = {
        "model": lm.name,
        "layers": layers,
        "mean_attribution": averaged,
        "mean_delta_metric": sum(deltas) / len(deltas),
        "n_pairs": n,
        "top_dimensions": {
            str(layer): top_dimensions(vectors[layer] / n, k=args.top_k) for layer in layers
        },
        "provenance": provenance(lm.device),
    }
    write_json(args.out, payload)
    torch.save({layer: vectors[layer] / n for layer in layers}, Path(args.out).with_suffix(".pt"))
    ranked = sorted(averaged.items(), key=lambda kv: abs(kv[1]), reverse=True)
    for layer, value in ranked[:10]:
        print(f"  layer {layer:>2d}  attribution={value:+.4f}")


def cmd_metrics(args) -> None:
    rows = list(read_jsonl(args.results))
    if args.max_per_group:
        trimmed, counts = [], {}
        for row in rows:
            key = (row.get("cell"), row["prompt_id"])
            counts[key] = counts.get(key, 0) + 1
            if counts[key] <= args.max_per_group:
                trimmed.append(row)
        rows = trimmed
    if not rows:
        raise SystemExit(f"no rows in {args.results}")

    lm = _load_model(args)
    artifact = None
    embed_layer = args.embed_layer
    if args.artifact:
        artifact = DirectionArtifact.load(args.artifact)
        if embed_layer is None:
            embed_layer = artifact.embed_layer
    if embed_layer is None:
        embed_layer = lm.n_layers // 2

    pairs = [(row["prompt"], row["text"]) for row in rows]
    embeddings = response_activations(
        lm, pairs, [embed_layer], batch_size=args.batch_size, max_length=args.max_length
    )[embed_layer]
    per_sample = {i: embeddings[i] for i in range(embeddings.shape[0])}

    with gpu_clock(lm.device) as stats:
        summary = aggregate_by_cell(rows, per_sample)

    if args.mauve_reference:
        reference = [row["text"] for row in read_jsonl(args.mauve_reference)]
        for cell in summary:
            texts = [rows[i]["text"] for i, row in enumerate(rows) if row.get("cell") == cell]
            summary[cell].update(mauve(texts, reference))

    payload = {
        "results": args.results,
        "model": lm.name,
        "embed_layer": embed_layer,
        "n_samples": len(rows),
        "gpu_stats": stats,
        "cells": summary,
    }
    write_json(args.out, payload)
    ordered = sorted(summary.items(), key=lambda kv: kv[1].get("vendi", 0), reverse=True)
    for cell, entry in ordered[:15]:
        print(
            f"  {cell:<28s} vendi={entry.get('vendi', 0):6.3f} "
            f"distinct2={entry.get('distinct_2', 0):.3f} "
            f"clusters={entry.get('n_clusters', 0):.2f} "
            f"completion={entry.get('completion_rate', 0):.2f}"
        )


def cmd_merge(args) -> None:
    seen = set()
    merged = []
    for path in args.inputs:
        for row in read_jsonl(path):
            key = (
                row.get("model"),
                row.get("cell"),
                row.get("prompt_id"),
                row.get("sample_id"),
                row.get("condition"),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    write_jsonl(args.out, merged, append=False)
    print(f"merged {len(merged)} unique samples from {len(args.inputs)} files -> {args.out}")


def cmd_lens(args) -> None:
    lm = _load_model(args)
    layers = resolve_layers(args.layers, lm.n_layers)
    report = logit_lens(lm, args.text, layers, top_k=args.top_k)
    for layer in layers:
        entry = report[layer]
        tokens = ", ".join(f"{t['token']!r}({t['logprob']:.2f})" for t in entry["top_tokens"])
        print(f"layer {layer:>2d}  H={entry['entropy']:.3f}  {tokens}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divspace",
        description="Locate and stabilise the latent subspace governing output diversity.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="measure throughput and peak VRAM")
    _add_model_args(probe)
    probe.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    probe.add_argument("--max-new-tokens", type=int, default=128)
    probe.add_argument("--probe-prompt", default="Write a short paragraph about the sea.")
    probe.add_argument("--no-chat-template", action="store_true")
    probe.add_argument("--out", default="results/probe.json")
    probe.set_defaults(func=cmd_probe)

    sample_cmd = sub.add_parser("sample", help="baseline sampling")
    _add_model_args(sample_cmd)
    _add_prompt_args(sample_cmd)
    _add_shard_args(sample_cmd)
    sample_cmd.add_argument("--n-samples", type=int, default=8)
    sample_cmd.add_argument("--temperature", type=float, default=0.9)
    sample_cmd.add_argument("--top-p", type=float, default=0.95)
    sample_cmd.add_argument("--max-new-tokens", type=int, default=96)
    sample_cmd.add_argument("--batch-size", type=int, default=8)
    sample_cmd.add_argument("--seed", type=int, default=0)
    sample_cmd.add_argument("--out", required=True)
    sample_cmd.set_defaults(func=cmd_sample)

    extract = sub.add_parser("extract", help="fit diversity directions")
    _add_model_args(extract)
    _add_prompt_args(extract)
    _add_shard_args(extract)
    extract.add_argument("--layers", default="middle")
    extract.add_argument("--embed-layer", type=int, default=None)
    extract.add_argument("--embed-layer-frac", type=float, default=0.5)
    extract.add_argument("--pool", default="mean", choices=["mean", "last"])
    extract.add_argument("--n-samples", type=int, default=16)
    extract.add_argument("--temperature", type=float, default=0.9)
    extract.add_argument("--top-p", type=float, default=0.95)
    extract.add_argument("--max-new-tokens", type=int, default=96)
    extract.add_argument("--hi-quantile", type=float, default=0.8)
    extract.add_argument("--lo-quantile", type=float, default=0.2)
    extract.add_argument("--min-samples", type=int, default=8)
    extract.add_argument("--raw-direction", action="store_true")
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--max-length", type=int, default=768)
    extract.add_argument("--seed", type=int, default=0)
    extract.add_argument("--out", required=True)
    extract.set_defaults(func=cmd_extract)

    steer = sub.add_parser("steer", help="layer x coefficient sweep")
    _add_model_args(steer)
    _add_prompt_args(steer)
    _add_shard_args(steer)
    steer.add_argument("--artifact", required=True)
    steer.add_argument("--layers", default=None)
    steer.add_argument("--coefs", type=float, nargs="+", default=[-2.0, -1.0, 1.0, 2.0])
    steer.add_argument("--positions", default="all")
    steer.add_argument("--control", default="none", choices=["none", "random", "shuffled", "orthogonal"])
    steer.add_argument("--condition", default="steer")
    steer.add_argument("--mode", default="inject", choices=["inject", "ablate"])
    steer.add_argument("--include-baseline", action="store_true")
    steer.add_argument("--n-samples", type=int, default=8)
    steer.add_argument("--temperature", type=float, default=0.9)
    steer.add_argument("--top-p", type=float, default=0.95)
    steer.add_argument("--max-new-tokens", type=int, default=96)
    steer.add_argument("--batch-size", type=int, default=8)
    steer.add_argument("--seed", type=int, default=0)
    steer.add_argument("--verbose", action="store_true")
    steer.add_argument("--out", required=True)
    steer.set_defaults(func=cmd_steer)

    attribute = sub.add_parser("attribute", help="attribution patching over layers")
    _add_model_args(attribute)
    attribute.add_argument("--artifact", required=True)
    attribute.add_argument("--layers", default=None)
    attribute.add_argument("--n-pairs", type=int, default=32)
    attribute.add_argument("--top-k", type=int, default=20)
    attribute.add_argument("--max-length", type=int, default=512)
    attribute.add_argument("--out", default="results/attribution.json")
    attribute.set_defaults(func=cmd_attribute)

    metrics_cmd = sub.add_parser("metrics", help="score a results file")
    _add_model_args(metrics_cmd)
    metrics_cmd.add_argument("--results", required=True)
    metrics_cmd.add_argument("--artifact", default=None)
    metrics_cmd.add_argument("--embed-layer", type=int, default=None)
    metrics_cmd.add_argument("--max-per-group", type=int, default=16)
    metrics_cmd.add_argument("--batch-size", type=int, default=8)
    metrics_cmd.add_argument("--max-length", type=int, default=768)
    metrics_cmd.add_argument("--mauve-reference", default=None)
    metrics_cmd.add_argument("--out", default="results/metrics.json")
    metrics_cmd.set_defaults(func=cmd_metrics)

    merge = sub.add_parser("merge", help="concatenate shard outputs, dropping duplicates")
    merge.add_argument("inputs", nargs="+")
    merge.add_argument("--out", required=True)
    merge.set_defaults(func=cmd_merge)

    lens = sub.add_parser("lens", help="logit lens on a single string")
    _add_model_args(lens)
    lens.add_argument("--text", required=True)
    lens.add_argument("--layers", default="middle")
    lens.add_argument("--top-k", type=int, default=5)
    lens.set_defaults(func=cmd_lens)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    Path("results").mkdir(exist_ok=True)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
