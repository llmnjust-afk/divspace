"""Do per-prompt directions agree with one another?

If the pooled vector is a sum of components that all point somewhere similar, the
per-prompt fits should correlate positively with each other. Values near zero mean
each prompt says something different, so their average is not a summary of a shared
axis. Reads the ``prompt_directions`` block of ``stability_suite.json``.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path


def mean_pairwise_cosine(vectors: list[list[float]]) -> float:
    if len(vectors) < 2:
        return float("nan")
    cosines = []
    for left, right in itertools.combinations(vectors, 2):
        dot = sum(a * b for a, b in zip(left, right))
        norm_left = sum(a * a for a in left) ** 0.5
        norm_right = sum(b * b for b in right) ** 0.5
        if norm_left > 0 and norm_right > 0:
            cosines.append(dot / (norm_left * norm_right))
    return sum(cosines) / len(cosines) if cosines else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/PAIRWISE.md")
    args = parser.parse_args()

    lines = ["# Mean pairwise cosine between per-prompt directions", ""]
    lines.append(
        "One direction is fitted per prompt from that prompt's own samples; the entry is "
        "the mean cosine over all prompt pairs. Values near zero mean the prompts disagree "
        "about which way diversity points."
    )
    lines.append("")

    paths = sorted(glob.glob(str(Path(args.results) / "*" / "stability_suite.json")))
    columns: list[str] = []
    table: list[tuple[str, dict[str, float]]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        directions = payload.get("prompt_directions", {})
        if not directions:
            continue
        columns = columns or sorted(directions, key=int)
        values = {
            layer: mean_pairwise_cosine(directions[layer]) for layer in columns
        }
        table.append((payload.get("model", Path(path).parent.name), values))

    if not table:
        lines.append("_no prompt_directions found_")
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.out} (empty)")
        return

    header = ["model"] + [f"layer {layer}" for layer in columns]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for model, values in table:
        row = [model] + [f"{values[layer]:+.3f}" for layer in columns]
        lines.append("| " + " | ".join(row) + " |")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
