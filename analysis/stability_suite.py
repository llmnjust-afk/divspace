"""Is the within-prompt null a property of prompt mixing or of the sample budget?

Reads the ``stability_suite.json`` files written by ``stability-suite`` and reports,
for every model: the mean within-prompt split-half cosine, the pooled cosine as the
number of pooled prompts grows at a fixed per-arm budget, and the cosine between
the pooled direction and a direction fitted from prompt-level mean dispersion.
A pooled cosine that is already high at ``m = 1`` and barely moves as ``m`` grows
means budget, not prompt mixing, explains the contrast; a curve that rises with
``m`` means prompt identity does.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path

from scipy import stats


def t_interval(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        return (math.nan, math.nan)
    mean = statistics.mean(values)
    half = float(stats.t.ppf(0.975, n - 1)) * statistics.stdev(values) / math.sqrt(n)
    return (mean - half, mean + half)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/STABILITY_SUITE.md")
    args = parser.parse_args()

    lines = ["# Within-prompt, budget-matched and between-prompt stability", ""]
    lines.append(
        "`within` is the mean split-half cosine over prompts when both halves come from "
        "one prompt. `pooled(m)` is the same statistic when the halves draw on `m` prompts "
        "at an identical per-arm budget, so `m = 1` is the within-prompt case and the rise "
        "along `m` isolates prompt mixing from sample size. `between` is the cosine between "
        "the pooled direction and a direction fitted from prompt-level mean dispersion."
    )
    lines.append("")

    paths = sorted(glob.glob(str(Path(args.results) / "*" / "stability_suite.json")))
    if not paths:
        lines.append("_no stability_suite.json found_")
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.out} (empty)")
        return

    sizes: list[str] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        sizes = sizes or [key for key in payload.get("budget", {})]

    header = ["model", "layer", "within", "95% CI"] + [f"pooled m={s}" for s in sizes] + ["between"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    rows = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        layer = str(payload.get("primary_layer"))
        within = payload.get("within", {}).get(layer, [])
        low, high = t_interval(within)
        curve = {
            size: statistics.mean(values) if values else math.nan
            for size, values in payload.get("budget", {}).items()
        }
        between = payload.get("between", {}).get(layer, math.nan)
        row = [
            payload.get("model", Path(path).parent.name),
            layer,
            f"{statistics.mean(within):+.3f}" if within else "-",
            f"[{low:+.3f}, {high:+.3f}]" if within else "-",
        ]
        row += [f"{curve.get(size, math.nan):+.3f}" for size in sizes]
        row.append(f"{between:+.3f}")
        lines.append("| " + " | ".join(row) + " |")
        if within and payload.get("budget"):
            biggest = max(payload["budget"])
            pooled = payload["budget"][biggest]
            if pooled:
                rows.append((statistics.mean(within), statistics.mean(pooled)))

    lines.append("")
    if rows:
        lines.append("## Within-prompt versus fully pooled")
        lines.append("")
        lines.append("| model comparison | mean within | mean pooled (largest m) |")
        lines.append("| --- | --- | --- |")
        for index, (within_mean, pooled_mean) in enumerate(rows):
            lines.append(f"| model {index} | {within_mean:+.3f} | {pooled_mean:+.3f} |")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
