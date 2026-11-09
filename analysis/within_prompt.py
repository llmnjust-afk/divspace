"""Is the diversity direction reproducible inside a single prompt?

Compares the pooled split-half cosine (between prompts mixed) with the
within-prompt split-half cosine (two disjoint halves of one prompt's samples).
The gap between the two is the paper's central measurement: a stable pooled
direction can be entirely between-prompt identity structure.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

from scipy import stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/WITHIN_PROMPT.md")
    args = parser.parse_args()

    lines = ["# Pooled versus within-prompt stability", ""]
    lines.append(
        "Pooled: cosine between directions fitted on two halves of the whole sample set "
        "(all prompts mixed). Within: cosine between two halves of one prompt's own samples, "
        "averaged over prompts; brackets are 95% t-intervals over prompts."
    )
    lines.append("")
    header = ["model", "pooled (middle layer)", "within-prompt mean", "95% CI",
              "prompts", "cos > 0.5", "t vs 0", "p"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for path in sorted(glob.glob(f"{args.results}/*/within_prompt.jsonl")):
        slug = Path(path).parent.name
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        values = [row["cosine"] for row in rows]
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            stderr = stats.sem(values)
            tstat, pvalue = stats.ttest_1samp(values, 0.0)
            half = 1.96 * stderr
        else:
            half = float("nan")
            tstat, pvalue = float("nan"), float("nan")
        pooled = math.nan
        summary = Path(path).parent / "directions.json"
        if summary.exists():
            payload = json.loads(summary.read_text(encoding="utf-8"))
            stability = payload.get("stability", {})
            layers = payload.get("layers", [])
            if stability and layers:
                middle = str(layers[len(layers) // 2])
                pooled = stability.get(middle, {}).get("cosine", math.nan)
        above = sum(1 for v in values if v > 0.5)
        lines.append(
            f"| {slug} | {pooled:+.3f} | {mean:+.3f} | [{mean - half:+.3f}, {mean + half:+.3f}] | "
            f"{n} | {above}/{n} | {tstat:+.2f} | {pvalue:.3f} |"
        )

    lines.append("")
    lines.append(
        "A direction that could support steering needs a cosine close to 1: only then does a "
        "fit on one sample half predict what the other half measures."
    )
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
