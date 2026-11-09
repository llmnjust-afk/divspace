"""Dose-response and directional-specificity tests on the calibrated sweeps.

Two questions, both asked per model and then combined across models:
  1. does the effect grow with the coefficient (Spearman rho per model)?
  2. at a matched coefficient, does the fitted direction beat a random vector
     (paired difference, sign test across models)?
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics
from pathlib import Path

from scipy import stats

PAT = re.compile(r"_c([+-]\d+\.\d+)$")


def cells_by_coef(path: str) -> tuple[float, dict[float, tuple[float, float]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
    base = payload.get("baseline")
    out = {}
    for name, cell in payload.items():
        match = PAT.search(name)
        if match:
            out[float(match.group(1))] = (
                cell["vendi"],
                cell.get("vendi_stderr", 0.0),
            )
    return (base["vendi"] if base else math.nan), out


def delta(entry: tuple[float, float], base: float) -> float:
    return entry[0] - base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/DOSE.md")
    args = parser.parse_args()

    lines = ["# Dose-response and specificity", ""]
    lines.append("## Within-model dose-response (Spearman rho between coefficient and dVendi)")
    lines.append("")
    lines.append("| model | arm | n cells | rho | p |")
    lines.append("| --- | --- | --- | --- | --- |")

    rhos = {"real": [], "random": []}
    paired = []
    for path in sorted(glob.glob(f"{args.results}/*/metrics_sweep_cal.json")):
        slug = Path(path).parent.name
        base_real, real = cells_by_coef(path)
        random_path = path.replace("metrics_sweep_cal.json", "metrics_sweep_cal_random.json")
        base_rand, rand = cells_by_coef(random_path) if Path(random_path).exists() else (math.nan, {})
        for arm, cells, base in (("real", real, base_real), ("random", rand, base_rand)):
            if not cells or math.isnan(base):
                continue
            coefs = sorted(cells)
            values = [delta(cells[c], base) for c in coefs]
            rho, p = stats.spearmanr(coefs, values)
            rhos[arm].append(rho)
            lines.append(f"| {slug} | {arm} | {len(coefs)} | {rho:+.2f} | {p:.3f} |")
        shared = sorted(set(real) & set(rand))
        if shared:
            diffs = [delta(real[c], base_real) - delta(rand[c], base_rand) for c in shared]
            paired.append((slug, sum(diffs) / len(diffs), len(diffs)))

    lines.append("")
    lines.append("## Across-model summary")
    lines.append("")
    for arm, values in rhos.items():
        if len(values) >= 3:
            stat, p = stats.wilcoxon(values) if any(values) else (0.0, 1.0)
            lines.append(
                f"- {arm}: median Spearman rho = {statistics.median(values):+.2f} "
                f"over {len(values)} models, Wilcoxon p = {p:.3f}"
            )
    if rhos["real"]:
        pos = sum(1 for r in rhos["real"] if r > 0)
        lines.append(
            f"- real arm: {pos}/{len(rhos['real'])} models have a positive (diversity-increasing) slope"
        )
    lines.append("")
    lines.append("## Fitted direction minus random direction at matched coefficients")
    lines.append("")
    lines.append("| model | mean dVendi (fitted) - dVendi (random) | n coefs |")
    lines.append("| --- | --- | --- |")
    diffs = []
    for slug, value, n in paired:
        lines.append(f"| {slug} | {value:+.3f} | {n} |")
        diffs.append(value)
    if len(diffs) >= 3:
        stat, p = stats.wilcoxon(diffs)
        lines.append("")
        lines.append(
            f"Across {len(diffs)} models the fitted direction is {sum(diffs)/len(diffs):+.3f} "
            f"Vendi relative to a random vector (Wilcoxon p = {p:.3f})."
        )
    elif diffs:
        lines.append("")
        lines.append(f"Mean difference across {len(diffs)} models: {sum(diffs)/len(diffs):+.3f}.")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
