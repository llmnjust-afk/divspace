"""Compare real directions against placebo controls at matched (layer, coefficient) cells."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

CELL = re.compile(r"^(?P<kind>[a-z_]+)_L(?P<layer>\d+)_c(?P<coef>[+-]\d+\.\d+)$")
METRICS = ["vendi", "pc1_ratio", "distinct_2", "participation_ratio"]
CONTROLS = ["random", "shuffled", "orthogonal"]

SETTINGS = {
    "inject T=0.9": ("metrics_sweep.json", "metrics_control_{}.json"),
    "ablate": ("metrics_ablate.json", "metrics_ablate_{}.json"),
    "inject low temperature": ("metrics_sweep_lt.json", "metrics_lt_control_{}.json"),
}


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def parse(cell: str) -> tuple[int, float] | None:
    match = CELL.match(cell)
    if not match:
        return None
    return int(match["layer"]), float(match["coef"])


def cell_table(payload: dict) -> dict[tuple[int, float], dict]:
    return {
        key: value
        for name, value in payload["cells"].items()
        if (key := parse(name)) is not None
    }


def combine(a: dict, b: dict, metric: str) -> tuple[float, float]:
    """Difference of means and its standard error, treating the two cells as independent."""
    var = a.get(f"{metric}_stderr", 0.0) ** 2 + b.get(f"{metric}_stderr", 0.0) ** 2
    return a[metric] - b[metric], math.sqrt(var)


def compare(model_dir: Path, sweep_file: str, control_pattern: str) -> dict | None:
    sweep = load(model_dir / sweep_file)
    if not sweep or "cells" not in sweep:
        return None
    treated = cell_table(sweep)
    baseline = sweep["cells"].get("baseline", {})
    out = {"n_treated_cells": len(treated), "baseline": baseline.get("vendi"), "vs": {}}

    for control in CONTROLS:
        payload = load(model_dir / control_pattern.format(control))
        if not payload:
            continue
        placebo = cell_table(payload)
        shared = sorted(set(placebo) & set(treated))
        if not shared:
            continue
        entry = {"n_cells": len(shared)}
        for metric in METRICS:
            diffs = [combine(treated[key], placebo[key], metric) for key in shared]
            mean = sum(d for d, _ in diffs) / len(diffs)
            se = math.sqrt(sum(s * s for _, s in diffs)) / len(diffs)
            entry[metric] = {"delta": mean, "ci95": [mean - 1.96 * se, mean + 1.96 * se]}
        out["vs"][control] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/CONTROLS.md")
    args = parser.parse_args()

    lines = ["# Real direction vs placebo controls", ""]
    lines.append(
        "Delta = mean(treated cell) - mean(placebo cell) over matched (layer, coefficient) "
        "cells; positive means the real direction produced more diversity than the placebo. "
        "Brackets are 95% intervals over the matched cells."
    )
    everything = {}

    for setting, (sweep_file, pattern) in SETTINGS.items():
        rows = []
        for model_dir in sorted(p for p in Path(args.results).iterdir() if p.is_dir()):
            item = compare(model_dir, sweep_file, pattern)
            if item and item["vs"]:
                rows.append({"model": model_dir.name, **item})
                everything.setdefault(setting, []).append({"model": model_dir.name, **item})
        if not rows:
            continue
        lines += ["", f"## {setting}", ""]
        header = ["model", "baseline vendi"] + [f"vs {c}" for c in CONTROLS]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows:
            cells = [row["model"], f"{row['baseline']:.3f}" if row["baseline"] else "-"]
            for control in CONTROLS:
                entry = row["vs"].get(control)
                if entry is None:
                    cells.append("-")
                else:
                    delta = entry["vendi"]
                    lo, hi = delta["ci95"]
                    cells.append(f"{delta['delta']:+.3f} [{lo:+.3f},{hi:+.3f}]")
            lines.append("| " + " | ".join(cells) + " |")

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(everything, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
