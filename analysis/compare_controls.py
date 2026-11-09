"""Compare real steering against the placebo controls at matched (layer, coefficient) cells."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

CELL = re.compile(r"^(?P<kind>[a-z]+)_L(?P<layer>\d+)_c(?P<coef>[+-]\d+\.\d+)$")
METRICS = ["vendi", "pc1_ratio", "distinct_2", "participation_ratio"]
CONTROLS = ["random", "shuffled", "orthogonal"]


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


def cell_table(payload: dict, kind: str) -> dict[tuple[int, float], dict]:
    return {
        key: value
        for name, value in payload["cells"].items()
        if (key := parse(name)) is not None
    }


def combine(a: dict, b: dict, metric: str) -> tuple[float, float]:
    """Difference of means and its standard error, treating the two cells as independent."""
    mean = a[metric] - b[metric]
    var = a.get(f"{metric}_stderr", 0.0) ** 2 + b.get(f"{metric}_stderr", 0.0) ** 2
    return mean, math.sqrt(var)


def summarise(model_dir: Path) -> dict | None:
    sweep = load(model_dir / "metrics_sweep.json")
    if not sweep or "baseline" not in sweep["cells"]:
        return None
    steer = cell_table(sweep, "steer")
    baseline = sweep["cells"]["baseline"]
    out = {"model": model_dir.name, "n_steer_cells": len(steer), "baseline": {}, "deltas": {}}

    for metric in METRICS:
        out["baseline"][metric] = baseline[metric]

    for control in CONTROLS:
        payload = load(model_dir / f"metrics_control_{control}.json")
        if not payload:
            continue
        cells = cell_table(payload, control)
        shared = sorted(set(cells) & set(steer))
        if not shared:
            continue
        for metric in METRICS:
            diffs = [combine(steer[key], cells[key], metric) for key in shared]
            mean = sum(d for d, _ in diffs) / len(diffs)
            se = math.sqrt(sum(s * s for _, s in diffs)) / len(diffs)
            out["deltas"].setdefault(metric, {})[control] = {
                "delta": mean,
                "se": se,
                "ci95": [mean - 1.96 * se, mean + 1.96 * se],
                "n_cells": len(shared),
            }

    ablate = load(model_dir / "metrics_ablate.json")
    if ablate:
        out["deltas"].setdefault("vendi", {})["ablate"] = {
            "delta": (
                sum(c["vendi"] for c in ablate["cells"].values()) / len(ablate["cells"])
                - baseline["vendi"]
            ),
            "se": 0.0,
            "ci95": [0.0, 0.0],
            "n_cells": len(ablate["cells"]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/CONTROLS.md")
    args = parser.parse_args()

    rows = []
    for model_dir in sorted(p for p in Path(args.results).iterdir() if p.is_dir()):
        item = summarise(model_dir)
        if item:
            rows.append(item)

    lines = ["# Real direction vs placebo controls", ""]
    lines.append(
        "Delta = mean(steer cell) - mean(control cell) over matched (layer, coefficient) cells; "
        "positive delta means the real direction produced *more* diversity than the placebo."
    )
    lines.append("")
    header = ["model", "baseline vendi", "vs random", "vs shuffled", "vs orthogonal", "ablate"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for item in rows:
        cells = [item["model"], f"{item['baseline']['vendi']:.3f}"]
        for control in CONTROLS:
            entry = item["deltas"].get("vendi", {}).get(control)
            if entry is None:
                cells.append("-")
            else:
                lo, hi = entry["ci95"]
                cells.append(f"{entry['delta']:+.3f} [{lo:+.3f},{hi:+.3f}]")
        ablate = item["deltas"].get("vendi", {}).get("ablate")
        cells.append(f"{ablate['delta']:+.3f}" if ablate else "-")
        lines.append("| " + " | ".join(cells) + " |")

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
