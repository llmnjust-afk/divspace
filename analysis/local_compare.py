"""Prompt-local steering compared against its two controls within the same run."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

CELL = re.compile(r"^(?P<kind>[a-z]+)_L(?P<layer>\d+)_c(?P<coef>[+-]\d+\.\d+)$")
METRICS = ["vendi", "pc1_ratio", "distinct_2"]


def by_coef(cells: dict) -> dict[str, dict[float, dict]]:
    out: dict[str, dict[float, dict]] = {}
    for name, value in cells.items():
        match = CELL.match(name)
        if match:
            out.setdefault(match["kind"], {})[float(match["coef"])] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/LOCAL.md")
    args = parser.parse_args()

    lines = ["# Prompt-local steering vs controls", ""]
    lines.append(
        "Each prompt is steered with a direction fitted from its own samples. Controls are a "
        "random direction and the direction from the previous prompt. Delta = local - control "
        "at the same coefficient; brackets are 95% intervals over coefficients."
    )
    lines.append("")
    header = ["model", "baseline", "coef", "local", "vs random", "vs other prompt"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for path in sorted(Path(args.results).glob("*/metrics_local.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells = payload["cells"]
        table = by_coef(cells)
        if "local" not in table:
            continue
        baseline = cells.get("baseline", {}).get("vendi")
        for coef in sorted(table["local"]):
            local = table["local"][coef]
            row = [path.parent.name, f"{baseline:.2f}" if baseline else "-", f"{coef:+.1f}",
                   f"{local['vendi']:.3f}"]
            for control in ("localrandom", "localother"):
                other = table.get(control, {}).get(coef)
                if other is None:
                    row.append("-")
                    continue
                delta = local["vendi"] - other["vendi"]
                se = math.sqrt(
                    local.get("vendi_stderr", 0.0) ** 2 + other.get("vendi_stderr", 0.0) ** 2
                )
                row.append(f"{delta:+.3f} [{delta - 1.96 * se:+.3f},{delta + 1.96 * se:+.3f}]")
            lines.append("| " + " | ".join(row) + " |")

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
