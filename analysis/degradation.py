"""Separate a real diversity change from generic degradation of the text.

A steering vector that shortens and repeats generations will lower any diversity
metric without telling us anything about diversity control. This script reports
each cell's diversity move next to its length and repetition move, and re-tests
the diversity change on the subset of cells that stay close to the baseline in
both.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

PAT = re.compile(r"_c([+-]\d+\.\d+)$")
QUALITY = ("mean_words", "rep_3", "self_bleu")


def z(delta: float, se_a: float, se_b: float) -> float:
    total = math.sqrt(se_a**2 + se_b**2)
    return delta / total if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--pattern", default="metrics_sweep_cal*.json")
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--out", default="results/DEGRADATION.md")
    args = parser.parse_args()

    lines = ["# Diversity change versus text degradation", ""]
    lines.append(
        "For every cell we report the Vendi change beside the relative change in mean "
        "length and in trigram repetition. Cells that move length or repetition by more "
        f"than {args.tolerance:.0%} are classed as degraded; the last two columns re-test "
        "the Vendi change on the surviving cells only."
    )
    lines.append("")
    header = ["model", "arm", "coef", "dVendi", "z", "dLen", "dRep3", "verdict"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    summary = []
    for path in sorted(glob.glob(str(Path(args.results) / "*" / args.pattern))):
        slug = Path(path).parent.name
        arm = "random" if "random" in path else "real"
        cells = json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
        base = cells.get("baseline")
        if not base:
            continue
        for name, cell in cells.items():
            match = PAT.search(name)
            if not match:
                continue
            coef = float(match.group(1))
            dv = cell["vendi"] - base["vendi"]
            zz = z(dv, cell.get("vendi_stderr", 0.0), base.get("vendi_stderr", 0.0))
            dlen = (cell.get("mean_words", 0.0) - base.get("mean_words", 0.0)) / max(
                base.get("mean_words", 1.0), 1e-6
            )
            drep = (cell.get("rep_3", 0.0) - base.get("rep_3", 0.0)) / max(
                base.get("rep_3", 1e-6), 1e-6
            )
            degraded = abs(dlen) > args.tolerance or abs(drep) > args.tolerance
            verdict = "degraded" if degraded else "clean"
            lines.append(
                "| "
                + " | ".join(
                    [
                        slug,
                        arm,
                        f"{coef:.1f}",
                        f"{dv:+.3f}",
                        f"{zz:+.2f}",
                        f"{dlen:+.0%}",
                        f"{drep:+.0%}",
                        verdict,
                    ]
                )
                + " |"
            )
            summary.append((slug, arm, coef, dv, zz, degraded))

    lines.append("")
    lines.append("## Vendi change on non-degraded cells only")
    lines.append("")
    lines.append("| model | arm | clean cells | mean dVendi | max |z| |")
    lines.append("| --- | --- | --- | --- | --- |")
    groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for slug, arm, coef, dv, zz, degraded in summary:
        if not degraded:
            groups.setdefault((slug, arm), []).append((dv, zz))
    for (slug, arm), values in sorted(groups.items()):
        mean_dv = sum(v for v, _ in values) / len(values)
        max_z = max(abs(zv) for _, zv in values)
        lines.append(f"| {slug} | {arm} | {len(values)} | {mean_dv:+.3f} | {max_z:.2f} |")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
