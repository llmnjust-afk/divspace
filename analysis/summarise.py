"""Build a human-readable summary of an experiment directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEY_METRICS = [
    "vendi",
    "pc1_ratio",
    "participation_ratio",
    "distinct_2",
    "self_bleu",
    "completion_rate",
    "mean_words",
]


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def best_cell(cells: dict, baseline: dict | None) -> tuple[str, dict]:
    """Pick the cell with the highest Vendi score that does not lose quality."""
    floor = 0.0
    if baseline is not None:
        floor = 0.9 * float(baseline.get("completion_rate", 0.0))
    candidates = [(name, entry) for name, entry in cells.items() if name != "baseline"]
    eligible = [
        (name, entry)
        for name, entry in candidates
        if float(entry.get("completion_rate", 0.0)) >= floor
    ]
    pool = eligible or candidates or list(cells.items())
    return max(pool, key=lambda kv: float(kv[1].get("vendi", 0.0)))


def format_row(values: list) -> str:
    return "| " + " | ".join(str(v) for v in values) + " |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/SUMMARY.md")
    args = parser.parse_args()

    root = Path(args.results)
    lines = ["# divspace experiment summary", ""]

    manifest = load_json(root / "manifest.json")
    if manifest:
        lines += [
            f"- commit: `{manifest.get('git_commit', 'unknown')}`",
            f"- host: `{manifest.get('hostname', 'unknown')}`",
            f"- gpus: {manifest.get('gpus', 'unknown')}",
            f"- generated: {manifest.get('timestamp', 'unknown')}",
            "",
        ]

    rows = [format_row(["model", "condition", "cell", *KEY_METRICS])]
    rows.append(format_row(["---"] * (3 + len(KEY_METRICS))))

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for metrics_path in sorted(model_dir.glob("metrics_*.json")):
            payload = load_json(metrics_path)
            if not payload or "cells" not in payload:
                continue
            cells = payload["cells"]
            baseline = cells.get("baseline")
            if baseline:
                rows.append(
                    format_row(
                        [
                            model_dir.name,
                            metrics_path.stem.replace("metrics_", ""),
                            "baseline",
                            *[f"{float(baseline.get(k, 0.0)):.3f}" for k in KEY_METRICS],
                        ]
                    )
                )
            name, entry = best_cell(cells, baseline)
            rows.append(
                format_row(
                    [
                        model_dir.name,
                        metrics_path.stem.replace("metrics_", ""),
                        name,
                        *[f"{float(entry.get(k, 0.0)):.3f}" for k in KEY_METRICS],
                    ]
                )
            )

    lines += ["## Diversity at the best intervention per file", "", *rows, ""]

    attribution_rows = [format_row(["model", "top layers (sign, magnitude)"])]
    attribution_rows.append(format_row(["---", "---"]))
    for path in sorted(root.glob("*/attribution.json")):
        payload = load_json(path)
        if not payload or "mean_attribution" not in payload:
            continue
        ranked = sorted(
            payload["mean_attribution"].items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:3]
        attribution_rows.append(
            format_row(
                [
                    path.parent.name,
                    ", ".join(f"L{layer} ({value:+.3f})" for layer, value in ranked),
                ]
            )
        )
    lines += ["## Attribution", "", *attribution_rows, ""]

    stability_rows = [format_row(["model", "layer", "split-half cosine"])]
    stability_rows.append(format_row(["---", "---", "---"]))
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        summary = load_json(model_dir / "directions.json")
        if not summary:
            continue
        for layer, entry in sorted(summary.get("stability", {}).items()):
            stability_rows.append(
                format_row([model_dir.name, layer, f"{entry.get('cosine', float('nan')):.3f}"])
            )
    if len(stability_rows) > 2:
        lines += ["## Direction stability", "", *stability_rows, ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
