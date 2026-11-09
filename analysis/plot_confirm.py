"""Four-panel figure for the confirmatory evidence."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
PAT = re.compile(r"_c([+-]\d+\.\d+)$")
SHORT = {
    "EleutherAI_pythia-410m": "pythia-410m",
    "EleutherAI_pythia-1b": "pythia-1b",
    "EleutherAI_pythia-2.8b": "pythia-2.8b",
    "Qwen_Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "Qwen_Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "allenai_OLMo-2-1124-7B-Instruct": "OLMo2-7B",
    "HuggingFaceTB_SmolLM2-1.7B-Instruct": "SmolLM2-1.7B",
}


def load_cells(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["cells"]


def series(cells: dict, prefix: str) -> tuple[list[float], list[float], list[float]]:
    xs, ys, es = [], [], []
    for name, cell in cells.items():
        match = PAT.search(name)
        if match and name.startswith(prefix):
            xs.append(float(match.group(1)))
            ys.append(cell["vendi"])
            es.append(cell.get("vendi_stderr", 0.0))
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order], [es[i] for i in order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures/confirm")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))

    ax = axes[0][0]
    for index, path in enumerate(sorted(glob.glob(f"{args.results}/*/yesno_cal.json"))):
        slug = Path(path).parent.name
        recs = json.loads(Path(path).read_text(encoding="utf-8"))
        base = next((r["yes_rate"] for r in recs if r["cell"] == "baseline"), None)
        if base is None:
            continue
        points = []
        for rec in recs:
            match = PAT.search(rec["cell"])
            if match:
                points.append((float(match.group(1)), rec["yes_rate"] - base, rec["n"]))
        if not points:
            continue
        points.sort()
        xs = [0.0] + [p[0] for p in points]
        ys = [0.0] + [p[1] for p in points]
        ax.plot(xs, ys, marker="o", color=OKABE[index % len(OKABE)],
                label=SHORT.get(slug, slug), linewidth=1.6, markersize=4)
    ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
    ax.set_title("(a) Labelled yes/no control responds to dose", fontsize=10)
    ax.set_xlabel("steering coefficient")
    ax.set_ylabel("yes-rate minus baseline")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[0][1]
    for index, path in enumerate(sorted(glob.glob(f"{args.results}/*/directions.json"))):
        slug = Path(path).parent.name
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stability = payload.get("stability", {})
        if not stability:
            continue
        layers = sorted(stability, key=int)
        ax.plot(range(len(layers)), [stability[k]["cosine"] for k in layers],
                marker="o", color=OKABE[index % len(OKABE)], label=SHORT.get(slug, slug),
                linewidth=1.6, markersize=4)
    ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
    ax.set_title("(b) Split-half stability of the fitted direction", fontsize=10)
    ax.set_xlabel("layer index (low to high)")
    ax.set_ylabel("cosine between halves")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1][0]
    for index, path in enumerate(sorted(glob.glob(f"{args.results}/*/metrics_sweep_cal.json"))):
        slug = Path(path).parent.name
        cells = load_cells(path)
        base = cells.get("baseline", {})
        xs, ys, es = series(cells, "steer_")
        if not xs:
            continue
        ax.errorbar(xs, [y - base.get("vendi", 0.0) for y in ys], yerr=es, marker="o",
                    capsize=2, color=OKABE[index % len(OKABE)], label=SHORT.get(slug, slug),
                    linewidth=1.4, markersize=4)
    ax.axhline(0.0, color="0.3", linewidth=0.9, linestyle="--")
    ax.set_title("(c) Fitted diversity direction: no dose-response", fontsize=10)
    ax.set_xlabel("steering coefficient")
    ax.set_ylabel("Vendi minus baseline")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1][1]
    for index, path in enumerate(sorted(glob.glob(f"{args.results}/*/metrics_sweep_cal_random.json"))):
        slug = Path(path).parent.name
        cells = load_cells(path)
        base = cells.get("baseline", {})
        xs, ys, es = series(cells, "randcal_")
        if not xs:
            continue
        ax.errorbar(xs, [y - base.get("vendi", 0.0) for y in ys], yerr=es, marker="s",
                    capsize=2, color=OKABE[index % len(OKABE)], label=SHORT.get(slug, slug),
                    linewidth=1.4, markersize=4)
    ax.axhline(0.0, color="0.3", linewidth=0.9, linestyle="--")
    ax.set_title("(d) Random directions move diversity, fitted ones do not", fontsize=10)
    ax.set_xlabel("steering coefficient")
    ax.set_ylabel("Vendi minus baseline")
    ax.legend(fontsize=7, frameon=False)

    for row in axes:
        for axis in row:
            axis.tick_params(labelsize=8)
            axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(f"{args.out}.{suffix}", dpi=200)
    print(f"wrote {args.out}.pdf and .png")


if __name__ == "__main__":
    main()
