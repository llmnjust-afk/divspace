"""Figures for the paper: attribution profile, coefficient response, Pareto frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_attribution(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layers = [int(layer) for layer in payload["mean_attribution"]]
    values = [payload["mean_attribution"][str(layer)] for layer in layers]
    order = sorted(zip(layers, values))
    return {
        "layers": [l for l, _ in order],
        "values": [v for _, v in order],
        "model": payload.get("model", path.parent.name),
    }


def plot_attribution(files: list[Path], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for path in files:
        data = load_attribution(path)
        ax.plot(data["layers"], data["values"], marker="o", markersize=3, label=data["model"])
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("layer")
    ax.set_ylabel("attribution to response log-likelihood")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


def plot_coefficient_response(metrics_path: Path, out: Path, layer: int | None = None) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    points = []
    for cell, entry in payload["cells"].items():
        if not cell.startswith("steer_"):
            continue
        parts = dict(piece.split("L") if piece.startswith("L") else ("coef", piece) for piece in cell.split("_")[1:])
        try:
            cell_layer = int(parts["L"])
            coef = float(parts["coef"])
        except (KeyError, ValueError):
            continue
        if layer is not None and cell_layer != layer:
            continue
        points.append((coef, entry.get("vendi", 0.0), entry.get("completion_rate", 0.0), cell_layer))
    if not points:
        print(f"no steering cells in {metrics_path}")
        return
    points.sort()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label="Vendi")
    ax.plot([p[0] for p in points], [p[2] for p in points], marker="s", label="completion")
    ax.set_xlabel("steering coefficient")
    ax.set_ylabel("score")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


def plot_pareto(metrics_files: list[Path], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for path in metrics_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        xs, ys, labels = [], [], []
        for cell, entry in payload["cells"].items():
            xs.append(entry.get("vendi", 0.0))
            ys.append(entry.get("completion_rate", 0.0))
            labels.append(cell)
        ax.scatter(xs, ys, s=12, alpha=0.7, label=payload.get("model", path.stem))
    ax.set_xlabel("Vendi score (effective number of distinct samples)")
    ax.set_ylabel("completion rate")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution", nargs="*", type=Path, default=[])
    parser.add_argument("--metrics", nargs="*", type=Path, default=[])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    if args.attribution:
        plot_attribution(args.attribution, args.out_dir / "attribution.pdf")
    if args.metrics:
        plot_pareto(args.metrics, args.out_dir / "pareto.pdf")
        plot_coefficient_response(args.metrics[0], args.out_dir / "coefficient_response.pdf", args.layer)


if __name__ == "__main__":
    main()
