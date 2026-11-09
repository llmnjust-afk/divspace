"""Write a provenance manifest for an experiment run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_inventory() -> list[str]:
    output = _run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"]
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/manifest.json")
    parser.add_argument("--results", default="results")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    root = Path(args.results)
    files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "gpus": gpu_inventory(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "n_files": len(files),
        "total_mb": round(sum(f.stat().st_size for f in files) / 1024**2, 2),
        "note": args.note,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
